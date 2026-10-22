//+------------------------------------------------------------------+
//|                                             NeuroNet_D2Skill.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#ifndef NEURONET_D2SKILL_MQH
#define NEURONET_D2SKILL_MQH
//+------------------------------------------------------------------+
//| Reuse boundary: D2Skill keeps its own flat GPU bank because its  |
//| signed corrections, direction agreement, utility/age eviction,   |
//| candidate promotion and Task/Step dual routing are not           |
//| represented by CScenarioCodebook. RAG score/Top-K and            |
//| persistence conventions are reused where their contracts match;  |
//| CNeuronRAGMemory is intentionally not used because its publish   |
//| path is CPU copy-on-write plus re-upload.                        |
//+------------------------------------------------------------------+
void CD2SkillItem::RebindItemBuffer(CBufferFloat &buffer, COpenCLMy *obj)
  {
   if(!obj)
      return;
   if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
      buffer.BufferFree();
   if(buffer.Total() > 0)
      buffer.BufferCreate(obj);
  }
//+------------------------------------------------------------------+
//| Constructor: initializes the correction pointer, representation  |
//| mode and EMA smoothing defaults.                                 |
//+------------------------------------------------------------------+
void CD2SkillItem::CD2SkillItem(void) : m_correction(NULL),
   m_representation(D2SkillDirectionMagnitude),
   m_beta_correction(0.05f), m_beta_direction(0.05f),
   m_beta_magnitude(0.05f), m_item_dimension(0)
  {
  }
//+------------------------------------------------------------------+
//| Destructor.                                                      |
//+------------------------------------------------------------------+
void CD2SkillItem::~CD2SkillItem(void)
  {
  }
//+------------------------------------------------------------------+
//| Initializes the item and allocates its GPU-resident correction,  |
//| direction, scale, observation and mass buffers.                  |
//+------------------------------------------------------------------+
bool CD2SkillItem::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint numNeurons, ENUM_OPTIMIZATION optimization_type,
                        uint batch)
  {
   m_correction = NULL;
   m_item_dimension = numNeurons;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons,
                            optimization_type, batch))
      ReturnFalse;
   return(EnsureItemState());
  }
//+------------------------------------------------------------------+
//| Attaches the external correction buffer when its layout matches  |
//| the item output.                                                 |
//+------------------------------------------------------------------+
bool CD2SkillItem::SetCorrectionBuffer(CBufferFloat *correction)
  {
   if(!correction || CheckPointer(correction) == POINTER_INVALID)
     {
      m_correction = NULL;
      return(true);
     }
   if(!Output || correction.Total() != Output.Total() ||
      correction.GetOpenCL() != OpenCL)
      ReturnFalse;
   m_correction = correction;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the attached correction buffer.                          |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::GetCorrectionBuffer(void) const
  {
   return(m_correction);
  }
//+------------------------------------------------------------------+
//| Sets the representation mode of the item.                        |
//+------------------------------------------------------------------+
void CD2SkillItem::SetRepresentation(const ED2SkillRepresentation mode)
  {
   m_representation = mode;
  }
//+------------------------------------------------------------------+
//| Returns the current representation mode.                         |
//+------------------------------------------------------------------+
ED2SkillRepresentation CD2SkillItem::Representation(void) const
  {
   return(m_representation);
  }
//+------------------------------------------------------------------+
//| Sets the EMA smoothing factors for correction, direction and     |
//| magnitude, clamped to [0,1].                                     |
//+------------------------------------------------------------------+
void CD2SkillItem::SetEMA(const float beta_correction, const float beta_direction,
                          const float beta_magnitude)
  {
   m_beta_correction = MathMin(MathMax(beta_correction, 0.0f), 1.0f);
   m_beta_direction = MathMin(MathMax(beta_direction, 0.0f), 1.0f);
   m_beta_magnitude = MathMin(MathMax(beta_magnitude, 0.0f), 1.0f);
  }
//+------------------------------------------------------------------+
//| Returns the correction EMA smoothing factor.                     |
//+------------------------------------------------------------------+
float CD2SkillItem::BetaCorrection(void) const
  {
   return(m_beta_correction);
  }
//+------------------------------------------------------------------+
//| Returns the direction EMA smoothing factor.                      |
//+------------------------------------------------------------------+
float CD2SkillItem::BetaDirection(void) const
  {
   return(m_beta_direction);
  }
//+------------------------------------------------------------------+
//| Returns the magnitude EMA smoothing factor.                      |
//+------------------------------------------------------------------+
float CD2SkillItem::BetaMagnitude(void) const
  {
   return(m_beta_magnitude);
  }
//+------------------------------------------------------------------+
//| Returns the item correction buffer pointer.                      |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Correction(void)
  {
   return(m_correction);
  }
//+------------------------------------------------------------------+
//| Returns the item direction buffer.                               |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Direction(void)
  {
   return(GetPointer(m_item_direction));
  }
//+------------------------------------------------------------------+
//| Returns the item scale buffer.                                   |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Scale(void)
  {
   return(GetPointer(m_item_scale));
  }
//+------------------------------------------------------------------+
//| Returns the item observation counter buffer.                     |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Observations(void)
  {
   return(GetPointer(m_item_observations));
  }
//+------------------------------------------------------------------+
//| Returns the item mass buffer.                                    |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Mass(void)
  {
   return(GetPointer(m_item_mass));
  }
//+------------------------------------------------------------------+
//| No-op: a skill item exposes no trainable weights.                |
//+------------------------------------------------------------------+
bool CD2SkillItem::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the item neuron type identifier (defNeuronD2SkillItem).  |
//+------------------------------------------------------------------+
int CD2SkillItem::Type(void) const
  {
   return(defNeuronD2SkillItem);
  }
//+------------------------------------------------------------------+
//| Rebinds a bank GPU buffer to the given OpenCL object.            |
//+------------------------------------------------------------------+
void CD2SkillBank::RebindBuffer(CBufferFloat &buffer, COpenCLMy *obj)
  {
   if(!obj)
      return;
   if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
      buffer.BufferFree();
   buffer.BufferCreate(obj);
  }
//+------------------------------------------------------------------+
//| Checks that every bank buffer matches the required               |
//| slot-by-dimension layout.                                        |
//+------------------------------------------------------------------+
bool CD2SkillBank::ValidateBuffers(void)
  {
   const long total = (long)m_slots * (long)m_dimension;
   if(total <= 0 || total > 2147483647 ||
      !Output || !Gradient || Output.Total() != (int)m_dimension ||
      Gradient.Total() != (int)m_dimension)
      ReturnFalse;
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
          m_influence.Total() == (int)m_slots &&
          m_utility_applied.Total() == (int)m_slots &&
          m_gradient_influence_total.Total() == 1 &&
          m_retrieved.Total() == (int)m_slots &&
          m_selected_correction.Total() == (int)m_dimension &&
          m_selected_slot.Total() == 1 &&
          m_selected_score.Total() == 1 &&
          m_diagnostics.Total() == D2SKILL_DIAGNOSTICS &&
          m_utility_distribution.Total() == D2SKILL_DISTRIBUTION_BINS &&
          m_usage_distribution.Total() == D2SKILL_DISTRIBUTION_BINS);
  }
//+------------------------------------------------------------------+
//| Initializes a bank GPU buffer with the given total and fill      |
//| value.                                                           |
//+------------------------------------------------------------------+
bool CD2SkillBank::InitBuffer(CBufferFloat &buffer, const int total, const float value)
  {
   return(buffer.BufferInit(total, value) && buffer.BufferCreate(OpenCL));
  }
//+------------------------------------------------------------------+
//| Constructor: initializes thresholds, EMAs, lifecycle and policy  |
//| defaults.                                                        |
//+------------------------------------------------------------------+
CD2SkillBank::CD2SkillBank(void) : m_slots(0), m_dimension(0), m_enabled(false),
   m_similarity_threshold(0.65f), m_direction_threshold(0.0f),
   m_utility_weight(0.05f), m_beta_key(0.05f),
   m_beta_utility(0.05f), m_alpha(1.0f),
   m_min_utility(-1.0f), m_utility_aware(false),
   m_max_correction(10.0f), m_min_confirmations(3),
   m_protection_age(32), m_inactivity_age(256),
   m_active_slot(UINT_MAX)
  {
  }
//+------------------------------------------------------------------+
//| Destructor (CD2SkillBank).                                       |
//+------------------------------------------------------------------+
CD2SkillBank::~CD2SkillBank(void)
  {
  }
//+------------------------------------------------------------------+
//| Returns the bank neuron type identifier (defNeuronD2SkillBank).  |
//+------------------------------------------------------------------+
int CD2SkillBank::Type(void) const
  {
   return(defNeuronD2SkillBank);
  }
//+------------------------------------------------------------------+
//| Enables or disables the bank.                                    |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetEnabled(const bool enabled)
  {
   m_enabled = enabled;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns whether the bank is enabled.                             |
//+------------------------------------------------------------------+
bool CD2SkillBank::Enabled(void) const
  {
   return(m_enabled);
  }
//+------------------------------------------------------------------+
//| Sets the similarity and direction thresholds used for slot       |
//| scoring.                                                         |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetThresholds(const float similarity, const float direction)
  {
   if(!MathIsValidNumber(similarity) || !MathIsValidNumber(direction))
      ReturnFalse;
   m_similarity_threshold = similarity;
   m_direction_threshold = direction;
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the utility weighting applied when scoring slots.           |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetUtilityWeight(const float value)
  {
   if(!MathIsValidNumber(value))
      ReturnFalse;
   m_utility_weight = value;
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the residual blend factor alpha, clamped to [0,1].          |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetAlpha(const float value)
  {
   if(!MathIsValidNumber(value) || value < 0.0f || value > 1.0f)
      ReturnFalse;
   m_alpha = value;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the residual blend factor alpha.                         |
//+------------------------------------------------------------------+
float CD2SkillBank::Alpha(void) const
  {
   return(m_alpha);
  }
//+------------------------------------------------------------------+
//| Enables utility-aware slot scoring and sets the minimum utility. |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetUtilityPolicy(const bool utility_aware, const float min_utility)
  {
   if(!MathIsValidNumber(min_utility) || min_utility < -1.0f || min_utility > 1.0f)
      ReturnFalse;
   m_utility_aware = utility_aware;
   m_min_utility = min_utility;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns whether utility-aware scoring is enabled.                |
//+------------------------------------------------------------------+
bool CD2SkillBank::UtilityAware(void) const
  {
   return(m_utility_aware);
  }
//+------------------------------------------------------------------+
//| Returns the minimum utility required for slot selection.         |
//+------------------------------------------------------------------+
float CD2SkillBank::MinUtility(void) const
  {
   return(m_min_utility);
  }
//+------------------------------------------------------------------+
//| Sets the confirmation, protection and inactivity ages of the     |
//| slot lifecycle.                                                  |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetLifecycle(const uint confirmations, const uint protection_age,
                                const uint inactivity_age)
  {
   if(confirmations == 0)
      ReturnFalse;
   m_min_confirmations = confirmations;
   m_protection_age = protection_age;
   m_inactivity_age = MathMax(inactivity_age, protection_age + 1);
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the maximum magnitude allowed for a slot correction.        |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetMaxCorrection(const float value)
  {
   if(!MathIsValidNumber(value) || value < 0.0f)
      ReturnFalse;
   m_max_correction = value;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the maximum allowed correction magnitude.                |
//+------------------------------------------------------------------+
float CD2SkillBank::MaxCorrection(void) const
  {
   return(m_max_correction);
  }
//+------------------------------------------------------------------+
//| Returns the number of slots in the bank.                         |
//+------------------------------------------------------------------+
uint CD2SkillBank::Slots(void) const
  {
   return(m_slots);
  }
//+------------------------------------------------------------------+
//| Returns the dimension of each slot vector.                       |
//+------------------------------------------------------------------+
uint CD2SkillBank::Dimension(void) const
  {
   return(m_dimension);
  }
//+------------------------------------------------------------------+
//| Returns the currently selected slot index (UINT_MAX when none).  |
//+------------------------------------------------------------------+
uint CD2SkillBank::ActiveSlot(void) const
  {
   return(m_active_slot);
  }
//+------------------------------------------------------------------+
//| Returns the buffer holding the selected slot correction.         |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedCorrection(void) { return(GetPointer(m_selected_correction)); }
//+------------------------------------------------------------------+
//| Returns the buffer holding the selected slot index.              |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedSlot(void) { return(GetPointer(m_selected_slot)); }
//+------------------------------------------------------------------+
//| Returns the buffer holding the selected slot score.              |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedScore(void) { return(GetPointer(m_selected_score)); }
//+------------------------------------------------------------------+
//| Returns the slot key buffer.                                     |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Keys(void) { return(GetPointer(m_keys)); }
//+------------------------------------------------------------------+
//| Returns the per-slot correction buffer.                          |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Corrections(void) { return(GetPointer(m_corrections)); }
//+------------------------------------------------------------------+
//| Returns the per-slot direction buffer.                           |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Directions(void) { return(GetPointer(m_directions)); }
//+------------------------------------------------------------------+
//| Returns the per-slot scale buffer.                               |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Scales(void) { return(GetPointer(m_scales)); }
//+------------------------------------------------------------------+
//| Returns the per-slot observation counter buffer.                 |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::ObservationsBank(void) { return(GetPointer(m_observations)); }
//+------------------------------------------------------------------+
//| Returns the per-slot use counter buffer.                         |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Uses(void) { return(GetPointer(m_uses)); }
//+------------------------------------------------------------------+
//| Returns the per-slot mass buffer.                                |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::MassBank(void) { return(GetPointer(m_mass)); }
//+------------------------------------------------------------------+
//| Returns the per-slot utility buffer.                             |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Utility(void) { return(GetPointer(m_utility)); }
//+------------------------------------------------------------------+
//| Returns the per-slot used flag buffer.                           |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Used(void) { return(GetPointer(m_used)); }
//+------------------------------------------------------------------+
//| Returns the per-slot state buffer.                               |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::State(void) { return(GetPointer(m_state)); }
//+------------------------------------------------------------------+
//| Returns the per-slot age buffer.                                 |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Ages(void) { return(GetPointer(m_age)); }
//+------------------------------------------------------------------+
//| Returns the per-slot protection counter buffer.                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Protection(void) { return(GetPointer(m_protection)); }
//+------------------------------------------------------------------+
//| Returns the bank diagnostics buffer.                             |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Diagnostics(void) { return(GetPointer(m_diagnostics)); }
//+------------------------------------------------------------------+
//| Returns the utility distribution histogram buffer.               |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::UtilityDistribution(void) { return(GetPointer(m_utility_distribution)); }
//+------------------------------------------------------------------+
//| Returns the usage distribution histogram buffer.                 |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::UsageDistribution(void) { return(GetPointer(m_usage_distribution)); }
//+------------------------------------------------------------------+
//| Returns the per-slot influence buffer.                           |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Influence(void) { return(GetPointer(m_influence)); }
//+------------------------------------------------------------------+
//| Returns the per-slot retrieved flag buffer.                      |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Retrieved(void) { return(GetPointer(m_retrieved)); }
//+------------------------------------------------------------------+
//| Constructor: starts unmaterialized with both banks disabled.     |
//+------------------------------------------------------------------+
CD2Skill::CD2Skill(void) : m_ready(false), m_task_bank_materialized(false),
   m_step_bank_materialized(false), m_online_direction_update(false), m_mode(D2_DISABLED)
  {
  }
//+------------------------------------------------------------------+
//| Destructor (CD2Skill).                                           |
//+------------------------------------------------------------------+
CD2Skill::~CD2Skill(void)
  {
  }
//+------------------------------------------------------------------+
//| Returns the skill neuron type identifier (defNeuronD2Skill).     |
//+------------------------------------------------------------------+
int CD2Skill::Type(void) const
  {
   return(defNeuronD2Skill);
  }
//+------------------------------------------------------------------+
//| Returns the current skill runtime mode.                          |
//+------------------------------------------------------------------+
ENUM_D2SKILL_MODE CD2Skill::Mode(void) const
  {
   return(m_mode);
  }
//+------------------------------------------------------------------+
//| Returns whether the skill is ready after a successful            |
//| initialization.                                                  |
//+------------------------------------------------------------------+
bool CD2Skill::Ready(void) const
  {
   return(m_ready);
  }
//+------------------------------------------------------------------+
//| Returns the Task bank pointer when ready, otherwise NULL.        |
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::TaskBank(void)
  {
   return(m_ready ? GetPointer(m_task_bank) : NULL);
  }
//+------------------------------------------------------------------+
//| Returns the Step bank pointer when ready, otherwise NULL.        |
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::StepBank(void)
  {
   return(m_ready ? GetPointer(m_step_bank) : NULL);
  }
//+------------------------------------------------------------------+
//| Returns the primary (Task) bank pointer when ready, otherwise    |
//| NULL.                                                            |
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::Bank(void)
  {
   return(TaskBank());
  }
//+------------------------------------------------------------------+
//| Returns the primary (Task) bank pointer (const) when ready,      |
//| otherwise NULL.                                                  |
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::Bank(void) const
  {
   return(TaskBank());
  }
//+------------------------------------------------------------------+
//| Returns the Task bank pointer (const) when ready, otherwise      |
//| NULL.                                                            |
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::TaskBank(void) const
  {
   return(m_ready ? GetPointer(m_task_bank) : NULL);
  }
//+------------------------------------------------------------------+
//| Returns the Step bank pointer (const) when ready, otherwise      |
//| NULL.                                                            |
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::StepBank(void) const
  {
   return(m_ready ? GetPointer(m_step_bank) : NULL);
  }
//+------------------------------------------------------------------+
//| Checks D2 composition readiness, including deferred banks.       |
//+------------------------------------------------------------------+
bool CD2Skill::SupportsOpenCLChecked(void) const
  {
//--- Validate the exact D2 identity and complete Scenario-owned tree.
   if(Type() != defNeuronD2Skill || !m_ready || bCritic ||
      !ValidateScenarioOwnedTreeChecked())
      return(false);
//--- Validate each Bank using its current materialization state.
   if(m_task_bank_materialized && !m_task_bank.SupportsOpenCLChecked())
      return(false);
   if(!m_task_bank_materialized && !m_task_bank.SupportsDeferredOpenCLChecked())
      return(false);
   if(m_step_bank_materialized && !m_step_bank.SupportsOpenCLChecked())
      return(false);
   if(!m_step_bank_materialized && !m_step_bank.SupportsDeferredOpenCLChecked())
      return(false);
//--- Bound tensor products before comparing derived dimensions.
   const long max_elements = INT_MAX;
   if((long)iQueries > max_elements / (long)iScenarios ||
      (long)iQueries * (long)iScenarios > max_elements / (long)iDimension ||
      (long)iScenarios > max_elements / (long)iVariables ||
      (long)iScenarios * (long)iVariables > max_elements / (long)iHorizon ||
      (long)iScenarios * (long)iVariables * (long)iHorizon > max_elements / (long)iDimension)
      return(false);
//--- Confirm the established Task and Step dimensions and slot counts.
   return(m_task_bank.Dimension() == (uint)((long)iQueries * iScenarios * iDimension) &&
          m_task_bank.Slots() == iScenarios &&
          m_step_bank.Dimension() == (uint)((long)iScenarios * iVariables * iHorizon * iDimension) &&
          m_step_bank.Slots() == iScenarios);
  }
//+------------------------------------------------------------------+
//| Transfers both banks before publishing the Scenario parent.      |
//+------------------------------------------------------------------+
bool CD2Skill::SetOpenCLChecked(COpenCLMy *obj)
  {
//--- Validate the exact model and both Bank states before either Bank is prepared.
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      ReturnFalseEx("checked target or capability");
   if(m_task_bank_materialized && !m_task_bank.SupportsOpenCLChecked(obj))
      ReturnFalseEx("checked Task prevalidation");
   if(!m_task_bank_materialized && !m_task_bank.SupportsDeferredOpenCLChecked(obj))
      ReturnFalseEx("checked Task prevalidation");
   if(m_step_bank_materialized && !m_step_bank.SupportsOpenCLChecked(obj))
      ReturnFalseEx("checked Step prevalidation");
   if(!m_step_bank_materialized && !m_step_bank.SupportsDeferredOpenCLChecked(obj))
      ReturnFalseEx("checked Step prevalidation");
//--- Prepare and transfer Task before publishing its readiness.
   if(!m_task_bank_materialized && !m_task_bank.PrepareOpenCLChecked(obj))
      ReturnFalseEx("checked Task prepare");
   if(!m_task_bank.SetOpenCLChecked(obj))
      ReturnFalseEx("checked Task context");
   if(!m_task_bank_materialized)
      m_task_bank_materialized = true;
//--- Prepare and transfer Step before publishing its readiness.
   if(!m_step_bank_materialized && !m_step_bank.PrepareOpenCLChecked(obj))
      ReturnFalseEx("checked Step prepare");
   if(!m_step_bank.SetOpenCLChecked(obj))
      ReturnFalseEx("checked Step context");
   if(!m_step_bank_materialized)
      m_step_bank_materialized = true;
//--- Publish the Scenario parent only after both Bank transfers succeed.
   if(!CNeuronScenarioCrossAttention::SetOpenCLChecked(obj))
      ReturnFalseEx("checked Scenario context");
   return(true);
  }
//+------------------------------------------------------------------+
//| Enables the Task and/or Step branch after checked preparation.   |
//+------------------------------------------------------------------+
bool CD2Skill::Enable(const bool task, const bool step)
  {
//--- Validate every requested Bank before preparing either deferred branch.
   if(!m_ready || !OpenCL)
      ReturnFalse;
   if(task && m_task_bank_materialized && !m_task_bank.SupportsOpenCLChecked(OpenCL))
      ReturnFalse;
   if(task && !m_task_bank_materialized && !m_task_bank.SupportsDeferredOpenCLChecked(OpenCL))
      ReturnFalse;
   if(step && m_step_bank_materialized && !m_step_bank.SupportsOpenCLChecked(OpenCL))
      ReturnFalse;
   if(step && !m_step_bank_materialized && !m_step_bank.SupportsDeferredOpenCLChecked(OpenCL))
      ReturnFalse;
//--- Prepare and transfer requested deferred Banks without changing enabled state.
   if(task && !m_task_bank_materialized)
     {
      if(!m_task_bank.PrepareOpenCLChecked(OpenCL))
         ReturnFalse;
      if(!m_task_bank.SetOpenCLChecked(OpenCL))
         ReturnFalse;
      if(!m_task_bank.SetCorrectionBuffer(m_task_bank.SelectedCorrection()))
         ReturnFalse;
      m_task_bank_materialized = true;
     }
   if(step && !m_step_bank_materialized)
     {
      if(!m_step_bank.PrepareOpenCLChecked(OpenCL))
         ReturnFalse;
      if(!m_step_bank.SetOpenCLChecked(OpenCL))
         ReturnFalse;
      if(!m_step_bank.SetCorrectionBuffer(m_step_bank.SelectedCorrection()))
         ReturnFalse;
      m_step_bank_materialized = true;
     }
//--- Bind selected correction aliases only after all requested transfers succeed.
   if(task && m_task_bank.GetCorrectionBuffer() == NULL &&
      !m_task_bank.SetCorrectionBuffer(m_task_bank.SelectedCorrection()))
      ReturnFalse;
   if(step && m_step_bank.GetCorrectionBuffer() == NULL &&
      !m_step_bank.SetCorrectionBuffer(m_step_bank.SelectedCorrection()))
      ReturnFalse;
   m_task_bank.SetEnabled(task);
   m_step_bank.SetEnabled(step);
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the Task bank feed-forward on the given source.             |
//+------------------------------------------------------------------+
bool CD2Skill::TaskFeedForward(CNeuronBaseOCL *source)
  {
   return(m_ready && m_task_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
//| Runs the Step bank feed-forward on the given source.             |
//+------------------------------------------------------------------+
bool CD2Skill::StepFeedForward(CNeuronBaseOCL *source)
  {
   return(m_ready && m_step_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
//| Applies the pseudo-residual update to the Task bank.             |
//+------------------------------------------------------------------+
bool CD2Skill::TaskUpdate(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(m_ready && m_task_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
//| Applies the pseudo-residual update to the Step bank.             |
//+------------------------------------------------------------------+
bool CD2Skill::StepUpdate(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(m_ready && m_step_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
//| Applies the utility update to the Task bank; reports whether it  |
//| was applied.                                                     |
//+------------------------------------------------------------------+
bool CD2Skill::TaskUtility(const float delta_j, bool &applied)
  {
   applied = false;
   if(!m_ready || (m_mode != D2_EVALUATE && m_mode != D2_ONLINE_CALIBRATION))
      ReturnFalse;
   return(m_task_bank.UpdateUtility(delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Applies the utility update to the Task bank.                     |
//+------------------------------------------------------------------+
bool CD2Skill::TaskUtility(const float delta_j)
  {
   bool applied = false;
   return(TaskUtility(delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Applies the utility update to the Step bank; reports whether it  |
//| was applied.                                                     |
//+------------------------------------------------------------------+
bool CD2Skill::StepUtility(const float delta_j, bool &applied)
  {
   applied = false;
   if(!m_ready || (m_mode != D2_EVALUATE && m_mode != D2_ONLINE_CALIBRATION))
      ReturnFalse;
   return(m_step_bank.UpdateUtility(delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Applies the utility update to the Step bank.                     |
//+------------------------------------------------------------------+
bool CD2Skill::StepUtility(const float delta_j)
  {
   bool applied = false;
   return(StepUtility(delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Runs the second-pass backward propagation through the skill      |
//| highway.                                                         |
//+------------------------------------------------------------------+
bool CD2Skill::BackwardSecond(CNeuronBaseOCL *source, CNeuronBaseOCL *forecast)
  {
   return(m_ready && calcInputGradientsSecond(source, forecast));
  }
//+------------------------------------------------------------------+
//| Initializes the skill, its scenario-attention base and both      |
//| Task/Step banks.                                                 |
//+------------------------------------------------------------------+
bool CD2Skill::Init(uint numOutputs, uint myIndex, COpenCLMy *opencl,
                    uint scenarios, uint horizon, uint latent, bool critic,
                    ENUM_OPTIMIZATION optimization_type, uint batch,
                    uint variables, uint stack_size, uint top_k)
  {
   if(critic)
      ReturnFalse;
   m_ready = false;
   m_mode = D2_DISABLED;
   if(!CNeuronScenarioCrossAttention::Init(numOutputs, myIndex, opencl,
                                           scenarios, horizon, latent, false,
                                           optimization_type, batch, variables,
                                           stack_size, top_k))
      ReturnFalse;
   if(!InitBanks(optimization_type, batch))
      ReturnFalse;
   m_task_bank_materialized = true;
   m_step_bank_materialized = true;
   m_task_bank.SetEnabled(false);
   m_step_bank.SetEnabled(false);
   m_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Creates the Task and Step banks with dimensions derived from the |
//| attention layout.                                                |
//+------------------------------------------------------------------+
bool CD2Skill::InitBanks(ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   const uint task_dimension = iQueries * iScenarios * iDimension;
   const uint step_dimension = iScenarios * iVariables * iHorizon * iDimension;
   if(task_dimension == 0 || step_dimension == 0 || !OpenCL)
      ReturnFalse;
   if(!m_task_bank.Init(task_dimension, 100000, OpenCL, iScenarios,
                        optimization_type, batch))
      ReturnFalse;
   if(!m_step_bank.Init(step_dimension, 100001, OpenCL, iScenarios,
                        optimization_type, batch))
      ReturnFalse;
   m_task_bank.SetActivationFunction(None);
   m_step_bank.SetActivationFunction(None);
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns whether the Task branch is active in the current runtime |
//| mode.                                                            |
//+------------------------------------------------------------------+
bool CD2Skill::TaskSkillEnabled(void) const
  {
   return(m_ready && m_task_bank.Enabled() &&
          (m_mode == D2_EVALUATE || m_mode == D2_INFERENCE ||
           m_mode == D2_ONLINE_CALIBRATION));
  }
//+------------------------------------------------------------------+
//| Returns whether the Step branch is active in the current runtime |
//| mode.                                                            |
//+------------------------------------------------------------------+
bool CD2Skill::StepSkillEnabled(void) const
  {
   return(m_ready && m_step_bank.Enabled() &&
          (m_mode == D2_EVALUATE || m_mode == D2_INFERENCE ||
           m_mode == D2_ONLINE_CALIBRATION));
  }
//+------------------------------------------------------------------+
//| Returns whether bank weights are writable (D2_COLLECT mode       |
//| only).                                                           |
//+------------------------------------------------------------------+
bool CD2Skill::SkillUpdatesEnabled(void) const
  {
   return(m_mode == D2_COLLECT);
  }
//+------------------------------------------------------------------+
//| Enables only the opt-in online direction EMA.                    |
//+------------------------------------------------------------------+
bool CD2Skill::OnlineDirectionUpdatesEnabled(void) const
  {
   return(m_mode == D2_ONLINE_CALIBRATION && m_online_direction_update);
  }
//+------------------------------------------------------------------+
//| Updates one selected bank direction without mutating its         |
//| lifecycle.                                                       |
//+------------------------------------------------------------------+
bool CD2Skill::UpdateOnlineDirection(CD2SkillBank *bank, CBufferFloat *gradient)
  {
   if(!OnlineDirectionUpdatesEnabled() || !bank || !bank.Enabled())
      return(true);
   if(!gradient)
      ReturnFalse;
   return(bank.UpdateDirectionFromGradient(gradient));
  }
//+------------------------------------------------------------------+
//| Runs the second-pass feed-forward: optional banks, token         |
//| attention and history gates.                                     |
//+------------------------------------------------------------------+
bool CD2Skill::feedForwardSecond(CNeuronBaseOCL *NeuronOCL,
                                 CNeuronBaseOCL *Forecast)
  {
   if(!OpenCL || !NeuronOCL || !Forecast || Forecast.Type() != defNeuronScenarioForecast ||
      NeuronOCL.Neurons() != int(bCritic ? 19 : 13))
      ReturnFalse;
   CNeuronScenarioForecast *forecast = Forecast;
   if(forecast.Variables() != iVariables || forecast.Scenarios() != iScenarios ||
      forecast.Horizon() != iHorizon || forecast.Dimension() != iDimension ||
      Forecast.Neurons() != int(iScenarios * iVariables * iHorizon * iDimension) ||
      !forecast.GetLogU() || forecast.GetLogU().Total() != int(iScenarios * iVariables * iHorizon) ||
      !forecast.GetPi() || forecast.GetPi().Total() != int(iScenarios))
      ReturnFalse;
   if(!cQueryProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   CNeuronBaseOCL *task_source = cQueryProjection.AsObject();
   if(TaskSkillEnabled())
     {
      if(!m_task_bank.FeedForward(task_source))
         ReturnFalse;
      task_source = m_task_bank.AsObject();
     }
   else
      if(m_mode == D2_COLLECT && m_task_bank.Enabled() &&
         !m_task_bank.Observe(task_source))
         ReturnFalse;
   if(!cWz.FeedForward(Forecast))
      ReturnFalse;
   CNeuronBaseOCL *step_source = cWz.AsObject();
   if(StepSkillEnabled())
     {
      if(!m_step_bank.FeedForward(step_source))
         ReturnFalse;
      step_source = m_step_bank.AsObject();
     }
   else
      if(m_mode == D2_COLLECT && m_step_bank.Enabled() &&
         !m_step_bank.Observe(step_source))
         ReturnFalse;
   if(!cWu.FeedForwardBuffer(forecast.GetLogU()))
      ReturnFalse;
   if(!SumAndNormalize(step_source.getOutput(), cWu.getOutput(), cTokenSum.getOutput(),
                       1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cTokenPE.FeedForward(cTokenSum.AsObject()))
      ReturnFalse;
   if(!cTokensTranspose.FeedForward(cTokenPE.AsObject()))
      ReturnFalse;
   if(!Concat(cTokensTranspose.getOutput(), cTokensTranspose.getOutput(), cKV.getOutput(),
              iScenarios * iDimension, iScenarios * iDimension, iVariables * iHorizon))
      ReturnFalse;
   if(!cAttention.FeedForward(task_source, cKV.getOutput()))
      ReturnFalse;
   if(!cAttentionTranspose.FeedForward(cAttention.AsObject()))
      ReturnFalse;
   if(!bCritic)
     {
      if(!cHistoryStack.FeedForward(GetPointer(this)))
         ReturnFalse;
      if(!cHistoryTTM.FeedForward(cAttentionTranspose.AsObject(), cHistoryStack.getOutput()))
         ReturnFalse;
      if(!cHistoryTokenProjection.FeedForward(cTokenPE.AsObject()))
         ReturnFalse;
      if(!cHistoryMarket.FeedForward(cHistoryStack.AsObject(), cHistoryTokenProjection.getOutput()))
         ReturnFalse;
      if(!cHistoryContext.FeedForward(cHistoryTTM.AsObject(), cHistoryMarket.getOutput()))
         ReturnFalse;
      if(!cScenarioHistory.FeedForward(cAttentionTranspose.AsObject(), cHistoryContext.getOutput()))
         ReturnFalse;
      if(!ScalarToVector(forecast.GetPi(), cScenarioHistory.getOutput(), cHeadGate.getOutput(),
                         iQueries * iDimension))
         ReturnFalse;
     }
   else
      if(!ScalarToVector(forecast.GetPi(), cAttentionTranspose.getOutput(), cHeadGate.getOutput(),
                         iQueries * iDimension))
         ReturnFalse;
   if(!cHeadGateTranspose.FeedForward(cHeadGate.AsObject()))
      ReturnFalse;
   if(!cOutputProjection.FeedForward(cHeadGateTranspose.AsObject()))
      ReturnFalse;
   if(!cQueryResidual.FeedForward(task_source))
      ReturnFalse;
   if(!SumAndNormalize(cOutputProjection.getOutput(), cQueryResidual.getOutput(), cMHAOutput.getOutput(),
                       iDimension, true, 0, 0, 0, 1))
      ReturnFalse;
   return(CNeuronMSRes::feedForward(cMHAOutput.AsObject()));
  }
//+------------------------------------------------------------------+
//| Back-propagates the second-pass gradients, including bank        |
//| influence accumulation.                                          |
//+------------------------------------------------------------------+
bool CD2Skill::calcInputGradientsSecond(CNeuronBaseOCL *NeuronOCL,
                                        CNeuronBaseOCL *Forecast)
  {
   if(!OpenCL || !NeuronOCL || !Forecast || Forecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
   CNeuronScenarioForecast *forecast = Forecast;
   if(!forecast.GetPi())
      ReturnFalse;
   if(!CNeuronMSRes::calcInputGradients(cMHAOutput.AsObject()))
      ReturnFalse;
   if(!DeActivation(cOutputProjection.getOutput(), cOutputProjection.getGradient(), cMHAOutput.getGradient(),
                    cOutputProjection.Activation()))
      ReturnFalse;
   if(!DeActivation(cQueryResidual.getOutput(), cQueryResidual.getGradient(), cMHAOutput.getGradient(),
                    cQueryResidual.Activation()))
      ReturnFalse;
   if(!cHeadGateTranspose.CalcHiddenGradients(cOutputProjection.AsObject()))
      ReturnFalse;
   if(!cHeadGate.CalcHiddenGradients(cHeadGateTranspose.AsObject()))
      ReturnFalse;
   if(!bCritic)
     {
      if(!ScalarToVector(forecast.GetPi(), cHeadGate.getGradient(), cScenarioHistory.getGradient(),
                         iQueries * iDimension))
         ReturnFalse;
      if(!cAttentionTranspose.CalcHiddenGradients(cScenarioHistory.AsObject(), cHistoryContext.getOutput(),
            cHistoryContext.getGradient(), None))
         ReturnFalse;
      if(!cHistoryTTM.CalcHiddenGradients(cHistoryContext.AsObject(), cHistoryMarket.getOutput(),
                                          cHistoryMarket.getGradient(), None))
         ReturnFalse;
      if(!cAttentionTranspose.CalcHiddenGradients(cHistoryTTM.AsObject(), cHistoryStack.getOutput(), NULL, None))
         ReturnFalse;
      if(!cHistoryStack.CalcHiddenGradients(cHistoryMarket.AsObject(), cHistoryTokenProjection.getOutput(),
                                            cHistoryTokenProjection.getGradient(), None))
         ReturnFalse;
      if(!cTokenPE.CalcHiddenGradients(cHistoryTokenProjection.AsObject()))
         ReturnFalse;
     }
   else
      if(!ScalarToVector(forecast.GetPi(), cHeadGate.getGradient(), cAttentionTranspose.getGradient(),
                         iQueries * iDimension))
         ReturnFalse;
   if(!cAttention.CalcHiddenGradients(cAttentionTranspose.AsObject()))
      ReturnFalse;
   if(!cQueryGradient.CalcHiddenGradients(cQueryResidual.AsObject()))
      ReturnFalse;
   if(TaskSkillEnabled())
     {
      //--- Attention owns the two-input gradient dispatcher.  Passing the Bank as
      //--- its query input writes dL/d(Bank output) to the Bank without weights.
      if(!m_task_bank.CalcHiddenGradients(cAttention.AsObject(), cKV.getOutput(),
                                          cKV.getGradient(), None))
         ReturnFalse;
      if((m_mode == D2_EVALUATE || m_mode == D2_ONLINE_CALIBRATION) &&
         !m_task_bank.AccumulateInfluenceFromGradient(m_task_bank.getGradient()))
         ReturnFalse;
      if(!cQueryProjection.CalcHiddenGradients(m_task_bank.AsObject()))
         ReturnFalse;
     }
   else
      if(!cQueryProjection.CalcHiddenGradients(cAttention.AsObject(), cKV.getOutput(),
            cKV.getGradient(), None))
         ReturnFalse;
   if(!SumAndNormalize(cQueryProjection.getGradient(), cQueryGradient.getGradient(), cQueryProjection.getGradient(),
                       1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!DeConcat(cKVGradK.getGradient(), cKVGradV.getGradient(), cKV.getGradient(),
                iScenarios * iDimension, iScenarios * iDimension, iVariables * iHorizon))
      ReturnFalse;
   if(!SumAndNormalize(cKVGradK.getGradient(), cKVGradV.getGradient(),
                       cTokensTranspose.getGradient(), 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!cTokenPE.CalcHiddenGradients(cTokensTranspose.AsObject()))
      ReturnFalse;
   if(!cTokenSum.CalcHiddenGradients(cTokenPE.AsObject()))
      ReturnFalse;
   if(StepSkillEnabled())
     {
      //--- As for Task, Step has no weights.  Its output is the corrected cWz,
      //--- therefore the token-sum gradient is passed through unchanged to cWz.
      if(!CopyBufferRaw(cTokenSum.getGradient(), m_step_bank.getGradient(),
                        m_step_bank.Neurons()) ||
         !m_step_bank.calcInputGradients(cWz.AsObject()))
         ReturnFalse;
      if((m_mode == D2_EVALUATE || m_mode == D2_ONLINE_CALIBRATION) &&
         !m_step_bank.AccumulateInfluenceFromGradient(m_step_bank.getGradient()))
         ReturnFalse;
     }
   else
      if(!Concat(cTokenSum.getGradient(), cTokenSum.getGradient(), cWz.getGradient(),
                 iScenarios * iVariables * iHorizon * iDimension, 0, 1))
         ReturnFalse;
   if(!Concat(cTokenSum.getGradient(), cTokenSum.getGradient(), cWu.getGradient(),
              iScenarios * iVariables * iHorizon * iDimension, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cQueryProjection.AsObject()))
      ReturnFalse;
   if(SkillUpdatesEnabled() && m_task_bank.Enabled() &&
      !m_task_bank.UpdateFromGradient(cQueryProjection.getOutput(), cQueryProjection.getGradient()))
      ReturnFalse;
   if(SkillUpdatesEnabled() && m_step_bank.Enabled() &&
      !m_step_bank.UpdateFromGradient(cWz.getOutput(), cWz.getGradient()))
      ReturnFalse;
   if(!UpdateOnlineDirection(GetPointer(m_task_bank), m_task_bank.getGradient()) ||
      !UpdateOnlineDirection(GetPointer(m_step_bank), m_step_bank.getGradient()))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      return(DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(),
                          NeuronOCL.Activation()));
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates the second-pass attention weights; bank weights stay     |
//| untouched.                                                       |
//+------------------------------------------------------------------+
bool CD2Skill::updateInputWeightsSecond(CNeuronBaseOCL *NeuronOCL,
                                        CNeuronBaseOCL *Forecast)
  {
   if(!NeuronOCL || !Forecast || Forecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
//--- Collection records bank observations but freezes the Actor highway.
//--- Evaluation, inference, and online calibration are read-only for Actor weights.
   if(m_mode == D2_COLLECT || m_mode == D2_EVALUATE || m_mode == D2_INFERENCE ||
      m_mode == D2_ONLINE_CALIBRATION)
      return(true);
   CNeuronScenarioForecast *forecast = Forecast;
   if(!forecast.GetLogU())
      ReturnFalse;
   if(!cQueryProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cWz.UpdateInputWeights(Forecast))
      ReturnFalse;
   if(!cWu.UpdateInputWeightsBuffer(forecast.GetLogU()))
      ReturnFalse;
   if(!cTokenPE.UpdateInputWeights(cTokenSum.AsObject()))
      ReturnFalse;
   if(!cOutputProjection.UpdateInputWeights(cHeadGateTranspose.AsObject()))
      ReturnFalse;
   if(!cQueryResidual.UpdateInputWeights(cQueryProjection.AsObject()))
      ReturnFalse;
   if(!bCritic)
     {
      if(!cHistoryTokenProjection.UpdateInputWeights(cTokenPE.AsObject()))
         ReturnFalse;
      if(!cHistoryMarket.UpdateInputWeights(cHistoryStack.AsObject(), cHistoryTokenProjection.getOutput()))
         ReturnFalse;
      if(!cHistoryContext.UpdateInputWeights(cHistoryTTM.AsObject(), cHistoryMarket.getOutput()))
         ReturnFalse;
      if(!cScenarioHistory.UpdateInputWeights(cAttentionTranspose.AsObject(), cHistoryContext.getOutput()))
         ReturnFalse;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the skill and both banks to the file handle.               |
//+------------------------------------------------------------------+
bool CD2Skill::Save(const int file_handle)
  {
   if(!m_ready || !CNeuronScenarioCrossAttention::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, D2SKILL_LAYER_FORMAT);
   return(m_task_bank.Save(file_handle) && m_step_bank.Save(file_handle));
  }
//+------------------------------------------------------------------+
//| Restores the skill and both banks, validating restored bank      |
//| shapes.                                                          |
//+------------------------------------------------------------------+
bool CD2Skill::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronScenarioCrossAttention::Load(file_handle))
      ReturnFalse;
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_LAYER_FORMAT || bCritic || !OpenCL)
      ReturnFalse;
   m_ready = false;
   m_task_bank_materialized = false;
   m_step_bank_materialized = false;
//--- Serialized bank shapes are restored by Load().  SetOpenCL() creates
//--- scratch and is invalid before those dimensions exist.
   m_task_bank.AttachOpenCL(OpenCL);
   m_step_bank.AttachOpenCL(OpenCL);
   if(FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank ||
      !m_task_bank.Load(file_handle) ||
      FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank ||
      !m_step_bank.Load(file_handle))
      ReturnFalse;
   const uint task_dimension = iQueries * iScenarios * iDimension;
   const uint step_dimension = iScenarios * iVariables * iHorizon * iDimension;
   if(m_task_bank.Dimension() != task_dimension || m_task_bank.Slots() != iScenarios ||
      m_step_bank.Dimension() != step_dimension || m_step_bank.Slots() != iScenarios)
      ReturnFalse;
   m_ready = true;
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the skill from file and attaches the given OpenCL       |
//| context.                                                         |
//+------------------------------------------------------------------+
bool CD2Skill::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl || FileReadInteger(file_handle) != Type())
      ReturnFalse;
   OpenCL = opencl;
   return(Load(file_handle));
  }
//+------------------------------------------------------------------+
//| Rebinds the skill GPU buffers to the given OpenCL context.       |
//+------------------------------------------------------------------+
void CD2Skill::SetOpenCL(COpenCLMy *opencl)
  {
   CNeuronScenarioCrossAttention::SetOpenCL(opencl);
//--- Disabled checkpoint banks retain their host state but defer GPU scratch
//--- until the runtime mode explicitly enables the respective branch.  Do not
//--- AttachOpenCL() here: it would replace only the bank pointer, causing the
//--- later SetOpenCL() to skip the inherited Output/Gradient rebind.
   if(m_task_bank_materialized)
      m_task_bank.SetOpenCL(opencl);
   if(m_step_bank_materialized)
      m_step_bank.SetOpenCL(opencl);
  }
//+------------------------------------------------------------------+
//| Propagates the training flag to the skill and both banks.        |
//+------------------------------------------------------------------+
void CD2Skill::TrainMode(bool flag)
  {
   CNeuronScenarioCrossAttention::TrainMode(flag);
   m_task_bank.TrainMode(flag);
   m_step_bank.TrainMode(flag);
  }
//+------------------------------------------------------------------+
//| Polyak-averages the base attention weights; bank parameters are  |
//| excluded.                                                        |
//+------------------------------------------------------------------+
bool CD2Skill::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
//--- Skill banks are accumulated model parameters, not trainable copies of
//--- the scenario-attention weights.  Polyak update applies only to the base
//--- attention highway.
   return(CNeuronScenarioCrossAttention::WeightsUpdate(source, tau));
  }
//+------------------------------------------------------------------+
//| Preserves COLLECT bank state while CNet weights are frozen.      |
//+------------------------------------------------------------------+
bool CD2Skill::UpdateInputStateSecond(CObject *SourceObject,
                                      CNeuronBaseOCL *Forecast)
  {
   if(CheckPointer(SourceObject) == POINTER_INVALID || !Forecast || !m_ready)
      ReturnFalse;
   CNeuronBaseOCL *source = SourceObject;
   return(updateInputWeightsSecond(source, Forecast));
  }
//+------------------------------------------------------------------+
//| Clears the recurrent execution state; banks survive as durable   |
//| model parameters.                                                |
//+------------------------------------------------------------------+
bool CD2Skill::Clear(void)
  {
//--- Clear resets only the recurrent execution state inherited from the
//--- attention layer.  Banks are durable model parameters and survive it.
   const bool ready = m_ready;
   if(!CNeuronScenarioCrossAttention::Clear())
      ReturnFalse;
   m_ready = ready;
   return(true);
  }
//+------------------------------------------------------------------+
//| Sets the skill runtime mode.                                     |
//+------------------------------------------------------------------+
bool CD2Skill::SetMode(const ENUM_D2SKILL_MODE mode)
  {
   if(mode < D2_DISABLED || mode > D2_ONLINE_CALIBRATION)
      ReturnFalse;
   m_mode = mode;
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures the optional online direction-only EMA.               |
//+------------------------------------------------------------------+
bool CD2Skill::SetOnlineDirectionUpdate(const bool enabled)
  {
   m_online_direction_update = enabled;
   return(true);
  }
//+------------------------------------------------------------------+
//| Resets the episode influence counters of both banks.             |
//+------------------------------------------------------------------+
bool CD2Skill::ResetEpisodeInfluence(void)
  {
   if(!m_ready || !m_task_bank.ResetEpisodeInfluence() ||
      !m_step_bank.ResetEpisodeInfluence())
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Combines the source with the item correction (or the source      |
//| alone) into the output.                                          |
//+------------------------------------------------------------------+
bool CD2SkillItem::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Output) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *source = NeuronOCL.getOutput();
   if(!source || source.Total() != Output.Total() || source.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(CheckPointer(m_correction) == POINTER_INVALID)
      return(SumAndNormalize(source, source, Output, source.Total(), false,
                             0, 0, 0, 0.5f));
   return(SumAndNormalize(source, m_correction, Output, source.Total(), false,
                          0, 0, 0, 1.0f));
  }
//+------------------------------------------------------------------+
//| Passes the incoming gradient through, preserving gradients       |
//| accumulated by other branches.                                   |
//+------------------------------------------------------------------+
bool CD2SkillItem::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Gradient) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *source_gradient = NeuronOCL.getGradient();
   if(!source_gradient || source_gradient.Total() != Gradient.Total() ||
      source_gradient.GetOpenCL() != OpenCL)
      ReturnFalse;
//--- A skill item has no weights: dL/dx receives dL/dy.  Preserve any
//--- gradient already accumulated by another branch (for example the
//--- Actor history path) and add the bank output gradient on the device.
   return(SumAndNormalize(source_gradient, Gradient, source_gradient,
                          Gradient.Total(), false, 0, 0, 0, 1.0f));
  }
//+------------------------------------------------------------------+
//| Allocates and binds all item GPU buffers.                        |
//+------------------------------------------------------------------+
bool CD2SkillItem::EnsureItemState(void)
  {
   if(!OpenCL || m_item_dimension == 0)
      ReturnFalse;
   if(m_item_correction.Total() != (int)m_item_dimension &&
      !m_item_correction.BufferInit((int)m_item_dimension, 0.0f))
      ReturnFalse;
   if((m_item_correction.GetOpenCL() != OpenCL || m_item_correction.GetIndex() < 0) &&
      !m_item_correction.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_correction == NULL)
     {
      m_correction = GetPointer(m_item_correction);
     }
   if(m_item_direction.Total() != (int)m_item_dimension &&
      !m_item_direction.BufferInit((int)m_item_dimension, 0.0f))
      ReturnFalse;
   if((m_item_direction.GetOpenCL() != OpenCL || m_item_direction.GetIndex() < 0) &&
      !m_item_direction.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_item_scale.Total() != 1 && !m_item_scale.BufferInit(1, 0.0f))
      ReturnFalse;
   if((m_item_scale.GetOpenCL() != OpenCL || m_item_scale.GetIndex() < 0) &&
      !m_item_scale.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_item_observations.Total() != 1 && !m_item_observations.BufferInit(1, 0.0f))
      ReturnFalse;
   if((m_item_observations.GetOpenCL() != OpenCL || m_item_observations.GetIndex() < 0) &&
      !m_item_observations.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_item_mass.Total() != 1 && !m_item_mass.BufferInit(1, 0.0f))
      ReturnFalse;
   if((m_item_mass.GetOpenCL() != OpenCL || m_item_mass.GetIndex() < 0) &&
      !m_item_mass.BufferCreate(OpenCL))
      ReturnFalse;
   return(m_correction && m_correction.GetOpenCL() == OpenCL &&
          m_correction.Total() == (int)m_item_dimension &&
          m_correction.GetIndex() >= 0 && m_item_direction.GetIndex() >= 0 &&
          m_item_scale.GetIndex() >= 0 && m_item_observations.GetIndex() >= 0 &&
          m_item_mass.GetIndex() >= 0 && EnsureItemScratch());
  }
//+------------------------------------------------------------------+
//| Prepares legitimately absent Item-owned host buffers only.       |
//+------------------------------------------------------------------+
bool CD2SkillItem::PrepareItemOwnedCheckedState(void)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || m_item_dimension == 0 ||
      CheckPointer(Output) == POINTER_INVALID || CheckPointer(Gradient) == POINTER_INVALID ||
      Output.Total() != (int)m_item_dimension || Gradient.Total() != (int)m_item_dimension ||
      (Output.GetIndex() >= 0 && CheckPointer(Output.GetOpenCL()) == POINTER_INVALID) ||
      (Gradient.GetIndex() >= 0 && CheckPointer(Gradient.GetOpenCL()) == POINTER_INVALID))
      return(false);
   uint local_size    = 0;
   uint groups        = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count))
      return(false);
   CBufferFloat *buffers[7];
   int           totals[7];
   string        slots[7] = {"correction", "direction", "scale", "observations",
                             "mass", "norm partials", "reduced norm"
                            };
   buffers[0] = GetPointer(m_item_correction);
   totals[0] = (int)m_item_dimension;
   buffers[1] = GetPointer(m_item_direction);
   totals[1] = (int)m_item_dimension;
   buffers[2] = GetPointer(m_item_scale);
   totals[2] = 1;
   buffers[3] = GetPointer(m_item_observations);
   totals[3] = 1;
   buffers[4] = GetPointer(m_item_mass);
   totals[4] = 1;
   buffers[5] = GetPointer(m_item_partials);
   totals[5] = (int)partial_count;
   buffers[6] = GetPointer(m_item_reduced_norm);
   totals[6] = 2;
   for(int i = 0; i < 7; i++)
     {
      if(CheckPointer(buffers[i]) == POINTER_INVALID ||
         (buffers[i].Total() != totals[i] &&
          (buffers[i].Total() != 0 || buffers[i].GetIndex() >= 0)) ||
         (buffers[i].GetIndex() >= 0 &&
          CheckPointer(buffers[i].GetOpenCL()) == POINTER_INVALID))
         ReturnFalseEx("item " + slots[i] + " checked state");
     }
   for(int i = 0; i < 7; i++)
      if(buffers[i].Total() == 0 && !buffers[i].BufferInit(totals[i], 0.0f))
         ReturnFalseEx("item " + slots[i] + " host init");
   return(true);
  }
//+------------------------------------------------------------------+
//| Computes the GPU launch layout and norm-reduction scratch size   |
//| for the item.                                                    |
//+------------------------------------------------------------------+
bool CD2SkillItem::ItemLaunchLayout(uint &local_size, uint &groups,
                                    uint &partial_count) const
  {
   if(!OpenCL || m_item_dimension == 0)
      ReturnFalse;
   long limit = OpenCL.GetMaxWorkGroupSize();
   limit = MathMin(limit, OpenCL.GetMaxLocalSize(0));
   if(limit <= 0)
      ReturnFalse;
   local_size = 1;
   while(local_size <= (uint)limit / 2)
      local_size *= 2;
   groups = (m_item_dimension + local_size - 1) / local_size;
   ulong count = m_item_dimension;
   ulong total = count;
   while(count > 1)
     {
      count = (count + 1) / 2;
      if(count > 1)
         total += count;
     }
   if(total == 0 || total > 2147483647)
      ReturnFalse;
   partial_count = (uint)total;
   return(true);
  }
//+------------------------------------------------------------------+
//| Allocates the item norm-reduction scratch buffers.               |
//+------------------------------------------------------------------+
bool CD2SkillItem::EnsureItemScratch(void)
  {
   uint local_size = 0;
   uint groups = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count))
      ReturnFalse;
   if(m_item_partials.Total() != (int)partial_count &&
      !m_item_partials.BufferInit(partial_count, 0.0f))
      ReturnFalse;
   if((m_item_partials.GetOpenCL() != OpenCL || m_item_partials.GetIndex() < 0) &&
      !m_item_partials.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_item_reduced_norm.Total() != 2 && !m_item_reduced_norm.BufferInit(2, 0.0f))
      ReturnFalse;
   if((m_item_reduced_norm.GetOpenCL() != OpenCL || m_item_reduced_norm.GetIndex() < 0) &&
      !m_item_reduced_norm.BufferCreate(OpenCL))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the staged GPU reduction that computes the item vector      |
//| norm.                                                            |
//+------------------------------------------------------------------+
bool CD2SkillItem::CalculateItemNorm(CBufferFloat *source, const uint local_size,
                                     const uint groups)
  {
   if(!source || source.GetOpenCL() != OpenCL || source.GetIndex() < 0 ||
      local_size == 0 || groups == 0)
      ReturnFalse;
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
   uint input_count = m_item_dimension;
   uint input_offset = 0;
   uint output_offset = m_item_dimension;
   const int reduce_kernel = def_k_D2SkillItemReduce;
   while(input_count > 0)
     {
      const uint reduce_groups = (input_count + 1) / 2;
      const int final_output = (reduce_groups == 1 ? 1 : 0);
      setBuffer(reduce_kernel, def_k_d2ir_partials, m_item_partials.GetIndex())
      setBuffer(reduce_kernel, def_k_d2ir_reduced, m_item_reduced_norm.GetIndex())
      setArgument(reduce_kernel, def_k_d2ir_input_count, (int)input_count)
      setArgument(reduce_kernel, def_k_d2ir_input_offset, (int)input_offset)
      setArgument(reduce_kernel, def_k_d2ir_output_offset, (int)output_offset)
      setArgument(reduce_kernel, def_k_d2ir_final_output, final_output)
      const uint padded = ((reduce_groups + local_size - 1) / local_size) * local_size;
      uint reduce_global[] = {padded};
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
//| Applies the EMA update of correction, direction and magnitude    |
//| from the gradient.                                               |
//+------------------------------------------------------------------+
bool CD2SkillItem::UpdateFromGradient(CBufferFloat *gradient)
  {
   if(!gradient || !OpenCL || gradient.GetOpenCL() != OpenCL ||
      gradient.Total() != (int)m_item_dimension || gradient.GetIndex() < 0 ||
      !EnsureItemState())
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count) ||
      !CalculateItemNorm(gradient, local_size, groups))
      ReturnFalse;
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
         ReturnFalse;
      setArgument(kernel, def_k_d2ia_phase, 1)
      kernelExecuteLoc(kernel, offset, global, local)
     }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the item state to the file handle.                         |
//+------------------------------------------------------------------+
bool CD2SkillItem::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Save(file_handle) ||
      m_item_dimension == 0 || !EnsureItemState())
      ReturnFalse;
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
//| Restores the item state from the file handle.                    |
//+------------------------------------------------------------------+
bool CD2SkillItem::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_ITEM_FORMAT)
      ReturnFalse;
   m_item_dimension = (uint)FileReadInteger(file_handle);
   m_representation = (ED2SkillRepresentation)FileReadInteger(file_handle);
   dFileReadUFloat(file_handle, m_beta_correction);
   dFileReadUFloat(file_handle, m_beta_direction);
   dFileReadUFloat(file_handle, m_beta_magnitude);
   if(m_item_dimension == 0 || m_representation < D2SkillFullResidual ||
      m_representation > D2SkillDirectionMagnitude)
      ReturnFalse;
   if(!m_item_correction.Load(file_handle) || !m_item_direction.Load(file_handle) ||
      !m_item_scale.Load(file_handle) || !m_item_observations.Load(file_handle) ||
      !m_item_mass.Load(file_handle))
      ReturnFalse;
   if(m_item_correction.Total() != (int)m_item_dimension ||
      m_item_direction.Total() != (int)m_item_dimension || m_item_scale.Total() != 1 ||
      m_item_observations.Total() != 1 || m_item_mass.Total() != 1)
      ReturnFalse;
   SetOpenCL(OpenCL);
   m_correction = GetPointer(m_item_correction);
   return(m_correction.GetOpenCL() == OpenCL && m_correction.GetIndex() >= 0);
  }
//+------------------------------------------------------------------+
//| Restores the item state and attaches the given OpenCL context.   |
//+------------------------------------------------------------------+
bool CD2SkillItem::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl ||
      FileReadInteger(file_handle, INT_VALUE) != Type())
      ReturnFalse;
   AttachOpenCL(opencl);
   return(Load(file_handle));
  }
//+------------------------------------------------------------------+
//| Rebinds all item GPU buffers to the given OpenCL context.        |
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
//| Validates Item-owned checked state without alias policy.         |
//+------------------------------------------------------------------+
bool CD2SkillItem::ValidateItemOwnedCheckedState(const bool allow_missing) const
  {
//--- Validate the fixed item geometry and the inherited live owners.
   if(CheckPointer(OpenCL) == POINTER_INVALID ||
      m_item_dimension == 0 || m_item_dimension > (uint)INT_MAX ||
      CheckPointer(Output) == POINTER_INVALID || CheckPointer(Gradient) == POINTER_INVALID ||
      Output.Total() != (int)m_item_dimension || Gradient.Total() != (int)m_item_dimension ||
      (Output.GetIndex() >= 0 && CheckPointer(Output.GetOpenCL()) == POINTER_INVALID) ||
      (Gradient.GetIndex() >= 0 && CheckPointer(Gradient.GetOpenCL()) == POINTER_INVALID))
      return(false);
//--- Obtain the existing reduction layout without allocating or rebinding scratch.
   uint local_size    = 0;
   uint groups        = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count))
      return(false);
//--- Permit only wholly absent deferred Item slots; allocated slots retain live owners.
   const CBufferFloat *buffers[7];
   int                 totals[7];
   buffers[0] = GetPointer(m_item_correction);
   totals[0] = (int)m_item_dimension;
   buffers[1] = GetPointer(m_item_direction);
   totals[1] = (int)m_item_dimension;
   buffers[2] = GetPointer(m_item_scale);
   totals[2] = 1;
   buffers[3] = GetPointer(m_item_observations);
   totals[3] = 1;
   buffers[4] = GetPointer(m_item_mass);
   totals[4] = 1;
   buffers[5] = GetPointer(m_item_partials);
   totals[5] = (int)partial_count;
   buffers[6] = GetPointer(m_item_reduced_norm);
   totals[6] = 2;
   for(int i = 0; i < 7; i++)
     {
      if(CheckPointer(buffers[i]) == POINTER_INVALID ||
         (buffers[i].Total() != totals[i] &&
          (!allow_missing || buffers[i].Total() != 0 || buffers[i].GetIndex() >= 0)) ||
         (buffers[i].GetIndex() >= 0 &&
          CheckPointer(buffers[i].GetOpenCL()) == POINTER_INVALID))
         return(false);
     }
//--- Finalize the Item-owned state validation without an alias policy.
   return(true);
  }
//+------------------------------------------------------------------+
//| Reports checked transfer support for an exact D2 skill item.     |
//+------------------------------------------------------------------+
bool CD2SkillItem::SupportsOpenCLChecked(void) const
  {
//--- Retain the exact Item type gate and the current correction-alias policy.
   if(Type() != defNeuronD2SkillItem || !ValidateItemOwnedCheckedState())
      return(false);
//--- Validate the optional correction alias without constraining borrowed owners to the outer target.
   if(m_correction != NULL)
     {
      if(CheckPointer(m_correction) == POINTER_INVALID ||
         m_correction.Total() != (int)m_item_dimension)
         return(false);
      if(m_correction != GetPointer(m_item_correction) &&
         (CheckPointer(m_correction.GetOpenCL()) == POINTER_INVALID ||
          m_correction.GetIndex() < 0))
         return(false);
     }
//--- Finalize without constraining an external correction alias to the outer owner.
   return(true);
  }
//+------------------------------------------------------------------+
//| Transfers owned item buffers before inherited context            |
//| publication.                                                     |
//+------------------------------------------------------------------+
bool CD2SkillItem::SetOpenCLChecked(COpenCLMy *obj)
  {
//--- Reject an invalid target, incomplete item or foreign borrowed correction before mutation.
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      ReturnFalseEx("checked item target or capability");
   if(m_correction != NULL && m_correction != GetPointer(m_item_correction) &&
      (CheckPointer(m_correction) == POINTER_INVALID ||
       m_correction.GetOpenCL() != obj ||
       m_correction.Total() != (int)m_item_dimension || m_correction.GetIndex() < 0))
      ReturnFalseEx("borrowed item correction target");
//--- Read every allocated foreign direct buffer before any item buffer is recreated.
   if(m_item_correction.GetIndex() >= 0 &&
      CheckPointer(m_item_correction.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item correction owner");
   if(m_item_correction.GetIndex() >= 0 && m_item_correction.GetOpenCL() != obj &&
      !m_item_correction.BufferRead())
      ReturnFalseEx("item correction read");
   if(m_item_direction.GetIndex() >= 0 &&
      CheckPointer(m_item_direction.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item direction owner");
   if(m_item_direction.GetIndex() >= 0 && m_item_direction.GetOpenCL() != obj &&
      !m_item_direction.BufferRead())
      ReturnFalseEx("item direction read");
   if(m_item_scale.GetIndex() >= 0 &&
      CheckPointer(m_item_scale.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item scale owner");
   if(m_item_scale.GetIndex() >= 0 && m_item_scale.GetOpenCL() != obj &&
      !m_item_scale.BufferRead())
      ReturnFalseEx("item scale read");
   if(m_item_observations.GetIndex() >= 0 &&
      CheckPointer(m_item_observations.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item observations owner");
   if(m_item_observations.GetIndex() >= 0 && m_item_observations.GetOpenCL() != obj &&
      !m_item_observations.BufferRead())
      ReturnFalseEx("item observations read");
   if(m_item_mass.GetIndex() >= 0 &&
      CheckPointer(m_item_mass.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item mass owner");
   if(m_item_mass.GetIndex() >= 0 && m_item_mass.GetOpenCL() != obj &&
      !m_item_mass.BufferRead())
      ReturnFalseEx("item mass read");
   if(m_item_partials.GetIndex() >= 0 &&
      CheckPointer(m_item_partials.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item partials owner");
   if(m_item_partials.GetIndex() >= 0 && m_item_partials.GetOpenCL() != obj &&
      !m_item_partials.BufferRead())
      ReturnFalseEx("item partials read");
   if(m_item_reduced_norm.GetIndex() >= 0 &&
      CheckPointer(m_item_reduced_norm.GetOpenCL()) == POINTER_INVALID)
      ReturnFalseEx("item reduced norm owner");
   if(m_item_reduced_norm.GetIndex() >= 0 && m_item_reduced_norm.GetOpenCL() != obj &&
      !m_item_reduced_norm.BufferRead())
      ReturnFalseEx("item reduced norm read");
//--- Recreate only direct item buffers that are foreign to the target or host-only.
   if(m_item_correction.GetOpenCL() != obj || m_item_correction.GetIndex() < 0)
      if(!m_item_correction.BufferCreate(obj))
         ReturnFalseEx("item correction create");
   if(m_item_direction.GetOpenCL() != obj || m_item_direction.GetIndex() < 0)
      if(!m_item_direction.BufferCreate(obj))
         ReturnFalseEx("item direction create");
   if(m_item_scale.GetOpenCL() != obj || m_item_scale.GetIndex() < 0)
      if(!m_item_scale.BufferCreate(obj))
         ReturnFalseEx("item scale create");
   if(m_item_observations.GetOpenCL() != obj || m_item_observations.GetIndex() < 0)
      if(!m_item_observations.BufferCreate(obj))
         ReturnFalseEx("item observations create");
   if(m_item_mass.GetOpenCL() != obj || m_item_mass.GetIndex() < 0)
      if(!m_item_mass.BufferCreate(obj))
         ReturnFalseEx("item mass create");
   if(m_item_partials.GetOpenCL() != obj || m_item_partials.GetIndex() < 0)
      if(!m_item_partials.BufferCreate(obj))
         ReturnFalseEx("item partials create");
   if(m_item_reduced_norm.GetOpenCL() != obj || m_item_reduced_norm.GetIndex() < 0)
      if(!m_item_reduced_norm.BufferCreate(obj))
         ReturnFalseEx("item reduced norm create");
//--- Publish inherited owners only after every owned item buffer reached the target.
   if(!CNeuronBaseOCL::SetOpenCLChecked(obj))
      ReturnFalseEx("inherited item base context");
//--- Finalize the checked item transfer without replacing the correction alias.
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears all item buffers and drops the external correction        |
//| reference.                                                       |
//+------------------------------------------------------------------+
bool CD2SkillItem::Clear(void)
  {
   m_correction = NULL;
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   return((m_item_correction.Total() == 0 || m_item_correction.Fill(0)) &&
          (m_item_direction.Total() == 0 || m_item_direction.Fill(0)) &&
          (m_item_scale.Total() == 0 || m_item_scale.Fill(0)) &&
          (m_item_observations.Total() == 0 || m_item_observations.Fill(0)) &&
          (m_item_mass.Total() == 0 || m_item_mass.Fill(0)) &&
          (m_item_partials.Total() == 0 || m_item_partials.Fill(0)) &&
          (m_item_reduced_norm.Total() == 0 || m_item_reduced_norm.Fill(0)));
  }
//+------------------------------------------------------------------+
//| Choose a D2 local size without changing the generic RAG width.   |
//+------------------------------------------------------------------+
uint CD2SkillBank::D2LaunchLocalSize(const uint work_items) const
  {
   if(!OpenCL || work_items == 0)
      ReturnFalse;
   const long device_limit = MathMin(OpenCL.GetMaxWorkGroupSize(),
                                     OpenCL.GetMaxLocalSize(0));
   if(device_limit <= 0)
      ReturnFalse;
   const ulong capped_items = MathMin((ulong)device_limit,
                                      (ulong)MathMax(work_items, (uint)32));
   if(capped_items == 0)
      ReturnFalse;
   uint local_size = 1;
   while(local_size <= (uint)capped_items / 2)
      local_size *= 2;
   return(local_size);
  }
//+------------------------------------------------------------------+
//| Computes the GPU launch layout for the retrieval and Top-1       |
//| stage.                                                           |
//+------------------------------------------------------------------+
bool CD2SkillBank::RetrievalLaunchLayout(uint &local_size, uint &partial_count) const
  {
   if(!OpenCL || m_slots == 0 || m_dimension == 0 || m_slots > RAG_FLOAT_INDEX_LIMIT ||
      OpenCL.GetMaxWorkGroupSize() < RAG_TOPK_LOCAL_WIDTH ||
      OpenCL.GetMaxLocalSize(0) < RAG_TOPK_LOCAL_WIDTH)
      ReturnFalse;
   const ulong score_items = (ulong)m_slots * m_dimension;
   if(score_items == 0 || score_items > UINT_MAX)
      ReturnFalse;
   local_size = D2LaunchLocalSize((uint)score_items);
   partial_count = (m_slots + RAG_TOPK_LOCAL_WIDTH - 1) / RAG_TOPK_LOCAL_WIDTH;
   return(local_size > 0 && partial_count > 0);
  }
//+------------------------------------------------------------------+
//| Ensures a GPU buffer of the given size exists on the bank        |
//| context.                                                         |
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureRetrievalBuffer(CBufferFloat &buffer, const int total)
  {
   if(!OpenCL || total <= 0)
      ReturnFalse;
   if(buffer.Total() != total && !buffer.BufferInit(total, 0.0f))
      ReturnFalse;
   if(buffer.GetOpenCL() == OpenCL && buffer.GetIndex() >= 0)
      return(true);
   if(buffer.GetOpenCL() != NULL && buffer.GetIndex() >= 0)
      buffer.BufferFree();
   return(buffer.BufferCreate(OpenCL));
  }
//+------------------------------------------------------------------+
//| Computes the staged radix-8 reduction scratch size.              |
//+------------------------------------------------------------------+
bool CD2SkillBank::ReductionScratchCount(const uint items, const uint batches,
      const uint fields, uint &total) const
  {
   const uint reduction_radix = 8;
   if(items == 0 || batches == 0 || fields == 0)
      ReturnFalse;
   ulong stage = items;
   ulong count = (ulong)batches * fields * stage;
   ulong required = count;
   while(stage > 1)
     {
      stage = (stage + reduction_radix - 1) / reduction_radix;
      if(stage > 1)
        {
         count = (ulong)batches * fields * stage;
         if(required > 2147483647 - count)
            ReturnFalse;
         required += count;
        }
     }
   if(required == 0 || required > 2147483647)
      ReturnFalse;
   total = (uint)required;
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the staged radix-8 GPU reduction from partials into reduced |
//| metrics.                                                         |
//+------------------------------------------------------------------+
bool CD2SkillBank::RunStagedReduction(CBufferFloat &partials, CBufferFloat &reduced,
                                      const uint items, const uint batches, const uint fields,
                                      const uint local_size, const int max_field_a,
                                      const int max_field_b)
  {
   const uint reduction_radix = 8;
   uint required = 0;
   const ulong reduced_total = (ulong)batches * fields;
   if(!OpenCL || local_size == 0 || !ReductionScratchCount(items, batches, fields, required) ||
      partials.GetOpenCL() != OpenCL || partials.GetIndex() < 0 ||
      partials.Total() < (int)required || reduced_total > 2147483647 ||
      reduced.GetOpenCL() != OpenCL || reduced.GetIndex() < 0 ||
      reduced.Total() < (int)reduced_total)
      ReturnFalse;
   uint input_count = items;
   uint input_offset = 0;
   uint output_offset = batches * items * fields;
   uint offset[1] = {0};
   const int kernel = def_k_D2SkillMetricReduce;
   while(input_count > 0)
     {
      const uint output_count = (input_count + reduction_radix - 1) / reduction_radix;
      const int final_output = (output_count == 1 ? 1 : 0);
      setBuffer(kernel, def_k_d2mr_partials, partials.GetIndex())
      setBuffer(kernel, def_k_d2mr_metrics, reduced.GetIndex())
      setArgument(kernel, def_k_d2mr_batch_count, (int)batches)
      setArgument(kernel, def_k_d2mr_input_count, (int)input_count)
      setArgument(kernel, def_k_d2mr_fields, (int)fields)
      setArgument(kernel, def_k_d2mr_input_offset, (int)input_offset)
      setArgument(kernel, def_k_d2mr_output_offset, (int)output_offset)
      setArgument(kernel, def_k_d2mr_max_field_a, max_field_a)
      setArgument(kernel, def_k_d2mr_max_field_b, max_field_b)
      setArgument(kernel, def_k_d2mr_final_output, final_output)
      const ulong active = (ulong)batches * output_count * fields;
      if(active == 0 || active > UINT_MAX)
         ReturnFalse;
      const uint stage_local_size = D2LaunchLocalSize((uint)active);
      if(stage_local_size == 0)
         ReturnFalse;
      const ulong padded = ((active + stage_local_size - 1) / stage_local_size) * stage_local_size;
      if(padded > UINT_MAX)
         ReturnFalse;
      uint global[1] = {(uint)padded};
      uint local[1] = {stage_local_size};
      kernelExecuteLoc(kernel, offset, global, local)
      if(final_output != 0)
         break;
      input_count = output_count;
      input_offset = output_offset;
      output_offset += batches * output_count * fields;
     }
   return(true);
  }
//+------------------------------------------------------------------+
//| Copies the source into the partials buffer, then runs the staged |
//| reduction.                                                       |
//+------------------------------------------------------------------+
bool CD2SkillBank::RunStagedReduction(CBufferFloat &source, CBufferFloat &partials,
                                      CBufferFloat &reduced, const uint items,
                                      const uint batches, const uint fields,
                                      const uint local_size, const int max_field_a,
                                      const int max_field_b)
  {
   if(!OpenCL || items == 0 || source.GetOpenCL() != OpenCL ||
      source.GetIndex() < 0 || source.Total() != (int)items)
      ReturnFalse;
   if(source.GetIndex() != partials.GetIndex())
     {
      uint offset[1] = {0};
      uint global[1] = {items};
      setBuffer(def_k_CopyBufferRaw, def_k_copy_raw_source, source.GetIndex())
      setBuffer(def_k_CopyBufferRaw, def_k_copy_raw_destination, partials.GetIndex())
      setArgument(def_k_CopyBufferRaw, def_k_copy_raw_total, (int)items)
      kernelExecute(def_k_CopyBufferRaw, offset, global)
     }
   return(RunStagedReduction(partials, reduced, items, batches, fields,
                             local_size, max_field_a, max_field_b));
  }
//+------------------------------------------------------------------+
//| Allocates all retrieval-stage scratch buffers.                   |
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureRetrievalScratch(void)
  {
   uint local_size = 0;
   uint partial_count = 0;
   uint score_scratch = 0;
   if(!RetrievalLaunchLayout(local_size, partial_count) || partial_count > INT_MAX / 2 ||
      !ReductionScratchCount(m_dimension, m_slots, 4, score_scratch) ||
      m_slots > INT_MAX / 4)
      ReturnFalse;
   const int pairs = (int)(2 * partial_count);
   return(EnsureRetrievalBuffer(m_retrieval_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_score_partials, (int)score_scratch) &&
          EnsureRetrievalBuffer(m_score_metrics, (int)(4 * m_slots)) &&
          EnsureRetrievalBuffer(m_retrieval_partial, pairs) &&
          EnsureRetrievalBuffer(m_retrieval_merge, pairs) &&
          EnsureRetrievalBuffer(m_retrieval_top, 2));
  }
//+------------------------------------------------------------------+
//| Computes the GPU launch layout for the bank update stage.        |
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateLaunchLayout(uint &local_size, uint &groups) const
  {
   if(!OpenCL || m_dimension == 0)
      ReturnFalse;
   uint limit = (uint)MathMin(OpenCL.GetMaxWorkGroupSize(), OpenCL.GetMaxLocalSize(0));
   if(limit == 0)
      ReturnFalse;
   local_size = 1;
   while(local_size <= limit / 2)
      local_size *= 2;
   groups = (m_dimension + local_size - 1) / local_size;
   return(groups > 0);
  }
//+------------------------------------------------------------------+
//| Allocates all update-stage scratch buffers.                      |
//+------------------------------------------------------------------+
bool CD2SkillBank::EnsureUpdateScratch(void)
  {
   uint local_size = 0;
   uint groups = 0;
   uint metric_scratch = 0;
   uint diagnostic_norm_scratch = 0;
   uint diagnostic_scratch = 0;
   uint influence_scratch = 0;
   uint gradient_influence_scratch = 0;
   if(!UpdateLaunchLayout(local_size, groups) ||
      !ReductionScratchCount(m_dimension, 1, D2SKILL_METRICS, metric_scratch) ||
      !ReductionScratchCount(m_dimension, m_slots, 1, diagnostic_norm_scratch) ||
      !ReductionScratchCount(m_slots, 1, D2SKILL_DIAGNOSTIC_PARTIALS, diagnostic_scratch) ||
      !ReductionScratchCount(m_slots, 1, 1, influence_scratch) ||
      !ReductionScratchCount(m_dimension, 1, 1, gradient_influence_scratch))
      ReturnFalse;
   return(EnsureRetrievalBuffer(m_free_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_eviction_scores, (int)m_slots) &&
          EnsureRetrievalBuffer(m_free_top, 2) && EnsureRetrievalBuffer(m_eviction_top, 2) &&
          EnsureRetrievalBuffer(m_metric_partials, (int)metric_scratch) &&
          EnsureRetrievalBuffer(m_metrics, D2SKILL_METRICS) &&
          EnsureRetrievalBuffer(m_decision, 4) &&
          EnsureRetrievalBuffer(m_diagnostics_norm_partials, (int)diagnostic_norm_scratch) &&
          EnsureRetrievalBuffer(m_diagnostics_norms, (int)m_slots) &&
          EnsureRetrievalBuffer(m_diagnostics_partials, (int)diagnostic_scratch) &&
          EnsureRetrievalBuffer(m_diagnostics_totals, D2SKILL_DIAGNOSTIC_PARTIALS) &&
          EnsureRetrievalBuffer(m_influence_partials, (int)influence_scratch) &&
          EnsureRetrievalBuffer(m_gradient_influence_partials, (int)gradient_influence_scratch) &&
          EnsureRetrievalBuffer(m_gradient_influence_total, 1));
  }
//+------------------------------------------------------------------+
//| Finds the best scoring slot using the generic RAG Top-1          |
//| partial/merge kernels.                                           |
//+------------------------------------------------------------------+
bool CD2SkillBank::RunTop1(CBufferFloat &scores, CBufferFloat &top)
  {
   uint local_size = 0;
   uint partial_count = 0;
   if(!RetrievalLaunchLayout(local_size, partial_count) || !EnsureRetrievalScratch())
      ReturnFalse;
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
//| Computes the per-slot metrics for the given state and gradient.  |
//+------------------------------------------------------------------+
bool CD2SkillBank::RunMetrics(CBufferFloat *state, CBufferFloat *gradient,
                              const uint local_size, const uint groups)
  {
   if(!state || !gradient)
      ReturnFalse;
   uint offset[1] = {0};
   const int partial_kernel = def_k_D2SkillMetricPartial;
   setBuffer(partial_kernel, def_k_d2mp_query, state.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_gradient, gradient.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_selected_slot, m_selected_slot.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_keys, m_keys.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_directions, m_directions.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_used, m_used.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_top, m_retrieval_top.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_key, m_keys.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_candidate_direction, m_directions.GetIndex())
   setBuffer(partial_kernel, def_k_d2mp_partials, m_metric_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2mp_slots, (int)m_slots)
   setArgument(partial_kernel, def_k_d2mp_dimension, (int)m_dimension)
   uint partial_global[1] = {groups * local_size};
   uint partial_local[1] = {local_size};
   kernelExecuteLoc(partial_kernel, offset, partial_global, partial_local)
   return(RunStagedReduction(m_metric_partials, m_metrics, m_dimension, 1,
                             D2SKILL_METRICS, local_size));
  }
//+------------------------------------------------------------------+
//| Computes the bank vector norm for the active representation.     |
//+------------------------------------------------------------------+
bool CD2SkillBank::CalculateBankVectorNorm(const uint local_size, const uint groups)
  {
   if(local_size == 0 || groups == 0 || !EnsureItemScratch())
      ReturnFalse;
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
//--- The item scratch is sized for the wider binary Item norm reduction, so it
//--- also contains every radix-8 stage required for this one-field bank norm.
   return(RunStagedReduction(m_item_partials, m_item_reduced_norm, m_dimension,
                             1, 1, local_size));
  }
//+------------------------------------------------------------------+
//| Initializes the bank with the given slot count and dimension,    |
//| allocating every GPU buffer.                                     |
//+------------------------------------------------------------------+
bool CD2SkillBank::Init(uint dimension, uint myIndex, COpenCLMy *open_cl,
                        uint slots, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!open_cl || dimension == 0 || slots == 0 || slots > RAG_FLOAT_INDEX_LIMIT ||
      !CD2SkillItem::Init(0, myIndex, open_cl, dimension, optimization_type, batch))
      ReturnFalse;
   m_dimension = dimension;
   m_slots = slots;
   m_active_slot = UINT_MAX;
   const long total = (long)m_slots * (long)m_dimension;
   if(total <= 0 || total > 2147483647)
      ReturnFalse;
   if(!InitBuffer(m_keys, (int)total) || !InitBuffer(m_corrections, (int)total) ||
      !InitBuffer(m_directions, (int)total) || !InitBuffer(m_scales, (int)m_slots) ||
      !InitBuffer(m_utility, (int)m_slots) || !InitBuffer(m_observations, (int)m_slots) ||
      !InitBuffer(m_uses, (int)m_slots) || !InitBuffer(m_mass, (int)m_slots) ||
      !InitBuffer(m_used, (int)m_slots) || !InitBuffer(m_state, (int)m_slots) ||
      !InitBuffer(m_age, (int)m_slots) || !InitBuffer(m_protection, (int)m_slots) ||
      !InitBuffer(m_influence, (int)m_slots) || !InitBuffer(m_utility_applied, (int)m_slots) ||
      !InitBuffer(m_influence_total, 1) ||
      !InitBuffer(m_gradient_influence_total, 1) ||
      !InitBuffer(m_retrieved, (int)m_slots) ||
      !InitBuffer(m_selected_correction, (int)m_dimension) ||
      !InitBuffer(m_selected_slot, 1, -1.0f) ||
      !InitBuffer(m_selected_score, 1, -3.402823e+38f) ||
      !InitBuffer(m_diagnostics, D2SKILL_DIAGNOSTICS) ||
      !InitBuffer(m_utility_distribution, D2SKILL_DISTRIBUTION_BINS) ||
      !InitBuffer(m_usage_distribution, D2SKILL_DISTRIBUTION_BINS) ||
      !EnsureRetrievalScratch() || !EnsureUpdateScratch())
      ReturnFalse;
   if(!SetCorrectionBuffer(GetPointer(m_selected_correction)))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Rebinds every bank GPU buffer to the given OpenCL context and    |
//| re-exposes the selected correction.                              |
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
   RebindBuffer(m_influence, obj);
   RebindBuffer(m_utility_applied, obj);
   RebindBuffer(m_influence_total, obj);
   RebindBuffer(m_influence_partials, obj);
   RebindBuffer(m_gradient_influence_total, obj);
   RebindBuffer(m_gradient_influence_partials, obj);
   RebindBuffer(m_retrieved, obj);
   RebindBuffer(m_selected_correction, obj);
   RebindBuffer(m_selected_slot, obj);
   RebindBuffer(m_selected_score, obj);
   RebindBuffer(m_diagnostics, obj);
   RebindBuffer(m_utility_distribution, obj);
   RebindBuffer(m_usage_distribution, obj);
   RebindBuffer(m_retrieval_scores, obj);
   RebindBuffer(m_score_partials, obj);
   RebindBuffer(m_score_metrics, obj);
   RebindBuffer(m_retrieval_partial, obj);
   RebindBuffer(m_retrieval_merge, obj);
   RebindBuffer(m_retrieval_top, obj);
   RebindBuffer(m_free_scores, obj);
   RebindBuffer(m_eviction_scores, obj);
   RebindBuffer(m_free_top, obj);
   RebindBuffer(m_eviction_top, obj);
   RebindBuffer(m_metric_partials, obj);
   RebindBuffer(m_metrics, obj);
   RebindBuffer(m_decision, obj);
   RebindBuffer(m_diagnostics_norm_partials, obj);
   RebindBuffer(m_diagnostics_norms, obj);
   RebindBuffer(m_diagnostics_partials, obj);
   RebindBuffer(m_diagnostics_totals, obj);
   EnsureRetrievalScratch();
   EnsureUpdateScratch();
//--- CD2SkillItem::SetOpenCL binds its standalone correction buffer.
//--- A bank must instead expose its selected slot correction to the item.
   if(m_selected_correction.Total() == (int)m_dimension &&
      m_selected_correction.GetOpenCL() == obj && m_selected_correction.GetIndex() >= 0)
      m_correction = GetPointer(m_selected_correction);
  }
//+------------------------------------------------------------------+
//| Reports checked transfer support for an exact D2 skill bank.     |
//+------------------------------------------------------------------+
bool CD2SkillBank::SupportsOpenCLChecked(void) const
  {
   return(Type() == defNeuronD2SkillBank && ValidateBankOwnedCheckedState(false));
  }
//+------------------------------------------------------------------+
//| Checks strict Bank transfer support for a specific target.       |
//+------------------------------------------------------------------+
bool CD2SkillBank::SupportsOpenCLChecked(const COpenCLMy *target) const
  {
   return(Type() == defNeuronD2SkillBank &&
          ValidateBankOwnedCheckedState(false, target));
  }
//+------------------------------------------------------------------+
//| Validates the Bank-owned checked state before any mutation.      |
//+------------------------------------------------------------------+
bool CD2SkillBank::ValidateBankOwnedCheckedState(const bool allow_missing_transient,
      const COpenCLMy *target) const
  {
//--- Validate fixed geometry before deriving any scratch-buffer sizes.
   if(!ValidateItemOwnedCheckedState(allow_missing_transient) ||
      m_slots == 0 || m_dimension == 0 || m_dimension != m_item_dimension ||
      m_slots > RAG_FLOAT_INDEX_LIMIT)
      return(false);
   const long vector_total = (long)m_slots * (long)m_dimension;
   if(vector_total <= 0 || vector_total > INT_MAX || m_slots > (uint)INT_MAX ||
      m_slots > (uint)INT_MAX / 4)
      return(false);
//--- Derive every existing scratch size without allocation or mutable validation.
   uint retrieval_local             = 0;
   uint retrieval_partials          = 0;
   uint score_partials              = 0;
   uint metric_partials             = 0;
   uint diagnostics_norm_partials   = 0;
   uint diagnostics_partials        = 0;
   uint influence_partials          = 0;
   uint gradient_influence_partials = 0;
   if(!RetrievalLaunchLayout(retrieval_local, retrieval_partials) ||
      retrieval_partials > (uint)INT_MAX / 2 ||
      !ReductionScratchCount(m_dimension, m_slots, 4, score_partials) ||
      !ReductionScratchCount(m_dimension, 1, D2SKILL_METRICS, metric_partials) ||
      !ReductionScratchCount(m_dimension, m_slots, 1, diagnostics_norm_partials) ||
      !ReductionScratchCount(m_slots, 1, D2SKILL_DIAGNOSTIC_PARTIALS, diagnostics_partials) ||
      !ReductionScratchCount(m_slots, 1, 1, influence_partials) ||
      !ReductionScratchCount(m_dimension, 1, 1, gradient_influence_partials) ||
      score_partials > (uint)INT_MAX || metric_partials > (uint)INT_MAX ||
      diagnostics_norm_partials > (uint)INT_MAX || diagnostics_partials > (uint)INT_MAX ||
      influence_partials > (uint)INT_MAX || gradient_influence_partials > (uint)INT_MAX)
      return(false);
//--- Map all Bank-owned slots to their established host shapes.
   const CBufferFloat *buffers[42];
   int                 totals[42];
   buffers[0] = GetPointer(m_keys);
   totals[0] = (int)vector_total;
   buffers[1] = GetPointer(m_corrections);
   totals[1] = (int)vector_total;
   buffers[2] = GetPointer(m_directions);
   totals[2] = (int)vector_total;
   buffers[3] = GetPointer(m_scales);
   totals[3] = (int)m_slots;
   buffers[4] = GetPointer(m_utility);
   totals[4] = (int)m_slots;
   buffers[5] = GetPointer(m_observations);
   totals[5] = (int)m_slots;
   buffers[6] = GetPointer(m_uses);
   totals[6] = (int)m_slots;
   buffers[7] = GetPointer(m_mass);
   totals[7] = (int)m_slots;
   buffers[8] = GetPointer(m_used);
   totals[8] = (int)m_slots;
   buffers[9] = GetPointer(m_state);
   totals[9] = (int)m_slots;
   buffers[10] = GetPointer(m_age);
   totals[10] = (int)m_slots;
   buffers[11] = GetPointer(m_protection);
   totals[11] = (int)m_slots;
   buffers[12] = GetPointer(m_influence);
   totals[12] = (int)m_slots;
   buffers[13] = GetPointer(m_utility_applied);
   totals[13] = (int)m_slots;
   buffers[14] = GetPointer(m_retrieved);
   totals[14] = (int)m_slots;
   buffers[15] = GetPointer(m_retrieval_scores);
   totals[15] = (int)m_slots;
   buffers[16] = GetPointer(m_free_scores);
   totals[16] = (int)m_slots;
   buffers[17] = GetPointer(m_eviction_scores);
   totals[17] = (int)m_slots;
   buffers[18] = GetPointer(m_diagnostics_norms);
   totals[18] = (int)m_slots;
   buffers[19] = GetPointer(m_influence_total);
   totals[19] = 1;
   buffers[20] = GetPointer(m_gradient_influence_total);
   totals[20] = 1;
   buffers[21] = GetPointer(m_selected_slot);
   totals[21] = 1;
   buffers[22] = GetPointer(m_selected_score);
   totals[22] = 1;
   buffers[23] = GetPointer(m_selected_correction);
   totals[23] = (int)m_dimension;
   buffers[24] = GetPointer(m_diagnostics);
   totals[24] = D2SKILL_DIAGNOSTICS;
   buffers[25] = GetPointer(m_utility_distribution);
   totals[25] = D2SKILL_DISTRIBUTION_BINS;
   buffers[26] = GetPointer(m_usage_distribution);
   totals[26] = D2SKILL_DISTRIBUTION_BINS;
   buffers[27] = GetPointer(m_score_partials);
   totals[27] = (int)score_partials;
   buffers[28] = GetPointer(m_score_metrics);
   totals[28] = (int)(4 * m_slots);
   buffers[29] = GetPointer(m_retrieval_partial);
   totals[29] = (int)(2 * retrieval_partials);
   buffers[30] = GetPointer(m_retrieval_merge);
   totals[30] = (int)(2 * retrieval_partials);
   buffers[31] = GetPointer(m_retrieval_top);
   totals[31] = 2;
   buffers[32] = GetPointer(m_free_top);
   totals[32] = 2;
   buffers[33] = GetPointer(m_eviction_top);
   totals[33] = 2;
   buffers[34] = GetPointer(m_metric_partials);
   totals[34] = (int)metric_partials;
   buffers[35] = GetPointer(m_metrics);
   totals[35] = D2SKILL_METRICS;
   buffers[36] = GetPointer(m_decision);
   totals[36] = 4;
   buffers[37] = GetPointer(m_diagnostics_norm_partials);
   totals[37] = (int)diagnostics_norm_partials;
   buffers[38] = GetPointer(m_diagnostics_partials);
   totals[38] = (int)diagnostics_partials;
   buffers[39] = GetPointer(m_diagnostics_totals);
   totals[39] = D2SKILL_DIAGNOSTIC_PARTIALS;
   buffers[40] = GetPointer(m_influence_partials);
   totals[40] = (int)influence_partials;
   buffers[41] = GetPointer(m_gradient_influence_partials);
   totals[41] = (int)gradient_influence_partials;
//--- Deferred loads may omit only transient slots; every present slot keeps its shape and owner.
   for(int i = 0; i < 42; i++)
     {
      const bool transient = (i >= 13 && i <= 20) || i >= 27;
      const bool absent = buffers[i].Total() == 0 && buffers[i].GetIndex() < 0;
      if(CheckPointer(buffers[i]) == POINTER_INVALID ||
         (buffers[i].Total() != totals[i] &&
          (!allow_missing_transient || !transient || !absent)) ||
         (buffers[i].GetIndex() >= 0 &&
          CheckPointer(buffers[i].GetOpenCL()) == POINTER_INVALID))
         return(false);
     }
//--- Only Item and selected-Bank aliases may remain host-only for checked repair.
   if(m_correction != NULL && m_correction != GetPointer(m_item_correction) &&
      m_correction != GetPointer(m_selected_correction) &&
      (CheckPointer(m_correction) == POINTER_INVALID ||
       m_correction.Total() != (int)m_dimension || m_correction.GetIndex() < 0 ||
       CheckPointer(m_correction.GetOpenCL()) == POINTER_INVALID ||
       (target != NULL && m_correction.GetOpenCL() != target)))
      return(false);
//--- Finalize exact Bank capability after every owned slot is checked.
   return(true);
  }
//+------------------------------------------------------------------+
//| Checks a loaded Bank before its missing transient slots exist.   |
//+------------------------------------------------------------------+
bool CD2SkillBank::SupportsDeferredOpenCLChecked(const COpenCLMy *target) const
  {
   return(Type() == defNeuronD2SkillBank &&
          ValidateBankOwnedCheckedState(true, target));
  }
//+------------------------------------------------------------------+
//| Initializes absent loaded-bank transient host slots only.        |
//+------------------------------------------------------------------+
bool CD2SkillBank::PrepareOpenCLChecked(COpenCLMy *target)
  {
//--- Complete target-aware prevalidation before any host-only slot is initialized.
   if(!SupportsDeferredOpenCLChecked(target))
      return(false);
   if(!PrepareItemOwnedCheckedState())
      return(false);
//--- Derive every transient host-buffer shape without allocating device storage.
   uint retrieval_local              = 0;
   uint retrieval_partials           = 0;
   uint score_partials               = 0;
   uint metric_partials              = 0;
   uint diagnostic_norm_partials     = 0;
   uint diagnostic_partials          = 0;
   uint influence_partials           = 0;
   uint gradient_influence_partials  = 0;
   if(!RetrievalLaunchLayout(retrieval_local, retrieval_partials))
      ReturnFalseEx("bank retrieval launch layout");
   if(!ReductionScratchCount(m_dimension, m_slots, 4, score_partials))
      ReturnFalseEx("bank score partials shape");
   if(!ReductionScratchCount(m_dimension, 1, D2SKILL_METRICS, metric_partials))
      ReturnFalseEx("bank metric partials shape");
   if(!ReductionScratchCount(m_dimension, m_slots, 1, diagnostic_norm_partials))
      ReturnFalseEx("bank diagnostic norm partials shape");
   if(!ReductionScratchCount(m_slots, 1, D2SKILL_DIAGNOSTIC_PARTIALS, diagnostic_partials))
      ReturnFalseEx("bank diagnostic partials shape");
   if(!ReductionScratchCount(m_slots, 1, 1, influence_partials))
      ReturnFalseEx("bank influence partials shape");
   if(!ReductionScratchCount(m_dimension, 1, 1, gradient_influence_partials))
      ReturnFalseEx("bank gradient influence partials shape");
//--- Map transient Bank slots to their established host shapes.
   CBufferFloat *buffers[23];
   int           totals[23];
   string        slots[23] = {"utility applied", "influence total",
                              "gradient influence total", "retrieved", "retrieval scores",
                              "free scores", "eviction scores", "diagnostic norms",
                              "score partials", "score metrics", "retrieval partial",
                              "retrieval merge", "retrieval top", "free top", "eviction top",
                              "metric partials", "metrics", "decision",
                              "diagnostic norm partials", "diagnostic partials",
                              "diagnostic totals", "influence partials",
                              "gradient influence partials"
                             };
   buffers[0] = GetPointer(m_utility_applied);
   totals[0] = (int)m_slots;
   buffers[1] = GetPointer(m_influence_total);
   totals[1] = 1;
   buffers[2] = GetPointer(m_gradient_influence_total);
   totals[2] = 1;
   buffers[3] = GetPointer(m_retrieved);
   totals[3] = (int)m_slots;
   buffers[4] = GetPointer(m_retrieval_scores);
   totals[4] = (int)m_slots;
   buffers[5] = GetPointer(m_free_scores);
   totals[5] = (int)m_slots;
   buffers[6] = GetPointer(m_eviction_scores);
   totals[6] = (int)m_slots;
   buffers[7] = GetPointer(m_diagnostics_norms);
   totals[7] = (int)m_slots;
   buffers[8] = GetPointer(m_score_partials);
   totals[8] = (int)score_partials;
   buffers[9] = GetPointer(m_score_metrics);
   totals[9] = (int)(4 * m_slots);
   buffers[10] = GetPointer(m_retrieval_partial);
   totals[10] = (int)(2 * retrieval_partials);
   buffers[11] = GetPointer(m_retrieval_merge);
   totals[11] = (int)(2 * retrieval_partials);
   buffers[12] = GetPointer(m_retrieval_top);
   totals[12] = 2;
   buffers[13] = GetPointer(m_free_top);
   totals[13] = 2;
   buffers[14] = GetPointer(m_eviction_top);
   totals[14] = 2;
   buffers[15] = GetPointer(m_metric_partials);
   totals[15] = (int)metric_partials;
   buffers[16] = GetPointer(m_metrics);
   totals[16] = D2SKILL_METRICS;
   buffers[17] = GetPointer(m_decision);
   totals[17] = 4;
   buffers[18] = GetPointer(m_diagnostics_norm_partials);
   totals[18] = (int)diagnostic_norm_partials;
   buffers[19] = GetPointer(m_diagnostics_partials);
   totals[19] = (int)diagnostic_partials;
   buffers[20] = GetPointer(m_diagnostics_totals);
   totals[20] = D2SKILL_DIAGNOSTIC_PARTIALS;
   buffers[21] = GetPointer(m_influence_partials);
   totals[21] = (int)influence_partials;
   buffers[22] = GetPointer(m_gradient_influence_partials);
   totals[22] = (int)gradient_influence_partials;
//--- Reject malformed optional slots before initializing only their absent host mirrors.
   for(int i = 0; i < 23; i++)
      if(CheckPointer(buffers[i]) == POINTER_INVALID ||
         (buffers[i].Total() != totals[i] &&
          (buffers[i].Total() != 0 || buffers[i].GetIndex() >= 0)) ||
         (buffers[i].GetIndex() >= 0 &&
          CheckPointer(buffers[i].GetOpenCL()) == POINTER_INVALID))
         ReturnFalseEx("bank " + slots[i] + " checked state");
//--- Initialize only the accepted unallocated transient buffers on the CPU.
   for(int i = 0; i < 23; i++)
      if(buffers[i].Total() == 0 && !buffers[i].BufferInit(totals[i], 0.0f))
         ReturnFalseEx("bank " + slots[i] + " host init");
   return(true);
  }
//+------------------------------------------------------------------+
//| Transfers all Bank buffers before the qualified Item transfer.   |
//+------------------------------------------------------------------+
bool CD2SkillBank::SetOpenCLChecked(COpenCLMy *obj)
  {
//--- Reject a foreign borrowed alias before any owned Bank mutation.
   if(CheckPointer(obj) == POINTER_INVALID || !SupportsOpenCLChecked())
      ReturnFalseEx("checked bank target or capability");
   if(m_correction != NULL && m_correction != GetPointer(m_item_correction) &&
      m_correction != GetPointer(m_selected_correction) &&
      (m_correction.GetOpenCL() != obj || m_correction.GetIndex() < 0 ||
       m_correction.Total() != (int)m_dimension ||
       CheckPointer(m_correction.GetOpenCL()) == POINTER_INVALID))
      ReturnFalseEx("borrowed bank correction target");
//--- Read every foreign allocated Bank buffer before any Bank buffer is recreated.
   CBufferFloat *buffers[42];
   string       labels[42];
   buffers[0] = GetPointer(m_keys);
   labels[0] = "bank keys";
   buffers[1] = GetPointer(m_corrections);
   labels[1] = "bank corrections";
   buffers[2] = GetPointer(m_directions);
   labels[2] = "bank directions";
   buffers[3] = GetPointer(m_scales);
   labels[3] = "bank scales";
   buffers[4] = GetPointer(m_utility);
   labels[4] = "bank utility";
   buffers[5] = GetPointer(m_observations);
   labels[5] = "bank observations";
   buffers[6] = GetPointer(m_uses);
   labels[6] = "bank uses";
   buffers[7] = GetPointer(m_mass);
   labels[7] = "bank mass";
   buffers[8] = GetPointer(m_used);
   labels[8] = "bank used";
   buffers[9] = GetPointer(m_state);
   labels[9] = "bank state";
   buffers[10] = GetPointer(m_age);
   labels[10] = "bank age";
   buffers[11] = GetPointer(m_protection);
   labels[11] = "bank protection";
   buffers[12] = GetPointer(m_influence);
   labels[12] = "bank influence";
   buffers[13] = GetPointer(m_utility_applied);
   labels[13] = "bank utility applied";
   buffers[14] = GetPointer(m_retrieved);
   labels[14] = "bank retrieved";
   buffers[15] = GetPointer(m_retrieval_scores);
   labels[15] = "bank retrieval scores";
   buffers[16] = GetPointer(m_free_scores);
   labels[16] = "bank free scores";
   buffers[17] = GetPointer(m_eviction_scores);
   labels[17] = "bank eviction scores";
   buffers[18] = GetPointer(m_diagnostics_norms);
   labels[18] = "bank diagnostics norms";
   buffers[19] = GetPointer(m_influence_total);
   labels[19] = "bank influence total";
   buffers[20] = GetPointer(m_gradient_influence_total);
   labels[20] = "bank gradient influence total";
   buffers[21] = GetPointer(m_selected_slot);
   labels[21] = "bank selected slot";
   buffers[22] = GetPointer(m_selected_score);
   labels[22] = "bank selected score";
   buffers[23] = GetPointer(m_selected_correction);
   labels[23] = "bank selected correction";
   buffers[24] = GetPointer(m_diagnostics);
   labels[24] = "bank diagnostics";
   buffers[25] = GetPointer(m_utility_distribution);
   labels[25] = "bank utility distribution";
   buffers[26] = GetPointer(m_usage_distribution);
   labels[26] = "bank usage distribution";
   buffers[27] = GetPointer(m_score_partials);
   labels[27] = "bank score partials";
   buffers[28] = GetPointer(m_score_metrics);
   labels[28] = "bank score metrics";
   buffers[29] = GetPointer(m_retrieval_partial);
   labels[29] = "bank retrieval partial";
   buffers[30] = GetPointer(m_retrieval_merge);
   labels[30] = "bank retrieval merge";
   buffers[31] = GetPointer(m_retrieval_top);
   labels[31] = "bank retrieval top";
   buffers[32] = GetPointer(m_free_top);
   labels[32] = "bank free top";
   buffers[33] = GetPointer(m_eviction_top);
   labels[33] = "bank eviction top";
   buffers[34] = GetPointer(m_metric_partials);
   labels[34] = "bank metric partials";
   buffers[35] = GetPointer(m_metrics);
   labels[35] = "bank metrics";
   buffers[36] = GetPointer(m_decision);
   labels[36] = "bank decision";
   buffers[37] = GetPointer(m_diagnostics_norm_partials);
   labels[37] = "bank diagnostics norm partials";
   buffers[38] = GetPointer(m_diagnostics_partials);
   labels[38] = "bank diagnostics partials";
   buffers[39] = GetPointer(m_diagnostics_totals);
   labels[39] = "bank diagnostics totals";
   buffers[40] = GetPointer(m_influence_partials);
   labels[40] = "bank influence partials";
   buffers[41] = GetPointer(m_gradient_influence_partials);
   labels[41] = "bank gradient influence partials";
   for(int i = 0; i < 42; i++)
     {
      if(buffers[i].GetIndex() >= 0 &&
         CheckPointer(buffers[i].GetOpenCL()) == POINTER_INVALID)
         ReturnFalseEx(labels[i] + " owner");
      if(buffers[i].GetIndex() >= 0 && buffers[i].GetOpenCL() != obj &&
         !buffers[i].BufferRead())
         ReturnFalseEx(labels[i] + " read");
     }
//--- Recreate only foreign or host-only Bank slots, retaining target-resident values.
   for(int i = 0; i < 42; i++)
      if(buffers[i].GetOpenCL() != obj || buffers[i].GetIndex() < 0)
         if(!buffers[i].BufferCreate(obj))
            ReturnFalseEx(labels[i] + " create");
//--- Transfer Item-owned state and publish the outer context only after Bank success.
   if(!CD2SkillItem::SetOpenCLChecked(obj))
      ReturnFalseEx("checked bank item transfer");
//--- Finalize after the qualified Item transfer publishes the outer context.
   return(true);
  }
//+------------------------------------------------------------------+
//| Scores all slots against the state and gathers the winning       |
//| correction into the selected buffers.                            |
//+------------------------------------------------------------------+
bool CD2SkillBank::Retrieve(CBufferFloat *state, const bool mark_retrieved)
  {
   if(!m_enabled || !state || state.Total() != (int)m_dimension ||
      state.GetOpenCL() != OpenCL || state.GetIndex() < 0 ||
      !EnsureRetrievalScratch())
      ReturnFalse;
   uint score_local_size = 0;
   uint partial_count = 0;
   if(!RetrievalLaunchLayout(score_local_size, partial_count))
      ReturnFalse;
   uint offset[1] = {0};
   const int score_kernel = def_k_D2SkillScoreSlots;
   setBuffer(score_kernel, def_k_d2ss_query, state.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_keys, m_keys.GetIndex())
   setBuffer(score_kernel, def_k_d2ss_partials, m_score_partials.GetIndex())
   setArgument(score_kernel, def_k_d2ss_slots, (int)m_slots)
   setArgument(score_kernel, def_k_d2ss_dimension, (int)m_dimension)
   const uint score_items = m_slots * m_dimension;
   const uint score_padded = ((score_items + score_local_size - 1) / score_local_size) * score_local_size;
   uint score_global[1] = {score_padded};
   uint score_local[1] = {score_local_size};
   kernelExecuteLoc(score_kernel, offset, score_global, score_local)
   if(!RunStagedReduction(m_score_partials, m_score_metrics, m_dimension, m_slots, 4,
                          score_local_size))
      ReturnFalse;
   const int finalize_kernel = def_k_D2SkillScoreFinalize;
   setBuffer(finalize_kernel, def_k_d2sf_metrics, m_score_metrics.GetIndex())
   setBuffer(finalize_kernel, def_k_d2sf_utility, m_utility.GetIndex())
   setBuffer(finalize_kernel, def_k_d2sf_used, m_used.GetIndex())
   setBuffer(finalize_kernel, def_k_d2sf_state, m_state.GetIndex())
   setBuffer(finalize_kernel, def_k_d2sf_scores, m_retrieval_scores.GetIndex())
   setArgument(finalize_kernel, def_k_d2sf_slots, (int)m_slots)
   setArgument(finalize_kernel, def_k_d2sf_similarity_threshold, m_similarity_threshold)
   setArgument(finalize_kernel, def_k_d2sf_utility_weight, m_utility_weight)
   setArgument(finalize_kernel, def_k_d2sf_min_utility, m_min_utility)
   setArgument(finalize_kernel, def_k_d2sf_utility_policy, (int)m_utility_aware)
   const uint finalize_local_size = D2LaunchLocalSize(m_slots);
   if(finalize_local_size == 0)
      ReturnFalse;
   const uint finalize_padded = ((m_slots + finalize_local_size - 1) /
                                 finalize_local_size) * finalize_local_size;
   uint finalize_global[1] = {finalize_padded};
   uint finalize_local[1] = {finalize_local_size};
   kernelExecuteLoc(finalize_kernel, offset, finalize_global, finalize_local)
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
   setBuffer(gather_kernel, def_k_d2gc_retrieved, m_retrieved.GetIndex())
   setArgument(gather_kernel, def_k_d2gc_mark_retrieved, (int)mark_retrieved)
   uint gather_global[1] = {m_dimension};
   kernelExecute(gather_kernel, offset, gather_global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Retrieves the best correction for the source and applies the     |
//| item residual.                                                   |
//+------------------------------------------------------------------+
bool CD2SkillBank::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || CheckPointer(NeuronOCL) == POINTER_INVALID ||
      CheckPointer(Output) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *source = NeuronOCL.getOutput();
   if(!source || source.Total() != (int)m_dimension || source.GetOpenCL() != OpenCL)
     {
      PrintFormat("D2SkillBank forward contract: source=%p total=%d expected=%u source_cl=%p bank_cl=%p",
                  source, (source ? source.Total() : -1), m_dimension,
                  (source ? source.GetOpenCL() : NULL), OpenCL);
      ReturnFalse;
     }
   if(!m_enabled)
      return(CopyBufferRaw(source, Output, m_dimension));
   if(!Retrieve(source))
      ReturnFalse;
//--- CD2SkillItem owns the residual application.  The bank only selects
//--- m_selected_correction and exposes it through the inherited buffer.
   return(CD2SkillItem::feedForward(NeuronOCL));
  }
//+------------------------------------------------------------------+
//| Select the current slot without routing its correction.          |
//+------------------------------------------------------------------+
bool CD2SkillBank::Observe(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !m_enabled || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   CBufferFloat *source = NeuronOCL.getOutput();
   if(!source || source.Total() != (int)m_dimension || source.GetOpenCL() != OpenCL ||
      source.GetIndex() < 0)
      ReturnFalse;
   return(Retrieve(source, false));
  }
//+------------------------------------------------------------------+
//| Accumulates g dot (alpha * correction) for the forward winner.   |
//+------------------------------------------------------------------+
bool CD2SkillBank::AccumulateInfluenceFromGradient(CBufferFloat *gradient)
  {
   if(!m_enabled || !OpenCL || !gradient || gradient.Total() != (int)m_dimension ||
      gradient.GetOpenCL() != OpenCL || gradient.GetIndex() < 0 ||
      m_selected_correction.GetIndex() < 0 || m_selected_slot.GetIndex() < 0 ||
      m_influence.GetIndex() < 0 || !EnsureUpdateScratch())
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      ReturnFalse;
   uint offset[1] = {0};
   uint local[1] = {local_size};
   const int partial_kernel = def_k_D2SkillInfluencePartial;
   setBuffer(partial_kernel, def_k_d2sip_gradient, gradient.GetIndex())
   setBuffer(partial_kernel, def_k_d2sip_correction, m_selected_correction.GetIndex())
   setBuffer(partial_kernel, def_k_d2sip_partials, m_gradient_influence_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2sip_dimension, (int)m_dimension)
   uint global[1] = {groups * local_size};
   kernelExecuteLoc(partial_kernel, offset, global, local)
   if(!RunStagedReduction(m_gradient_influence_partials, m_gradient_influence_total,
                          m_dimension, 1, 1, local_size))
      ReturnFalse;
   const int accumulate_kernel = def_k_D2SkillInfluenceAccumulate;
   setBuffer(accumulate_kernel, def_k_d2sia_selected_slot, m_selected_slot.GetIndex())
   setBuffer(accumulate_kernel, def_k_d2sia_reduced_influence,
             m_gradient_influence_total.GetIndex())
   setBuffer(accumulate_kernel, def_k_d2sia_influence, m_influence.GetIndex())
   setArgument(accumulate_kernel, def_k_d2sia_slots, (int)m_slots)
   uint scalar_global[1] = {1};
   kernelExecute(accumulate_kernel, offset, scalar_global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Updates only the direction EMA of the retrieved stable slot.     |
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateDirectionFromGradient(CBufferFloat *gradient)
  {
   if(!m_enabled || !OpenCL || !gradient || gradient.Total() != (int)m_dimension ||
      gradient.GetOpenCL() != OpenCL || gradient.GetIndex() < 0 ||
      m_selected_slot.GetIndex() < 0 || !EnsureItemState())
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   uint partial_count = 0;
   if(!ItemLaunchLayout(local_size, groups, partial_count))
      ReturnFalse;
   const int normalize_kernel = def_k_D2SkillNormalizeDirection;
   setBuffer(normalize_kernel, def_k_d2nd_directions, m_directions.GetIndex())
   setBuffer(normalize_kernel, def_k_d2nd_selected_slot, m_selected_slot.GetIndex())
   setBuffer(normalize_kernel, def_k_d2nd_used, m_used.GetIndex())
   setBuffer(normalize_kernel, def_k_d2nd_state, m_state.GetIndex())
   setBuffer(normalize_kernel, def_k_d2nd_partials, m_item_partials.GetIndex())
   setBuffer(normalize_kernel, def_k_d2nd_reduced_norm, m_item_reduced_norm.GetIndex())
   setArgument(normalize_kernel, def_k_d2nd_slots, (int)m_slots)
   setArgument(normalize_kernel, def_k_d2nd_dimension, (int)m_dimension)
   setArgument(normalize_kernel, def_k_d2nd_phase, 0)
   uint offset[1] = {0};
   uint global[1] = {groups * local_size};
   uint local[1] = {local_size};
   kernelExecuteLoc(normalize_kernel, offset, global, local)
   if(!RunStagedReduction(m_item_partials, m_item_reduced_norm, m_dimension, 1, 1,
                          local_size))
      ReturnFalse;
   setArgument(normalize_kernel, def_k_d2nd_phase, 1)
   kernelExecuteLoc(normalize_kernel, offset, global, local)
   if(!CalculateItemNorm(gradient, local_size, groups))
      ReturnFalse;
   const int kernel = def_k_D2SkillDirectionEMA;
   setBuffer(kernel, def_k_d2de_gradient, gradient.GetIndex())
   setBuffer(kernel, def_k_d2de_directions, m_directions.GetIndex())
   setBuffer(kernel, def_k_d2de_selected_slot, m_selected_slot.GetIndex())
   setBuffer(kernel, def_k_d2de_used, m_used.GetIndex())
   setBuffer(kernel, def_k_d2de_state, m_state.GetIndex())
   setBuffer(kernel, def_k_d2de_reduced_norm, m_item_reduced_norm.GetIndex())
   setBuffer(kernel, def_k_d2de_partials, m_item_partials.GetIndex())
   setArgument(kernel, def_k_d2de_slots, (int)m_slots)
   setArgument(kernel, def_k_d2de_dimension, (int)m_dimension)
   setArgument(kernel, def_k_d2de_beta_direction, m_beta_direction)
   setArgument(kernel, def_k_d2de_phase, 0)
   kernelExecuteLoc(kernel, offset, global, local)
   if(!RunStagedReduction(m_item_partials, m_item_reduced_norm, m_dimension, 1, 1,
                          local_size))
      ReturnFalse;
   setArgument(kernel, def_k_d2de_phase, 1)
   kernelExecuteLoc(kernel, offset, global, local)
   return(true);
  }
//+------------------------------------------------------------------+
//| Runs the full bank update: lifecycle ages, metrics, candidate    |
//| promotion and EMA vector update.                                 |
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
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      ReturnFalse;
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
   setBuffer(age_kernel, def_k_d2al_retrieved, m_retrieved.GetIndex())
   setBuffer(age_kernel, def_k_d2al_candidate_scores, m_retrieval_scores.GetIndex())
   uint slot_global[1] = {m_slots};
   kernelExecute(age_kernel, offset, slot_global)
   if(!RunTop1(m_free_scores, m_free_top) || !RunTop1(m_eviction_scores, m_eviction_top) ||
      !RunTop1(m_retrieval_scores, m_retrieval_top) ||
      !RunMetrics(state, gradient, local_size, groups))
      ReturnFalse;
   const int leader_kernel = def_k_D2SkillCandidateLeader;
   setBuffer(leader_kernel, def_k_d2cl_selected_slot, m_selected_slot.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_used, m_used.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_scales, m_scales.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_observations, m_observations.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_mass, m_mass.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_age, m_age.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_streak, m_observations.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_valid, m_state.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_scale, m_scales.GetIndex())
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
   setBuffer(leader_kernel, def_k_d2cl_candidate_top, m_retrieval_top.GetIndex())
   setBuffer(leader_kernel, def_k_d2cl_candidate_protection, m_protection.GetIndex())
   uint scalar_global[1] = {1};
   kernelExecute(leader_kernel, offset, scalar_global)
   const int vector_kernel = def_k_D2SkillVectorUpdate;
   setBuffer(vector_kernel, def_k_d2vu_query, state.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_gradient, gradient.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_keys, m_keys.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_corrections, m_corrections.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_directions, m_directions.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_scales, m_scales.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_key, m_keys.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_direction, m_directions.GetIndex())
   setBuffer(vector_kernel, def_k_d2vu_candidate_scale, m_scales.GetIndex())
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
      ReturnFalse;
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
   setBuffer(promote_kernel, def_k_d2ps_candidate_streak, m_observations.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_candidate_valid, m_state.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_candidate_scale, m_scales.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_decision, m_decision.GetIndex())
   setBuffer(promote_kernel, def_k_d2ps_diagnostics, m_diagnostics.GetIndex())
   setArgument(promote_kernel, def_k_d2ps_slots, (int)m_slots)
   setArgument(promote_kernel, def_k_d2ps_protection_age, (int)m_protection_age)
   kernelExecute(promote_kernel, offset, scalar_global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Applies the utility EMA from episode influence and reports       |
//| whether it was applied.                                          |
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateUtility(const float delta_j, bool &applied)
  {
   applied = false;
   if(!m_enabled || !OpenCL || !MathIsValidNumber(delta_j) || !EnsureUpdateScratch())
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      ReturnFalse;
   const int total_kernel = def_k_D2SkillInfluenceTotal;
   setBuffer(total_kernel, def_k_d2it_influence, m_influence.GetIndex())
   setBuffer(total_kernel, def_k_d2it_used, m_used.GetIndex())
   setBuffer(total_kernel, def_k_d2it_state, m_state.GetIndex())
   setBuffer(total_kernel, def_k_d2it_partials, m_influence_partials.GetIndex())
   setArgument(total_kernel, def_k_d2it_slots, (int)m_slots)
   uint offset[1] = {0};
   const uint slot_padded = ((m_slots + local_size - 1) / local_size) * local_size;
   uint slot_global[1] = {slot_padded};
   uint local[1] = {local_size};
   kernelExecuteLoc(total_kernel, offset, slot_global, local)
   if(!RunStagedReduction(m_influence_partials, m_influence_total, m_slots, 1, 1,
                          local_size))
      ReturnFalse;
   const int kernel = def_k_D2SkillUtility;
   setBuffer(kernel, def_k_d2ut_influence, m_influence.GetIndex())
   setBuffer(kernel, def_k_d2ut_influence_total, m_influence_total.GetIndex())
   setBuffer(kernel, def_k_d2ut_utility, m_utility.GetIndex())
   setBuffer(kernel, def_k_d2ut_used, m_used.GetIndex())
   setBuffer(kernel, def_k_d2ut_state, m_state.GetIndex())
   setBuffer(kernel, def_k_d2ut_diagnostics, m_diagnostics.GetIndex())
   setBuffer(kernel, def_k_d2ut_applied, m_utility_applied.GetIndex())
   setArgument(kernel, def_k_d2ut_delta, delta_j)
   setArgument(kernel, def_k_d2ut_beta, m_beta_utility)
   setArgument(kernel, def_k_d2ut_slots, (int)m_slots)
   kernelExecuteLoc(kernel, offset, slot_global, local)
   if(!RunStagedReduction(m_utility_applied, m_influence_partials,
                          m_influence_total, m_slots, 1, 1,
                          local_size) || !m_influence_total.BufferRead())
      ReturnFalse;
   applied = (m_influence_total[0] > 0.5f);
   return(true);
  }
//+------------------------------------------------------------------+
//| Applies the utility EMA from episode influence.                  |
//+------------------------------------------------------------------+
bool CD2SkillBank::UpdateUtility(const float delta_j)
  {
   bool applied = false;
   return(UpdateUtility(delta_j, applied));
  }
//+------------------------------------------------------------------+
//| Zeroes the per-slot influence counters for the current episode.  |
//+------------------------------------------------------------------+
bool CD2SkillBank::ResetEpisodeInfluence(void)
  {
//--- BASE/disabled banks have no influence buffer by design.  Episode cleanup
//--- is therefore a successful no-op and must not materialize one.
   if(!m_enabled)
      return(true);
   if(!OpenCL || m_slots == 0 || m_influence.GetIndex() < 0)
      ReturnFalse;
   const int kernel = def_k_D2SkillResetInfluence;
   setBuffer(kernel, def_k_d2ri_influence, m_influence.GetIndex())
   setArgument(kernel, def_k_d2ri_slots, (int)m_slots)
   uint offset[1] = {0};
   uint global[1] = {m_slots};
   kernelExecute(kernel, offset, global)
   return(true);
  }
//+------------------------------------------------------------------+
//| Recomputes bank diagnostics, norms and the utility/usage         |
//| distributions.                                                   |
//+------------------------------------------------------------------+
bool CD2SkillBank::RefreshDiagnostics(void)
  {
//--- OOS reporting may inspect every branch.  Disabled branches report their
//--- caller-provided zero counters without allocating diagnostics scratch.
   if(!m_enabled)
      return(true);
   if(!OpenCL || !ValidateBuffers() || !EnsureUpdateScratch())
      ReturnFalse;
   uint local_size = 0;
   uint groups = 0;
   if(!UpdateLaunchLayout(local_size, groups))
      ReturnFalse;
   uint offset[1] = {0};
   uint local[1] = {local_size};
   const uint vector_items = m_slots * m_dimension;
   const uint vector_padded = ((vector_items + local_size - 1) / local_size) * local_size;
   const int norm_kernel = def_k_D2SkillDiagnosticsNormPartial;
   setBuffer(norm_kernel, def_k_d2dnp_corrections, m_corrections.GetIndex())
   setBuffer(norm_kernel, def_k_d2dnp_partials, m_diagnostics_norm_partials.GetIndex())
   setArgument(norm_kernel, def_k_d2dnp_slots, (int)m_slots)
   setArgument(norm_kernel, def_k_d2dnp_dimension, (int)m_dimension)
   uint norm_global[1] = {vector_padded};
   kernelExecuteLoc(norm_kernel, offset, norm_global, local)
   if(!RunStagedReduction(m_diagnostics_norm_partials, m_diagnostics_norms,
                          m_dimension, m_slots, 1, local_size))
      ReturnFalse;
   const int partial_kernel = def_k_D2SkillDiagnosticsPartial;
   setBuffer(partial_kernel, def_k_d2dp_norms, m_diagnostics_norms.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_utility, m_utility.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_uses, m_uses.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_used, m_used.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_state, m_state.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_age, m_age.GetIndex())
   setBuffer(partial_kernel, def_k_d2dp_partials, m_diagnostics_partials.GetIndex())
   setArgument(partial_kernel, def_k_d2dp_slots, (int)m_slots)
   const uint slot_padded = ((m_slots + local_size - 1) / local_size) * local_size;
   uint partial_global[1] = {slot_padded};
   kernelExecuteLoc(partial_kernel, offset, partial_global, local)
   if(!RunStagedReduction(m_diagnostics_partials, m_diagnostics_totals, m_slots, 1,
                          D2SKILL_DIAGNOSTIC_PARTIALS, local_size, 9, 10))
      ReturnFalse;
   const int reduce_kernel = def_k_D2SkillDiagnosticsReduce;
   setBuffer(reduce_kernel, def_k_d2dr_totals, m_diagnostics_totals.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_diagnostics, m_diagnostics.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_utility_distribution, m_utility_distribution.GetIndex())
   setBuffer(reduce_kernel, def_k_d2dr_usage_distribution, m_usage_distribution.GetIndex())
   setArgument(reduce_kernel, def_k_d2dr_slots, (int)m_slots)
   const uint finalize_padded = ((23 + local_size - 1) / local_size) * local_size;
   uint reduce_global[1] = {finalize_padded};
   kernelExecuteLoc(reduce_kernel, offset, reduce_global, local)
   return(true);
  }
//+------------------------------------------------------------------+
//| Saves the bank state and diagnostics to the file handle.         |
//+------------------------------------------------------------------+
bool CD2SkillBank::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !RefreshDiagnostics() || !CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
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
          m_influence.Save(file_handle) && m_selected_correction.Save(file_handle) &&
          m_selected_slot.Save(file_handle) && m_selected_score.Save(file_handle) &&
          m_diagnostics.Save(file_handle) &&
          m_utility_distribution.Save(file_handle) &&
          m_usage_distribution.Save(file_handle));
  }
//+------------------------------------------------------------------+
//| Restores the bank state from the file handle, deferring device   |
//| attachment.                                                      |
//+------------------------------------------------------------------+
bool CD2SkillBank::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_FORMAT)
      ReturnFalse;
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
      ReturnFalse;
   const long total = (long)m_slots * (long)m_dimension;
   if(!m_keys.Load(file_handle) || !m_corrections.Load(file_handle) ||
      !m_directions.Load(file_handle) || !m_scales.Load(file_handle) ||
      !m_utility.Load(file_handle) || !m_observations.Load(file_handle) ||
      !m_uses.Load(file_handle) || !m_mass.Load(file_handle) || !m_used.Load(file_handle) ||
      !m_state.Load(file_handle) || !m_age.Load(file_handle) || !m_protection.Load(file_handle) ||
      !m_influence.Load(file_handle) || !m_selected_correction.Load(file_handle) ||
      !m_selected_slot.Load(file_handle) || !m_selected_score.Load(file_handle))
      ReturnFalse;
   if(!m_diagnostics.Load(file_handle))
      ReturnFalse;
   if(m_diagnostics.Total() != D2SKILL_DIAGNOSTICS)
      if(!InitBuffer(m_diagnostics, D2SKILL_DIAGNOSTICS))
         ReturnFalse;
   if(!m_utility_distribution.Load(file_handle) ||
      !m_usage_distribution.Load(file_handle) || !m_utility_applied.BufferInit((int)m_slots, 0.0f) ||
      !m_influence_total.BufferInit(1, 0.0f) ||
      !m_gradient_influence_total.BufferInit(1, 0.0f) ||
      !m_retrieved.BufferInit((int)m_slots, 0.0f))
      ReturnFalse;
   if(total > 2147483647 || !ValidateBuffers())
      ReturnFalse;
//--- Device attachment is intentionally deferred to CD2Skill::Enable().
//--- This keeps disabled Task/Step banks out of a baseline checkpoint load.
   m_correction = NULL;
   return(true);
  }
//+------------------------------------------------------------------+
//| Restores the bank state and attaches the given OpenCL context.   |
//+------------------------------------------------------------------+
bool CD2SkillBank::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl ||
      FileReadInteger(file_handle, INT_VALUE) != defNeuronD2SkillBank)
      ReturnFalse;
   OpenCL = opencl;
   if(!Load(file_handle))
      ReturnFalse;
   SetOpenCL(opencl);
   return(true);
  }
//+------------------------------------------------------------------+
//| Clears all bank buffers and resets the active slot.              |
//+------------------------------------------------------------------+
bool CD2SkillBank::Clear(void)
  {
   m_correction = NULL;
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   m_active_slot = UINT_MAX;
   if(!(m_retrieved.Fill(0) && m_influence.Fill(0) && m_utility_applied.Fill(0) &&
        m_influence_total.Fill(0) && m_gradient_influence_total.Fill(0) &&
        m_selected_correction.Fill(0) && m_selected_slot.Fill(-1.0f) &&
        m_selected_score.Fill(-3.402823e+38f) && m_diagnostics.Fill(0) &&
        m_utility_distribution.Fill(0) && m_usage_distribution.Fill(0) &&
        (m_retrieval_scores.Total() == 0 || m_retrieval_scores.Fill(0)) &&
        (m_score_partials.Total() == 0 || m_score_partials.Fill(0)) &&
        (m_score_metrics.Total() == 0 || m_score_metrics.Fill(0)) &&
        (m_retrieval_partial.Total() == 0 || m_retrieval_partial.Fill(0)) &&
        (m_retrieval_merge.Total() == 0 || m_retrieval_merge.Fill(0)) &&
        (m_retrieval_top.Total() == 0 || m_retrieval_top.Fill(0)) &&
        (m_free_scores.Total() == 0 || m_free_scores.Fill(0)) &&
        (m_eviction_scores.Total() == 0 || m_eviction_scores.Fill(0)) &&
        (m_free_top.Total() == 0 || m_free_top.Fill(0)) &&
        (m_eviction_top.Total() == 0 || m_eviction_top.Fill(0)) &&
        (m_influence_partials.Total() == 0 || m_influence_partials.Fill(0)) &&
        (m_gradient_influence_partials.Total() == 0 || m_gradient_influence_partials.Fill(0)) &&
        (m_metric_partials.Total() == 0 || m_metric_partials.Fill(0)) &&
        (m_metrics.Total() == 0 || m_metrics.Fill(0)) &&
        (m_decision.Total() == 0 || m_decision.Fill(0)) &&
        (m_diagnostics_norm_partials.Total() == 0 || m_diagnostics_norm_partials.Fill(0)) &&
        (m_diagnostics_norms.Total() == 0 || m_diagnostics_norms.Fill(0)) &&
        (m_diagnostics_partials.Total() == 0 || m_diagnostics_partials.Fill(0)) &&
        (m_diagnostics_totals.Total() == 0 || m_diagnostics_totals.Fill(0))))
      ReturnFalse;
   return(SetCorrectionBuffer(GetPointer(m_selected_correction)));
  }
//+------------------------------------------------------------------+
//| Explicit destructive reset used only by isolated fixture tests.  |
//+------------------------------------------------------------------+
bool CD2SkillBank::ResetDurableState(void)
  {
   if(!Clear())
      ReturnFalse;
   if(!(m_keys.Fill(0) && m_corrections.Fill(0) && m_directions.Fill(0) &&
        m_scales.Fill(0) && m_utility.Fill(0) && m_observations.Fill(0) &&
        m_uses.Fill(0) && m_mass.Fill(0) && m_used.Fill(0) && m_state.Fill(0) &&
        m_age.Fill(0) && m_protection.Fill(0) && m_influence.Fill(0) &&
        m_utility_applied.Fill(0) &&
        m_influence_total.Fill(0) && m_gradient_influence_total.Fill(0) &&
        m_retrieved.Fill(0) &&
        m_selected_correction.Fill(0) && m_selected_slot.Fill(-1.0f) &&
        m_selected_score.Fill(-3.402823e+38f) && m_diagnostics.Fill(0) &&
        m_utility_distribution.Fill(0) && m_usage_distribution.Fill(0) &&
        (m_retrieval_scores.Total() == 0 || m_retrieval_scores.Fill(0)) &&
        (m_score_partials.Total() == 0 || m_score_partials.Fill(0)) &&
        (m_score_metrics.Total() == 0 || m_score_metrics.Fill(0)) &&
        (m_retrieval_partial.Total() == 0 || m_retrieval_partial.Fill(0)) &&
        (m_retrieval_merge.Total() == 0 || m_retrieval_merge.Fill(0)) &&
        (m_retrieval_top.Total() == 0 || m_retrieval_top.Fill(0)) &&
        (m_free_scores.Total() == 0 || m_free_scores.Fill(0)) &&
        (m_eviction_scores.Total() == 0 || m_eviction_scores.Fill(0)) &&
        (m_free_top.Total() == 0 || m_free_top.Fill(0)) &&
        (m_eviction_top.Total() == 0 || m_eviction_top.Fill(0)) &&
        (m_influence_partials.Total() == 0 || m_influence_partials.Fill(0)) &&
        (m_gradient_influence_partials.Total() == 0 || m_gradient_influence_partials.Fill(0)) &&
        (m_metric_partials.Total() == 0 || m_metric_partials.Fill(0)) &&
        (m_metrics.Total() == 0 || m_metrics.Fill(0)) &&
        (m_decision.Total() == 0 || m_decision.Fill(0)) &&
        (m_diagnostics_norm_partials.Total() == 0 || m_diagnostics_norm_partials.Fill(0)) &&
        (m_diagnostics_norms.Total() == 0 || m_diagnostics_norms.Fill(0)) &&
        (m_diagnostics_partials.Total() == 0 || m_diagnostics_partials.Fill(0)) &&
        (m_diagnostics_totals.Total() == 0 || m_diagnostics_totals.Fill(0))))
      ReturnFalse;
   return(SetCorrectionBuffer(GetPointer(m_selected_correction)));
  }
#endif // NEURONET_D2SKILL_MQH
//+------------------------------------------------------------------+