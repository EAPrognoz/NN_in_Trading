//+------------------------------------------------------------------+
//|                                             NeuroNet_D2Skill.mqh |
//|                                  GPU-resident D2Skill primitives |
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
//|                                                                  |
//+------------------------------------------------------------------+
void CD2SkillItem::CD2SkillItem(void) : m_correction(NULL),
   m_representation(D2SkillDirectionMagnitude),
   m_beta_correction(0.05f), m_beta_direction(0.05f),
   m_beta_magnitude(0.05f), m_item_dimension(0)
  {
  }
//+------------------------------------------------------------------+
void CD2SkillItem::~CD2SkillItem(void)
  {
  }
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
CBufferFloat *CD2SkillItem::GetCorrectionBuffer(void) const
  {
   return(m_correction);
  }
//+------------------------------------------------------------------+
void CD2SkillItem::SetRepresentation(const ED2SkillRepresentation mode)
  {
   m_representation = mode;
  }
//+------------------------------------------------------------------+
ED2SkillRepresentation CD2SkillItem::Representation(void) const
  {
   return(m_representation);
  }
//+------------------------------------------------------------------+
void CD2SkillItem::SetEMA(const float beta_correction, const float beta_direction,
                          const float beta_magnitude)
  {
   m_beta_correction = MathMin(MathMax(beta_correction, 0.0f), 1.0f);
   m_beta_direction = MathMin(MathMax(beta_direction, 0.0f), 1.0f);
   m_beta_magnitude = MathMin(MathMax(beta_magnitude, 0.0f), 1.0f);
  }
//+------------------------------------------------------------------+
float CD2SkillItem::BetaCorrection(void) const
  {
   return(m_beta_correction);
  }
//+------------------------------------------------------------------+
float CD2SkillItem::BetaDirection(void) const
  {
   return(m_beta_direction);
  }
//+------------------------------------------------------------------+
float CD2SkillItem::BetaMagnitude(void) const
  {
   return(m_beta_magnitude);
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Correction(void)
  {
   return(m_correction);
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Direction(void)
  {
   return(GetPointer(m_item_direction));
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Scale(void)
  {
   return(GetPointer(m_item_scale));
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Observations(void)
  {
   return(GetPointer(m_item_observations));
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillItem::Mass(void)
  {
   return(GetPointer(m_item_mass));
  }
//+------------------------------------------------------------------+
bool CD2SkillItem::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   return(true);
  }
//+------------------------------------------------------------------+
int CD2SkillItem::Type(void) const
  {
   return(defNeuronD2SkillItem);
  }
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
bool CD2SkillBank::InitBuffer(CBufferFloat &buffer, const int total, const float value)
  {
   return(buffer.BufferInit(total, value) && buffer.BufferCreate(OpenCL));
  }
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
CD2SkillBank::~CD2SkillBank(void)
  {
  }
//+------------------------------------------------------------------+
int CD2SkillBank::Type(void) const
  {
   return(defNeuronD2SkillBank);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::SetEnabled(const bool enabled)
  {
   m_enabled = enabled;
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::Enabled(void) const
  {
   return(m_enabled);
  }
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
bool CD2SkillBank::SetUtilityWeight(const float value)
  {
   if(!MathIsValidNumber(value))
      ReturnFalse;
   m_utility_weight = value;
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::SetAlpha(const float value)
  {
   if(!MathIsValidNumber(value) || value < 0.0f || value > 1.0f)
      ReturnFalse;
   m_alpha = value;
   return(true);
  }
//+------------------------------------------------------------------+
float CD2SkillBank::Alpha(void) const
  {
   return(m_alpha);
  }
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
bool CD2SkillBank::UtilityAware(void) const
  {
   return(m_utility_aware);
  }
//+------------------------------------------------------------------+
float CD2SkillBank::MinUtility(void) const
  {
   return(m_min_utility);
  }
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
bool CD2SkillBank::SetMaxCorrection(const float value)
  {
   if(!MathIsValidNumber(value) || value < 0.0f)
      ReturnFalse;
   m_max_correction = value;
   return(true);
  }
//+------------------------------------------------------------------+
float CD2SkillBank::MaxCorrection(void) const
  {
   return(m_max_correction);
  }
//+------------------------------------------------------------------+
uint CD2SkillBank::Slots(void) const
  {
   return(m_slots);
  }
//+------------------------------------------------------------------+
uint CD2SkillBank::Dimension(void) const
  {
   return(m_dimension);
  }
//+------------------------------------------------------------------+
uint CD2SkillBank::ActiveSlot(void) const
  {
   return(m_active_slot);
  }
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedCorrection(void) { return(GetPointer(m_selected_correction)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedSlot(void) { return(GetPointer(m_selected_slot)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::SelectedScore(void) { return(GetPointer(m_selected_score)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Keys(void) { return(GetPointer(m_keys)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Corrections(void) { return(GetPointer(m_corrections)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Directions(void) { return(GetPointer(m_directions)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Scales(void) { return(GetPointer(m_scales)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::ObservationsBank(void) { return(GetPointer(m_observations)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Uses(void) { return(GetPointer(m_uses)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::MassBank(void) { return(GetPointer(m_mass)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Utility(void) { return(GetPointer(m_utility)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Used(void) { return(GetPointer(m_used)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::State(void) { return(GetPointer(m_state)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Ages(void) { return(GetPointer(m_age)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Protection(void) { return(GetPointer(m_protection)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Diagnostics(void) { return(GetPointer(m_diagnostics)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::UtilityDistribution(void) { return(GetPointer(m_utility_distribution)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::UsageDistribution(void) { return(GetPointer(m_usage_distribution)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Influence(void) { return(GetPointer(m_influence)); }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat *CD2SkillBank::Retrieved(void) { return(GetPointer(m_retrieved)); }
//+------------------------------------------------------------------+
CD2Skill::CD2Skill(void) : m_ready(false), m_online_direction_update(false),
   m_mode(D2_DISABLED)
  {
  }
//+------------------------------------------------------------------+
CD2Skill::~CD2Skill(void)
  {
  }
//+------------------------------------------------------------------+
int CD2Skill::Type(void) const
  {
   return(defNeuronD2Skill);
  }
//+------------------------------------------------------------------+
ENUM_D2SKILL_MODE CD2Skill::Mode(void) const
  {
   return(m_mode);
  }
//+------------------------------------------------------------------+
bool CD2Skill::Ready(void) const
  {
   return(m_ready);
  }
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::TaskBank(void)
  {
   return(m_ready ? GetPointer(m_task_bank) : NULL);
  }
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::StepBank(void)
  {
   return(m_ready ? GetPointer(m_step_bank) : NULL);
  }
//+------------------------------------------------------------------+
CD2SkillBank *CD2Skill::Bank(void)
  {
   return(TaskBank());
  }
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::Bank(void) const
  {
   return(TaskBank());
  }
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::TaskBank(void) const
  {
   return(m_ready ? GetPointer(m_task_bank) : NULL);
  }
//+------------------------------------------------------------------+
const CD2SkillBank *CD2Skill::StepBank(void) const
  {
   return(m_ready ? GetPointer(m_step_bank) : NULL);
  }
//+------------------------------------------------------------------+
bool CD2Skill::Enable(const bool task, const bool step)
  {
   if(!m_ready)
      ReturnFalse;
   m_task_bank.SetEnabled(task);
   m_step_bank.SetEnabled(step);
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2Skill::TaskFeedForward(CNeuronBaseOCL *source)
  {
   return(m_ready && m_task_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
bool CD2Skill::StepFeedForward(CNeuronBaseOCL *source)
  {
   return(m_ready && m_step_bank.FeedForward(source));
  }
//+------------------------------------------------------------------+
bool CD2Skill::TaskUpdate(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(m_ready && m_task_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
bool CD2Skill::StepUpdate(CBufferFloat *state, CBufferFloat *gradient)
  {
   return(m_ready && m_step_bank.UpdateFromGradient(state, gradient));
  }
//+------------------------------------------------------------------+
bool CD2Skill::TaskUtility(const float delta_j)
  {
   if(!m_ready || (m_mode != D2_EVALUATE && m_mode != D2_ONLINE_CALIBRATION))
      ReturnFalse;
   return(m_task_bank.UpdateUtility(delta_j));
  }
//+------------------------------------------------------------------+
bool CD2Skill::StepUtility(const float delta_j)
  {
   if(!m_ready || (m_mode != D2_EVALUATE && m_mode != D2_ONLINE_CALIBRATION))
      ReturnFalse;
   return(m_step_bank.UpdateUtility(delta_j));
  }
//+------------------------------------------------------------------+
bool CD2Skill::BackwardSecond(CNeuronBaseOCL *source, CNeuronBaseOCL *forecast)
  {
   return(m_ready && calcInputGradientsSecond(source, forecast));
  }
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
bool CD2Skill::TaskSkillEnabled(void) const
  {
   return(m_ready && m_task_bank.Enabled() &&
          (m_mode == D2_EVALUATE || m_mode == D2_INFERENCE ||
           m_mode == D2_ONLINE_CALIBRATION));
  }
//+------------------------------------------------------------------+
bool CD2Skill::StepSkillEnabled(void) const
  {
   return(m_ready && m_step_bank.Enabled() &&
          (m_mode == D2_EVALUATE || m_mode == D2_INFERENCE ||
           m_mode == D2_ONLINE_CALIBRATION));
  }
//+------------------------------------------------------------------+
bool CD2Skill::SkillUpdatesEnabled(void) const
  {
   return(m_mode == D2_COLLECT);
  }
//+------------------------------------------------------------------+
//| Enables only the opt-in online direction EMA.                   |
//+------------------------------------------------------------------+
bool CD2Skill::OnlineDirectionUpdatesEnabled(void) const
  {
   return(m_mode == D2_ONLINE_CALIBRATION && m_online_direction_update);
  }
//+------------------------------------------------------------------+
//| Updates one selected bank direction without mutating its lifecycle. |
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
                       1, true, 0, 0, 0, 1))
      ReturnFalse;
   return(CNeuronMSRes::feedForward(cMHAOutput.AsObject()));
  }
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
bool CD2Skill::updateInputWeightsSecond(CNeuronBaseOCL *NeuronOCL,
                                        CNeuronBaseOCL *Forecast)
  {
   if(!NeuronOCL || !Forecast || Forecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
//--- Collection records bank observations but freezes the Actor highway.
//--- Evaluation and inference are read-only for Actor and market weights.
   if(m_mode == D2_COLLECT || m_mode == D2_EVALUATE || m_mode == D2_INFERENCE)
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
bool CD2Skill::Save(const int file_handle)
  {
   if(!m_ready || !CNeuronScenarioCrossAttention::Save(file_handle))
      ReturnFalse;
   dFileWriteUInt(file_handle, D2SKILL_LAYER_FORMAT);
   return(m_task_bank.Save(file_handle) && m_step_bank.Save(file_handle));
  }
//+------------------------------------------------------------------+
bool CD2Skill::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !CNeuronScenarioCrossAttention::Load(file_handle))
      ReturnFalse;
   const uint format = (uint)FileReadInteger(file_handle);
   if(format != D2SKILL_LAYER_FORMAT || bCritic || !OpenCL)
      ReturnFalse;
   m_ready = false;
   m_task_bank.SetOpenCL(OpenCL);
   m_step_bank.SetOpenCL(OpenCL);
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
bool CD2Skill::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl || FileReadInteger(file_handle) != Type())
      ReturnFalse;
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
bool CD2Skill::SetMode(const ENUM_D2SKILL_MODE mode)
  {
   if(mode < D2_DISABLED || mode > D2_ONLINE_CALIBRATION)
      ReturnFalse;
   m_mode = mode;
   return(true);
  }
//+------------------------------------------------------------------+
//| Configures the optional online direction-only EMA.              |
//+------------------------------------------------------------------+
bool CD2Skill::SetOnlineDirectionUpdate(const bool enabled)
  {
   m_online_direction_update = enabled;
   return(true);
  }
bool CD2Skill::ResetEpisodeInfluence(void)
  {
   if(!m_ready || !m_task_bank.ResetEpisodeInfluence() ||
      !m_step_bank.ResetEpisodeInfluence())
      ReturnFalse;
   return(true);
  }
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
bool CD2SkillItem::Load(const int file_handle, COpenCLMy *opencl)
  {
   if(file_handle == INVALID_HANDLE || !opencl ||
      FileReadInteger(file_handle, INT_VALUE) != Type())
      ReturnFalse;
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
//| Choose a D2 local size without changing the generic RAG width. |
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
      !InitBuffer(m_influence, (int)m_slots) || !InitBuffer(m_influence_total, 1) ||
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
//| Accumulates g dot (alpha * correction) for the forward winner.  |
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
//| Updates only the direction EMA of the retrieved stable slot.    |
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
bool CD2SkillBank::UpdateUtility(const float delta_j)
  {
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
   setArgument(kernel, def_k_d2ut_delta, delta_j)
   setArgument(kernel, def_k_d2ut_beta, m_beta_utility)
   setArgument(kernel, def_k_d2ut_slots, (int)m_slots)
   kernelExecuteLoc(kernel, offset, slot_global, local)
   return(true);
  }
//+------------------------------------------------------------------+
bool CD2SkillBank::ResetEpisodeInfluence(void)
  {
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
bool CD2SkillBank::RefreshDiagnostics(void)
  {
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
      !m_usage_distribution.Load(file_handle) || !InitBuffer(m_influence_total, 1) ||
      !InitBuffer(m_gradient_influence_total, 1) ||
      !InitBuffer(m_retrieved, (int)m_slots))
      ReturnFalse;
   if(total > 2147483647 || !ValidateBuffers())
      ReturnFalse;
   SetOpenCL(OpenCL);
   if(!EnsureRetrievalScratch() || !EnsureUpdateScratch() ||
      !SetCorrectionBuffer(GetPointer(m_selected_correction)))
      ReturnFalse;
   return(true);
  }
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
bool CD2SkillBank::Clear(void)
  {
   m_correction = NULL;
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   m_active_slot = UINT_MAX;
   if(!(m_retrieved.Fill(0) && m_influence.Fill(0) &&
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
//| Explicit destructive reset used only by isolated fixture tests. |
//+------------------------------------------------------------------+
bool CD2SkillBank::ResetDurableState(void)
  {
   if(!Clear())
      ReturnFalse;
   if(!(m_keys.Fill(0) && m_corrections.Fill(0) && m_directions.Fill(0) &&
        m_scales.Fill(0) && m_utility.Fill(0) && m_observations.Fill(0) &&
        m_uses.Fill(0) && m_mass.Fill(0) && m_used.Fill(0) && m_state.Fill(0) &&
        m_age.Fill(0) && m_protection.Fill(0) && m_influence.Fill(0) &&
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
