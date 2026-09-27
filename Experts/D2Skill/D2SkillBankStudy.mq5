//+------------------------------------------------------------------+
//|                                             D2SkillBankStudy.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property description "Харнесс изучения банка D2-навыков малой размерности."
#property description "Формирует банк из двух слотов и сохраняет его в файл D2SkillBankStudy.bin."
#property description "Только для тестов; ордера не выставляет."

#include "..\\NeuroNet_DNG\\NeuroNet.mqh"

const uint   D2_STUDY_DIM     = 2;
const uint   D2_STUDY_SLOTS   = 2;
const float  D2_STUDY_EPSILON = 0.000001f;
const string D2_STUDY_FILE    = "D2SkillBankStudy.bin";

CNet           g_net;
CNeuronBaseOCL g_source;
bool           g_completed = false;
//+------------------------------------------------------------------+
//| Implements CreateHost.                                           |
//+------------------------------------------------------------------+
bool CreateHost(void)
  {
   CArrayObj *description = new CArrayObj();
   if(!description)
      ReturnFalse;
   description.FreeMode(true);
   for(int i = 0; i < 2; i++)
     {
      CLayerDescription *layer = new CLayerDescription();
      if(!layer)
        {
         DeleteObjAndFalse(description);
        }
      layer.type = defNeuronBaseOCL;
      layer.count = D2_STUDY_DIM;
      layer.activation = None;
      layer.optimization = ADAM;
      if(!description.Add(layer))
        {
         DeleteObj(layer);
         DeleteObjAndFalse(description);
        }
     }
   const bool created = g_net.Create(description);
   delete description;
   return(created && g_net.GetOpenCL() != NULL);
  }
//+------------------------------------------------------------------+
//| Implements Put.                                                  |
//+------------------------------------------------------------------+
bool Put(CBufferFloat *buffer, const float first, const float second)
  {
   float values[2] = {first, second};
   return(buffer && buffer.AssignArray(values) && buffer.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Implements ReadSelection.                                        |
//+------------------------------------------------------------------+
bool ReadSelection(CD2SkillBank &bank)
  {
   return(bank.SelectedSlot().BufferRead() && bank.SelectedScore().BufferRead() &&
          bank.SelectedCorrection().BufferRead());
  }
//+------------------------------------------------------------------+
//| Confirmation data use a stable account-side query.               |
//+------------------------------------------------------------------+
bool Train(CD2SkillBank &bank, const float gradient_x, const float gradient_y)
  {
   if(!Put(g_source.getOutput(), 1.0f, 0.0f) || !bank.FeedForward(g_source.AsObject()) ||
      !Put(g_source.getGradient(), gradient_x, gradient_y) ||
      !bank.UpdateFromGradient(g_source.getOutput(), g_source.getGradient()))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements ReadLifecycle.                                        |
//+------------------------------------------------------------------+
bool ReadLifecycle(CD2SkillBank &bank)
  {
   return(bank.RefreshDiagnostics() && bank.Diagnostics().BufferRead() &&
          bank.Used().BufferRead() && bank.State().BufferRead());
  }
//+------------------------------------------------------------------+
//| Two close keys intentionally own corrections with opposite sign. |
//+------------------------------------------------------------------+
bool SeedCloseOpposingCorrections(CD2SkillBank &bank)
  {
   float keys[4] = {1.00f, 0.00f, 0.98f, 0.20f};
   float corrections[4] = {0.40f, 0.00f, -0.40f, 0.00f};
   float directions[4] = {1.00f, 0.00f, -1.00f, 0.00f};
   float scales[2] = {0.40f, 0.40f};
   float utility[2] = {0.00f, 0.00f};
   float observations[2] = {2.00f, 2.00f};
   float uses[2] = {0.00f, 0.00f};
   float mass[2] = {1.00f, 1.00f};
   float used[2] = {1.00f, 1.00f};
   float states[2] = {(float)D2SKILL_ACTIVE, (float)D2SKILL_ACTIVE};
   float ages[2] = {0.00f, 0.00f};
   float protection[2] = {0.00f, 0.00f};
   float influence[2] = {0.00f, 0.00f};
   return(bank.Keys().AssignArray(keys) && bank.Keys().BufferWrite() &&
          bank.Corrections().AssignArray(corrections) && bank.Corrections().BufferWrite() &&
          bank.Directions().AssignArray(directions) && bank.Directions().BufferWrite() &&
          bank.Scales().AssignArray(scales) && bank.Scales().BufferWrite() &&
          bank.Utility().AssignArray(utility) && bank.Utility().BufferWrite() &&
          bank.ObservationsBank().AssignArray(observations) && bank.ObservationsBank().BufferWrite() &&
          bank.Uses().AssignArray(uses) && bank.Uses().BufferWrite() &&
          bank.MassBank().AssignArray(mass) && bank.MassBank().BufferWrite() &&
          bank.Used().AssignArray(used) && bank.Used().BufferWrite() &&
          bank.State().AssignArray(states) && bank.State().BufferWrite() &&
          bank.Ages().AssignArray(ages) && bank.Ages().BufferWrite() &&
          bank.Protection().AssignArray(protection) && bank.Protection().BufferWrite() &&
          bank.Influence().AssignArray(influence) && bank.Influence().BufferWrite());
  }
//+------------------------------------------------------------------+
//| Nearby account states must select their matching signed          |
//| residual.                                                        |
//+------------------------------------------------------------------+
bool RunCorrectionSelection(void)
  {
   CD2SkillBank bank;
   if(!bank.Init(D2_STUDY_DIM, 100, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !bank.SetThresholds(0.50f, 0.50f) || !bank.SetAlpha(1.00f) ||
      !bank.SetUtilityPolicy(false, -1.00f) || !bank.SetEnabled(true) ||
      !SeedCloseOpposingCorrections(bank))
      ReturnFalse;
   if(!Put(g_source.getOutput(), 1.00f, 0.00f) || !bank.FeedForward(g_source.AsObject()) ||
      !ReadSelection(bank))
      ReturnFalse;
   const float aligned_slot = bank.SelectedSlot()[0];
   const float aligned_correction = bank.SelectedCorrection()[0];
   if(!Put(g_source.getOutput(), 0.98f, 0.20f) || !bank.FeedForward(g_source.AsObject()) ||
      !ReadSelection(bank))
      ReturnFalse;
   const float opposing_slot = bank.SelectedSlot()[0];
   const float opposing_correction = bank.SelectedCorrection()[0];
   PrintFormat("D2SkillBankStudy: corrections aligned_slot=%.0f aligned=%.4f "
               "opposing_slot=%.0f opposing=%.4f",
               aligned_slot, aligned_correction, opposing_slot, opposing_correction);
   return(aligned_slot == 0.0f && aligned_correction > 0.0f &&
          opposing_slot == 1.0f && opposing_correction < 0.0f);
  }
//+------------------------------------------------------------------+
//| Candidate becomes an ordinary slot, then evicts only inactive.   |
//+------------------------------------------------------------------+
bool RunCandidateAndEviction(void)
  {
   CD2SkillBank bank;
   if(!bank.Init(D2_STUDY_DIM, 200, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !bank.SetThresholds(0.80f, 0.80f) || !bank.SetLifecycle(2, 0, 2) ||
      !bank.SetUtilityPolicy(false, -1.00f) || !bank.SetEnabled(true) ||
      !Train(bank, 1.00f, 0.00f) || !Train(bank, 1.00f, 0.00f) ||
      !Train(bank, -1.00f, 0.00f) || !Train(bank, -1.00f, 0.00f) ||
      !Train(bank, 0.00f, 1.00f) || !Train(bank, 0.00f, 1.00f) ||
      !ReadLifecycle(bank))
      ReturnFalse;
   const float candidate_before = bank.Diagnostics()[10];
   const float evicted_before = bank.Diagnostics()[7];
   if(candidate_before != 1.0f || evicted_before != 0.0f)
      ReturnFalse;
   if(!Train(bank, 0.00f, 1.00f) || !ReadLifecycle(bank))
      ReturnFalse;
   const float evicted_after = bank.Diagnostics()[7];
   const float inactive_after = bank.Diagnostics()[13];
   PrintFormat("D2SkillBankStudy: candidate candidate=%.0f evicted_before=%.0f "
               "evicted_after=%.0f inactive_after=%.0f",
               candidate_before, evicted_before, evicted_after, inactive_after);
   return(evicted_after >= 1.0f && inactive_after <= 1.0f);
  }
//+------------------------------------------------------------------+
//| A promoted DirectionMagnitude candidate keeps scale as its norm. |
//+------------------------------------------------------------------+
bool RunDirectionMagnitudePromotion(void)
  {
   CD2SkillBank bank;
   if(!bank.Init(D2_STUDY_DIM, 250, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !bank.SetThresholds(0.80f, -1.00f) || !bank.SetLifecycle(2, 0, 2) ||
      !bank.SetMaxCorrection(10.00f) || !bank.SetUtilityPolicy(false, -1.00f) ||
      !bank.SetEnabled(true) || !Train(bank, 1.00f, 0.00f) ||
      !Train(bank, 1.00f, 1.00f) || !Put(g_source.getOutput(), 1.00f, 0.00f) ||
      !bank.FeedForward(g_source.AsObject()) || !ReadSelection(bank) ||
      !bank.Directions().BufferRead() || !bank.Scales().BufferRead())
      ReturnFalse;
   const int slot = (int)bank.SelectedSlot()[0];
   if(slot < 0 || slot >= (int)D2_STUDY_SLOTS)
      ReturnFalse;
   const int offset = slot * (int)D2_STUDY_DIM;
   const float direction_norm = (float)MathSqrt(bank.Directions()[offset] *
                                bank.Directions()[offset] + bank.Directions()[offset + 1] *
                                bank.Directions()[offset + 1]);
   const float correction_norm = (float)MathSqrt(bank.SelectedCorrection()[0] *
                                 bank.SelectedCorrection()[0] + bank.SelectedCorrection()[1] *
                                 bank.SelectedCorrection()[1]);
   const float scale = bank.Scales()[slot];
   PrintFormat("D2SkillBankStudy: direction_magnitude slot=%d direction_norm=%.8f "
               "correction_norm=%.8f scale=%.8f", slot, direction_norm,
               correction_norm, scale);
   return(MathAbs(direction_norm - 1.00f) <= D2_STUDY_EPSILON &&
          MathAbs(correction_norm - scale) <= D2_STUDY_EPSILON);
  }
//+------------------------------------------------------------------+
//| Protected slots reject an unrelated confirmed candidate.         |
//+------------------------------------------------------------------+
bool RunProtectedRejection(void)
  {
   CD2SkillBank bank;
   if(!bank.Init(D2_STUDY_DIM, 300, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !bank.SetThresholds(0.80f, 0.80f) || !bank.SetLifecycle(2, 100, 1000) ||
      !bank.SetUtilityPolicy(false, -1.00f) || !bank.SetEnabled(true) ||
      !Train(bank, 1.00f, 0.00f) || !Train(bank, 1.00f, 0.00f) ||
      !Train(bank, -1.00f, 0.00f) || !Train(bank, -1.00f, 0.00f) ||
      !Train(bank, 0.00f, 1.00f) || !Train(bank, 0.00f, 1.00f) ||
      !ReadLifecycle(bank))
      ReturnFalse;
   const float rejected = bank.Diagnostics()[5];
   const float evicted = bank.Diagnostics()[7];
   const float protected_slots = bank.Diagnostics()[11];
   PrintFormat("D2SkillBankStudy: protected rejected=%.0f evicted=%.0f protected=%.0f",
               rejected, evicted, protected_slots);
   return(rejected >= 1.0f && evicted == 0.0f && protected_slots == D2_STUDY_SLOTS);
  }
//+------------------------------------------------------------------+
//| A query outside the acceptance gate must take the zero NoSkill   |
//| path.                                                            |
//+------------------------------------------------------------------+
bool RunNoSkill(void)
  {
   CD2SkillBank bank;
   if(!bank.Init(D2_STUDY_DIM, 400, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !bank.SetThresholds(0.50f, 0.50f) || !bank.SetUtilityPolicy(false, -1.00f) ||
      !bank.SetEnabled(true) || !SeedCloseOpposingCorrections(bank) ||
      !Put(g_source.getOutput(), -1.00f, 0.00f) || !bank.FeedForward(g_source.AsObject()) ||
      !ReadSelection(bank))
      ReturnFalse;
   const float correction_norm = bank.SelectedCorrection()[0] * bank.SelectedCorrection()[0] +
                                 bank.SelectedCorrection()[1] * bank.SelectedCorrection()[1];
   PrintFormat("D2SkillBankStudy: noskill slot=%.0f correction_norm=%.8f",
               bank.SelectedSlot()[0], correction_norm);
   return(bank.SelectedSlot()[0] == -1.0f && correction_norm <= D2_STUDY_EPSILON);
  }
//+------------------------------------------------------------------+
//| Save/Load preserves durable slots and gives the same next        |
//| winner.                                                          |
//+------------------------------------------------------------------+
bool RunPersistenceWinner(void)
  {
   CD2SkillBank original;
   if(!original.Init(D2_STUDY_DIM, 500, g_net.GetOpenCL(), D2_STUDY_SLOTS, ADAM, 1) ||
      !original.SetThresholds(0.50f, 0.50f) || !original.SetAlpha(1.00f) ||
      !original.SetUtilityPolicy(false, -1.00f) || !original.SetEnabled(true) ||
      !SeedCloseOpposingCorrections(original) || !Put(g_source.getOutput(), 0.98f, 0.20f) ||
      !original.FeedForward(g_source.AsObject()) || !ReadSelection(original))
      ReturnFalse;
   const float expected_slot = original.SelectedSlot()[0];
   const float expected_score = original.SelectedScore()[0];
   const float expected_correction = original.SelectedCorrection()[0];
   int handle = FileOpen(D2_STUDY_FILE, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   if(!original.Save(handle))
     {
      FileClose(handle);
      ReturnFalse;
     }
   FileClose(handle);
   CD2SkillBank restored;
   handle = FileOpen(D2_STUDY_FILE, FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   if(!restored.Load(handle, g_net.GetOpenCL()))
     {
      FileClose(handle);
      ReturnFalse;
     }
   FileClose(handle);
   if(!Put(g_source.getOutput(), 0.98f, 0.20f) || !restored.FeedForward(g_source.AsObject()) ||
      !ReadSelection(restored))
      ReturnFalse;
   const float slot_delta = MathAbs(restored.SelectedSlot()[0] - expected_slot);
   const float score_delta = MathAbs(restored.SelectedScore()[0] - expected_score);
   const float correction_delta = MathAbs(restored.SelectedCorrection()[0] - expected_correction);
   PrintFormat("D2SkillBankStudy: persistence winner=%.0f slot_delta=%.8f "
               "score_delta=%.8f correction_delta=%.8f",
               restored.SelectedSlot()[0], slot_delta, score_delta, correction_delta);
   return(slot_delta <= D2_STUDY_EPSILON && score_delta <= D2_STUDY_EPSILON &&
          correction_delta <= D2_STUDY_EPSILON);
  }
//+------------------------------------------------------------------+
//| Implements OnInit.                                               |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   if(!CreateHost() ||
      !g_source.Init(0, 10, g_net.GetOpenCL(), D2_STUDY_DIM, ADAM, 1) ||
      !RunCorrectionSelection() || !RunCandidateAndEviction() ||
      !RunDirectionMagnitudePromotion() ||
      !RunProtectedRejection() || !RunNoSkill() || !RunPersistenceWinner())
     {
      PrintFormat("D2SkillBankStudy: FAIL error=%d", GetLastError());
      return(INIT_FAILED);
     }
   Print("D2SkillBankStudy: PASS corrections candidate direction protected eviction noskill persistence");
   g_completed = true;
   if(!EventSetMillisecondTimer(1))
      return(INIT_FAILED);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Implements OnTimer.                                              |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   if(g_completed)
      ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Implements OnDeinit.                                             |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   g_source.Clear();
  }
//+---+