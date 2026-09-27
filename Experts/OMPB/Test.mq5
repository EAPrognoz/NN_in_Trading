//+------------------------------------------------------------------+
//| Test.mq5                                                         |
//| Copyright 2026, DNG                                              |
//| https://www.mql5.com/ru/users/dng                                |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property strict
#property description "Инференс обученной политики D2Skill (OMPB) на счёте."
#property description "Оценка вне выборки (OOS) и банковский режим инференса."
#property description "Параметры заморожены; обучение не запускается."
#property description "Прогон на тестовом счёте."
#define  D2SKILL
#include "Trajectory.mqh"
//---
input group                       "---- D2Skill OOS ----"
input ENUM_D2SKILL_STAGE          InpD2SkillStage = D2Skill_D2_STAGE_AUGMENTED; //Lifecycle stage
input ENUM_D2SKILL_MODE           InpD2SkillExecutionMode = D2_INFERENCE; //Bank execution mode
input ED2SkillRepresentation      InpD2SkillRepresentation = D2SkillDirectionMagnitude; //Bank representation
input bool                        InpD2SkillResetBanksOnRepresentationMismatch =
   false; //Reset incompatible banks
input bool                        InpD2SkillRecreateIncompatibleCheckpoint =
   false; //Recreate incompatible policy checkpoint
datetime Start = 0;
datetime End = 0;
int Epochs = 0;
CNet Actor;
CBufferFloat State, TimeState, Account, Action;
double PreviousBalance = 0, PreviousEquity = 0;
bool D2SkillTestInitialized = false;
ulong OOSTransitions = 0, OOSExecutableActions = 0;
double OOSInitialBalance = 0;
datetime OOSStartTime = 0;
double OOSInitialEquity = 0, OOSPeakEquity = 0, OOSMaximumDrawdown = 0;
double OOSReward = 0;
ulong OOSTradeCount = 0;
ulong OOSGpuBoundaryLatencyTotal = 0, OOSGpuBoundaryLatencyMaximum = 0;
ulong OOSGpuBoundaryLatencySamples = 0;
//+------------------------------------------------------------------+
//| Implements D2SkillWriteD2OOSHeader.                              |
//+------------------------------------------------------------------+
bool D2SkillWriteD2OOSHeader(const int handle)
  {
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   if(FileWrite(handle,
                "run_time", "symbol", "timeframe", "stage", "stage_name",
                "mode", "mode_name", "execution_mode", "execution_mode_name",
                "utility_aware", "min_utility", "utility_scale",
                "utility_source", "initial_balance", "final_balance", "initial_equity",
                "final_equity", "reward_equity_delta", "pnl_balance_delta", "max_drawdown",
                "max_drawdown_percent", "closed_trades", "transitions", "executable_actions",
                "comparison_eligible", "task_diagnostics_valid", "task_diagnostics_error",
                "task_retrievals", "task_no_skill", "task_retrieval_rate", "task_no_skill_rate",
                "task_promotions", "task_replacements", "step_diagnostics_valid",
                "step_diagnostics_error", "step_retrievals", "step_no_skill",
                "step_retrieval_rate", "step_no_skill_rate", "step_promotions",
                "step_replacements", "gpu_boundary_latency_us_avg", "gpu_boundary_latency_us_max",
                "paired_terminal_updates", "task_selection_valid", "selected_task_slot",
                "selected_task_score", "step_selection_valid", "selected_step_slot",
                "selected_step_score") <= 0)
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2StageToString.                               |
//+------------------------------------------------------------------+
string D2SkillD2StageToString(const int stage)
  {
   if(stage == D2Skill_D2_STAGE_BASE)
      return("base");
   if(stage == D2Skill_D2_STAGE_FORMATION)
      return("formation");
   if(stage == D2Skill_D2_STAGE_HINDSIGHT)
      return("hindsight");
   if(stage == D2Skill_D2_STAGE_AUGMENTED)
      return("augmented");
   if(stage == D2Skill_D2_STAGE_ONLINE)
      return("online");
   return("unknown");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2ModeToString.                                |
//+------------------------------------------------------------------+
string D2SkillD2ModeToString(const ENUM_D2SKILL_BANK_MODE mode)
  {
   if(mode == D2Skill_D2_MODE_BASE)
      return("base");
   if(mode == D2Skill_D2_MODE_TASK)
      return("task");
   if(mode == D2Skill_D2_MODE_STEP)
      return("step");
   if(mode == D2Skill_D2_MODE_FULL)
      return("full");
   return("unknown");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2ExecutionModeToString.                       |
//+------------------------------------------------------------------+
string D2SkillD2ExecutionModeToString(const ENUM_D2SKILL_MODE mode)
  {
   switch(mode)
     {
      case D2_DISABLED:
         return("disabled");
      case D2_COLLECT:
         return("collect");
      case D2_EVALUATE:
         return("evaluate");
      case D2_INFERENCE:
         return("inference");
      case D2_ONLINE_CALIBRATION:
         return("online_calibration");
     }
   return("unknown");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2UtilitySource.                               |
//+------------------------------------------------------------------+
string D2SkillD2UtilitySource(void)
  {
   if(D2SkillD2UtilityMode == D2Skill_D2_UTILITY_PAIRED_HINDSIGHT)
      return("paired_terminal");
   return("disabled");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillReadD2Selection.                               |
//+------------------------------------------------------------------+
bool D2SkillReadD2Selection(const bool task_bank, int &selected_slot, float &selected_score)
  {
   selected_slot = -1;
   selected_score = 0.0f;
   CNeuronBaseOCL *layer = Actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill || !skill.Ready())
      ReturnFalse;
   CD2SkillBank *bank = (task_bank ? skill.TaskBank() : skill.StepBank());
   if(!bank)
      ReturnFalse;
//--- BASE/disabled branches intentionally expose no device selection state.
//--- OOS renders this normal control flow as selection_unavailable / NA.
   if(!bank.Enabled())
      return(false);
   if(!bank.SelectedSlot() || !bank.SelectedScore())
      ReturnFalse;
   if(!bank.SelectedSlot().BufferRead() || !bank.SelectedScore().BufferRead() ||
      bank.SelectedSlot().Total() != 1 || bank.SelectedScore().Total() != 1)
      ReturnFalse;
   selected_slot = (int)bank.SelectedSlot()[0];
   selected_score = bank.SelectedScore()[0];
   return(selected_slot >= 0 && (uint)selected_slot < bank.Slots() &&
          MathIsValidNumber(double(selected_score)) && selected_score > -1.0e30f);
  }
//+------------------------------------------------------------------+
//| Reads diagnostic counters only at the OOS reporting boundary.    |
//+------------------------------------------------------------------+
bool D2SkillReadD2OOSBankStats(const bool task_bank, ulong &retrievals,
                               ulong &no_skill, ulong &promotions,
                               ulong &replacements)
  {
   retrievals = 0;
   no_skill = 0;
   promotions = 0;
   replacements = 0;
   CNeuronBaseOCL *layer = Actor.Layer(1);
   if(!layer || layer.Type() != defNeuronD2Skill)
      ReturnFalse;
   CD2Skill *skill = (CD2Skill*)layer;
   if(!skill || !skill.Ready())
      ReturnFalse;
   CD2SkillBank *bank = (task_bank ? skill.TaskBank() : skill.StepBank());
//--- Base/disabled branches intentionally have no device diagnostics.  Their
//--- counters remain zero and OOS reporting must not allocate scratch to read
//--- them.
   if(!bank)
      ReturnFalse;
   if(!bank.Enabled())
      return(true);
   if(!bank.RefreshDiagnostics() || !bank.Diagnostics() ||
      !bank.Diagnostics().BufferRead() || bank.Diagnostics().Total() != D2SKILL_DIAGNOSTICS)
      ReturnFalse;
   CBufferFloat *diagnostics = bank.Diagnostics();
   retrievals = (ulong)MathMax(0.0, double(diagnostics[0]));
   no_skill = (ulong)MathMax(0.0, double(diagnostics[1]));
   promotions = (ulong)MathMax(0.0, double(diagnostics[4]));
   replacements = (ulong)MathMax(0.0, double(diagnostics[7]));
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2OOSCounterText.                              |
//+------------------------------------------------------------------+
string D2SkillD2OOSCounterText(const bool valid, const ulong value)
  {
   return(valid ? StringFormat("%I64u", value) : "NA");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2OOSRateText.                                 |
//+------------------------------------------------------------------+
string D2SkillD2OOSRateText(const bool valid, const ulong numerator, const ulong denominator)
  {
   if(!valid)
      return("NA");
   return(StringFormat("%.8f", (denominator > 0 ? double(numerator) / denominator : 0.0)));
  }
//+------------------------------------------------------------------+
//| Implements D2SkillD2OOSReadError.                                |
//+------------------------------------------------------------------+
string D2SkillD2OOSReadError(const bool valid)
  {
   return(valid ? "none" : "bank_or_diagnostics_unavailable");
  }
//+------------------------------------------------------------------+
//| Implements D2SkillWriteD2OOSBankConfiguration.                   |
//+------------------------------------------------------------------+
bool D2SkillWriteD2OOSBankConfiguration(const int handle, const string prefix,
                                        CD2SkillBank *bank)
  {
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   if(!bank)
     {
      FileWrite(handle, prefix + "_configuration_valid=false");
      FileWrite(handle, prefix + "_configuration_error=bank_unavailable");
      ReturnFalse;
     }
   FileWrite(handle, prefix + "_configuration_valid=true");
   FileWrite(handle, StringFormat("%s_slots=%u", prefix, bank.Slots()));
   FileWrite(handle, StringFormat("%s_dimension=%u", prefix, bank.Dimension()));
   FileWrite(handle, StringFormat("%s_representation=%d", prefix, bank.Representation()));
   FileWrite(handle, StringFormat("%s_alpha=%f", prefix, bank.Alpha()));
   FileWrite(handle, StringFormat("%s_utility_aware=%s", prefix,
                                  (bank.UtilityAware() ? "true" : "false")));
   FileWrite(handle, StringFormat("%s_min_utility=%f", prefix, bank.MinUtility()));
   FileWrite(handle, StringFormat("%s_max_correction=%f", prefix, bank.MaxCorrection()));
   FileWrite(handle, StringFormat("%s_beta_correction=%f", prefix, bank.BetaCorrection()));
   FileWrite(handle, StringFormat("%s_beta_direction=%f", prefix, bank.BetaDirection()));
   FileWrite(handle, StringFormat("%s_beta_magnitude=%f", prefix, bank.BetaMagnitude()));
//--- The following values are the fixed D2SkillEnableD2Banks reconstruction
//--- contract; the Bank API intentionally has no duplicate read-only getters.
   FileWrite(handle, prefix + "_top_k=1");
   FileWrite(handle, prefix + "_similarity_threshold=0.65");
   FileWrite(handle, prefix + "_direction_threshold=0.00");
   FileWrite(handle, prefix + "_utility_weight=0.05");
   FileWrite(handle, prefix + "_beta_key=0.05");
   FileWrite(handle, prefix + "_beta_utility=0.05");
   FileWrite(handle, prefix + "_min_confirmations=3");
   FileWrite(handle, prefix + "_protection_age=32");
   FileWrite(handle, prefix + "_inactivity_age=256");
   return(true);
  }
//+------------------------------------------------------------------+
//| Counts closed position exits in the exact tested symbol window.  |
//+------------------------------------------------------------------+
ulong D2SkillOOSClosedTrades(void)
  {
   if(OOSStartTime <= 0 || !HistorySelect(OOSStartTime, TimeCurrent()))
      return(0);
   ulong trades = 0;
   const uint total = HistoryDealsTotal();
   for(uint index = 0; index < total; index++)
     {
      const ulong deal = HistoryDealGetTicket(index);
      if(deal == 0 || HistoryDealGetString(deal, DEAL_SYMBOL) != _Symbol)
         continue;
      const ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal, DEAL_ENTRY);
      if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY || entry == DEAL_ENTRY_INOUT)
         trades++;
     }
   return(trades);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillAppendOOSResultRow.                            |
//+------------------------------------------------------------------+
bool D2SkillAppendOOSResultRow(const double final_balance, const double final_equity)
  {
   const int handle = FileOpen(D2Skill_OOS_FILE,
                               FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   const bool need_header = (FileSize(handle) <= 0);
   FileSeek(handle, 0, SEEK_END);
   if(need_header && !D2SkillWriteD2OOSHeader(handle))
     {
      FileClose(handle);
      ReturnFalse;
     }
   int task_slot = -1, step_slot = -1;
   float task_score = 0.0f;
   float step_score = 0.0f;
   const bool task_selection_valid = D2SkillReadD2Selection(true, task_slot, task_score);
   const bool step_selection_valid = D2SkillReadD2Selection(false, step_slot, step_score);
   ulong task_retrievals = 0, task_no_skill = 0, task_promotions = 0, task_replacements = 0;
   ulong step_retrievals = 0, step_no_skill = 0, step_promotions = 0, step_replacements = 0;
   const bool task_diagnostics_valid = D2SkillReadD2OOSBankStats(true, task_retrievals, task_no_skill,
                                       task_promotions, task_replacements);
   const bool step_diagnostics_valid = D2SkillReadD2OOSBankStats(false, step_retrievals, step_no_skill,
                                       step_promotions, step_replacements);
   const ulong task_total = task_retrievals + task_no_skill;
   const ulong step_total = step_retrievals + step_no_skill;
   if(!task_diagnostics_valid || !step_diagnostics_valid)
      PrintFormat("D2Skill D2 OOS diagnostics unavailable: task=%s step=%s",
                  D2SkillD2OOSReadError(task_diagnostics_valid),
                  D2SkillD2OOSReadError(step_diagnostics_valid));
   if(!task_selection_valid || !step_selection_valid)
      PrintFormat("D2Skill D2 OOS selection unavailable: task=%s step=%s",
                  (task_selection_valid ? "none" : "selection_unavailable"),
                  (step_selection_valid ? "none" : "selection_unavailable"));
   const double drawdown_percent = (OOSPeakEquity > 0.0 ?
                                    100.0 * OOSMaximumDrawdown / OOSPeakEquity : 0.0);
   const double average_latency = (OOSGpuBoundaryLatencySamples > 0 ?
                                   double(OOSGpuBoundaryLatencyTotal) / OOSGpuBoundaryLatencySamples : 0.0);
   FileWrite(handle,
             TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS),
             _Symbol,
             EnumToString((ENUM_TIMEFRAMES)Period()),
             D2SkillD2Stage,
             D2SkillD2StageToString(D2SkillD2Stage),
             D2SkillD2Mode,
             D2SkillD2ModeToString(D2SkillD2Mode),
             D2SkillD2ExecutionMode,
             D2SkillD2ExecutionModeToString(D2SkillD2ExecutionMode),
             (D2SkillD2UtilityAware ? "true" : "false"),
             D2SkillD2MinUtility,
             D2SkillD2UtilityScale,
             D2SkillD2UtilitySource(),
             OOSInitialBalance,
             final_balance,
             OOSInitialEquity,
             final_equity,
             OOSReward,
             (final_balance - OOSInitialBalance),
             OOSMaximumDrawdown,
             drawdown_percent,
             OOSTradeCount,
             OOSTransitions,
             OOSExecutableActions,
             (OOSExecutableActions > 0 ? "true" : "false"),
             (task_diagnostics_valid ? "true" : "false"),
             D2SkillD2OOSReadError(task_diagnostics_valid),
             D2SkillD2OOSCounterText(task_diagnostics_valid, task_retrievals),
             D2SkillD2OOSCounterText(task_diagnostics_valid, task_no_skill),
             D2SkillD2OOSRateText(task_diagnostics_valid, task_retrievals, task_total),
             D2SkillD2OOSRateText(task_diagnostics_valid, task_no_skill, task_total),
             D2SkillD2OOSCounterText(task_diagnostics_valid, task_promotions),
             D2SkillD2OOSCounterText(task_diagnostics_valid, task_replacements),
             (step_diagnostics_valid ? "true" : "false"),
             D2SkillD2OOSReadError(step_diagnostics_valid),
             D2SkillD2OOSCounterText(step_diagnostics_valid, step_retrievals),
             D2SkillD2OOSCounterText(step_diagnostics_valid, step_no_skill),
             D2SkillD2OOSRateText(step_diagnostics_valid, step_retrievals, step_total),
             D2SkillD2OOSRateText(step_diagnostics_valid, step_no_skill, step_total),
             D2SkillD2OOSCounterText(step_diagnostics_valid, step_promotions),
             D2SkillD2OOSCounterText(step_diagnostics_valid, step_replacements),
             average_latency,
             OOSGpuBoundaryLatencyMaximum,
             D2SkillD2PairedUtilityUpdates,
             (task_selection_valid ? "true" : "false"),
             (task_selection_valid ? IntegerToString(task_slot) : "NA"),
             (task_selection_valid ? DoubleToString(task_score, 8) : "NA"),
             (step_selection_valid ? "true" : "false"),
             (step_selection_valid ? IntegerToString(step_slot) : "NA"),
             (step_selection_valid ? DoubleToString(step_score, 8) : "NA"));
   FileClose(handle);
   Print("D2Skill D2 OOS table: row appended to ", D2Skill_OOS_FILE);
   return(true);
  }
//+------------------------------------------------------------------+
//| Implements D2SkillWriteD2OOSManifest.                            |
//+------------------------------------------------------------------+
bool D2SkillWriteD2OOSManifest(const double final_balance, const double final_equity)
  {
   const int handle = FileOpen(D2Skill_OOS_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=D2Skill_D2_OOS");
   FileWrite(handle, StringFormat("run_time=%s", TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS)));
   FileWrite(handle, StringFormat("symbol=%s", _Symbol));
   FileWrite(handle, StringFormat("timeframe=%s", EnumToString((ENUM_TIMEFRAMES)Period())));
   FileWrite(handle, StringFormat("stage=%d", D2SkillD2Stage));
   FileWrite(handle, StringFormat("stage_name=%s", D2SkillD2StageToString(D2SkillD2Stage)));
   FileWrite(handle, StringFormat("mode=%d", D2SkillD2Mode));
   FileWrite(handle, StringFormat("mode_name=%s", D2SkillD2ModeToString(D2SkillD2Mode)));
   FileWrite(handle, StringFormat("execution_mode=%d", D2SkillD2ExecutionMode));
   FileWrite(handle, StringFormat("execution_mode_name=%s",
                                  D2SkillD2ExecutionModeToString(D2SkillD2ExecutionMode)));
   FileWrite(handle, StringFormat("utility_aware=%s", (D2SkillD2UtilityAware ? "true" : "false")));
   FileWrite(handle, StringFormat("min_utility=%f", D2SkillD2MinUtility));
   FileWrite(handle, StringFormat("utility_scale=%f", D2SkillD2UtilityScale));
   FileWrite(handle, StringFormat("utility_source=%s", D2SkillD2UtilitySource()));
   FileWrite(handle, StringFormat("initial_balance=%f", OOSInitialBalance));
   FileWrite(handle, StringFormat("final_balance=%f", final_balance));
   FileWrite(handle, StringFormat("pnl=%f", final_balance - OOSInitialBalance));
   FileWrite(handle, "reward_definition=equity_delta_accumulated_after_each_executed_transition");
   FileWrite(handle, StringFormat("initial_equity=%f", OOSInitialEquity));
   FileWrite(handle, StringFormat("final_equity=%f", final_equity));
   FileWrite(handle, StringFormat("reward=%f", OOSReward));
   FileWrite(handle, StringFormat("max_drawdown=%f", OOSMaximumDrawdown));
   FileWrite(handle, StringFormat("max_drawdown_percent=%f",
                                  (OOSPeakEquity > 0.0 ? 100.0 * OOSMaximumDrawdown / OOSPeakEquity : 0.0)));
   FileWrite(handle, StringFormat("closed_trades=%I64u", OOSTradeCount));
   FileWrite(handle, StringFormat("oos_transitions=%I64u", OOSTransitions));
   FileWrite(handle, StringFormat("executable_actions=%I64u", OOSExecutableActions));
   FileWrite(handle, StringFormat("comparison_eligible=%s",
                                  (OOSExecutableActions > 0 ? "true" : "false")));
   ulong task_retrievals = 0, task_no_skill = 0, task_promotions = 0, task_replacements = 0;
   ulong step_retrievals = 0, step_no_skill = 0, step_promotions = 0, step_replacements = 0;
   const bool task_diagnostics_valid = D2SkillReadD2OOSBankStats(true, task_retrievals, task_no_skill,
                                       task_promotions, task_replacements);
   const bool step_diagnostics_valid = D2SkillReadD2OOSBankStats(false, step_retrievals, step_no_skill,
                                       step_promotions, step_replacements);
   const ulong task_total = task_retrievals + task_no_skill;
   const ulong step_total = step_retrievals + step_no_skill;
   FileWrite(handle, StringFormat("task_diagnostics_valid=%s", (task_diagnostics_valid ? "true" : "false")));
   FileWrite(handle, "task_diagnostics_error=" + D2SkillD2OOSReadError(task_diagnostics_valid));
   FileWrite(handle, "task_retrievals=" + D2SkillD2OOSCounterText(task_diagnostics_valid, task_retrievals));
   FileWrite(handle, "task_no_skill=" + D2SkillD2OOSCounterText(task_diagnostics_valid, task_no_skill));
   FileWrite(handle, "task_retrieval_rate=" +
             D2SkillD2OOSRateText(task_diagnostics_valid, task_retrievals, task_total));
   FileWrite(handle, "task_no_skill_rate=" + D2SkillD2OOSRateText(task_diagnostics_valid, task_no_skill, task_total));
   FileWrite(handle, "task_promotions=" + D2SkillD2OOSCounterText(task_diagnostics_valid, task_promotions));
   FileWrite(handle, "task_replacements=" + D2SkillD2OOSCounterText(task_diagnostics_valid, task_replacements));
   FileWrite(handle, StringFormat("step_diagnostics_valid=%s", (step_diagnostics_valid ? "true" : "false")));
   FileWrite(handle, "step_diagnostics_error=" + D2SkillD2OOSReadError(step_diagnostics_valid));
   FileWrite(handle, "step_retrievals=" + D2SkillD2OOSCounterText(step_diagnostics_valid, step_retrievals));
   FileWrite(handle, "step_no_skill=" + D2SkillD2OOSCounterText(step_diagnostics_valid, step_no_skill));
   FileWrite(handle, "step_retrieval_rate=" +
             D2SkillD2OOSRateText(step_diagnostics_valid, step_retrievals, step_total));
   FileWrite(handle, "step_no_skill_rate=" + D2SkillD2OOSRateText(step_diagnostics_valid, step_no_skill, step_total));
   FileWrite(handle, "step_promotions=" + D2SkillD2OOSCounterText(step_diagnostics_valid, step_promotions));
   FileWrite(handle, "step_replacements=" + D2SkillD2OOSCounterText(step_diagnostics_valid, step_replacements));
   FileWrite(handle, "gpu_latency_definition=host_elapsed_us_Actor_feedForward_plus_ReadAction");
   FileWrite(handle, StringFormat("gpu_boundary_latency_us_avg=%f",
                                  (OOSGpuBoundaryLatencySamples > 0 ?
                                   double(OOSGpuBoundaryLatencyTotal) / OOSGpuBoundaryLatencySamples : 0.0)));
   FileWrite(handle, StringFormat("gpu_boundary_latency_us_max=%I64u", OOSGpuBoundaryLatencyMaximum));
   FileWrite(handle, StringFormat("paired_utility_updates=%I64u", D2SkillD2PairedUtilityUpdates));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", D2SkillLastSignature));
   FileWrite(handle, StringFormat("actor_file=%s", D2Skill_ACTOR_FILE));
   FileWrite(handle, StringFormat("actor_hash_fnv64=%I64u",
                                  D2SkillHashFile(ulong(1469598103934665603), D2Skill_ACTOR_FILE)));
   FileWrite(handle, StringFormat("actor_critic_manifest=%s", D2Skill_AC_MANIFEST_FILE));
   CNeuronBaseOCL *layer = Actor.Layer(1);
   CD2Skill *skill = (layer && layer.Type() == defNeuronD2Skill ? (CD2Skill*)layer : NULL);
   if(!skill || !skill.Ready())
     {
      D2SkillWriteD2OOSBankConfiguration(handle, "task", NULL);
      D2SkillWriteD2OOSBankConfiguration(handle, "step", NULL);
     }
   else
     {
      D2SkillWriteD2OOSBankConfiguration(handle, "task", skill.TaskBank());
      D2SkillWriteD2OOSBankConfiguration(handle, "step", skill.StepBank());
     }
   FileClose(handle);
   Print("D2Skill D2 OOS manifest: ", D2Skill_OOS_MANIFEST_FILE);
   return(true);
  }
//+------------------------------------------------------------------+
//| Function OnInit.                                                 |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//--- Exact Forecast and six-artifact AC manifest/hash validation happens
//--- before loading/binding the Actor and before any possible policy forward.
   D2SkillTestInitialized = false;
   if(!D2SkillConfigureRuntime(InpD2SkillStage, InpD2SkillMode,
                               InpD2SkillExecutionMode, InpD2SkillUtilityAware,
                               InpD2SkillMinUtility, InpD2SkillUtilityScale,
                               false, false, InpD2SkillRepresentation,
                               InpD2SkillResetBanksOnRepresentationMismatch,
                               InpD2SkillRecreateIncompatibleCheckpoint))
     { Print("D2Skill test init: runtime configuration=FAIL"); return INIT_FAILED; }
   if(!D2SkillInitIndicators())
     { Print("D2Skill test init: indicators=FAIL"); return INIT_FAILED; }
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
     { Print("D2Skill test init: trade filling=FAIL"); return INIT_FAILED; }
   if(!D2SkillLoadForecastInference())
     { Print("D2Skill test init: forecast=FAIL"); return INIT_FAILED; }
   if(!D2SkillLoadInferenceActor(Actor))
     { Print("D2Skill test init: actor checkpoint=FAIL"); return INIT_FAILED; }
   if(!D2SkillVerifyFrozenForecastExact())
     { Print("D2Skill test init: frozen forecast=FAIL"); return INIT_FAILED; }
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   OOSInitialBalance = PreviousBalance;
   OOSStartTime = TimeCurrent();
   OOSInitialEquity = PreviousEquity;
   OOSPeakEquity = PreviousEquity;
   OOSMaximumDrawdown = 0;
   OOSReward = 0;
   OOSTradeCount = 0;
   OOSTransitions = 0;
   OOSExecutableActions = 0;
   OOSGpuBoundaryLatencyTotal = 0;
   OOSGpuBoundaryLatencyMaximum = 0;
   OOSGpuBoundaryLatencySamples = 0;
   D2SkillTestInitialized = true;
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(!D2SkillTestInitialized)
     {
      D2SkillForecast = NULL;
      return;
     }
   const double final_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   const double final_equity = AccountInfoDouble(ACCOUNT_EQUITY);
//--- Inference is pure: neither policy nor Forecast artifacts are saved.
   if(D2SkillTestInitialized && D2SkillForecast != NULL && !D2SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   if(D2SkillTestInitialized)
      PrintFormat("D2Skill OOS summary: transitions=%I64u executable_actions=%I64u " +
                  "initial_balance=%.2f final_balance=%.2f",
                  OOSTransitions, OOSExecutableActions, OOSInitialBalance,
                  final_balance);
   OOSTradeCount = D2SkillOOSClosedTrades();
   const double drawdown_percent = (OOSPeakEquity > 0.0 ?
                                    100.0 * OOSMaximumDrawdown / OOSPeakEquity : 0.0);
   const double average_latency = (OOSGpuBoundaryLatencySamples > 0 ?
                                   double(OOSGpuBoundaryLatencyTotal) / OOSGpuBoundaryLatencySamples : 0.0);
   Print("D2 OOS_TABLE stage mode execution_mode utility_aware min_utility utility_scale " +
         "transitions executable_actions initial_balance final_balance pnl");
   PrintFormat("D2 OOS_TABLE_ROW stage=%d(%s) mode=%d(%s) execution=%d(%s) aware=%s " +
               "min_utility=%.6f utility_scale=%.6f utility_source=%s reward=%.2f pnl=%.2f " +
               "drawdown=%.2f drawdown_percent=%.6f closed_trades=%I64u " +
               "gpu_boundary_latency_us_avg=%.3f gpu_boundary_latency_us_max=%I64u " +
               "paired_terminal_updates=%I64u transitions=%I64u executable=%I64u " +
               "initial_balance=%.2f final_balance=%.2f",
               D2SkillD2Stage, D2SkillD2StageToString(D2SkillD2Stage),
               D2SkillD2Mode, D2SkillD2ModeToString(D2SkillD2Mode),
               D2SkillD2ExecutionMode, D2SkillD2ExecutionModeToString(D2SkillD2ExecutionMode),
               (D2SkillD2UtilityAware ? "true" : "false"),
               D2SkillD2MinUtility,
               D2SkillD2UtilityScale, D2SkillD2UtilitySource(),
               OOSReward, final_balance - OOSInitialBalance, OOSMaximumDrawdown,
               drawdown_percent, OOSTradeCount, average_latency,
               OOSGpuBoundaryLatencyMaximum,
               D2SkillD2PairedUtilityUpdates,
               OOSTransitions, OOSExecutableActions, OOSInitialBalance,
               final_balance);
   if(OOSExecutableActions == 0)
      Print("D2Skill D2 OOS REJECT: executable_actions=0; Base/Task/Step/Full comparison is ineligible");
   if(!D2SkillAppendOOSResultRow(final_balance, final_equity))
      Print("D2Skill D2 OOS table: append failed");
   if(!D2SkillWriteD2OOSManifest(final_balance, final_equity))
      Print("D2Skill D2 OOS manifest: write failed");
   D2SkillForecast = NULL;
  }
//+------------------------------------------------------------------+
//| Function OnTick.                                                 |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
   if(!D2SkillRefreshLiveMarket(GetPointer(State), GetPointer(TimeState)) ||
      !D2SkillMarket.Clear() ||
      !D2SkillForwardForecastState(GetPointer(State)))
     { PrintFormat("%s -> %d market/forecast", __FUNCTION__, __LINE__); return; }
   double buy_value = 0, sell_value = 0;
   if(!D2SkillBuildLiveAccount(PreviousBalance, PreviousEquity, Rates[0].time,
                               GetPointer(Account), buy_value, sell_value))
     { PrintFormat("%s -> %d account", __FUNCTION__, __LINE__); return; }
   const ulong forward_started = GetMicrosecondCount();
   if(!Actor.feedForward(GetPointer(Account), 1, false, GetPointer(D2SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(Action)))
     { PrintFormat("%s -> %d actor/read", __FUNCTION__, __LINE__); return; }
   const ulong forward_elapsed = GetMicrosecondCount() - forward_started;
   OOSGpuBoundaryLatencyTotal += forward_elapsed;
   if(forward_elapsed > OOSGpuBoundaryLatencyMaximum)
      OOSGpuBoundaryLatencyMaximum = forward_elapsed;
   OOSGpuBoundaryLatencySamples++;
   if(!D2SkillExecuteAction(GetPointer(Action), buy_value, sell_value))
     { PrintFormat("%s -> %d actor/execute", __FUNCTION__, __LINE__); return; }
   OOSTransitions++;
   if(IsExecutableOrder(double(Action[0]), double(Action[1]), double(Action[2])) ||
      IsExecutableOrder(double(Action[3]), double(Action[4]), double(Action[5])))
      OOSExecutableActions++;
   const double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   OOSReward += current_equity - PreviousEquity;
   if(current_equity > OOSPeakEquity)
      OOSPeakEquity = current_equity;
   else
      if(OOSPeakEquity - current_equity > OOSMaximumDrawdown)
         OOSMaximumDrawdown = OOSPeakEquity - current_equity;
   PreviousBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PreviousEquity = current_equity;
  }
//+---+