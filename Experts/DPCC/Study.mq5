//+------------------------------------------------------------------+
//|                                                        Study.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property version   "1.00"
#property description "DPCC offline Actor and Critic training with a frozen Market encoder"
#property description "Uses historical bars and Oracle targets for supervised policy learning"
#property description "Saves the latest Actor and Critic weights and optimizer moments"
#property strict
//--- This Expert Advisor advances one committed history row per timer event.
#define DPCC_TRAJECTORY_ENTRY_STUDY
#include "Trajectory.mqh"
#undef DPCC_TRAJECTORY_ENTRY_STUDY

input group "---- История и период обучения ----"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_CURRENT;      // PERIOD_CURRENT использует таймфрейм графика.
input datetime InpTrainingStart = D'2025.01.01 00:00';    // Серверное время открытия бара; начало включительно.
//--- Конец не включается; 12 будущих баров должны быть внутри периода.
input datetime InpTrainingEnd = D'2026.01.01 00:00';
//--- История захватывается автоматически из выбранных закрытых баров.

input group "---- Лимит и режим запуска ----"
//--- Всего обновлений; 0 — один проход. При превышении окна оно повторяется.
input ulong InpStopAfterUpdates = 0;
//--- Начать с текущих весов или инициализировать новый запуск.
input ENUM_DPCC_RESUME_MODE InpResumeMode = DPCC_RESUME_NEW;
input ENUM_DPCC_MODE InpMode = DPCC_P0;                         // Режим DPCC P0/P1/P2.
//--- Только подготовить и проверить входы, без обновления весов.
input bool InpPreflightOnly = false;

input group "---- Actor: параметры генерации действий ----"
input ulong InpDiffusionSeed = 20261007;     // Начальное значение генератора случайных чисел.
input double InpDiffusionS = 0.008;          // Масштаб шума при генерации торгового плана.
input double InpDiffusionBetaCap = 0.999;    // Верхняя граница beta расписания диффузии.

input group "---- Critic: цель и дисконтирование ----"
input int InpRiskType = defSRM_MEAN_CVAR;    // Функция риска для распределительной цели Critic.
input double InpRiskParameter = 0.1;         // Параметр выбранной функции риска.
input double InpMeanWeight = 0.0;            // Вес средней доходности в SRM-цели.
input double InpGamma = 0.5;                 // Коэффициент дисконтирования будущей награды.

input group "---- Oracle: поиск целевого плана ----"
input uint InpProjectionMaxIterations = 16;                // Максимум итераций локального поиска.
input uint InpProjectionMaxEvaluations = 160;              // Максимум оценок кандидатов за одну строку.
input uint InpProjectionLookbackSteps = 12;                // Число шагов горизонта, используемых проекцией.
input double InpProjectionImprovementAtol = 0.000001;      // Минимальное абсолютное улучшение для продолжения поиска.
input double InpProjectionDistanceTieAtol = 0.00000001;    // Допуск расстояния при выборе равных кандидатов.
input double InpOracleDelta1 = 0.1;                        // Шаг локального изменения координаты операции 1.
input double InpOracleDelta2 = 0.1;                        // Шаг локального изменения координаты операции 2.
input double InpOracleDelta3 = 0.1;                        // Шаг локального изменения координаты операции 3.
input double InpOracleDelta4 = 0.1;                        // Шаг локального изменения координаты операции 4.
input double InpOracleDelta5 = 0.1;                        // Шаг локального изменения координаты операции 5.
input double InpOracleDelta6 = 0.1;                        // Шаг локального изменения координаты операции 6.

input group "---- Ограничения счёта и защиты ----"
input double InpMinEquity = 50.0;        // Остановить обучение при эквити ниже этого значения.
input double InpMaxSLPoints = 1000.0;    // Максимальное расстояние Stop Loss в points.
input double InpMaxTPPoints = 1000.0;    // Максимальное расстояние Take Profit в points.

const string DPCC_PACKAGE_ROOT = "DPCC/StudyForecast_M05_20261017";
const string DPCC_STUDY_OUTPUT_STEM = "M/Study";
const bool   DPCC_RESTORE_LATEST_TRAINING_WEIGHTS = true;
const string DPCC_TRAINING_STATE_PATH = "DPCC/StudyAction6.latest";
const bool   DPCC_COMMON_FILES = true;

string g_dpcc_market_path = "";
string g_dpcc_market_profile_text = "";
ENUM_TIMEFRAMES g_dpcc_timeframe = PERIOD_CURRENT;
SDPCCAPNativeLoadContract g_dpcc_market_contract;
float g_dpcc_condition_scale[], g_dpcc_condition_bias[];
SDPCCRuntimeConfig g_dpcc_runtime;
float g_dpcc_limits[];

SDPCCOracleHistoryInput g_dpcc_input_source;
CDPCCMarketClock g_dpcc_clock;
CDPCCStudyStageConsumer g_dpcc_runner;
CDPCCStopButton g_dpcc_stop_button;
bool g_dpcc_initialized = false;
bool g_dpcc_finalize_pending = false;
bool g_dpcc_policy_state_restored = false;
bool g_dpcc_floor_reached = false;
bool g_dpcc_stop_save_failed = false;
ulong g_dpcc_completed = 0;
ulong g_dpcc_update_target = 0;
ulong g_dpcc_completed_epochs = 0;
uint g_dpcc_epoch_row = 0;
//+------------------------------------------------------------------+
//| Study progress metrics                                           |
//+------------------------------------------------------------------+
double EXTDPCCRewardSum       = 0.0;
double EXTDPCCTeacherScoreSum = 0.0;
double EXTDPCCLastReward      = 0.0;
double EXTDPCCLastTeacherScore = 0.0;
//+------------------------------------------------------------------+
//| Initialize source, model graphs, and optional weights-only state |
//+------------------------------------------------------------------+
bool DPCCStudyInitialize(string &reason)
  {
   reason = "";
   float oracle_deltas[6];
   oracle_deltas[0] = (float)InpOracleDelta1;
   oracle_deltas[1] = (float)InpOracleDelta2;
   oracle_deltas[2] = (float)InpOracleDelta3;
   oracle_deltas[3] = (float)InpOracleDelta4;
   oracle_deltas[4] = (float)InpOracleDelta5;
   oracle_deltas[5] = (float)InpOracleDelta6;
   if(!DPCCBuildAction6RuntimeFromInputs(InpMode, InpDiffusionSeed, InpDiffusionS,
                                         InpDiffusionBetaCap, InpRiskType, InpRiskParameter, InpMeanWeight, InpGamma,
                                         InpProjectionMaxIterations, InpProjectionMaxEvaluations,
                                         InpProjectionLookbackSteps,
                                         InpProjectionImprovementAtol, InpProjectionDistanceTieAtol, InpMinEquity,
                                         InpMaxSLPoints, InpMaxTPPoints, g_dpcc_runtime, g_dpcc_limits, reason))
     {
      reason = "action6_runtime=" + reason;
      return(false);
     }
   const long period_seconds = (long)PeriodSeconds(g_dpcc_timeframe);
   const long warmup_seconds = (long)DPCC_ENCODER_WARMUP_HISTORY * period_seconds;
   const datetime capture_from = (datetime)((long)InpTrainingStart - warmup_seconds);
   string raw_path = "", raw_sha = "", capture_reason = "";
   uint raw_rows = 0;
   if(period_seconds <= 0 || capture_from <= 0 ||
      !DPCCCaptureStudyForecastRaw(_Symbol, g_dpcc_timeframe, capture_from, InpTrainingEnd,
                                   raw_path, raw_sha, raw_rows, capture_reason, false,
                                   RSIPrice, CCIPrice, MACDPrice, true, DPCC_COMMON_FILES))
     {
      reason = "source_capture=" + (capture_reason == "" ? "invalid_history_range" : capture_reason);
      return(false);
     }
   Print("DPCC_STUDY_RAW_INPUT_CAPTURE timeframe=", EnumToString(g_dpcc_timeframe),
         " rows=", raw_rows, " from=", (string)capture_from, " to=", (string)InpTrainingEnd);
   if(!DPCCLoadOracleHistoryInput(DPCC_PACKAGE_ROOT, InpTrainingStart, InpTrainingEnd, g_dpcc_runtime, g_dpcc_limits,
                                  raw_path, DPCC_STUDY_OUTPUT_STEM, oracle_deltas,
                                  DPCC_COMMON_FILES, g_dpcc_timeframe,
                                  g_dpcc_market_path, g_dpcc_market_profile_text,
                                  g_dpcc_market_contract,
                                  g_dpcc_input_source, reason))
     {
      if(reason == "")
         reason = "terminal_files_required";
      return(false);
     }
   long server_opens[];
   const int count = ArraySize(g_dpcc_input_source.raw_bars);
   if(ArrayResize(server_opens, count) != count)
     {
      reason = "server_calendar_capacity";
      return(false);
     }
   for(int i = 0; i < count; i++)
      server_opens[i] = g_dpcc_input_source.raw_bars[i].server_open;
   if(!g_dpcc_clock.InitializeHistoricalServerCalendar(server_opens, reason, g_dpcc_timeframe))
      return(false);
   if(!g_dpcc_runner.InitializeAction6History(g_dpcc_input_source.config,
         (uint)ArraySize(g_dpcc_input_source.decision_bars),
         g_dpcc_input_source.deltas, g_dpcc_input_source.actor_beta,
         g_dpcc_input_source.actor_alpha_bar, g_dpcc_input_source.actor_sigma,
         InpMode, g_dpcc_timeframe, g_dpcc_input_source.raw_bars, g_dpcc_input_source.source_server,
         g_dpcc_input_source.raw_indicator_params, g_dpcc_input_source.raw_successor_open_server,
         g_dpcc_input_source.raw_path, g_dpcc_market_profile_text,
         g_dpcc_market_path, g_dpcc_market_contract, g_dpcc_condition_scale,
         g_dpcc_condition_bias, DPCC_COMMON_FILES, reason))
     {
      reason = "action6_init=" + reason;
      return(false);
     }
   if(!g_dpcc_runner.BindHistoryProvenance(g_dpcc_input_source, reason))
     {
      reason = "action6_provenance=" + reason;
      return(false);
     }
   if(!InpPreflightOnly && DPCC_RESTORE_LATEST_TRAINING_WEIGHTS)
     {
      const string checkpoint_manifest = DPCC_TRAINING_STATE_PATH + ".manifest";
      const int file_flags = (DPCC_COMMON_FILES ? FILE_COMMON : 0);
      if(FileIsExist(checkpoint_manifest, file_flags))
        {
         string restored_manifest = "";
         if(!g_dpcc_runner.LoadAction6TrainingStateOnly(DPCC_TRAINING_STATE_PATH,
               DPCC_COMMON_FILES, restored_manifest, reason))
           {
            reason = "training_state_restore=" + reason;
            return(false);
           }
         g_dpcc_policy_state_restored = true;
         Print("DPCC_STUDY_POLICY_STATE restored=1 scope=weights_and_adam_only manifest=",
               restored_manifest, " data_binding=NONE row_cursor=NOT_SAVED account=NOT_SAVED");
        }
      else
         Print("DPCC_STUDY_POLICY_STATE restored=0 source=initialized_parameters");
     }
   const int rows = ArraySize(g_dpcc_input_source.decision_bars);
   if(rows <= 0)
     {
      reason = "study.history.no_decision_rows";
      return(false);
     }
   g_dpcc_completed = 0;
   g_dpcc_completed_epochs = 0;
   g_dpcc_epoch_row = 0;
   g_dpcc_update_target = (InpStopAfterUpdates > 0 ? InpStopAfterUpdates : (ulong)rows);
   reason = "";
   return(true);
  }
//+------------------------------------------------------------------+
//| Emit the existing read-only Study preflight receipt              |
//+------------------------------------------------------------------+
void DPCCStudyLogPreflight(void)
  {
   Print("DPCC_STUDY_PREFLIGHT_COMPLETE rows=", ArraySize(g_dpcc_input_source.decision_bars),
         " no_training_updates=1 no_model_exports=1 no_orders=1");
   Print("DPCC_STUDY_PREFLIGHT_INPUT package_root=", DPCC_PACKAGE_ROOT,
         " raw_rows=", ArraySize(g_dpcc_input_source.raw_bars),
         " output_stem=", DPCC_STUDY_OUTPUT_STEM);
   Print("DPCC_STUDY_PREFLIGHT_SOURCE symbol=", _Symbol, " timeframe=", EnumToString(g_dpcc_timeframe),
         " server=", g_dpcc_input_source.source_server);
   Print("DPCC_STUDY_PREFLIGHT_CLOCK time_domain=", g_dpcc_clock.TimeDomain());
   Print("DPCC_STUDY_PREFLIGHT_MODEL market_profile_sha256=", g_dpcc_input_source.model_sha,
         " config_sha256=", g_dpcc_input_source.config_sha, " market=", g_dpcc_market_path,
         " market_profile_sha256=", g_dpcc_market_contract.profile_sha256);
   for(int i = 0; i < ArraySize(g_dpcc_input_source.source_labels); i++)
      Print("DPCC_STUDY_PREFLIGHT_PIN label=", g_dpcc_input_source.source_labels[i],
            " path=", g_dpcc_input_source.source_paths[i],
            " sha256=", g_dpcc_input_source.source_hashes[i]);
  }
//+------------------------------------------------------------------+
//| Render Study progress and committed learning metrics             |
//+------------------------------------------------------------------+
void DPCCStudyRenderProgress(const int rows, const int active_row)
  {
   const ulong target_updates = g_dpcc_update_target;
   if(rows <= 0 || target_updates == 0)
     {
      Comment("");
      return;
     }
   const double percent = 100.0 * (double)g_dpcc_completed / (double)target_updates;
   const ulong total_epochs = target_updates / (ulong)rows +
                              (target_updates % (ulong)rows > 0 ? 1 : 0);
   ulong current_epoch = g_dpcc_completed_epochs + 1;
   uint current_row = (active_row > 0 ? (uint)active_row : g_dpcc_epoch_row);
   if(current_row > (uint)rows)
      current_row = (uint)rows;
   if(g_dpcc_completed >= target_updates && target_updates % (ulong)rows == 0)
     {
      current_epoch = total_epochs;
      current_row = (uint)rows;
     }
   string status = "Study ACTION6: " + DoubleToString(percent, 2) + "% (" +
                   (string)g_dpcc_completed + "/" + (string)target_updates + ")";
   status += "\nEpoch: " + (string)current_epoch + "/" + (string)total_epochs +
             " | row: " + (string)current_row + "/" + (string)rows;
   if(active_row > 0)
      status += " | processing";
   if(g_dpcc_completed > 0)
     {
      const double count = (double)g_dpcc_completed;
      status += "\nReward last/mean: " + DoubleToString(EXTDPCCLastReward, 2) + " / " +
                DoubleToString(EXTDPCCRewardSum / count, 2);
      status += "\nOracle last/mean: " + DoubleToString(EXTDPCCLastTeacherScore, 2) + " / " +
                DoubleToString(EXTDPCCTeacherScoreSum / count, 2) +
                " | floor=" + (g_dpcc_floor_reached ? "true" : "false");
     }
   else
      status += "\nReward: waiting | Oracle: waiting";
   Comment(status);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
//| Save only current policy weights and Adam moments                |
//+------------------------------------------------------------------+
bool DPCCStudySaveTrainingState(const string cause)
  {
   if(g_dpcc_completed == 0)
      return(true);
   string manifest = "", sha = "", reason = "";
   const bool saved = g_dpcc_runner.SaveAction6TrainingStateOnly(DPCC_TRAINING_STATE_PATH,
                      DPCC_COMMON_FILES, manifest, sha, reason);
   Print("DPCC_STUDY_POLICY_STATE cause=", cause, " saved=", (saved ? 1 : 0),
         " updates=", g_dpcc_completed, " contents=actor_q1_q2_weights_and_adam",
         " row_cursor=NOT_SAVED account=NOT_SAVED data_binding=NONE",
         " manifest=", manifest, " sha256=", sha, " reason=", reason);
   return(saved);
  }
//+------------------------------------------------------------------+
//| Publish completed Study output only at an accepted phase         |
//| boundary                                                         |
//+------------------------------------------------------------------+
void DPCCStudyFinalize(void)
  {
   EventKillTimer();
   DPCCStudyRenderProgress(ArraySize(g_dpcc_input_source.decision_bars), 0);
   if(!DPCCStudySaveTrainingState("TRAINING_BOUNDARY"))
     {
      g_dpcc_stop_save_failed = true;
      g_dpcc_stop_button.SetText("Retry save and stop");
      DPCCStudyRenderProgress(ArraySize(g_dpcc_input_source.decision_bars), 0);
      Print("DPCC_STUDY_HISTORY_BLOCKED phase=training_state_save updates=", g_dpcc_completed,
            " action=await_button_retry");
      return;
     }
   const ulong target_updates = g_dpcc_update_target;
   const bool complete = (g_dpcc_completed >= target_updates);
   if(complete || g_dpcc_floor_reached)
     {
      string manifest = "", sha = "", reason = "";
      if(!g_dpcc_runner.SaveAction6HistoryModels(g_dpcc_completed, g_dpcc_completed_epochs,
            manifest, sha, reason))
         Print("DPCC_STUDY_HISTORY_BLOCKED phase=action6_export reason=", reason,
               " updates=", g_dpcc_completed, "/", target_updates);
      else
         Print("DPCC_STUDY_ACTION6_ROWS_COMPLETE updates=", g_dpcc_completed, "/", target_updates,
               " floor=", (g_dpcc_floor_reached ? "true" : "false"),
               " model_export=PASS manifest=", manifest, " sha256=", sha);
     }
   else
      Print("DPCC_STUDY_ACTION6_ROWS_COMPLETE updates=", g_dpcc_completed, "/", target_updates,
            " model_export=SKIPPED_partial_pass training_state=LATEST_WEIGHTS_ONLY");
   g_dpcc_stop_button.RemoveAdvisor();
  }
//+------------------------------------------------------------------+
//| Expert lifecycle                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   SDPCCExecutionControl control;
   string control_reason = "";
   if(!DPCCConfigureExecutionControl(DPCC_PACKAGE_ROOT, InpTimeframe, DPCC_STAGE_STUDY, "",
                                     InpStopAfterUpdates, InpResumeMode, InpMode,
                                     DPCC_STAGE_STUDY, control, control_reason, true))
     {
      Print("DPCC_STUDY_BLOCKED phase=profile_control reason=", control_reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   g_dpcc_timeframe = control.timeframe;
   if(!DPCCLoadMarketExecutionProfile(control, DPCC_COMMON_FILES, control_reason))
     {
      Print("DPCC_STUDY_BLOCKED phase=market_training_admission reason=", control_reason);
      return(INIT_PARAMETERS_INCORRECT);
     }
   g_dpcc_market_path = control.market_path;
   g_dpcc_market_profile_text = control.profile_text;
   g_dpcc_market_contract = control.market_contract;
   if(ArrayCopy(g_dpcc_condition_scale, control.condition_scale) != 109 ||
      ArrayCopy(g_dpcc_condition_bias, control.condition_bias) != 109)
     {
      Print("DPCC_STUDY_BLOCKED phase=market_condition_binding reason=condition_vector_shape");
      return(INIT_PARAMETERS_INCORRECT);
     }
   Print("DPCC_STUDY_MARKET_ADMITTED ready=", g_dpcc_market_contract.market_ready,
         " OMPB_calibration_completed=", g_dpcc_market_contract.calibration_completed,
         " OMPB_quality_pass=", g_dpcc_market_contract.stage02_quality_pass,
         " stage=", g_dpcc_market_contract.stage);
   if(!InpPreflightOnly && !g_dpcc_stop_button.Create("STUDY", "Stop and save"))
     {
      Print("DPCC_STUDY_BLOCKED phase=stop_button reason=chart_object_create_failed error=", GetLastError());
      return(INIT_FAILED);
     }
   if(!EventSetMillisecondTimer(1))
     {
      g_dpcc_stop_button.Delete();
      Print("DPCC_STUDY_TIMER_INIT_FAILED error=", GetLastError());
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit                                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment("");
   g_dpcc_stop_button.Delete();
  }
//+------------------------------------------------------------------+
//| Initialize once, then process at most one history bar per timer  |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   if(!g_dpcc_initialized)
     {
      if(!InpPreflightOnly)
         Comment("Study ACTION6: initializing history and model...");
      string reason = "";
      if(!DPCCStudyInitialize(reason))
        {
         EventKillTimer();
         Comment("");
         Print("DPCC_STUDY_HISTORY_BLOCKED ", reason);
         g_dpcc_stop_button.RemoveAdvisor();
         return;
        }
      g_dpcc_initialized = true;
      if(InpPreflightOnly)
        {
         Comment("");
         DPCCStudyLogPreflight();
         EventKillTimer();
         g_dpcc_stop_button.RemoveAdvisor();
         return;
        }
      Print("DPCC_STUDY_ACTION6_GRAPH_READY market=", g_dpcc_market_path,
            " actor=initialized q1=initialized q2=initialized legacy_z_route=inactive",
            " row_update=enabled mode=", EnumToString(InpMode),
            " updates_target=", g_dpcc_update_target,
            " restored_policy_weights=", (g_dpcc_policy_state_restored ? "true" : "false"),
            " stop_control=ready");
      DPCCStudyRenderProgress(ArraySize(g_dpcc_input_source.decision_bars), 0);
      return;
     }
   if(g_dpcc_finalize_pending)
     {
      DPCCStudyFinalize();
      return;
     }
   if(IsStopped())
     {
      EventKillTimer();
      Print("DPCC_STUDY_HISTORY_STOPPED rows=", g_dpcc_completed,
            " checkpoint=use_chart_stop_button_for_controlled_save");
      g_dpcc_stop_button.RemoveAdvisor();
      return;
     }
   const int rows = ArraySize(g_dpcc_input_source.decision_bars);
   if(g_dpcc_completed >= g_dpcc_update_target)
     {
      g_dpcc_finalize_pending = true;
      return;
     }
   if(g_dpcc_epoch_row >= (uint)rows)
     {
      string reason = "";
      if(!g_dpcc_runner.BeginAction6HistoryPass(reason))
        {
         EventKillTimer();
         Print("DPCC_STUDY_HISTORY_BLOCKED phase=epoch_reset epoch=",
               g_dpcc_completed_epochs + 1, " reason=", reason);
         g_dpcc_stop_button.RemoveAdvisor();
         return;
        }
      g_dpcc_completed_epochs++;
      g_dpcc_epoch_row = 0;
      g_dpcc_floor_reached = false;
      Print("DPCC_STUDY_EPOCH_COMPLETE epoch=", g_dpcc_completed_epochs,
            " rows=", rows, " updates=", g_dpcc_completed, "/", g_dpcc_update_target);
      DPCCStudyRenderProgress(rows, 1);
      return;
     }
   DPCCStudyRenderProgress(rows, (int)g_dpcc_epoch_row + 1);
   SDPCCOracleHistoryRow row;
   SDPCCOfflineRowResult result;
   string reason = "";
   if(!DPCCBuildOracleHistoryRow(g_dpcc_input_source, (int)g_dpcc_epoch_row,
                                 g_dpcc_clock, row, reason))
     {
      DPCCStudyRenderProgress(rows, 0);
      EventKillTimer();
      Print("DPCC_STUDY_HISTORY_BLOCKED epoch=", g_dpcc_completed_epochs + 1,
            " row=", g_dpcc_epoch_row,
            " phase=row_build reason=", reason);
      g_dpcc_stop_button.RemoveAdvisor();
      return;
     }
   if(!g_dpcc_runner.StepAction6History(g_dpcc_clock, row, result, reason) || !result.committed)
     {
      DPCCStudyRenderProgress(rows, 0);
      EventKillTimer();
      Print("DPCC_STUDY_HISTORY_BLOCKED epoch=", g_dpcc_completed_epochs + 1,
            " row=", g_dpcc_epoch_row,
            " phase=action6_update reason=", reason);
      g_dpcc_stop_button.RemoveAdvisor();
      return;
     }
   g_dpcc_epoch_row = (uint)result.next_row;
   g_dpcc_completed++;
   g_dpcc_floor_reached = result.floor_reached;
   EXTDPCCLastReward = result.reward;
   EXTDPCCLastTeacherScore = result.teacher_score;
   EXTDPCCRewardSum += result.reward;
   EXTDPCCTeacherScoreSum += result.teacher_score;
   DPCCStudyRenderProgress(rows, 0);
   if(result.next_row % 1000 == 0 || result.floor_reached || g_dpcc_epoch_row >= (uint)rows)
      Print("DPCC_STUDY_HISTORY_PROGRESS epoch=", g_dpcc_completed_epochs + 1,
            " row=", result.next_row, "/", rows,
            " updates=", g_dpcc_completed, "/", g_dpcc_update_target,
            " reward=", DoubleToString(result.reward, 2),
            " oracle_return=", DoubleToString(result.teacher_score, 2),
            " floor=", (result.floor_reached ? "true" : "false"));
   if(result.floor_reached || g_dpcc_completed >= g_dpcc_update_target)
      g_dpcc_finalize_pending = true;
  }
//+------------------------------------------------------------------+
//| Function OnChartEvent                                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(!g_dpcc_stop_button.Clicked(id, sparam))
      return;
   DPCCStudyRenderProgress(ArraySize(g_dpcc_input_source.decision_bars), 0);
   EventKillTimer();
   g_dpcc_stop_button.SetText(g_dpcc_stop_save_failed ?
                              "Retrying save..." : "Saving and stopping...");
   if(g_dpcc_initialized && g_dpcc_completed > 0)
     {
      if(!DPCCStudySaveTrainingState("CONTROLLED_STOP"))
        {
         g_dpcc_stop_save_failed = true;
         g_dpcc_stop_button.SetText("Retry save and stop");
         DPCCStudyRenderProgress(ArraySize(g_dpcc_input_source.decision_bars), 0);
         return;
        }
     }
   else
      Print("DPCC_STUDY_CONTROLLED_STOP no_policy_update=1 existing_checkpoint_unchanged=1");
   g_dpcc_stop_save_failed = false;
   Print("DPCC_STUDY_CONTROLLED_STOP saved=",
         (g_dpcc_completed > 0 ? "true" : "not_needed"),
         " action=remove_advisor");
   g_dpcc_stop_button.RemoveAdvisor();
  }
//+------------------------------------------------------------------+