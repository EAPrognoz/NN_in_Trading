//+------------------------------------------------------------------+
//|                                                         Test.mq5 |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
#property description "AC-SRM Stage 03/05: инференс замороженной политики с оценкой на LIVE-счете."
#property description "На старте бара Actor принимает решение; реальные ордера исполняются на тестерном счете."
#property description "Источник — LIVE-котировки; результат — журнал PASS/FAIL и манифест-проверка."
#define Online
#include "Trajectory.mqh"
//---
int                              Epochs = 0;
//---
input group                       "---- AC-SRM inference ----"
input int                         InpDealHorizon = 24; //Deal horizon (manifest validation symmetry)
input double                      MinBalance     = 50.0; //Minimum account balance (irrecoverable threshold)
//---
input group                       "---- SRM critic head ----"
input int                         InpRiskType    = defSRM_MEAN_CVAR; //Risk measure (defSRM_*)
input float                       InpRiskParameter = 0.1f; //Risk parameter (alpha/lambda/nu)
input float                       InpMeanWeight  = 0.0f; //Mean-CVaR omega
input int                         InpQuantiles   = 32; //Head quantiles (factory-fixed)
//---
input group                       "---- AC-SRM production checkpoint ----"
input ENUM_OMPB_STAGE             InpOMPBStage   = OMPB_STAGE_BASE_POLICY; //Chain stage: keep 03 (base policy)
//---
CNet           Actor;
CNet           Q1;
CNet           Q2;

CBufferFloat   State;
CBufferFloat   TimeState;
CBufferFloat   Account;
CBufferFloat   CurrentAction;
CBufferFloat   CriticInput;
CBufferFloat   SRMValue;
CBufferFloat   PrevAccount;
CBufferFloat   PrevAction;

int            TestBars = 0;
ulong          TestTerminal = 0;
ulong          TestExecutable = 0;
ulong          TestNoTrade = 0;
double         RewardSum = 0;
double         SRM1Sum = 0;
double         SRM2Sum = 0;
double         BalanceMin = 0;
double         BalanceMax = 0;
double         EquityMin = 0;
double         EquityMax = 0;
ulong          Irrecoverable = 0;
//--- LIVE stream state: previous bar post-execution snapshots.
double         LivePrevBalance = 0;
double         LivePrevEquity  = 0;
bool           LiveReady = false;
bool           LiveFinalized = false;
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
//+------------------------------------------------------------------------------------------------------------+
//| Inference requires the published Stage 03 checkpoint; creation  is intentionally impossible in Test mode.  |
//+------------------------------------------------------------------------------------------------------------+
bool LoadPoliciesInference(void)
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
   if(!FileIsExist(Skill_AC_MANIFEST_FILE, FILE_COMMON))
     {
      Print("Skill inference policy=FAIL reason=manifest_missing");
      ReturnFalse;
     }
   if(!SkillValidateACManifestFile(true, Skill_AC_MANIFEST_FILE, false) ||
      !SkillLoadPolicyNet(Actor, Skill_ACTOR_FILE) ||
      !SkillLoadPolicyNet(Q1, Skill_Q1_FILE) ||
      !SkillLoadPolicyNet(Q2, Skill_Q2_FILE))
     {
      Print("Skill inference policy=FAIL reason=tuple_invalid");
      ReturnFalse;
     }
   if(!SkillBindPolicyOpenCLChecked(Actor, Q1, Q2) ||
      !ValidatePolicyShapeACSRM(Actor, Q1, Q2))
      ReturnFalse;
   Actor.TrainMode(false);
   Q1.TrainMode(false);
   Q2.TrainMode(false);
   if(!Actor.SetWeightsUpdate(false) || !Q1.SetWeightsUpdate(false) ||
      !Q2.SetWeightsUpdate(false))
      ReturnFalse;
   return(true);
  }
//+------------------------------------------------------------------+
//| Returns the SRM risk score of one critic head distribution.      |
//+------------------------------------------------------------------+
bool CriticRiskValue(CNet &critic, double &value)
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
   value = double(SRMValue[0]);
   return(MathIsValidNumber(value));
  }
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Refreshes the live market state from the last CLOSED bar and the  scenario (forecast) context consumed by the actor cross-  attention.  SkillRefreshLiveMarket fills the Rates/indicator  window; the explicit Market feed refreshes z/u/pi.  |
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveRefreshState(CBufferFloat *state, CBufferFloat *time_state)
  {
   if(!SkillRefreshLiveMarket(state, time_state))
      ReturnFalse;
   return(SkillMarket.feedForward(state, 1, false, (CBufferFloat*)NULL) &&
          SkillForecast != NULL);
  }
//+------------------------------------------------------------------------------------------------------------------------------------+
//| Keeps the PREVIOUS (account, action) host copies: the action is  reported with its own bar and classified by executable contract.  |
//+------------------------------------------------------------------------------------------------------------------------------------+
bool LiveKeepPrevious(void)
  {
   if(PrevAccount.Total() != (int)AccountDescr &&
      !PrevAccount.BufferInit((int)AccountDescr, 0))
      ReturnFalse;
   if(PrevAction.Total() != (int)NActions &&
      !PrevAction.BufferInit((int)NActions, 0))
      ReturnFalse;
   return(PrevAccount.AssignArray(GetPointer(Account)) &&
          PrevAction.AssignArray(GetPointer(CurrentAction)));
  }
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| Executes the bar decision on the LIVE account: builds the real  account vector, runs the frozen Actor, retains this bar's  (state, action) critic tuple for next-bar diagnostics and re-  targets this expert's positions (CogDriver).  The post-execution  LIVE balance/equity snapshot is the reward base of this bar.  |
//+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveExecuteBar(const double previous_balance, const double previous_equity)
  {
   //--- Fresh CLOSED-bar market context: LiveRefreshState runs
//--- SkillRefreshLiveMarket (CopyRates offset 1 + Symb.RefreshRates)
//--- which SkillExecuteAction needs to derive valid SL/TP distances.
   if(!LiveRefreshState(GetPointer(State), GetPointer(TimeState)))
      ReturnFalse;
   datetime now = TimeCurrent();
   double buy_value = 0, sell_value = 0;
   if(!SkillBuildLiveAccount(previous_balance, previous_equity, now,
                             GetPointer(Account), buy_value, sell_value))
      ReturnFalse;
   if(Account.GetIndex() < 0 && !Account.BufferCreate(SkillMarket.GetOpenCL()))
      ReturnFalse;
   if(Account.GetIndex() >= 0 && !Account.BufferWrite())
      ReturnFalse;
   if(!Actor.Clear() || !Actor.feedForward(GetPointer(Account), 1, false,
                                           GetPointer(SkillMarket), -1) ||
      !ReadAction(Actor, GetPointer(CurrentAction)))
      ReturnFalse;
   CNeuronBaseOCL *context_layer = Actor.Layer(0);
   CNeuronBaseOCL *actor_layer = Actor.Layer(3);
   if(!context_layer || !actor_layer)
      ReturnFalse;
   if(!BuildCriticInput(context_layer.getOutput(), actor_layer.getOutput(),
                        GetPointer(CriticInput)))
      ReturnFalse;
   double margin_penalty = 0;
   bool market_closed = false;
   if(!SkillExecuteAction(GetPointer(CurrentAction), buy_value, sell_value,
                          margin_penalty, market_closed))
      ReturnFalse;
//--- Post-execution LIVE snapshot: the reward of this bar is evaluated
//--- from it on the next bar open.
   LivePrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   LivePrevEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
   return(LiveKeepPrevious());
  }
//+------------------------------------------------------------------+
//| First bar: no previous bar exists, only the opening decision.    |
//+------------------------------------------------------------------+
bool LiveFirstBar(void)
  {
   LiveReady = true;
   LivePrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   LivePrevEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(!LiveExecuteBar(LivePrevBalance, LivePrevEquity))
      ReturnFalse;
   PrintFormat("ACSRM_TEST_LIVE_BAR1 time=%s balance=%.2f equity=%.2f act=(%.8f %.8f %.8f %.8f %.8f %.8f)",
               TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
               LivePrevBalance, LivePrevEquity,
               double(PrevAction[0]), double(PrevAction[1]), double(PrevAction[2]),
               double(PrevAction[3]), double(PrevAction[4]), double(PrevAction[5]));
   return(true);
  }
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
//| One new-bar live transition: evaluates the just-closed bar from  the LIVE account (reward = equity change vs the post-execution  snapshot of the previous bar), emits critic SRM diagnostics on  the retained tuple and places the new bar order target.  |
//+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
bool LiveEvaluateBar(void)
  {
   if(!LiveReady)
      return(LiveFirstBar());
//--- LIVE result of the just-closed bar.
   const double balance_now = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity_now  = AccountInfoDouble(ACCOUNT_EQUITY);
   const double reward = equity_now - LivePrevEquity;
   if(!MathIsValidNumber(reward) || !MathIsValidNumber(balance_now) ||
      !MathIsValidNumber(equity_now))
      ReturnFalse;
   TestBars++;
   double srm1 = 0, srm2 = 0;
//--- Critic diagnostics on the retained previous (state, action) tuple.
   if(!Q1.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !Q2.feedForward(GetPointer(CriticInput), 1, false, GetPointer(SkillMarket), -1) ||
      !CriticRiskValue(Q1, srm1) || !CriticRiskValue(Q2, srm2))
      ReturnFalse;
   const double buy_lot = MathMax(0.0, double(PrevAction[0] - PrevAction[3]));
   const double sell_lot = MathMax(0.0, double(PrevAction[3] - PrevAction[0]));
   const bool executable = (IsExecutableOrder(buy_lot, PrevAction[1], PrevAction[2]) ||
                            IsExecutableOrder(sell_lot, PrevAction[4], PrevAction[5]));
   if(executable)
      TestExecutable++;
   else
      TestNoTrade++;
   RewardSum += reward;
   SRM1Sum += srm1;
   SRM2Sum += srm2;
   const bool terminal = (balance_now <= MinBalance);
   if(terminal)
      TestTerminal++;
   if(balance_now <= MinBalance)
      Irrecoverable++;
   if(TestBars == 1)
     {
      BalanceMin = balance_now;
      BalanceMax = balance_now;
      EquityMin = equity_now;
      EquityMax = equity_now;
     }
   else
     {
      BalanceMin = MathMin(BalanceMin, balance_now);
      BalanceMax = MathMax(BalanceMax, balance_now);
      EquityMin = MathMin(EquityMin, equity_now);
      EquityMax = MathMax(EquityMax, equity_now);
     }
   if(TestBars <= 4 || TestBars % 1000 == 0)
      PrintFormat("ACSRM_TEST_BAR bar=%d balance=%.2f equity=%.2f reward=%.8f srm1=%.8f srm2=%.8f act=(%.8f %.8f %.8f %.8f %.8f %.8f) terminal=%s",
                  TestBars, balance_now, equity_now, reward, srm1, srm2,
                  double(PrevAction[0]), double(PrevAction[1]), double(PrevAction[2]),
                  double(PrevAction[3]), double(PrevAction[4]), double(PrevAction[5]),
                  (terminal ? "true" : "false"));
//--- The new bar decision is made from the post-execution LIVE account
//--- of the previous bar (relative change fields) and the closed-bar
//--- market state.
   return(LiveExecuteBar(LivePrevBalance, LivePrevEquity));
  }
//+------------------------------------------------------------------+
//| Emits the final live evaluation report once.                     |
//+------------------------------------------------------------------+
void LiveFinalize(void)
  {
   if(LiveFinalized)
      return;
   LiveFinalized = true;
   if(TestBars <= 0)
     {
      Print("ACSRM_TEST_FAIL reason=no_bars");
      return;
     }
   const double r_mean = RewardSum / double(TestBars);
   const double srm1_mean = SRM1Sum / double(TestBars);
   const double srm2_mean = SRM2Sum / double(TestBars);
   PrintFormat("ACSRM_TEST_PASS bars=%d episodes=1 terminal=%I64u executable=%I64u no_trade=%I64u " +
               "r_mean=%.8f srm1_mean=%.8f srm2_mean=%.8f balance_min=%.2f balance_max=%.2f " +
               "equity_min=%.2f equity_max=%.2f irrecoverable=%I64u",
               TestBars, TestTerminal, TestExecutable, TestNoTrade,
               r_mean, srm1_mean, srm2_mean, BalanceMin, BalanceMax, EquityMin, EquityMax,
               Irrecoverable);
//--- Leave a clean tester account.
   CloseByDirection(POSITION_TYPE_BUY);
   CloseByDirection(POSITION_TYPE_SELL);
  }
//+------------------------------------------------------------------+
//| Initializes the Stage 03 inference Expert.                       |
//+------------------------------------------------------------------+
int OnInit()
  {
   ResetLastError();
//--- Pin the deal-evaluation horizon for manifest validation symmetry.
   SkillManifestDealHorizon = MathMax(1, InpDealHorizon);
//--- Confirm Stage 03, the frozen Forecast checkpoint and the policy tuple.
   if(InpOMPBStage != OMPB_STAGE_BASE_POLICY ||
      !SkillInitIndicators() || !SkillLoadForecastInference() ||
      !SkillConfigureProductionACSRMCheckpoint() ||
      !SkillValidateProductionACSRMCheckpoint() ||
      !SkillCaptureProductionACSRMFingerprints() || !LoadPoliciesInference() ||
      !SkillVerifyFrozenForecastExact() ||
      !SkillVerifyProductionACSRMFingerprints())
     {
      PrintFormat("ACSRM inference initialization failed at line %d error=%d",
                  __LINE__, GetLastError());
      return(INIT_FAILED);
     }
   PrintFormat("ACSRM_TEST_INIT_PASS tester=%d", MQLInfoInteger(MQL_TESTER));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Function OnDeinit.                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(MQLInfoInteger(MQL_TESTER) == 1 && !LiveFinalized)
     {
      PrintFormat("%s live finalize reason=%d bars=%d", ACSRM_LOG_PREFIX, reason, TestBars);
      LiveFinalize();
     }
   if(SkillForecast != NULL && !SkillVerifyFrozenForecastExact())
      PrintFormat("%s -> %d forecast mutation", __FUNCTION__, __LINE__);
   if(SkillProductionSignatureReady && !SkillVerifyProductionACSRMFingerprints())
      PrintFormat("%s -> %d ACSRM production signature mutation", __FUNCTION__, __LINE__);
   SkillForecast = NULL;
  }
//+--------------------------------------------------------------------------------------------------------------------------------+
//| LIVE bar-driven evaluation on the tester account stream.  Each  new bar triggers one decision + one next-bar live evaluation.  |
//+--------------------------------------------------------------------------------------------------------------------------------+
void OnTick(void)
  {
   if(!IsNewBar())
      return;
   if(!LiveEvaluateBar())
     {
      PrintFormat("ACSRM_TEST_FAIL reason=live_transition_failed bars=%d", TestBars);
      ExpertRemove();
     }
  }
//+---+