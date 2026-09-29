//+------------------------------------------------------------------+
//|                                                   Trajectory.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Reuse CogDriver indicators, data preparation and the canonical   |
//+------------------------------------------------------------------+
#include "..\CogDriver\Trajectory.mqh"
#ifndef ORION_FORECAST_STUDY
#define ORIONRecoverySmoke       false
#define ORIONRecoverySmokeBatches 2000
#define ORIONRecoverySmokeAge     512
#endif
#define ORION_FORMAT_VERSION     7
#define ORION_MARKET_FILE        "ORIONMkt.nnw"
#define ORION_TARGET_FILE        "ORIONTrg.nnw"
#define ORION_MANIFEST_FILE      "ORIONForecast.manifest"
#define ORION_ACTOR_FILE         "ORIONActor.nnw"
#define ORION_Q1_FILE            "ORIONQ1.nnw"
#define ORION_Q2_FILE            "ORIONQ2.nnw"
#define ORION_AC_MANIFEST_FILE   "ORIONActorCritic.manifest"
#define ORION_AC_FORMAT_VERSION  12
//+------------------------------------------------------------------+
//| ORION builds transitions directly from historical windows        |
//+------------------------------------------------------------------+
//| Base CNet provides Layer(int) and                                |
//| FeedForwardLayer(CNeuronBaseOCL*).                               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| A transient, non-owning layer view lets the library transpose    |
//+------------------------------------------------------------------+
class CORIONBufferView : public CNeuronBaseOCL
  {
public:
//--- Implements Bind.
   bool              Bind(CBufferFloat *source)
     {
      if(!source || source.GetIndex() < 0)
         ReturnFalse;
      if(Output != source)
         DeleteObj(Output);
      Output = source;
      return true;
     }
//--- Implements Unbind.
   void              Unbind(void)
     {
      Output = NULL;
     }
  };
//+------------------------------------------------------------------+
//| Device-only composition adapter, all operations reuse existing   |
//+------------------------------------------------------------------+
class CORIONDeviceOps : public CNeuronBaseOCL
  {
public:
//--- Implements Bind.
   bool              Bind(COpenCLMy *open_cl)
     {
      OpenCL = open_cl;
      return (CheckPointer(OpenCL) != POINTER_INVALID);
     }
//--- Creates and manages object lifecycle for Copy.
   bool              Copy(CBufferFloat *source, CBufferFloat *destination, const uint total)
     {
      if(!source || !destination || source.Total() != int(total) ||
         destination.Total() != int(total) || source.GetIndex() < 0 ||
         destination.GetIndex() < 0 || source.GetOpenCL() != OpenCL ||
         destination.GetOpenCL() != OpenCL)
         ReturnFalse;
      return CopyBufferRaw(source, destination, total);
     }
   bool              Join2(CBufferFloat *first, const uint first_total, CBufferFloat *second,
                           const uint second_total, CBufferFloat *destination)
     {
      if(!first || !second || !destination || first.Total() != int(first_total) ||
         second.Total() != int(second_total) ||
         destination.Total() != int(first_total + second_total) || first.GetIndex() < 0 ||
         second.GetIndex() < 0 || destination.GetIndex() < 0)
         ReturnFalse;
      return Concat(first, second, destination, first_total, second_total, 1);
     }
   bool              Join4(CBufferFloat *first, CBufferFloat *second, CBufferFloat *third,
                           CBufferFloat *fourth, CBufferFloat *destination, const uint block)
     {
      if(!first || !second || !third || !fourth || !destination ||
         first.Total() != int(block) || second.Total() != int(block) ||
         third.Total() != int(block) || fourth.Total() != int(block) ||
         destination.Total() != int(4 * block) || first.GetIndex() < 0 || second.GetIndex() < 0 ||
         third.GetIndex() < 0 || fourth.GetIndex() < 0 || destination.GetIndex() < 0)
         ReturnFalse;
      return Concat(first, second, third, fourth, destination, block, block, block, block, 1);
     }
   bool              Subtract(CBufferFloat *first, CBufferFloat *second, CBufferFloat *destination,
                              const uint dimension)
     {
      return Different(first, second, destination, dimension);
     }
   bool              BroadcastSum(CBufferFloat *vector_in, CBufferFloat *matrix_in,
                                  CBufferFloat *destination, const uint dimension, const uint variables)
     {
      return SumVecMatrix(vector_in, matrix_in, destination, dimension, variables);
     }
   bool              Add(CBufferFloat *first, CBufferFloat *second, CBufferFloat *destination,
                         const uint dimension)
     {
      return SumAndNormalize(first, second, destination, dimension, false, 0, 0, 0, 1.0f);
     }
   bool              Split2(CBufferFloat *first, CBufferFloat *second, CBufferFloat *source,
                            const uint first_total, const uint second_total)
     {
      return DeConcat(first, second, source, first_total, second_total, 1);
     }
  };
CNet            ORIONMarket;
CNet            ORIONTarget;
CNeuronScenarioForecast *ORIONForecast = NULL;
//---
CBufferFloat         ORIONState;
CBufferFloat         ORIONTime;
CBufferFloat         ORIONFuture;
CBufferFloat         ORIONLatentTarget;
CBufferFloat         ORIONLatentDelta;
CBufferFloat         ORIONProbeState;
CBufferFloat         ORIONProbeTime;
CORIONBufferView     ORIONFutureView;
CNeuronTransposeRCDOCL ORIONFutureTranspose;
CBufferFloat         ORIONLatentZero;
CBufferFloat         ORIONLatentNegativeMarket;
//---
ulong                ORIONInvalidBatches = 0;
ulong                ORIONBatches = 0;
ulong                ORIONResponsibilityMicroseconds = 0;
uint                 ORIONCompletedEpochs = 0;
ulong                ORIONLastSignature = 0;
bool                 ORIONReady = false;
CBufferFloat         ORIONFrozenGeneratorWeights;
CBufferFloat         ORIONFrozenRouterWeights;
CBufferFloat         ORIONFrozenConfidenceWeights;
CBufferFloat         ORIONFrozenPrototypes;
CBufferFloat         ORIONFrozenEMASums;
CBufferFloat         ORIONFrozenEMACounts;
CBufferFloat         ORIONFrozenUsage;
CBufferFloat         ORIONFrozenInactive;
CBufferFloat         ORIONFrozenInactivityAge;
bool                 ORIONFrozenBaselineReady = false;
CORIONDeviceOps      ORIONDevice;
//+------------------------------------------------------------------+
//| Implements ForecastRecoveryAge.                                  |
//+------------------------------------------------------------------+
uint ForecastRecoveryAge(void)
  {
   if(ORIONRecoverySmoke)
      return MathMax(1, int(ORIONRecoverySmokeAge));
   const int seconds = PeriodSeconds(TimeFrame);
   if(seconds <= 0)
      return 6240;
//--- 52 five-day trading weeks: one calendar-equivalent year on H1.
   return (uint)MathMax(1.0, MathRound(6240.0 * PeriodSeconds(PERIOD_H1) / seconds));
  }
//+------------------------------------------------------------------+
//| Creates and manages object lifecycle for ConfigureForecastRec... |
//+------------------------------------------------------------------+
bool ConfigureForecastRecoveryAge(void)
  {
   if(!ORIONForecast || !ORIONForecast.SetRecoveryAge(ForecastRecoveryAge()))
      ReturnFalse;
   PrintFormat("ORION %s recovery_age=%u bars", (ORIONRecoverySmoke ? "smoke" : "forecast"),
               ORIONForecast.RecoveryAge());
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONInitTrainingBuffers.                             |
//+------------------------------------------------------------------+
bool ORIONInitTrainingBuffers(void)
  {
   COpenCLMy *open_cl = ORIONMarket.GetOpenCL();
   if(!open_cl || !ORIONDevice.Bind(open_cl))
      ReturnFalse;
   if(!ORIONFutureTranspose.Init(0, 0, open_cl, NForecast, BarDescr, 1, ADAM, BatchSize) ||
      !ORIONFuture.BufferInit((NForecast * BarDescr), 0) ||
      !ORIONFuture.BufferCreate(open_cl) ||
      !ORIONLatentZero.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !ORIONLatentZero.BufferCreate(open_cl) ||
      !ORIONLatentNegativeMarket.BufferInit((BarDescr * EmbeddingSize), 0) ||
      !ORIONLatentNegativeMarket.BufferCreate(open_cl) ||
      !ORIONLatentTarget.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !ORIONLatentTarget.BufferCreate(open_cl) ||
      !ORIONLatentDelta.BufferInit((BarDescr * NForecast * EmbeddingSize), 0) ||
      !ORIONLatentDelta.BufferCreate(open_cl))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONAddBase.                                         |
//+------------------------------------------------------------------+
bool ORIONAddBase(CArrayObj *description, const uint count)
  {
   if(!description)
      ReturnFalse;
   CLayerDescription *descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronBaseOCL;
   descr.count = count;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return true;
   DeleteObjAndFalse(descr);
  }
//+------------------------------------------------------------------+
//| Implements ORIONNormalizeDescriptionBatch.                       |
//+------------------------------------------------------------------+
bool ORIONNormalizeDescriptionBatch(CArrayObj *description)
  {
   if(!description)
      ReturnFalse;
   for(int i = 0; i < description.Total(); i++)
     {
      CLayerDescription *layer = (CLayerDescription*)description.At(i);
      if(!layer)
         ReturnFalse;
      layer.batch = BatchSize;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONCreateDescriptions.                              |
//+------------------------------------------------------------------+
bool ORIONCreateDescriptions(CArrayObj *&market, CArrayObj *&target)
  {
   CArrayObj *legacy_decoder = NULL;
//--- Function if.
   if(!CreateStateDescriptions(market, legacy_decoder))
     {
      DeleteObj(legacy_decoder);
      ReturnFalse;
     }
   DeleteObj(legacy_decoder);
//--- Keep exactly CogDriver layers 0..4 (RankTCM is the boundary).
   int layer = market.Total() - 1;
//--- Function while.
   while(layer >= 0)
     {
      CLayerDescription* descr = market.At(layer);
//--- Function if.
      if(!descr || descr.type != defNeuronCogDriverRankTCM)
        {
         if(!market.Delete(layer))
            ReturnFalse;
         layer--;
        }
      else
         break;
     }
   if(market.Total() <= 0)
      ReturnFalse;
   if(!ORIONNormalizeDescriptionBatch(market))
      ReturnFalse;
//--- CreateBuffers already stores state feature-major as
//--- [BarDescr,HistoryBars], which is CogDriverData's required input layout.
   if(!market.Delete(layer) || !market.Delete(layer - 1))
      ReturnFalse;
   CLayerDescription *descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronCogDriverData;
   descr.window = HistoryBars;
   descr.count = BarDescr;
     {
      uint units[] = {StackSize, StackSize, Quantiles};
      if(ArrayCopy(descr.units, units, 0, 0, units.Size()) < int(units.Size()))
         DeleteObjAndFalse(descr);
     }
   descr.probability = 1.0f;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronCogDriverRankTCM;
   descr.window = HistoryBars * (2 * Quantiles + 1);
   descr.count = EmbeddingSize;
   descr.variables = BarDescr;
     {
      uint units[] = {StackSize, NHeads};
      if(ArrayCopy(descr.units, units, 0, 0, units.Size()) < int(units.Size()))
         DeleteObjAndFalse(descr);
     }
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronScenarioForecast;
   descr.count = NScenarios;
   descr.variables = TopK;
   descr.window_out = NForecast;
   descr.window = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!market.Add(descr))
      DeleteObjAndFalse(descr);
//---
   if(!target)
      target = new CArrayObj();
   else
      target.Clear();
   if(!target)
      ReturnFalse;
   target.FreeMode(true);
//--- Detached Target Encoder: one normalized future trajectory per BarDescr
//--- variable.  ConvOCL zero-fills the unavailable edge of each window, so
//--- window=3 preserves NForecast positions without an auxiliary pad buffer.
   if(!ORIONAddBase(target, (BarDescr * NForecast)))
      ReturnFalse;
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronPeriodNorm;
   descr.count = 1;
   descr.window = NForecast;
   descr.variables = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = 3;
   descr.step = 1;
   descr.window_out = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = GELU;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.step = EmbeddingSize;
   descr.window_out = 2 * EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = GELU;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronConvOCL;
   descr.count = NForecast;
   descr.window = 2 * EmbeddingSize;
   descr.step = 2 * EmbeddingSize;
   descr.window_out = EmbeddingSize;
   descr.layers = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   descr = new CLayerDescription();
   if(!descr)
      ReturnFalse;
   descr.type = defNeuronPeriodNorm;
   descr.count = NForecast;
   descr.window = EmbeddingSize;
   descr.variables = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(!target.Add(descr))
      DeleteObjAndFalse(descr);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONCreateNetworks.                                  |
//+------------------------------------------------------------------+
bool ORIONCreateNetworks(void)
  {
   CArrayObj *market = NULL, *target = NULL;
//--- Function if.
   if(!ORIONCreateDescriptions(market, target))
     {
      DeleteObj(market);
      DeleteObj(target);
      ReturnFalse;
     }
   const bool market_created = ORIONMarket.Create(market);
   PrintFormat("ORION Create market=%s batch=%u", (market_created ? "OK" : "FAIL"), uint(BatchSize));
   const bool target_created = (market_created && ORIONTarget.Create(target));
   PrintFormat("ORION Create target=%s", (target_created ? "OK" : "FAIL"));
   const bool created = (market_created && target_created);
   DeleteObj(market);
   DeleteObj(target);
   if(!created)
      ReturnFalse;
//--- CNet::Create initializes its OpenCL program.  Build each network first,
//--- then move its device buffers to the Market context; otherwise a later
//--- Create invalidates buffers belonging to an earlier network.
   ORIONTarget.SetOpenCL(ORIONMarket.GetOpenCL());
//--- Function if.
   if(!ORIONInitTrainingBuffers())
     {
      Print("ORION init: training buffers=FAIL");
      ReturnFalse;
     }
   ORIONForecast = (CNeuronScenarioForecast*)ORIONMarket.Layer(-1);
//--- Function if.
   if(!ORIONForecast || ORIONForecast.Type() != defNeuronScenarioForecast)
     {
      Print("ORION init: forecast layer=FAIL");
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
   ORIONTarget.TrainMode(false);
   ORIONMarket.TrainMode(true);
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONValidateShapes.                                  |
//+------------------------------------------------------------------+
bool ORIONValidateShapes(const bool training = true)
  {
   CNeuronBaseOCL *layer = ORIONMarket.Layer(0);
   CBufferFloat *buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != HistoryBars * BarDescr)
     {
      PrintFormat("ORION shape: market input expected=%d actual=%d", HistoryBars * BarDescr,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   layer = ORIONMarket.Layer(4);
   buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != (BarDescr * EmbeddingSize))
     {
      PrintFormat("ORION shape: market RankTCM expected=%d actual=%d", BarDescr * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
//--- Function if.
   if(training)
     {
      layer = ORIONTarget.Layer(0);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("ORION shape: target input expected=%d actual=%d", BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = ORIONTarget.Layer(1);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("ORION shape: target PeriodNorm expected=%d actual=%d", BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = ORIONTarget.Layer(2);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("ORION shape: target Conv3 expected=%d actual=%d", BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = ORIONTarget.Layer(3);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * 2 * EmbeddingSize))
        {
         PrintFormat("ORION shape: target Conv2 expected=%d actual=%d", BarDescr * NForecast * 2 * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = ORIONTarget.Layer(4);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("ORION shape: target final Conv expected=%d actual=%d", BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      layer = ORIONTarget.Layer(-1);
      buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast * EmbeddingSize))
        {
         PrintFormat("ORION shape: target output PeriodNorm expected=%d actual=%d", BarDescr * NForecast * EmbeddingSize,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
      buffer = ORIONFutureTranspose.getOutput();
//--- Function if.
      if(!buffer || buffer.Total() != (BarDescr * NForecast))
        {
         PrintFormat("ORION shape: future transpose expected=%d actual=%d", BarDescr * NForecast,
                     (buffer ? buffer.Total() : -1));
         ReturnFalse;
        }
     }
   layer = ORIONMarket.Layer(-1);
   buffer = (layer ? layer.getOutput() : NULL);
//--- Function if.
   if(!buffer || buffer.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize)
     {
      PrintFormat("ORION shape: Scenario Z expected=%d actual=%d", NScenarios * BarDescr * NForecast * EmbeddingSize,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = ORIONForecast.GetU();
//--- Function if.
   if(!buffer || buffer.Total() != (NScenarios * BarDescr * NForecast))
     {
      PrintFormat("ORION shape: Scenario U expected=%d actual=%d", NScenarios * BarDescr * NForecast,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   buffer = ORIONForecast.GetPi();
//--- Function if.
   if(!buffer || buffer.Total() != NScenarios)
     {
      PrintFormat("ORION shape: Scenario Pi expected=%d actual=%d", NScenarios,
                  (buffer ? buffer.Total() : -1));
      ReturnFalse;
     }
   if(ORIONForecast.Variables() != BarDescr || ORIONForecast.Scenarios() != NScenarios ||
      ORIONForecast.Horizon() != NForecast || ORIONForecast.Dimension() != EmbeddingSize ||
//--- Function ActiveTrajectories.
      ORIONForecast.ActiveTrajectories() > TopK)
     {
      PrintFormat("ORION shape: Forecast V=%d/%d K=%d/%d H=%d/%d D=%d/%d active=%d/%d",
                  ORIONForecast.Variables(), BarDescr, ORIONForecast.Scenarios(), NScenarios,
                  ORIONForecast.Horizon(), NForecast, ORIONForecast.Dimension(), EmbeddingSize,
                  ORIONForecast.ActiveTrajectories(), TopK);
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONHashUInt.                                        |
//+------------------------------------------------------------------+
ulong ORIONHashUInt(ulong hash, const ulong value)
  {
   hash ^= value;
   return hash * ulong(1099511628211);
  }
//+------------------------------------------------------------------+
//| Implements ORIONHashText.                                        |
//+------------------------------------------------------------------+
ulong ORIONHashText(ulong hash, const string text)
  {
   for(int i = 0; i < StringLen(text); i++)
      hash = ORIONHashUInt(hash, (ulong)StringGetCharacter(text, i));
   return hash;
  }
//+------------------------------------------------------------------+
//| Implements ORIONHashFile.                                        |
//+------------------------------------------------------------------+
ulong ORIONHashFile(ulong hash, const string file_name)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return 0;
   const ulong length = FileSize(handle);
   if(length == 0 || length > ulong(INT_MAX))
     { FileClose(handle); return 0; }
   uchar bytes[];
   if(ArrayResize(bytes, (int)length) != (int)length ||
       //--- Read file payload as bytes after successful allocation.
      FileReadArray(handle, bytes, 0, (int)length) != (int)length)
     { FileClose(handle); return 0; }
   FileClose(handle);
   for(int i = 0; i < (int)length; i++)
      hash = ORIONHashUInt(hash, (ulong)bytes[i]);
   return hash;
  }
//+------------------------------------------------------------------+
//| Exact signature of the unified Market/Scenario inference grap... |
//+------------------------------------------------------------------+
ulong ORIONForecastSignature(void)
  {
   if(!ORIONForecast)
      return 0;
   ulong hash = ulong(1469598103934665603);
   hash = ORIONHashFile(hash, ORION_MARKET_FILE);
   if(hash == 0)
      return 0;
   hash = ORIONHashUInt(hash, ORION_FORMAT_VERSION);
   hash = ORIONHashUInt(hash, BarDescr);
   hash = ORIONHashUInt(hash, NScenarios);
   hash = ORIONHashUInt(hash, TopK);
   hash = ORIONHashUInt(hash, NForecast);
   hash = ORIONHashUInt(hash, EmbeddingSize);
   hash = ORIONHashUInt(hash, ORIONForecast.ContractSignature());
   hash = ORIONHashText(hash, "z_layout=K,V,H,D;u_layout=K,V,H;pi_layout=K;codebook_layout=K,V,H,D");
   hash = ORIONHashText(hash, "market_layout=RankTCM_then_ScenarioForecast;variable_order=BarDescr_feature_series_0_to_8");
   hash = ORIONHashText(hash, "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   return hash;
  }
//+------------------------------------------------------------------+
//| Implements ORIONForecastTrainingSignature.                       |
//+------------------------------------------------------------------+
ulong ORIONForecastTrainingSignature(const ulong forecast_signature)
  {
   if(forecast_signature == 0)
      return 0;
   return ORIONHashFile(forecast_signature, ORION_TARGET_FILE);
  }
//+------------------------------------------------------------------+
//| Implements ORIONWriteManifest.                                   |
//+------------------------------------------------------------------+
bool ORIONWriteManifest(const uint completed_epochs)
  {
   if(!ORIONForecast)
      ReturnFalse;
   const ulong signature = ORIONForecastSignature();
   const ulong target_hash = ORIONHashFile(ulong(1469598103934665603), ORION_TARGET_FILE);
   const ulong training_signature = ORIONForecastTrainingSignature(signature);
   if(signature == 0 || target_hash == 0 || training_signature == 0)
      ReturnFalse;
   int handle = FileOpen(ORION_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=ORION_FORECAST");
   FileWrite(handle, StringFormat("version=%u", ORION_FORMAT_VERSION));
   FileWrite(handle, StringFormat("forecast_type=%d", defNeuronScenarioForecast));
   FileWrite(handle, StringFormat("variables=%u", BarDescr));
   FileWrite(handle, StringFormat("scenarios=%u", NScenarios));
   FileWrite(handle, StringFormat("top_k=%u", TopK));
   FileWrite(handle, StringFormat("horizon=%u", NForecast));
   FileWrite(handle, StringFormat("latent=%u", EmbeddingSize));
   FileWrite(handle, "z_layout=K,V,H,D");
   FileWrite(handle, "u_layout=K,V,H");
   FileWrite(handle, "pi_layout=K");
   FileWrite(handle, "codebook_layout=K,V,H,D");
   FileWrite(handle, "variable_order=BarDescr_feature_series_0_to_8");
   FileWrite(handle, StringFormat("contract_signature=%I64u", ORIONForecast.ContractSignature()));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", signature));
   FileWrite(handle, StringFormat("target_hash=%I64u", target_hash));
   FileWrite(handle, StringFormat("training_signature=%I64u", training_signature));
   FileWrite(handle, StringFormat("completed_epochs=%u", completed_epochs));
   FileWrite(handle, StringFormat("training_batches=%I64u", ORIONBatches));
   FileWrite(handle, StringFormat("invalid_batches=%I64u", ORIONInvalidBatches));
   FileWrite(handle, "normalization=OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
   FileClose(handle);
   ORIONLastSignature = signature;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONSaveCheckpoint.                                  |
//+------------------------------------------------------------------+
bool ORIONSaveCheckpoint(const uint completed_epochs)
  {
   const datetime now = TimeCurrent();
   if(!ORIONMarket.Save(ORION_MARKET_FILE, 0, 0, 0, now, true) ||
      !ORIONTarget.Save(ORION_TARGET_FILE, 0, 0, 0, now, true))
      ReturnFalse;
   return ORIONWriteManifest(completed_epochs);
  }
//+------------------------------------------------------------------+
//| Implements ORIONInitIndicators.                                  |
//+------------------------------------------------------------------+
bool ORIONInitIndicators(void)
  {
   return (Symb.Name(_Symbol) && Symb.Refresh() &&
           RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice) &&
           CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice) &&
           ATR.Create(Symb.Name(), TimeFrame, ATRPeriod) &&
           MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice));
  }
#ifdef ORION_FORECAST_STUDY
bool ORIONLoadForecastTraining(void);
//+------------------------------------------------------------------+
//| Creates or initializes ORIONForecastStudy.                       |
//+------------------------------------------------------------------+
bool CreateORIONForecastStudy(void)
  {
   ResetLastError();
   ORIONCompletedEpochs = 0;
   ORIONBatches = 0;
   ORIONInvalidBatches = 0;
//--- Always prefer an existing compatible checkpoint for continued training.
//--- A missing, incomplete or incompatible checkpoint falls back to a clean,
//--- independently randomized Forecast graph.
   bool resumed = ORIONLoadForecastTraining();
//--- Function if.
   if(!resumed)
     {
      const int load_error = GetLastError();
      ORIONForecast = NULL;
      PrintFormat("ORION init: forecast restore=FAIL error=%d; creating new random model", load_error);
      ResetLastError();
//--- Function if.
      if(!ORIONCreateNetworks())
        {
         PrintFormat("ORION init: forecast create=FAIL error=%d", GetLastError());
         ReturnFalse;
        }
     }
//--- Function if.
   if(!ORIONValidateShapes())
     {
      Print("ORION init: shapes=FAIL");
      ReturnFalse;
     }
//--- Function if.
   if(!ORIONInitIndicators())
     {
      PrintFormat("ORION init: indicators=FAIL error=%d", GetLastError());
      ReturnFalse;
     }
   PrintFormat("ORION init: forecast=%s completed_epochs=%u batches=%I64u invalid=%I64u",
               (resumed ? "RESUMED" : "NEW"), ORIONCompletedEpochs, ORIONBatches, ORIONInvalidBatches);
   ORIONReady = true;
//--- Function if.
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("ORION init: chart event=FAIL error=%d", GetLastError());
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ReleaseORIONForecastStudy.                            |
//+------------------------------------------------------------------+
void ReleaseORIONForecastStudy(const int reason)
  {
//--- Complete epochs save transactionally in TrainORIONForecast(). Never
//--- overwrite a valid checkpoint with a fresh, partial or failed run here.
   ORIONForecast = NULL;
   ORIONReady = false;
  }
//+------------------------------------------------------------------+
//| Implements ORIONPrepareLatentTarget.                             |
//+------------------------------------------------------------------+
bool ORIONPrepareLatentTarget(CBufferFloat *market_latent)
  {
   if(!market_latent || market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0 ||
      ORIONFuture.Total() != (NForecast * BarDescr) || ORIONFuture.GetIndex() < 0 ||
      ORIONLatentTarget.Total() != (BarDescr * NForecast * EmbeddingSize) || ORIONLatentTarget.GetIndex() < 0 ||
      ORIONLatentDelta.Total() != (BarDescr * NForecast * EmbeddingSize) || ORIONLatentDelta.GetIndex() < 0)
      ReturnFalse;
//--- ORIONFuture is decoder-native [H,B].  The existing transpose changes
//--- only this device layout to Target-native [B,H]; Market keeps its original
//--- [BarDescr,HistoryBars] representation and is never transposed.
   if(!ORIONFutureView.Bind(GetPointer(ORIONFuture)))
      ReturnFalse;
   const bool transposed = ORIONFutureTranspose.FeedForward(ORIONFutureView.AsObject());
   ORIONFutureView.Unbind();
   if(!transposed)
      ReturnFalse;
   CNeuronBaseOCL *future_latent = ORIONTarget.Layer(-1);
   if(!ORIONTarget.FeedForwardLayer(ORIONFutureTranspose.AsObject()))
      ReturnFalse;
   if(!future_latent || future_latent.getOutput().Total() != (BarDescr * NForecast * EmbeddingSize) ||
      future_latent.getOutputIndex() < 0)
      ReturnFalse;
//--- Forecast loss compares full future Z against full generator Z. Codebook
//--- EMA alone receives the detached delta future-current.
   if(!ORIONDevice.Copy(future_latent.getOutput(), GetPointer(ORIONLatentTarget),
                        BarDescr * NForecast * EmbeddingSize) ||
      !ORIONDevice.Subtract(GetPointer(ORIONLatentZero), market_latent,
                            GetPointer(ORIONLatentNegativeMarket), EmbeddingSize))
      ReturnFalse;
   if(!ORIONDevice.BroadcastSum(GetPointer(ORIONLatentNegativeMarket),
                                future_latent.getOutput(), GetPointer(ORIONLatentDelta),
                                EmbeddingSize, BarDescr))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONProbeNet.                                        |
//+------------------------------------------------------------------+
bool ORIONProbeNet(CNet &net, CBufferFloat *probe_state, double &latent[], const bool restore_training)
  {
   if(!probe_state || ArrayResize(latent, (BarDescr * EmbeddingSize)) != (BarDescr * EmbeddingSize))
      ReturnFalse;
//--- Measure both epoch boundaries in inference mode from an empty recurrent
//--- state.  Clear again afterwards so the diagnostic forward cannot seed the
//--- first training batch or inherit the final training batch state.
   if(!net.TrainMode(false))
      ReturnFalse;
   bool result = net.Clear();
   if(result)
      result = net.feedForward(probe_state, 1, false, (CBufferFloat*)NULL);
//--- Get the live OpenCL output owned by RankTCM.  GetLayerOutput() copies
//--- values into a new host CBufferFloat and therefore cannot be BufferRead().
   CNeuronBaseOCL *latent_layer = net.Layer(4);
   CBufferFloat *output = (latent_layer ? latent_layer.getOutput() : NULL);
   if(result)
      result = (output != NULL && output.GetIndex() >= 0 &&
                output.BufferRead() && output.Total() == (BarDescr * EmbeddingSize));
   for(uint d = 0; result && d < (BarDescr * EmbeddingSize); d++)
     {
      latent[d] = double(output[d]);
      if(!MathIsValidNumber(latent[d]))
         result = false;
     }
   const bool cleared = net.Clear();
   const bool restored = net.TrainMode(restore_training);
   return (result && cleared && restored);
  }
//+------------------------------------------------------------------+
//| Implements ORIONProbeDrift.                                      |
//+------------------------------------------------------------------+
double ORIONProbeDrift(const double &before[], const double &after[])
  {
   if(ArraySize(before) != (BarDescr * EmbeddingSize) || ArraySize(after) != (BarDescr * EmbeddingSize))
      return DBL_MAX;
   double square_sum = 0;
   for(uint d = 0; d < (BarDescr * EmbeddingSize); d++)
     {
      const double delta = after[d] - before[d];
      square_sum += delta * delta;
     }
   return MathSqrt(square_sum / double((BarDescr * EmbeddingSize)));
  }
//+------------------------------------------------------------------+
//| Implements FormatScenarioCounts.                                 |
//+------------------------------------------------------------------+
string FormatScenarioCounts(const uint &counts[])
  {
   string result = "[";
   for(int i = 0; i < ArraySize(counts); i++)
      result += (i > 0 ? "," : "") + IntegerToString((int)counts[i]);
  return result + "]";
  }
//+------------------------------------------------------------------+
//| Smoke-only responsibility audit; all tensor reads are host-side  |
//+------------------------------------------------------------------+
bool ORIONCollectRecoverySmoke(ulong &hits[], double &responsibility_sum[], double &scale_sum[],
                               double &absolute_sum[], double &absolute_over_scale_sum[])
  {
   if(!ORIONForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios)
      ReturnFalse;
   CBufferFloat *responsibilities = ORIONForecast.GetResponsibilities();
   CBufferFloat *confidence = ORIONForecast.GetConfidence();
   CBufferFloat *target = ORIONForecast.GetTarget();
   CBufferFloat *forecast = ORIONForecast.getOutput();
   const int target_total = BarDescr * NForecast * EmbeddingSize;
   const int confidence_total = NScenarios * BarDescr * NForecast;
   if(!responsibilities || !confidence || !target || !forecast || responsibilities.Total() != NScenarios ||
      confidence.Total() != confidence_total || target.Total() != target_total ||
      forecast.Total() != NScenarios * target_total || !responsibilities.BufferRead() || !confidence.BufferRead() ||
      !target.BufferRead() || !forecast.BufferRead())
      ReturnFalse;
   for(uint scenario = 0; scenario < NScenarios; scenario++)
     {
      const double responsibility = responsibilities[scenario];
      if(!MathIsValidNumber(responsibility) || responsibility < 0.0)
         ReturnFalse;
      if(responsibility > 0.0)
         hits[scenario]++;
      responsibility_sum[scenario] += responsibility;
      if(responsibility <= 0.0)
         continue;
      for(int coordinate = 0; coordinate < target_total; coordinate++)
        {
         const int token = coordinate / EmbeddingSize;
         const double scale = MathMax(0.001, MathMin(10.0, confidence[int(scenario) * BarDescr * NForecast + token]));
         const double absolute_error = MathAbs(target[coordinate] - forecast[int(scenario) * target_total + coordinate]);
         if(!MathIsValidNumber(scale) || !MathIsValidNumber(absolute_error))
            ReturnFalse;
         scale_sum[scenario] += responsibility * scale;
         absolute_sum[scenario] += responsibility * absolute_error;
         absolute_over_scale_sum[scenario] += responsibility * absolute_error / scale;
        }
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONLogRecoverySmoke.                                |
//+------------------------------------------------------------------+
bool ORIONLogRecoverySmoke(const ulong &hits[], const double &responsibility_sum[],
                           const double &scale_sum[], const double &absolute_sum[],
                           const double &absolute_over_scale_sum[],
                           const uint &recovered[], const uint &age_zero[])
  {
   if(!ORIONForecast || ArraySize(hits) != NScenarios || ArraySize(responsibility_sum) != NScenarios ||
      ArraySize(scale_sum) != NScenarios || ArraySize(absolute_sum) != NScenarios ||
      ArraySize(absolute_over_scale_sum) != NScenarios ||
      ArraySize(recovered) != NScenarios || ArraySize(age_zero) != NScenarios)
      ReturnFalse;
   CScenarioCodebook *codebook = ORIONForecast.GetCodebook();
   if(!codebook)
      ReturnFalse;
   CBufferFloat *ages = codebook.GetInactivityAge();
   CBufferFloat *counts = codebook.GetEMACounts();
   CBufferFloat *usage = codebook.GetUsage();
   CBufferFloat *inactive = codebook.GetInactive();
   if(!ages || !counts || !usage || !inactive || ages.Total() != NScenarios || counts.Total() != NScenarios ||
      usage.Total() != NScenarios || inactive.Total() != NScenarios || !ages.BufferRead() || !counts.BufferRead() ||
      !usage.BufferRead() || !inactive.BufferRead())
      ReturnFalse;
   for(uint scenario = 0; scenario < NScenarios; scenario++)
     {
      if(!MathIsValidNumber(ages[scenario]) || !MathIsValidNumber(counts[scenario]) ||
         !MathIsValidNumber(usage[scenario]) || !MathIsValidNumber(inactive[scenario]))
         ReturnFalse;
      const double denominator = responsibility_sum[scenario] * double(BarDescr * NForecast * EmbeddingSize);
      const double mean_scale = (denominator > 0.0 ? scale_sum[scenario] / denominator : 0.0);
      const double mean_absolute = (denominator > 0.0 ? absolute_sum[scenario] / denominator : 0.0);
      const double mean_absolute_over_scale = (denominator > 0.0 ? absolute_over_scale_sum[scenario] / denominator : 0.0);
      PrintFormat("ORION smoke state=%d hits=%I64u age_zero=%u responsibility_sum=%.8f mean_U=%.8f mean_abs_error=%.8f mean_abs_over_U=%.8f final_age=%.0f ema_count=%.8f usage=%.8f inactive=%d recovered=%u",
                  scenario, hits[scenario], age_zero[scenario], responsibility_sum[scenario], mean_scale, mean_absolute,
                  mean_absolute_over_scale, ages[scenario], counts[scenario], usage[scenario],
                  int(MathRound(inactive[scenario])), recovered[scenario]);
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONTrainBatch.                                      |
//+------------------------------------------------------------------+
bool ORIONTrainBatch(const int position)
  {
//--- Build exactly the same X<=t state as CreateBuffers(position,...,future),
//--- but do not touch any future sample until Market and Scenario are complete.
   if(!CreateBuffers(position + NForecast, GetPointer(ORIONState), GetPointer(ORIONTime), NULL))
      ReturnFalse;
   if(!ORIONMarket.feedForward(GetPointer(ORIONState), 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
//--- -1 is ScenarioForecast; the detached target is measured from the
//--- preceding RankTCM latent z_t.
   CNeuronBaseOCL *market_layer = ORIONMarket.Layer(-2);
   CBufferFloat *market_latent = (market_layer ? market_layer.getOutput() : NULL);
   if(!market_latent ||
      market_latent.Total() != (BarDescr * EmbeddingSize) || market_latent.GetIndex() < 0)
      ReturnFalse;
//--- Only now is any future window presented to the detached Target Encoder.
   if(!CreateBuffers(position, GetPointer(ORIONState), GetPointer(ORIONTime), GetPointer(ORIONFuture)))
      ReturnFalse;
   if(!ORIONPrepareLatentTarget(market_latent))
      ReturnFalse;
   const ulong responsibility_started = GetMicrosecondCount();
//--- Function if.
   if(!ORIONForecast.BuildResponsibilities(GetPointer(ORIONLatentTarget), GetPointer(ORIONLatentDelta)))
     {
      //--- Commit only a completed device validation.  Infrastructure failures leave
      //--- the transient control-valid flag false, so stale batch_control is ignored.
      if(ORIONForecast.LastBatchControlValid() && !ORIONForecast.CommitBatchDiagnostics())
         PrintFormat("%s -> %d invalid diagnostic commit failed", __FUNCTION__, __LINE__);
      ReturnFalse;
     }
   const ulong responsibility_elapsed = GetMicrosecondCount() - responsibility_started;
   if(!ORIONForecast.CommitBatchDiagnostics())
      ReturnFalse;
//--- The unified CNet updates ScenarioForecast and all preceding Market layers.
   if(!ORIONMarket.backPropGradient((CBufferFloat*)NULL, (CBufferFloat*)NULL, -1, true))
      ReturnFalse;
   ORIONResponsibilityMicroseconds += responsibility_elapsed;
   return true;
  }
//+------------------------------------------------------------------+
//| Current-epoch progress uses the existing 12-float device summ... |
//+------------------------------------------------------------------+
void ORIONShowForecastProgress(const double percent, const ulong failed_batches)
  {
   double lmix, router, trajectory, confidence, latent, observation;
   double valid, invalid, entropy, distance, inactive, recovered;
   if(ORIONForecast && ORIONForecast.ReadEpochDiagnostics(lmix, router, trajectory, confidence,
         latent, observation, valid, invalid, entropy, distance, inactive, recovered) && valid > 0.0)
     {
      Comment(StringFormat("ORION Forecast %6.2f%% L_mix %.8f latent %.8f invalid(current epoch) %I64u",
                           percent, lmix / valid, latent / valid, failed_batches));
      return;
     }
   Comment(StringFormat("ORION Forecast %6.2f%% L_mix n/a latent n/a invalid(current epoch) %I64u",
                        percent, failed_batches));
  }
//+------------------------------------------------------------------+
//| Implements TrainORIONForecast.                                   |
//+------------------------------------------------------------------+
void TrainORIONForecast(void)
  {
   if(!ORIONReady)
      return;
   int start = iBarShift(Symb.Name(), TimeFrame, Start);
   int end = iBarShift(Symb.Name(), TimeFrame, End);
   int bars = CopyRates(Symb.Name(), TimeFrame, 0, start, Rates);
   if(bars <= 0 || !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     { PrintFormat("%s -> %d", __FUNCTION__, __LINE__); return; }
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
     { PrintFormat("%s -> %d data unavailable", __FUNCTION__, __LINE__); return; }
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   if(!ArraySetAsSeries(Rates, true))
     { PrintFormat("%s -> %d data unavailable", __FUNCTION__, __LINE__); return; }
   bars -= end + HistoryBars + NForecast;
   if(bars < 0)
     { PrintFormat("%s -> %d insufficient history", __FUNCTION__, __LINE__); return; }
//--- With a forecast buffer CreateBuffers(position,...) starts Market state at
//--- position+H.  Reproduce that exact X<=t mapping with NULL, which also keeps
//--- every future target out of the diagnostic input.
   const int probe_position = MathMax(end - 1, 0);
   if(!CreateBuffers(probe_position + NForecast, GetPointer(ORIONProbeState),
                     //--- Gets Pointer.
                     GetPointer(ORIONProbeTime), NULL))
     { PrintFormat("%s -> %d fixed probe unavailable", __FUNCTION__, __LINE__); return; }
   uint ticks = GetTickCount();
   bool stop = false;
   const uint passes = (ORIONRecoverySmoke ? 1 : uint(MathMax(Epochs, 0)));
   const ulong smoke_limit = (ORIONRecoverySmoke ? ulong(MathMax(1, int(ORIONRecoverySmokeBatches))) : 0);
   for(uint pass = 0; pass < passes && !IsStopped() && !stop; pass++)
     {
      const uint epoch = ORIONCompletedEpochs;
      ulong epoch_failures = 0;
      ulong smoke_batches = 0;
      ulong smoke_hits[];
      double smoke_responsibility_sum[];
      double smoke_scale_sum[];
      double smoke_absolute_sum[];
      double smoke_absolute_over_scale_sum[];
      if(ORIONRecoverySmoke &&
         (ArrayResize(smoke_hits, NScenarios) != NScenarios || ArrayResize(smoke_responsibility_sum, NScenarios) != NScenarios ||
          ArrayResize(smoke_scale_sum, NScenarios) != NScenarios || ArrayResize(smoke_absolute_sum, NScenarios) != NScenarios ||
           //--- All temporary smoke buffers must resize to NScenarios.
          ArrayResize(smoke_absolute_over_scale_sum, NScenarios) != NScenarios))
        { PrintFormat("%s -> %d smoke counter allocation failed", __FUNCTION__, __LINE__); break; }
      ArrayInitialize(smoke_hits, 0);
      ArrayInitialize(smoke_responsibility_sum, 0.0);
      ArrayInitialize(smoke_scale_sum, 0.0);
      ArrayInitialize(smoke_absolute_sum, 0.0);
      ArrayInitialize(smoke_absolute_over_scale_sum, 0.0);
      if(!ORIONMarket.Clear() || !ORIONTarget.Clear())
        { PrintFormat("%s -> %d clear failed", __FUNCTION__, __LINE__); break; }
      ORIONTarget.TrainMode(false);
      double latent_before[], latent_after[];
      if(!ORIONProbeNet(ORIONMarket, GetPointer(ORIONProbeState), latent_before, true))
        { PrintFormat("%s -> %d fixed probe forward failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!ORIONForecast.ResetEpochDiagnostics())
        { PrintFormat("%s -> %d diagnostic reset failed", __FUNCTION__, __LINE__); stop = true; break; }
      ORIONResponsibilityMicroseconds = 0;
      for(int position = start - HistoryBars - NForecast - 1; position >= end && !IsStopped() && !stop; position--)
        {
//--- Function if.
         if(!ORIONTrainBatch(position))
           {
            epoch_failures++;
            PrintFormat("ORION invalid batch epoch=%d position=%d line=%d", epoch, position, __LINE__);
//--- Function if.
            if(GetTickCount() - ticks > 500)
              {
               const double percent = (double(pass) + 1.0 - double(position - end) / MathMax(start - end - HistoryBars - NForecast, 1)) * 100.0 / Epochs;
               ORIONShowForecastProgress(percent, epoch_failures);
               ticks = GetTickCount();
              }
            continue;
           }
//--- Function if.
         if(ORIONRecoverySmoke)
           {
            if(!ORIONCollectRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
                                          smoke_absolute_sum, smoke_absolute_over_scale_sum))
              { PrintFormat("%s -> %d smoke responsibility read failed", __FUNCTION__, __LINE__); stop = true; break; }
            smoke_batches++;
            if(smoke_batches >= smoke_limit)
               break;
           }
//--- Function if.
         if(GetTickCount() - ticks > 500)
           {
            const double percent = (double(pass) + 1.0 - double(position - end) / MathMax(start - end - HistoryBars - NForecast, 1)) * 100.0 / Epochs;
            ORIONShowForecastProgress(percent, epoch_failures);
            ticks = GetTickCount();
           }
        }
      double lmix, lrouter, ltrajectory, lconfidence, latent, observation;
      double valid_value, invalid_value, entropy, distance, inactive_value, recovered_value;
      if(!ORIONForecast.BuildEpochCodebookDiagnostics() ||
         !ORIONForecast.ReadEpochDiagnostics(lmix, lrouter, ltrajectory, lconfidence, latent, observation,
                                             valid_value, invalid_value, entropy, distance, inactive_value, recovered_value))
        { PrintFormat("%s -> %d diagnostic read failed", __FUNCTION__, __LINE__); stop = true; break; }
      const ulong valid = (ulong)MathRound(valid_value);
      const ulong invalid = (ulong)MathRound(invalid_value);
      const uint inactive = (uint)MathRound(inactive_value);
      const uint recovered = (uint)MathRound(recovered_value);
      uint recovered_by_scenario[];
      uint age_zero_by_scenario[];
      if(!ORIONForecast.ReadEpochRecoveryEvents(recovered_by_scenario))
        { PrintFormat("%s -> %d recovery-event read failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!ORIONForecast.ReadEpochAgeZeroEvents(age_zero_by_scenario))
        { PrintFormat("%s -> %d age-zero-event read failed", __FUNCTION__, __LINE__); stop = true; break; }
      ulong recovery_sum = 0;
      for(int i = 0; i < ArraySize(recovered_by_scenario); i++)
         recovery_sum += recovered_by_scenario[i];
      if(recovery_sum != (ulong)recovered)
        { PrintFormat("%s -> %d recovery-event mismatch total=%I64u diagnostics=%u", __FUNCTION__, __LINE__, recovery_sum, recovered); stop = true; break; }
      if(ORIONRecoverySmoke && valid != smoke_batches)
        { PrintFormat("%s -> %d smoke valid mismatch valid=%I64u collected=%I64u", __FUNCTION__, __LINE__, valid, smoke_batches); stop = true; break; }
      ORIONBatches += valid;
      ORIONInvalidBatches += invalid;
      if(valid == 0)
        { stop = true; break; }
      //--- Target is an independently initialized inference-only encoder.  It is
      //--- never copied from Market and is absent from every backward/update path.
      if(!ORIONProbeNet(ORIONMarket, GetPointer(ORIONProbeState), latent_after, true))
        { PrintFormat("%s -> %d Market epoch probe failed", __FUNCTION__, __LINE__); stop = true; break; }
      const double latent_drift = ORIONProbeDrift(latent_before, latent_after);
      PrintFormat("ORION epoch=%d batches=%I64u L_mix=%.8f L_router=%.8f L_trajectory=%.8f L_confidence=%.8f NLL_confidence=%.8f usage_entropy=%.8f pairwise_codebook=%.8f invalid=%I64u latent_drift=%.8f inactive=%u recovered=%u recovered_by_scenario=%s responsibility_ms=%.3f",
                  epoch + 1, valid, lmix / valid, lrouter / valid, ltrajectory / valid, latent / valid,
                  lconfidence / valid, entropy, distance, ORIONInvalidBatches,
                  latent_drift, inactive, recovered, FormatScenarioCounts(recovered_by_scenario),
                  double(ORIONResponsibilityMicroseconds) / (1000.0 * double(valid)));
      if(ORIONRecoverySmoke && !ORIONLogRecoverySmoke(smoke_hits, smoke_responsibility_sum, smoke_scale_sum,
                                                        smoke_absolute_sum, smoke_absolute_over_scale_sum,
                                                        recovered_by_scenario, age_zero_by_scenario))
        { PrintFormat("%s -> %d smoke state log failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(!ORIONRecoverySmoke && !ORIONSaveCheckpoint(epoch + 1))
        { PrintFormat("%s -> %d checkpoint failed", __FUNCTION__, __LINE__); stop = true; break; }
      if(ORIONRecoverySmoke)
         PrintFormat("ORION smoke complete valid=%I64u recovery_age=%u checkpoint=SKIPPED", valid,
                     ORIONForecast.RecoveryAge());
      ORIONCompletedEpochs = epoch + 1;
     }
   Comment("");
//--- Function if.
   if(!stop)
     {
      //--- Successful forecast stage is sealed for the later Actor-Critic stage.
      ORIONMarket.TrainMode(false);
      ORIONTarget.TrainMode(false);
      ORIONLastSignature = ORIONForecastSignature();
      PrintFormat("ORION forecast inference-only signature=%I64u batches=%I64u invalid=%I64u",
                  ORIONLastSignature, ORIONBatches, ORIONInvalidBatches);
     }
   ExpertRemove();
  }
#endif
//+------------------------------------------------------------------+
//| Actor-Critic inference and composition helpers Implements ORI... |
//+------------------------------------------------------------------+
string ORIONManifestValue(const string file_name, const string key)
  {
   int handle = FileOpen(file_name, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      return "";
   const string prefix = key + "=";
   string value = "";
//--- Function while.
   while(!FileIsEnding(handle))
     {
      const string line = FileReadString(handle);
//--- Function if.
      if(StringFind(line, prefix) == 0)
        {
         value = StringSubstr(line, StringLen(prefix));
         break;
        }
     }
   FileClose(handle);
   return value;
  }
//+------------------------------------------------------------------+
//| Implements ORIONValidateForecastManifestHeader. Static checkp... |
//+------------------------------------------------------------------+
bool ORIONValidateForecastManifestHeader(void)
  {
#define ORION_MANIFEST_HEADER_EQ(KEY,VALUE) if(ORIONManifestValue(ORION_MANIFEST_FILE,KEY)!=(VALUE)) ReturnFalse
   ORION_MANIFEST_HEADER_EQ("format", "ORION_FORECAST");
   ORION_MANIFEST_HEADER_EQ("version", IntegerToString(ORION_FORMAT_VERSION));
   ORION_MANIFEST_HEADER_EQ("forecast_type", IntegerToString(defNeuronScenarioForecast));
   ORION_MANIFEST_HEADER_EQ("variables", IntegerToString(BarDescr));
   ORION_MANIFEST_HEADER_EQ("scenarios", IntegerToString(NScenarios));
   ORION_MANIFEST_HEADER_EQ("top_k", IntegerToString(TopK));
   ORION_MANIFEST_HEADER_EQ("horizon", IntegerToString(NForecast));
   ORION_MANIFEST_HEADER_EQ("latent", IntegerToString(EmbeddingSize));
   ORION_MANIFEST_HEADER_EQ("z_layout", "K,V,H,D");
   ORION_MANIFEST_HEADER_EQ("u_layout", "K,V,H");
   ORION_MANIFEST_HEADER_EQ("pi_layout", "K");
   ORION_MANIFEST_HEADER_EQ("codebook_layout", "K,V,H,D");
   ORION_MANIFEST_HEADER_EQ("variable_order", "BarDescr_feature_series_0_to_8");
   ORION_MANIFEST_HEADER_EQ("normalization", "OHLC_deltas_from_open;tick_volume_div_1000;RSI_CCI_ATR_MACD_raw");
#undef ORION_MANIFEST_HEADER_EQ
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONValidateForecastManifest.                        |
//+------------------------------------------------------------------+
bool ORIONValidateForecastManifest(const ulong signature, const bool training = false)
  {
   if(signature == 0)
      ReturnFalse;
   if(!ORIONValidateForecastManifestHeader())
      ReturnFalse;
#define ORION_MANIFEST_EQ(KEY,VALUE) if(ORIONManifestValue(ORION_MANIFEST_FILE,KEY)!=(VALUE)) ReturnFalse
   ORION_MANIFEST_EQ("contract_signature", StringFormat("%I64u", ORIONForecast.ContractSignature()));
   ORION_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", signature));
//--- Function if.
   if(training)
     {
      const ulong target_hash = ORIONHashFile(ulong(1469598103934665603), ORION_TARGET_FILE);
      const ulong training_signature = ORIONForecastTrainingSignature(signature);
      ORION_MANIFEST_EQ("target_hash", StringFormat("%I64u", target_hash));
      ORION_MANIFEST_EQ("training_signature", StringFormat("%I64u", training_signature));
      if(target_hash == 0 || training_signature == 0 ||
         ORIONManifestValue(ORION_MANIFEST_FILE, "completed_epochs") == "" ||
         ORIONManifestValue(ORION_MANIFEST_FILE, "training_batches") == "" ||
         ORIONManifestValue(ORION_MANIFEST_FILE, "invalid_batches") == "")
         ReturnFalse;
     }
#undef ORION_MANIFEST_EQ
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONCaptureFrozenBuffer.                             |
//+------------------------------------------------------------------+
bool ORIONCaptureFrozenBuffer(CBufferFloat *source, CBufferFloat &baseline)
  {
   return (source && source.BufferRead() && source.Total() > 0 && baseline.AssignArray(source));
  }
//+------------------------------------------------------------------+
//| Implements ORIONFrozenBufferEqual.                               |
//+------------------------------------------------------------------+
bool ORIONFrozenBufferEqual(CBufferFloat *source, CBufferFloat &baseline)
  {
   if(!source || !source.BufferRead() || source.Total() != baseline.Total())
      ReturnFalse;
   for(int i = 0; i < source.Total(); i++)
      if(source[i] != baseline[i])
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONCaptureFrozenForecastBaseline.                   |
//+------------------------------------------------------------------+
bool ORIONCaptureFrozenForecastBaseline(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : ORIONForecast);
   ORIONFrozenBaselineReady = false;
   if(!current || !current.GetCodebook())
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   if(trainable <= 0 || trainable != int(current.TrainableWeights()) ||
      !ORIONCaptureFrozenBuffer(generator, ORIONFrozenGeneratorWeights) ||
      !ORIONCaptureFrozenBuffer(router, ORIONFrozenRouterWeights) ||
      !ORIONCaptureFrozenBuffer(confidence, ORIONFrozenConfidenceWeights) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetPrototypes(), ORIONFrozenPrototypes) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetEMASums(), ORIONFrozenEMASums) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetEMACounts(), ORIONFrozenEMACounts) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetUsage(), ORIONFrozenUsage) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetInactive(), ORIONFrozenInactive) ||
      !ORIONCaptureFrozenBuffer(current.GetCodebook().GetInactivityAge(), ORIONFrozenInactivityAge))
      ReturnFalse;
   ORIONFrozenBaselineReady = true;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONVerifyFrozenWeightsExact.                        |
//+------------------------------------------------------------------+
bool ORIONVerifyFrozenWeightsExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : ORIONForecast);
   if(!ORIONFrozenBaselineReady || !current)
      ReturnFalse;
   CBufferFloat *generator = current.GetGenerator().GetWeightsConv();
   CBufferFloat *router = current.GetRouter().GetWeightsConv();
   CBufferFloat *confidence = current.GetConfidenceHead().GetWeightsConv();
   const int trainable = (generator ? generator.Total() : 0) +
                         (router ? router.Total() : 0) +
                         (confidence ? confidence.Total() : 0);
   return (trainable > 0 && trainable == int(current.TrainableWeights()) &&
           ORIONFrozenBufferEqual(generator, ORIONFrozenGeneratorWeights) &&
           ORIONFrozenBufferEqual(router, ORIONFrozenRouterWeights) &&
           ORIONFrozenBufferEqual(confidence, ORIONFrozenConfidenceWeights));
  }
//+------------------------------------------------------------------+
//| Implements ORIONVerifyFrozenCodebookExact.                       |
//+------------------------------------------------------------------+
bool ORIONVerifyFrozenCodebookExact(CNeuronScenarioForecast *forecast = NULL)
  {
   CNeuronScenarioForecast *current = (forecast ? forecast : ORIONForecast);
   if(!ORIONFrozenBaselineReady || !current || !current.GetCodebook())
      ReturnFalse;
   return (ORIONFrozenBufferEqual(current.GetCodebook().GetPrototypes(), ORIONFrozenPrototypes) &&
           ORIONFrozenBufferEqual(current.GetCodebook().GetEMASums(), ORIONFrozenEMASums) &&
           ORIONFrozenBufferEqual(current.GetCodebook().GetEMACounts(), ORIONFrozenEMACounts) &&
           ORIONFrozenBufferEqual(current.GetCodebook().GetUsage(), ORIONFrozenUsage) &&
           ORIONFrozenBufferEqual(current.GetCodebook().GetInactive(), ORIONFrozenInactive) &&
           ORIONFrozenBufferEqual(current.GetCodebook().GetInactivityAge(), ORIONFrozenInactivityAge));
  }
//+------------------------------------------------------------------+
//| Implements ORIONVerifyFrozenForecastExact.                       |
//+------------------------------------------------------------------+
bool ORIONVerifyFrozenForecastExact(CNeuronScenarioForecast *forecast = NULL)
  {
   return (ORIONVerifyFrozenWeightsExact(forecast) && ORIONVerifyFrozenCodebookExact(forecast));
  }
//+------------------------------------------------------------------+
//| Implements ORIONLoadForecastInference.                           |
//+------------------------------------------------------------------+
bool ORIONLoadForecastInference(void)
  {
   ORIONFrozenBaselineReady = false;
   ORIONForecast = NULL;
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
//--- Function if.
   if(!ORIONMarket.Load(ORION_MARKET_FILE, error, undefine, forecast, studied, true))
     {
      Print("ORION inference: model load=FAIL");
      ReturnFalse;
     }
   ORIONForecast = (CNeuronScenarioForecast*)ORIONMarket.Layer(-1);
//--- Function if.
   if(!ORIONForecast || ORIONForecast.Type() != defNeuronScenarioForecast)
     {
      Print("ORION inference: forecast layer=FAIL");
      ReturnFalse;
     }
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//--- Function if.
   if(ORIONForecast.GetTopK() != TopK)
     {
      PrintFormat("ORION inference: TopK expected=%d actual=%d", TopK, ORIONForecast.GetTopK());
      ReturnFalse;
     }
//--- only the Market/Forecast path. Target and its future-window b...
   if(!ORIONValidateShapes(false))
     {
      Print("ORION inference: shape audit=FAIL");
      ReturnFalse;
     }
   ORIONMarket.TrainMode(false);
   const ulong signature = ORIONForecastSignature();
//--- Function if.
   if(!ORIONValidateForecastManifest(signature))
     {
      Print("ORION inference: manifest=FAIL");
      ReturnFalse;
     }
//--- Function if.
   if(!ORIONCaptureFrozenForecastBaseline())
     {
      Print("ORION inference: frozen baseline=FAIL");
      ReturnFalse;
     }
   ORIONLastSignature = signature;
   return true;
  }
//+------------------------------------------------------------------+
//| Full forecast-training restore. Target is required here becau... |
//+------------------------------------------------------------------+
bool ORIONLoadForecastTraining(void)
  {
   if(!FileIsExist(ORION_MANIFEST_FILE, FILE_COMMON) ||
      !FileIsExist(ORION_MARKET_FILE, FILE_COMMON) || !FileIsExist(ORION_TARGET_FILE, FILE_COMMON))
      ReturnFalse;
//--- Avoid a partial CNet::Load before the fallback random graph i...
   if(!ORIONValidateForecastManifestHeader())
     {
      Print("ORION restore: manifest static contract=FAIL");
      ReturnFalse;
     }
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   if(!ORIONMarket.Load(ORION_MARKET_FILE, error, undefine, forecast, studied, true) ||
      !ORIONTarget.Load(ORION_TARGET_FILE, error, undefine, forecast, studied, true))
      ReturnFalse;
   ORIONTarget.SetOpenCL(ORIONMarket.GetOpenCL());
   ORIONForecast = (CNeuronScenarioForecast*)ORIONMarket.Layer(-1);
   if(!ORIONForecast || ORIONForecast.Type() != defNeuronScenarioForecast)
      ReturnFalse;
   if(!ConfigureForecastRecoveryAge())
      ReturnFalse;
//--- These are transient training tensors and are intentionally absent from
//--- *.nnw. Recreate them before the common shape audit.
   if(!ORIONInitTrainingBuffers() || !ORIONValidateShapes())
      ReturnFalse;
   const ulong signature = ORIONForecastSignature();
   if(!ORIONValidateForecastManifest(signature, true))
      ReturnFalse;
   const string completed = ORIONManifestValue(ORION_MANIFEST_FILE, "completed_epochs");
   const string batches = ORIONManifestValue(ORION_MANIFEST_FILE, "training_batches");
   const string invalid = ORIONManifestValue(ORION_MANIFEST_FILE, "invalid_batches");
   if(completed == "" || batches == "" || invalid == "")
      ReturnFalse;
   ORIONCompletedEpochs = (uint)StringToInteger(completed);
   ORIONBatches = (ulong)StringToInteger(batches);
   ORIONInvalidBatches = (ulong)StringToInteger(invalid);
   ORIONTarget.TrainMode(false);
   ORIONMarket.TrainMode(true);
   ORIONLastSignature = signature;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONAddCross.                                        |
//+------------------------------------------------------------------+
bool ORIONAddCross(CArrayObj *description, const bool critic)
  {
   CLayerDescription *descr = new CLayerDescription();
   if(!description || !descr)
     { DeleteObj(descr); ReturnFalse; }
   descr.type = defNeuronScenarioCrossAttention;
   descr.count = NScenarios;
   descr.variables = (critic ? 5 : 3);
   descr.window_out = NForecast;
   descr.window = EmbeddingSize;
   descr.layers = BarDescr;
   descr.step = StackSize;
   descr.probability = TopK;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return true;
   DeleteObj(descr);
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Implements ORIONAddConv.                                         |
//+------------------------------------------------------------------+
bool ORIONAddConv(CArrayObj *description, const uint count, const uint window, const uint output,
                  const uint variables, const ENUM_ACTIVATION activation)
  {
   CLayerDescription *descr = new CLayerDescription();
   if(!description || !descr)
     { DeleteObj(descr); ReturnFalse; }
   descr.type = defNeuronConvOCL;
   descr.count = count;
   descr.window = window;
   descr.step = window;
   descr.window_out = output;
   descr.layers = variables;
   descr.activation = activation;
   descr.optimization = ADAM;
   descr.batch = BatchSize;
   if(description.Add(descr))
      return true;
   DeleteObj(descr);
   ReturnFalse;
  }
//+------------------------------------------------------------------+
//| Creates or initializes ORIONActorCriticDescriptions.             |
//+------------------------------------------------------------------+
bool CreateORIONActorCriticDescriptions(CArrayObj *&actor, CArrayObj *&critic)
  {
   actor = new CArrayObj();
   critic = new CArrayObj();
   if(!actor || !critic)
      ReturnFalse;
   actor.FreeMode(true);
   critic.FreeMode(true);
//--- Full K-major scenario tensors enter one learned trunk. There is no Pi sum.
   if(!ORIONAddBase(actor, AccountDescr) || !ORIONAddCross(actor, false) ||
      !ORIONAddConv(actor, 1, (3 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !ORIONAddConv(actor, 1, EmbeddingSize, 3, 2, SIGMOID))
      ReturnFalse;
   if(!ORIONAddBase(critic, AccountDescr + NActions) || !ORIONAddCross(critic, true) ||
      !ORIONAddConv(critic, 1, (5 * EmbeddingSize), EmbeddingSize, 1, GELU) ||
      !ORIONAddConv(critic, 1, EmbeddingSize, 1, 1, None))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONValidatePolicyShape.                             |
//+------------------------------------------------------------------+
bool ORIONValidatePolicyShape(CNet &net, const bool critic)
  {
   CNeuronScenarioCrossAttention *cross = (CNeuronScenarioCrossAttention*)net.Layer(1);
   if(!cross || cross.Type() != defNeuronScenarioCrossAttention ||
      cross.Variables() != BarDescr || cross.Scenarios() != NScenarios ||
      cross.Horizon() != NForecast || cross.Latent() != EmbeddingSize ||
      cross.Queries() != (critic ? 5 : 3) || cross.IsCritic() != critic ||
      cross.HistorySize() != StackSize || cross.HistoryTopK() != TopK)
      ReturnFalse;
   CNeuronBaseOCL *output_layer = net.Layer(3);
   CBufferFloat *buffer = (output_layer ? output_layer.getOutput() : NULL);
   if(!buffer || buffer.Total() != (critic ? 1 : NActions))
      ReturnFalse;
   CNeuronBaseOCL *input_layer = net.Layer(0);
   buffer = (input_layer ? input_layer.getOutput() : NULL);
   if(!buffer ||
      buffer.Total() != int(AccountDescr + (critic ? NActions : 0)))
      ReturnFalse;
   CNeuronBaseOCL *cross_layer = net.Layer(1);
   buffer = (cross_layer ? cross_layer.getOutput() : NULL);
   if(!buffer ||
      buffer.Total() != int(critic ? (5 * EmbeddingSize) : (3 * EmbeddingSize)))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONForwardForecast.                                 |
//+------------------------------------------------------------------+
bool ORIONForwardForecast(const int position, CBufferFloat *state, CBufferFloat *time)
  {
   if(!ORIONForecast || !CreateBuffers(position, state, time, NULL) ||
      !ORIONMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = ORIONForecast.GetZ(), *u = ORIONForecast.GetU(), *pi = ORIONForecast.GetPi();
   return (z != NULL && u != NULL && pi != NULL && z.Total() == NScenarios * BarDescr * NForecast * EmbeddingSize &&
           u.Total() == NScenarios * BarDescr * NForecast &&
           pi.Total() == NScenarios && z.GetIndex() >= 0 && u.GetIndex() >= 0 && pi.GetIndex() >= 0);
  }
//+------------------------------------------------------------------+
//| Builds CriticInput.                                              |
//+------------------------------------------------------------------+
bool BuildCriticInput(CBufferFloat *account, CBufferFloat *action, CBufferFloat *combined)
  {
   if(!account || !action || !combined || account.Total() != AccountDescr || action.Total() != NActions ||
      account.GetIndex() < 0 || action.GetIndex() < 0 ||
      (!combined.BufferInit(AccountDescr + NActions, 0)) ||
      (combined.GetIndex() < 0 && !combined.BufferCreate(ORIONMarket.GetOpenCL())))
      ReturnFalse;
   if(!ORIONDevice.Bind(ORIONMarket.GetOpenCL()))
      ReturnFalse;
   return ORIONDevice.Join2(account, AccountDescr, action, NActions, combined);
  }
//+------------------------------------------------------------------+
//| Implements EvaluateAction. lot instead of the former balance-... |
//+------------------------------------------------------------------+
double EvaluateAction(CBufferFloat *action, const double balance, const uint position)
  {
   const double reward = CheckAction(action, balance, position);
   if(!MathIsValidNumber(reward) || !action || action.Total() != NActions)
      return reward;
   const double buy = MathMax(0.0, double(action[0] - action[3]));
   const double sell = MathMax(0.0, double(action[3] - action[0]));
   const double min_lot = Symb.LotsMin();
   if(MathMax(buy, sell) >= min_lot || min_lot <= 0 || balance <= 0)
      return reward;
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Symb.Ask(), margin) || margin <= 0)
      return reward;
   const double virtual_lot = balance / (2.0 * margin);
   if(virtual_lot <= min_lot)
      return reward;
   return reward * (min_lot / virtual_lot);
  }
//+------------------------------------------------------------------+
//| Builds TeacherAction. raw future window here; the live Market... |
//+------------------------------------------------------------------+
bool BuildTeacherAction(const int position, CBufferFloat *account, CBufferFloat *action, double &reward)
  {
   reward = 0;
   if(!account || !action || position < 0 || position + HistoryBars + NForecast > int(Rates.Size()) ||
//--- Function Total.
      account.Total() != AccountDescr)
     {
      PrintFormat("BuildTeacherAction input position=%d rates=%d account=%d action=%d",
                  position, Rates.Size(), (account ? account.Total() : -1), (action ? action.Total() : -1));
      ReturnFalse;
     }
   CBufferFloat teacher_state, teacher_time;
//--- Function if.
   if(!CreateBuffers(position, GetPointer(teacher_state), GetPointer(teacher_time), GetPointer(ORIONFuture)))
     {
      PrintFormat("BuildTeacherAction CreateBuffers position=%d future=%d index=%d",
                  position, ORIONFuture.Total(), ORIONFuture.GetIndex());
      ReturnFalse;
     }
   vector<float> account_values;
//--- Function if.
   if(account.GetData(account_values) != AccountDescr)
     {
      PrintFormat("BuildTeacherAction account data total=%d index=%d", account.Total(), account.GetIndex());
      ReturnFalse;
     }
   const vector<float> teacher = OraculAction(account_values, GetPointer(ORIONFuture));
//--- Function if.
   if(teacher.Size() != NActions)
     {
      PrintFormat("BuildTeacherAction oracle size=%d expected=%d", teacher.Size(), NActions);
      ReturnFalse;
     }
//--- Function if.
   if(!action.AssignArray(teacher))
     {
      Print("BuildTeacherAction action AssignArray");
      ReturnFalse;
     }
//--- Function if.
   if(action.GetIndex() < 0 && !action.BufferCreate(ORIONMarket.GetOpenCL()))
     {
      PrintFormat("BuildTeacherAction action BufferCreate total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
//--- Function if.
   if(action.GetIndex() >= 0 && !action.BufferWrite())
     {
      PrintFormat("BuildTeacherAction action BufferWrite total=%d index=%d", action.Total(), action.GetIndex());
      ReturnFalse;
     }
   reward = EvaluateAction(action, MathMax(0.0, double(account_values[0]) * EtalonBalance), (uint)position);
//--- Function if.
   if(!MathIsValidNumber(reward))
     {
      PrintFormat("BuildTeacherAction reward invalid %.8f", reward);
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| A valid random policy action expands Critic coverage only. Bu... |
//+------------------------------------------------------------------+
bool BuildRandomAction(const int position, CBufferFloat *account, CBufferFloat *action, double &reward)
  {
   reward = 0;
   if(!account || !action || position < 0 || position >= int(Rates.Size()) ||
      account.Total() != AccountDescr)
      ReturnFalse;
   vector<float> account_values;
   if(account.GetData(account_values) != AccountDescr)
      ReturnFalse;
   const double balance = MathMax(0.0, double(account_values[0]) * EtalonBalance);
   double margin = 0;
   if(balance <= 0 || !OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Rates[position].open, margin) || margin <= 0)
      ReturnFalse;
   const double min_lot = Symb.LotsMin();
   const double max_lot = MathMin(Symb.LotsMax(), balance / (2.0 * margin));
   if(min_lot <= 0 || max_lot < min_lot)
      ReturnFalse;
   const double stop_points = MathMax(Symb.StopsLevel(), 10);
   const double min_tp = stop_points / MathMax(MaxTP, 1);
   const double min_sl = (stop_points + Symb.Spread()) / MathMax(MaxSL, 1);
   if(min_tp >= 1.0 || min_sl >= 1.0)
      ReturnFalse;
   const double uniform = MathRand() / 32767.0;
   const double lot = MathMin(max_lot, NormalizeLot(min_lot + (max_lot - min_lot) * uniform));
   const double tp = min_tp + (1.0 - min_tp) * (MathRand() / 32767.0);
   const double sl = min_sl + (1.0 - min_sl) * (MathRand() / 32767.0);
   vector<float> values = vector<float>::Zeros(NActions);
   if((MathRand() & 1) != 0)
      values[0] = float(lot);
   else
      values[3] = float(lot);
   values[1] = values[4] = float(tp);
   values[2] = values[5] = float(sl);
   if(!action.AssignArray(values))
      ReturnFalse;
   if(action.GetIndex() < 0 && !action.BufferCreate(ORIONMarket.GetOpenCL()))
      ReturnFalse;
   if(action.GetIndex() >= 0 && !action.BufferWrite())
      ReturnFalse;
   reward = EvaluateAction(action, balance, (uint)position);
   return MathIsValidNumber(reward);
  }
//+------------------------------------------------------------------+
//| Implements ORIONClampAction.                                     |
//+------------------------------------------------------------------+
bool ORIONClampAction(CBufferFloat *source, CBufferFloat *target)
  {
   if(!source || !target || !source.BufferRead() || source.Total() != NActions ||
      !target.BufferInit(NActions, 0))
      ReturnFalse;
   for(uint i = 0; i < NActions; i++)
     {
      if(!MathIsValidNumber(source[i]) || !target.Update(i, float(MathMax(0.0, MathMin(1.0, double(source[i]))))))
         ReturnFalse;
     }
   return (target.GetIndex() < 0 || target.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Implements ORIONWriteACManifest.                                 |
//+------------------------------------------------------------------+
bool ORIONWriteACManifest(const ulong forecast_signature)
  {
   if(forecast_signature == 0 || forecast_signature != ORIONLastSignature)
      ReturnFalse;
   const ulong actor_hash = ORIONHashFile(ulong(1469598103934665603), ORION_ACTOR_FILE);
   const ulong q1_hash = ORIONHashFile(ulong(1469598103934665603), ORION_Q1_FILE);
   const ulong q2_hash = ORIONHashFile(ulong(1469598103934665603), ORION_Q2_FILE);
   if(actor_hash == 0 || q1_hash == 0 || q2_hash == 0)
      ReturnFalse;
   int handle = FileOpen(ORION_AC_MANIFEST_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
   FileWrite(handle, "format=ORION_ACTOR_CRITIC");
   FileWrite(handle, StringFormat("version=%u", ORION_AC_FORMAT_VERSION));
   FileWrite(handle, StringFormat("forecast_signature=%I64u", forecast_signature));
   FileWrite(handle, StringFormat("actor_context=%u", (3 * EmbeddingSize)));
   FileWrite(handle, StringFormat("critic_context=%u", (5 * EmbeddingSize)));
   FileWrite(handle, "action_order=BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL");
   FileWrite(handle, "scenario_policy=preserve_KV_no_probability_aggregation");
   FileWrite(handle, "teacher_policy=realized_future_oracul_positive_actor_all_critic");
   FileWrite(handle, "no_trade_penalty=min_executable_lot");
   FileWrite(handle, "account_execution=target_position_tp_sl_bar_lifecycle");
   FileWrite(handle, "critic_target=direct_episode_return");
   FileWrite(handle, "actor_policy_gradient=executable_action_only");
   FileWrite(handle, "critic_coverage=policy_teacher_all_random_executable");
   FileWrite(handle, "actor_supervision=positive_teacher_and_random");
   FileWrite(handle, StringFormat("actor_hash=%I64u", actor_hash));
   FileWrite(handle, StringFormat("q1_hash=%I64u", q1_hash));
   FileWrite(handle, StringFormat("q2_hash=%I64u", q2_hash));
   FileWrite(handle, StringFormat("generation=%I64u", (ulong)TimeCurrent()));
   FileClose(handle);
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONValidateACManifest.                              |
//+------------------------------------------------------------------+
bool ORIONValidateACManifest(const bool required)
  {
   if(ORIONManifestValue(ORION_AC_MANIFEST_FILE, "format") == "")
      return !required;
#define ORION_AC_MANIFEST_EQ(KEY,VALUE) \
   { const string actual=ORIONManifestValue(ORION_AC_MANIFEST_FILE,KEY); const string expected=(VALUE); \
     if(actual!=expected) { PrintFormat("ORION policy manifest: %s expected=%s actual=%s",KEY,expected,actual); ReturnFalse; } }
   ORION_AC_MANIFEST_EQ("format", "ORION_ACTOR_CRITIC");
   ORION_AC_MANIFEST_EQ("version", IntegerToString(ORION_AC_FORMAT_VERSION));
   ORION_AC_MANIFEST_EQ("forecast_signature", StringFormat("%I64u", ORIONLastSignature));
   ORION_AC_MANIFEST_EQ("actor_context", IntegerToString((3 * EmbeddingSize)));
   ORION_AC_MANIFEST_EQ("critic_context", IntegerToString((5 * EmbeddingSize)));
   ORION_AC_MANIFEST_EQ("action_order", "BuyLot,BuyTP,BuySL,SellLot,SellTP,SellSL");
   ORION_AC_MANIFEST_EQ("scenario_policy", "preserve_KV_no_probability_aggregation");
   ORION_AC_MANIFEST_EQ("teacher_policy", "realized_future_oracul_positive_actor_all_critic");
   ORION_AC_MANIFEST_EQ("no_trade_penalty", "min_executable_lot");
   ORION_AC_MANIFEST_EQ("account_execution", "target_position_tp_sl_bar_lifecycle");
   ORION_AC_MANIFEST_EQ("critic_target", "direct_episode_return");
   ORION_AC_MANIFEST_EQ("actor_policy_gradient", "executable_action_only");
   ORION_AC_MANIFEST_EQ("critic_coverage", "policy_teacher_all_random_executable");
   ORION_AC_MANIFEST_EQ("actor_supervision", "positive_teacher_and_random");
   ORION_AC_MANIFEST_EQ("actor_hash", StringFormat("%I64u", ORIONHashFile(ulong(1469598103934665603), ORION_ACTOR_FILE)));
   ORION_AC_MANIFEST_EQ("q1_hash", StringFormat("%I64u", ORIONHashFile(ulong(1469598103934665603), ORION_Q1_FILE)));
   ORION_AC_MANIFEST_EQ("q2_hash", StringFormat("%I64u", ORIONHashFile(ulong(1469598103934665603), ORION_Q2_FILE)));
#undef ORION_AC_MANIFEST_EQ
//--- Function if.
   if(ORIONManifestValue(ORION_AC_MANIFEST_FILE, "generation") == "")
     {
      Print("ORION policy manifest: generation is absent");
      ReturnFalse;
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONLoadPolicyNet. Shared Actor-Critic lifecycle ... |
//+------------------------------------------------------------------+
bool ORIONLoadPolicyNet(CNet &net, const string file_name)
  {
   float error = 0, undefine = 0, forecast = 0;
   datetime studied = 0;
   return net.Load(file_name, error, undefine, forecast, studied, true);
  }
//+------------------------------------------------------------------+
//| Implements ORIONCreatePolicySet.                                 |
//+------------------------------------------------------------------+
bool ORIONCreatePolicySet(CNet &actor, CNet &q1, CNet &q2)
  {
   CArrayObj *actor_descr = NULL, *critic_descr = NULL;
   if(!CreateORIONActorCriticDescriptions(actor_descr, critic_descr))
     { DeleteObj(actor_descr); DeleteObj(critic_descr); ReturnFalse; }
   bool result = actor.Create(actor_descr);
   if(result)
      actor.SetOpenCL(ORIONMarket.GetOpenCL());
   if(result)
      result = q1.Create(critic_descr);
   if(result)
      q1.SetOpenCL(ORIONMarket.GetOpenCL());
   if(result)
      result = q2.Create(critic_descr);
   if(result)
      q2.SetOpenCL(ORIONMarket.GetOpenCL());
   DeleteObj(actor_descr);
   DeleteObj(critic_descr);
//--- Function if.
   if(result)
     {
      //--- A CNet temporary OpenCL context must be released before creating the
      //--- next policy net.  Otherwise nested MSRes buffers can retain stale
      //--- device handles from the preceding Create call.
      actor.SetOpenCL(ORIONMarket.GetOpenCL());
      q1.SetOpenCL(ORIONMarket.GetOpenCL());
      q2.SetOpenCL(ORIONMarket.GetOpenCL());
     }
   return result;
  }
//+------------------------------------------------------------------+
//| Implements ORIONLoadOrCreatePolicySet.                           |
//+------------------------------------------------------------------+
bool ORIONLoadOrCreatePolicySet(CNet &actor, CNet &q1, CNet &q2,
                                const bool allow_create)
  {
   const bool manifest_exists = FileIsExist(ORION_AC_MANIFEST_FILE, FILE_COMMON);
   const bool loaded = (manifest_exists && ORIONValidateACManifest(true) &&
                        ORIONLoadPolicyNet(actor, ORION_ACTOR_FILE) &&
                        ORIONLoadPolicyNet(q1, ORION_Q1_FILE) && ORIONLoadPolicyNet(q2, ORION_Q2_FILE));
//--- Function if.
   if(!loaded)
     {
      if(manifest_exists)
         Print("ORION policy restore=FAIL; creating new random policy set");
      if(!allow_create || !ORIONCreatePolicySet(actor, q1, q2))
         ReturnFalse;
     }
   actor.SetOpenCL(ORIONMarket.GetOpenCL());
   q1.SetOpenCL(ORIONMarket.GetOpenCL());
   q2.SetOpenCL(ORIONMarket.GetOpenCL());
   if(!ORIONValidatePolicyShape(actor, false) || !ORIONValidatePolicyShape(q1, true) ||
      !ORIONValidatePolicyShape(q2, true))
      ReturnFalse;
   actor.TrainMode(true);
   q1.TrainMode(true);
   q2.TrainMode(true);
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONSavePolicySet.                                   |
//+------------------------------------------------------------------+
bool ORIONSavePolicySet(CNet &actor, CNet &q1, CNet &q2)
  {
   if(!ORIONVerifyFrozenWeightsExact() || !ORIONVerifyFrozenCodebookExact() ||
      ORIONForecastSignature() != ORIONLastSignature)
      ReturnFalse;
   const datetime now = TimeCurrent();
   return (actor.Save(ORION_ACTOR_FILE, 0, 0, 0, now, true) &&
           q1.Save(ORION_Q1_FILE, q1.getRecentAverageError(), 0, 0, now, true) &&
           q2.Save(ORION_Q2_FILE, q2.getRecentAverageError(), 0, 0, now, true) &&
           ORIONWriteACManifest(ORIONLastSignature));
  }
//+------------------------------------------------------------------+
//| Implements ORIONLoadInferenceActor.                              |
//+------------------------------------------------------------------+
bool ORIONLoadInferenceActor(CNet &actor)
  {
   if(!ORIONValidateACManifest(true) || !ORIONLoadPolicyNet(actor, ORION_ACTOR_FILE))
      ReturnFalse;
   actor.SetOpenCL(ORIONMarket.GetOpenCL());
   if(!ORIONValidatePolicyShape(actor, false))
      ReturnFalse;
   actor.TrainMode(false);
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ReadAction.                                           |
//+------------------------------------------------------------------+
bool ReadAction(CNet &net, CBufferFloat *target)
  {
//--- This is the explicit device-to-CPU boundary for trade execution.
//--- The final actor layer is allowed to be host-only, therefore its live
//--- output must not be read through CBufferFloat::BufferRead().
   if(!target)
      ReturnFalse;
//--- CNet::getResults reuses a valid result object.  Passing target directly
//--- avoids an allocation and the subsequent CPU-to-CPU AssignArray copy.
   CBufferFloat *output = target;
   net.getResults(output);
   return (output == target && target.Total() == NActions);
  }
//+------------------------------------------------------------------+
//| Implements ORIONAdvanceAccountTime.                              |
//+------------------------------------------------------------------+
bool ORIONAdvanceAccountTime(CBufferFloat *current, const datetime next_time, CBufferFloat *next)
  {
   if(!current || !next || current.Total() != AccountDescr ||
      (current.GetIndex() >= 0 && !current.BufferRead()) ||
      !next.BufferInit(AccountDescr, 0))
      ReturnFalse;
   for(uint i = 0; i < 9; i++)
      if(!next.Update(i, current[i]))
         ReturnFalse;
   double x = next_time / (double)(D'2024.01.01' - D'2023.01.01');
   if(!next.Update(9, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_MN1);
   if(!next.Update(10, float(MathCos(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_W1);
   if(!next.Update(11, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   x = next_time / (double)PeriodSeconds(PERIOD_D1);
   if(!next.Update(12, float(MathSin(x != 0 ? 2.0 * M_PI*x : 0))))
      ReturnFalse;
   return (next.GetIndex() < 0 || next.BufferWrite());
  }
//+------------------------------------------------------------------+
//| Implements ORIONForwardForecastState.                            |
//+------------------------------------------------------------------+
bool ORIONForwardForecastState(CBufferFloat *state)
  {
//--- CNet accepts a host-only input buffer and uploads it through its first
//--- layer.  Do not require a device allocation from the caller here.
   if(!ORIONForecast || !state || state.Total() != HistoryBars * BarDescr)
      ReturnFalse;
   if(!ORIONMarket.feedForward(state, 1, false, (CBufferFloat*)NULL))
      ReturnFalse;
   CBufferFloat *z = ORIONForecast.GetZ(), *u = ORIONForecast.GetU(), *pi = ORIONForecast.GetPi();
   if(!z || !u || !pi || z.Total() != NScenarios * BarDescr * NForecast * EmbeddingSize ||
      u.Total() != NScenarios * BarDescr * NForecast || pi.Total() != NScenarios ||
      z.GetIndex() < 0 || u.GetIndex() < 0 || pi.GetIndex() < 0)
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONBuildLiveAccount.                                |
//+------------------------------------------------------------------+
bool ORIONBuildLiveAccount(const double previous_balance, const double previous_equity,
                           const datetime state_time, CBufferFloat *account,
                           double &buy_value, double &sell_value)
  {
   if(!account || !MathIsValidNumber(previous_balance) ||
      !MathIsValidNumber(previous_equity) || previous_balance <= 0 || previous_equity <= 0)
      ReturnFalse;
   double buy_profit = 0, sell_profit = 0, position_discount = 0;
   buy_value = 0;
   sell_value = 0;
   const datetime current = TimeCurrent();
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      const double profit = PositionGetDouble(POSITION_PROFIT);
      if((int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
        { buy_value += PositionGetDouble(POSITION_VOLUME); buy_profit += profit; }
      else
        { sell_value += PositionGetDouble(POSITION_VOLUME); sell_profit += profit; }
      position_discount += (current - PositionGetInteger(POSITION_TIME)) *
                           (1.0 / (60.0 * 60.0 * 10.0)) * MathAbs(profit);
     }
   vector<float> values = vector<float>::Zeros(AccountDescr);
   const double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   values[0] = float(balance / EtalonBalance);
   values[1] = float((balance - previous_balance) / previous_balance);
   values[2] = float(equity / previous_balance);
   values[3] = float((equity - previous_equity) / previous_equity);
   values[4] = float(buy_value);
   values[5] = float(sell_value);
   values[6] = float(buy_profit / previous_balance);
   values[7] = float(sell_profit / previous_balance);
   values[8] = float(position_discount / previous_balance);
   double x = state_time / (double)(D'2024.01.01' - D'2023.01.01');
   values[9] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_MN1);
   values[10] = float(MathCos(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_W1);
   values[11] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   x = state_time / (double)PeriodSeconds(PERIOD_D1);
   values[12] = float(MathSin(x != 0 ? 2.0 * M_PI*x : 0));
   for(uint i = 0; i < AccountDescr; i++)
      if(!MathIsValidNumber(values[i]))
         ReturnFalse;
   return (account.AssignArray(values) && (account.GetIndex() < 0 || account.BufferWrite()));
  }
//+------------------------------------------------------------------+
//| Implements ORIONRefreshLiveMarket.                               |
//+------------------------------------------------------------------+
bool ORIONRefreshLiveMarket(CBufferFloat *state, CBufferFloat *time_state)
  {
   const int requested = StackSize + HistoryBars;
//--- A new-bar event is evaluated from the last fully closed bar.  The
//--- forming bar must not leak unfinished OHLC/indicator values into policy.
   const int bars = CopyRates(Symb.Name(), TimeFrame, 1, requested, Rates);
   if(!state || !time_state || bars < HistoryBars || !ArraySetAsSeries(Rates, true) ||
      !RSI.BufferResize(bars) || !CCI.BufferResize(bars) ||
      !ATR.BufferResize(bars) || !MACD.BufferResize(bars) ||
      RSI.BarsCalculated() < bars || CCI.BarsCalculated() < bars ||
      ATR.BarsCalculated() < bars || MACD.BarsCalculated() < bars)
      ReturnFalse;
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
   return CreateBuffers(0, state, time_state, (CBufferFloat*)NULL);
  }
//+------------------------------------------------------------------+
//| Checks ExecutableOrder. One execution contract for historical... |
//+------------------------------------------------------------------+
bool IsExecutableOrder(const double lot, const double tp_fraction, const double sl_fraction)
  {
   const double stops = (MathMax(Symb.StopsLevel(), 1) + Symb.Spread()) * Symb.Point();
   return (lot >= Symb.LotsMin() && tp_fraction * MaxTP * Symb.Point() > 2.0 * stops &&
           sl_fraction * MaxSL * Symb.Point() > stops);
  }
//+------------------------------------------------------------------+
//| Implements NormalizeLot.                                         |
//+------------------------------------------------------------------+
double NormalizeLot(const double lot)
  {
   const double min_lot = Symb.LotsMin(), step_lot = Symb.LotsStep();
   return (step_lot > 0 ? min_lot + MathRound((lot - min_lot) / step_lot) * step_lot : lot);
  }
#ifndef Study
//+------------------------------------------------------------------+
//| Implements ORIONValidateAction.                                  |
//+------------------------------------------------------------------+
bool ORIONValidateAction(CBufferFloat *action)
  {
   if(!action || action.Total() != NActions ||
      (action.GetIndex() >= 0 && !action.BufferRead()))
      ReturnFalse;
   for(uint i = 0; i < NActions; i++)
      if(!MathIsValidNumber(action[i]) || action[i] < 0 || action[i] > 1)
         ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONExecuteAction.                                   |
//+------------------------------------------------------------------+
bool ORIONExecuteAction(CBufferFloat *action, double buy_value, double sell_value, double &margin_penalty,
                        bool &market_closed)
  {
   margin_penalty = 0;
   market_closed = false;
   if(!ORIONValidateAction(action))
      ReturnFalse;
//--- Canonical CogDriver mutual exclusion and broker constraints.
   if(action[0] >= action[3])
     { action.Update(0, action[0] - action[3]); action.Update(3, 0); }
   else
     { action.Update(3, action[3] - action[0]); action.Update(0, 0); }
   const double min_lot = Symb.LotsMin();
   if(!IsExecutableOrder(action[0], action[1], action[2]))
     { if(buy_value > 0) CloseByDirection(POSITION_TYPE_BUY); }
   else
     {
      const double lot = NormalizeLot(action[0]);
      const double tp = NormalizeDouble(Symb.Ask() + action[1] * MaxTP * Symb.Point(), Symb.Digits());
      const double sl = NormalizeDouble(Symb.Ask() - action[2] * MaxSL * Symb.Point(), Symb.Digits());
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, sl, tp);
      if((buy_value - lot) >= min_lot)
         ClosePartial(POSITION_TYPE_BUY, buy_value - lot);
      else
//--- Function if.
         if((lot - buy_value) >= min_lot && !Trade.Buy(lot - buy_value, Symb.Name(), Symb.Ask(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
//--- Function if.
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return true;
              }
//--- Function if.
            if(retcode != 10019)
              {
               PrintFormat("ORION buy execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
               ReturnFalse;
              }
            //--- Preserve CogDriver's insufficient-margin feedback, but keep the
            //--- transition alive: the order was rejected, not the account.
            margin_penalty -= 100.0 * (lot - buy_value);
           }
     }
   if(!IsExecutableOrder(action[3], action[4], action[5]))
     { if(sell_value > 0) CloseByDirection(POSITION_TYPE_SELL); }
   else
     {
      const double lot = NormalizeLot(action[3]);
      const double tp = NormalizeDouble(Symb.Bid() - action[4] * MaxTP * Symb.Point(), Symb.Digits());
      const double sl = NormalizeDouble(Symb.Bid() + action[5] * MaxSL * Symb.Point(), Symb.Digits());
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sl, tp);
      if((sell_value - lot) >= min_lot)
         ClosePartial(POSITION_TYPE_SELL, sell_value - lot);
      else
//--- Function if.
         if((lot - sell_value) >= min_lot && !Trade.Sell(lot - sell_value, Symb.Name(), Symb.Bid(), sl, tp))
           {
            const uint retcode = Trade.ResultRetcode();
//--- Function if.
            if(retcode == TRADE_RETCODE_MARKET_CLOSED)
              {
               market_closed = true;
               return true;
              }
//--- Function if.
            if(retcode != 10019)
              {
               PrintFormat("ORION sell execution failed: retcode=%u %s", retcode, Trade.ResultRetcodeDescription());
               ReturnFalse;
              }
            margin_penalty -= 100.0 * (lot - sell_value);
           }
     }
   return true;
  }
//+------------------------------------------------------------------+
//| Implements ORIONExecuteAction. Inference callers do not train... |
//+------------------------------------------------------------------+
bool ORIONExecuteAction(CBufferFloat *action, double buy_value, double sell_value)
  {
   double margin_penalty = 0;
   bool market_closed = false;
   return ORIONExecuteAction(action, buy_value, sell_value, margin_penalty, market_closed);
  }
#endif
//+------------------------------------------------------------------+
//| Implements PolicyBackward.                                       |
//+------------------------------------------------------------------+
bool PolicyBackward(CNet &actor, CNet &critic, CBufferFloat *actor_target,
                    CNet *scenario_net, const int scenario_layer)
  {
   CNeuronBaseOCL *critic_output = critic.Layer(3);
   CNeuronBaseOCL *critic_base = critic.Layer(0);
   CNeuronBaseOCL *actor_output = actor.Layer(3);
   if(!critic_output || !critic_base || !actor_output || !actor_target ||
      critic_output.getOutput().Total() != 1 || actor_output.getGradient().Total() != NActions ||
      !actor_target.BufferInit(1, 1.0f) ||
      (actor_target.GetIndex() < 0 && !actor_target.BufferCreate(ORIONMarket.GetOpenCL())) ||
      !ORIONDevice.Add(critic_output.getOutput(), actor_target, actor_target, 1))
      ReturnFalse;
   critic.TrainMode(false);
   bool result = critic.backProp(actor_target, scenario_net, scenario_layer);
   CBufferFloat account_gradient;
   if(!(result && critic_base.getGradient().Total() == AccountDescr + NActions &&
        account_gradient.BufferInit(AccountDescr, 0) &&
        (account_gradient.GetIndex() >= 0 || account_gradient.BufferCreate(ORIONMarket.GetOpenCL())) &&
        ORIONDevice.Split2(GetPointer(account_gradient), actor_output.getGradient(), critic_base.getGradient(),
                           AccountDescr, NActions) &&
        actor.backPropGradient(scenario_net, scenario_layer, -1, true)))
      result = false;
   critic.TrainMode(true);
   return result;
  }
//+------------------------------------------------------------------+
//| One historical-bar account transition.  CheckAction remains the  |
//+------------------------------------------------------------------+
bool AdvanceAccount(CBufferFloat *current, CBufferFloat *action, const int position,
                    const double min_balance, CBufferFloat *next, double &reward, bool &terminal)
  {
   terminal = true;
   reward = 0;
   if(!current || !action || !next || position <= 0 || position >= int(Rates.Size()) ||
      current.Total() != AccountDescr || action.Total() != NActions ||
      (current.GetIndex() >= 0 && !current.BufferRead()) ||
      (action.GetIndex() >= 0 && !action.BufferRead()))
      ReturnFalse;
   const double balance = MathMax(0.0, double(current[0]) * EtalonBalance);
   reward = EvaluateAction(action, balance, (uint)position);
   if(!MathIsValidNumber(reward))
      ReturnFalse;
   const double buy = MathMax(0.0, double(action[0] - action[3]));
   const double sell = MathMax(0.0, double(action[3] - action[0]));
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, Symb.Name(), 1, Rates[position].open, margin))
      ReturnFalse;
   const double min_lot = Symb.LotsMin();
//--- Function if.
   if(balance <= min_balance || balance < margin * min_lot)
     {
      terminal = true;
      return ORIONAdvanceAccountTime(current, Rates[position - 1].time, next);
     }
//--- Action is the target position for the next bar.  A valid same-direction
//--- action modifies its TP/SL; a valid opposite action closes then reopens;
//--- an invalid action closes both positions.
   const bool open_buy = IsExecutableOrder(buy, action[1], action[2]);
   const bool open_sell = IsExecutableOrder(sell, action[4], action[5]);
   const double target_buy = (open_buy ? NormalizeLot(buy) : 0.0);
   const double target_sell = (open_sell ? NormalizeLot(sell) : 0.0);
   const double point_cost = Symb.TickValue() / Symb.TickSize();
   const double entry = Rates[position].open;
   const double spread = Symb.Spread() * Symb.Point();
   const double current_buy = MathMax(0.0, double(current[4]));
   const double current_sell = MathMax(0.0, double(current[5]));
   double current_buy_profit = double(current[6]) * balance;
   double current_sell_profit = double(current[7]) * balance;
   double next_buy = 0, next_sell = 0;
   double next_buy_profit = 0, next_sell_profit = 0;
   double realized = 0;
//--- An invalid target liquidates every open position at the curre...
   if(!open_buy && !open_sell)
     {
      realized = current_buy_profit + current_sell_profit;
     }
   else
//--- Function if.
      if(open_buy)
        {
         //--- A reverse target closes the Sell; a same-side reduction realizes only its
         //--- proportional carried P/L.  The retained Buy keeps its marked-to-market P/L.
         realized += current_sell_profit;
//--- Function if.
         if(current_buy > target_buy && current_buy > 0)
           {
            const double closed = current_buy - target_buy;
            realized += current_buy_profit * closed / current_buy;
            current_buy_profit -= current_buy_profit * closed / current_buy;
           }
         const double added = MathMax(0.0, target_buy - current_buy);
         current_buy_profit -= spread * point_cost * added;
         const double tp = entry + (action[1] * MaxTP + Symb.Spread()) * Symb.Point();
         const double sl = entry - (action[2] * MaxSL + Symb.Spread()) * Symb.Point();
         const MqlRates bar = Rates[position - 1];
         if(sl >= bar.low)
            realized += current_buy_profit + (sl - entry) * point_cost * target_buy;
         else
            if(tp <= bar.high)
               realized += current_buy_profit + (tp - entry) * point_cost * target_buy;
            else
              {
               next_buy = target_buy;
               next_buy_profit = current_buy_profit + (bar.open - entry) * point_cost * target_buy;
              }
        }
      else
        {
         //--- Symmetric Sell lifecycle.  The SL-before-TP order matches CheckAction.
         realized += current_buy_profit;
//--- Function if.
         if(current_sell > target_sell && current_sell > 0)
           {
            const double closed = current_sell - target_sell;
            realized += current_sell_profit * closed / current_sell;
            current_sell_profit -= current_sell_profit * closed / current_sell;
           }
         const double added = MathMax(0.0, target_sell - current_sell);
         current_sell_profit -= spread * point_cost * added;
         const double tp = entry - (action[4] * MaxTP + Symb.Spread()) * Symb.Point();
         const double sl = entry + (action[5] * MaxSL - Symb.Spread()) * Symb.Point();
         const MqlRates bar = Rates[position - 1];
         if(sl <= bar.high)
            realized += current_sell_profit + (entry - sl) * point_cost * target_sell;
         else
            if(tp >= bar.low)
               realized += current_sell_profit + (entry - tp) * point_cost * target_sell;
            else
              {
               next_sell = target_sell;
               next_sell_profit = current_sell_profit + (entry - bar.open) * point_cost * target_sell;
              }
        }
   const double next_balance = MathMax(0.0, balance + realized);
   const double current_equity = balance + double(current[6]) * balance + double(current[7]) * balance;
   const double next_equity = next_balance + next_buy_profit + next_sell_profit;
   if(!ORIONAdvanceAccountTime(current, Rates[position - 1].time, next) ||
      (next.GetIndex() >= 0 && !next.BufferRead()))
      ReturnFalse;
   if(!next.Update(0, float(next_balance / EtalonBalance)) ||
      !next.Update(1, float((next_balance - balance) / MathMax(balance, 1.0))) ||
      !next.Update(2, float(next_equity / MathMax(balance, 1.0))) ||
      !next.Update(3, float((next_equity - current_equity) / MathMax(MathAbs(current_equity), 1.0))) ||
      !next.Update(4, float(next_buy)) || !next.Update(5, float(next_sell)) ||
      !next.Update(6, float(next_buy_profit / MathMax(next_balance, 1.0))) ||
      !next.Update(7, float(next_sell_profit / MathMax(next_balance, 1.0))) ||
      !next.Update(8, 0.0f))
      ReturnFalse;
   terminal = (next_balance <= min_balance || next_equity < margin * min_lot);
   return (next.GetIndex() < 0 || next.BufferWrite());
  }
//+------------------------------------------------------------------+