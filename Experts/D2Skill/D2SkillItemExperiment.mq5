//+------------------------------------------------------------------+
//|                                      D2SkillItemExperiment.mq5   |
//| Article 1: synthetic pseudo-residual representation experiment.  |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"

#include "..\\NeuroNet_DNG\\NeuroNet.mqh"

#define D2_ITEM_DIM       4
#define D2_ITEM_SAMPLES   32
#define D2_ITEM_EPOCHS    4
#define D2_ITEM_TOLERANCE 1.0e-5

CNet           g_net;
CNeuronBaseOCL g_source;
CBufferFloat   g_target;

//+------------------------------------------------------------------+
bool CreateOpenCLHost(void)
  {
   CArrayObj *description = new CArrayObj();
   if(!description)
      return(false);
   description.FreeMode(true);
   for(int i = 0; i < 2; i++)
     {
      CLayerDescription *layer = new CLayerDescription();
      if(!layer)
        {
         delete description;
         return(false);
        }
      layer.type = defNeuronBaseOCL;
      layer.count = D2_ITEM_DIM;
      layer.activation = None;
      layer.optimization = ADAM;
      if(!description.Add(layer))
        {
         delete layer;
         delete description;
         return(false);
        }
     }
   const bool result = g_net.Create(description);
   delete description;
   return(result && g_net.GetOpenCL() != NULL);
  }
//+------------------------------------------------------------------+
float TargetValue(const int sample, const int dimension)
  {
//--- Stable direction is +X; bounded noise is deliberately injected.
   float value = (dimension == 0 ? 0.80f : 0.0f);
   value += 0.05f * (float)MathSin(0.71 * sample + 0.37 * dimension);
//--- Two isolated outliers test robustness of both EMA modes.
   if(sample == 9 || sample == 22)
      value += (dimension == 0 ? -2.5f : (dimension == 1 ? 2.0f : 0.0f));
   return(value);
  }
//+------------------------------------------------------------------+
bool RunRepresentation(const ED2SkillRepresentation representation, const float beta)
  {
   COpenCLMy *opencl = g_net.GetOpenCL();
   CD2SkillItem item;
   if(!opencl || !item.Init(0, 300, opencl, D2_ITEM_DIM, ADAM, 1))
      return(false);
   item.SetRepresentation(representation);
   item.SetEMA(beta, beta, beta);
   float latent[D2_ITEM_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
   if(!g_source.getOutput().AssignArray(latent) ||
      !g_source.getOutput().BufferWrite() ||
      !g_target.BufferInit(D2_ITEM_DIM, 0.0f) ||
      !g_target.BufferCreate(opencl))
      return(false);
//--- Forward, loss-gradient and bank update stay on the GPU.  No output,
//--- gradient or slot buffer is read during this sample loop.
   const float stable[D2_ITEM_DIM] = {0.80f, 0.0f, 0.0f, 0.0f};
   float final_skill_loss = 0.0f;
   float final_base_loss = 0.0f;
   float final_cosine = 0.0f;
   float final_reconstruction_error = 0.0f;
   int stabilization_epoch = -1;
   float outlier_response = 0.0f;
   for(int epoch = 0; epoch < D2_ITEM_EPOCHS; epoch++)
     {
      float pseudo_norm_sum = 0.0f;
      for(int sample = 0; sample < D2_ITEM_SAMPLES; sample++)
        {
         const int sample_index = epoch * D2_ITEM_SAMPLES + sample;
         float target[D2_ITEM_DIM];
         float pseudo_norm = 0.0f;
         for(int d = 0; d < D2_ITEM_DIM; d++)
           {
            target[d] = TargetValue(sample_index, d);
            pseudo_norm += target[d] * target[d];
           }
         pseudo_norm_sum += MathSqrt(pseudo_norm);
         if(epoch == 0 && (sample == 0 || sample == 9 || sample == 22))
            PrintFormat("D2SkillItemExperiment: pseudo sample=%d target=(%.5f,%.5f,%.5f,%.5f) norm=%.5f outlier=%s",
                        sample_index, target[0], target[1], target[2], target[3],
                        MathSqrt(pseudo_norm), (sample == 9 || sample == 22 ? "true" : "false"));
         if(!g_target.AssignArray(target) || !g_target.BufferWrite())
            return(false);
         float error = 1.0f;
         if(!g_source.calcOutputGradients(&g_target, error) ||
            !item.UpdateFromGradient(g_source.getGradient()))
            return(false);
        }
      //----- Host readback is restricted to completed epoch boundaries.
      if(!item.feedForward(g_source.AsObject()) || !item.Correction() ||
         !item.Correction().BufferRead() || !item.Direction().BufferRead() ||
         !item.Scale().BufferRead() || !item.Observations().BufferRead() ||
         !item.Mass().BufferRead() || !item.getOutput().BufferRead())
         return(false);
      float correction_norm = 0.0f;
      float correction_error = 0.0f;
      float base_loss = 0.0f;
      float skill_loss = 0.0f;
      for(int d = 0; d < D2_ITEM_DIM; d++)
        {
         const float correction = item.Correction()[d];
         const float base_delta = latent[d] - stable[d];
         const float skill_delta = item.getOutput()[d] - stable[d];
         const float reconstruction_delta = correction - stable[d];
         correction_norm += correction * correction;
         correction_error += reconstruction_delta * reconstruction_delta;
         base_loss += 0.5f * base_delta * base_delta;
         skill_loss += 0.5f * skill_delta * skill_delta;
        }
      const float correction_length = MathSqrt(correction_norm);
      const float reconstruction_error = MathSqrt(correction_error);
      const float cosine = (correction_length > 1.0e-12f ?
                            item.Correction()[0] / correction_length : 0.0f);
      if(stabilization_epoch < 0 && cosine >= 0.90f)
         stabilization_epoch = epoch + 1;
      outlier_response = MathMax(outlier_response,
                                 MathAbs(item.Correction()[0] - stable[0]));
      PrintFormat("D2SkillItemExperiment: epoch=%d mode=%s beta=%.3f pseudo_norm=%.5f direction_cosine=%.5f reconstruction_error=%.5f outlier_response=%.5f observations=%.0f mass=%.5f base_loss=%.6f skill_loss=%.6f",
                  epoch + 1,
                  (representation == D2SkillFullResidual ? "full" : "direction-magnitude"),
                  beta, pseudo_norm_sum / (float)D2_ITEM_SAMPLES, cosine,
                  reconstruction_error, outlier_response, item.Observations()[0],
                  item.Mass()[0], base_loss, skill_loss);
      final_skill_loss = skill_loss;
      final_base_loss = base_loss;
      final_cosine = cosine;
      final_reconstruction_error = reconstruction_error;
     }
   PrintFormat("D2SkillItemExperiment: mode=%s beta=%.3f stabilization_epoch=%d final_cosine=%.5f reconstruction_error=%.5f outlier_response=%.5f",
               (representation == D2SkillFullResidual ? "full" : "direction-magnitude"),
               beta, stabilization_epoch, final_cosine, final_reconstruction_error,
               outlier_response);
   return(final_skill_loss < final_base_loss && final_cosine > 0.0f &&
          final_reconstruction_error >= 0.0f && stabilization_epoch > 0 &&
          outlier_response >= 0.0f);
  }
//+------------------------------------------------------------------+
//| Create an isolated OpenCL host for one arbitrary-dimension test. |
//+------------------------------------------------------------------+
bool CreateCoverageHost(CNet &net, const uint dimension)
  {
   CArrayObj *description = new CArrayObj();
   if(!description)
      return(false);
   description.FreeMode(true);
   for(int i = 0; i < 2; i++)
     {
      CLayerDescription *layer = new CLayerDescription();
      if(!layer)
        {
         delete description;
         return(false);
        }
      layer.type = defNeuronBaseOCL;
      layer.count = dimension;
      layer.activation = None;
      layer.optimization = ADAM;
      if(!description.Add(layer))
        {
         delete layer;
         delete description;
         return(false);
        }
     }
   const bool result = net.Create(description);
   delete description;
   return(result && net.GetOpenCL() != NULL);
  }
//+------------------------------------------------------------------+
//| Compare one deterministic GPU update with the complete CPU state.|
//+------------------------------------------------------------------+
bool RunArbitraryDimensionCoverage(void)
  {
   const int   dimensions[] = {1, 2, 4, 63, 64, 65, 257, 1025};
   const float beta_correction = 0.05f;
   const float beta_direction = 0.125f;
   const float beta_magnitude = 0.075f;
   const double tolerance = D2_ITEM_TOLERANCE;
   for(int test = 0; test < ArraySize(dimensions); test++)
     {
      const int dimension = dimensions[test];
      for(int mode = 0; mode < 2; mode++)
        {
         const ED2SkillRepresentation representation =
            (mode == 0 ? D2SkillFullResidual : D2SkillDirectionMagnitude);
         const string mode_name =
            (representation == D2SkillFullResidual ? "full" : "direction-magnitude");
         CNet net;
         CBufferFloat gradient;
         CD2SkillItem item;
         if(!CreateCoverageHost(net, (uint)dimension) ||
            !gradient.BufferInit(dimension, 0.0f) ||
            !gradient.BufferCreate(net.GetOpenCL()) ||
            !item.Init(0, 400 + (uint)(2 * test + mode), net.GetOpenCL(),
                       (uint)dimension, ADAM, 1))
            return(false);
         item.SetRepresentation(representation);
         item.SetEMA(beta_correction, beta_direction, beta_magnitude);
         vector<float> gradient_values = vector<float>::Zeros(dimension);
         vector<float> expected_correction = vector<float>::Zeros(dimension);
         vector<float> expected_direction = vector<float>::Zeros(dimension);
         float gradient_norm_square = 0.0f;
         for(int d = 0; d < dimension; d++)
           {
            const float sign = (d % 2 == 0 ? 1.0f : -1.0f);
            gradient_values[d] = sign * 0.001f * (float)(1 + d % 31);
           }
         //--- A nonzero tail distinguishes full coverage from workgroup truncation.
         gradient_values[dimension - 1] = 0.75f;
         for(int d = 0; d < dimension; d++)
            gradient_norm_square += gradient_values[d] * gradient_values[d];
         const float gradient_norm = (float)MathSqrt(gradient_norm_square);
         float expected_scale = 0.0f;
         float expected_direction_norm_square = 0.0f;
         for(int d = 0; d < dimension; d++)
           {
            if(representation == D2SkillFullResidual)
               expected_correction[d] = beta_correction * gradient_values[d];
            else
              {
               expected_direction[d] = (1.0e-12f < gradient_norm ?
                                        beta_direction * gradient_values[d] / gradient_norm : 0.0f);
               expected_direction_norm_square += expected_direction[d] * expected_direction[d];
              }
           }
         if(representation == D2SkillDirectionMagnitude)
           {
            expected_scale = beta_magnitude * gradient_norm;
            const float expected_direction_norm = (float)MathSqrt(expected_direction_norm_square);
            for(int d = 0; d < dimension; d++)
               expected_correction[d] = (1.0e-12f < expected_direction_norm ?
                                         expected_scale * expected_direction[d] /
                                         expected_direction_norm : 0.0f);
           }
         if(!gradient.AssignArray(gradient_values) || !gradient.BufferWrite() ||
            !item.UpdateFromGradient(GetPointer(gradient)) ||
            !item.Correction().BufferRead() || !item.Direction().BufferRead() ||
            !item.Scale().BufferRead() || !item.Observations().BufferRead() ||
            !item.Mass().BufferRead())
            return(false);
         for(int component = 0; component < dimension; component++)
           {
            if(MathAbs(item.Correction()[component] - expected_correction[component]) > tolerance)
              {
               PrintFormat("D2SkillItemExperiment: coverage correction failed dimension=%d mode=%s component=%d actual=%.8f expected=%.8f",
                           dimension, mode_name, component, item.Correction()[component],
                           expected_correction[component]);
               return(false);
              }
            if(MathAbs(item.Direction()[component] - expected_direction[component]) > tolerance)
              {
               PrintFormat("D2SkillItemExperiment: coverage direction failed dimension=%d mode=%s component=%d actual=%.8f expected=%.8f",
                           dimension, mode_name, component, item.Direction()[component],
                           expected_direction[component]);
               return(false);
              }
           }
         const float expected_observations = 1.0f;
         const float expected_mass = gradient_norm;
         if(MathAbs(item.Scale()[0] - expected_scale) > tolerance ||
            MathAbs(item.Observations()[0] - expected_observations) > tolerance ||
            MathAbs(item.Mass()[0] - expected_mass) > tolerance)
           {
            PrintFormat("D2SkillItemExperiment: coverage scalar failed dimension=%d mode=%s scale=%.8f/%.8f observations=%.8f/%.8f mass=%.8f/%.8f",
                        dimension, mode_name, item.Scale()[0], expected_scale,
                        item.Observations()[0], expected_observations, item.Mass()[0], expected_mass);
            return(false);
           }
         PrintFormat("D2SkillItemExperiment: coverage dimension=%d mode=%s components=%d tail=%.8f scale=%.8f observations=%.0f mass=%.8f",
                     dimension, mode_name, dimension, item.Correction()[dimension - 1],
                     item.Scale()[0], item.Observations()[0], item.Mass()[0]);
        }
     }
   return(true);
  }
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!CreateOpenCLHost() ||
      !g_source.Init(0, 100, g_net.GetOpenCL(), D2_ITEM_DIM, ADAM, 1))
      return(INIT_FAILED);
   if(!RunArbitraryDimensionCoverage() ||
      !RunRepresentation(D2SkillFullResidual, 0.08f) ||
      !RunRepresentation(D2SkillDirectionMagnitude, 0.08f) ||
      !RunRepresentation(D2SkillDirectionMagnitude, 0.02f))
     {
      PrintFormat("D2SkillItemExperiment: FAILED error=%d", GetLastError());
      return(INIT_FAILED);
     }
   Print("D2SkillItemExperiment: PASS");
   ExpertRemove();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   g_source.Clear();
  }
//+------------------------------------------------------------------+
