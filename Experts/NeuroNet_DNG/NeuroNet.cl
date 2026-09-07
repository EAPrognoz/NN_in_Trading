//+------------------------------------------------------------------+
//|                                                       NeuroNet.cl|
//|                              OpenCL kernels library for NeuroNet.|
//|                                               Copyright 2024, DNG|
//|                                 https://www.mql5.com/ru/users/dng|
//+------------------------------------------------------------------+
/// \file
/// \brief NeuroNet.cl
/// Library consist OpenCL kernels
/// \author <A HREF="https://www.mql5.com/en/users/dng"> DNG </A>
/// \copyright Copyright 2019, DNG
//---
//--- by default some GPU doesn't support floats
//--- cl_khr_fp64 directive is used to enable work with floats
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define l1 1.0e-4f
#define l2 1.0e-4f
#define MAX_GRAD 1.0e-2f
#define LOCAL_ARRAY_SIZE 64
#define SCENARIO_LOCAL_CAP 256
#define MAX_VALUE 3.4e37f
#define MIN_VALUE -MAX_VALUE
#define RAG_MAX_VALUE 3.402823466e+38f
#define RAG_INVALID_SCORE (-RAG_MAX_VALUE)
#define D2SKILL_FREE       0
#define D2SKILL_CANDIDATE  1
#define D2SKILL_PROTECTED  2
#define D2SKILL_ACTIVE     3
#define D2SKILL_INACTIVE   4
//--- Activation Functions
#define ActFunc_None         -1
#define ActFunc_TANH          0
#define ActFunc_SIGMOID       1
#define ActFunc_LReLU         2
#define ActFunc_SoftPlus      3
#define ActFunc_GELU          4
#define ActFunc_MinusSoftPlus 5
#define ActFunc_ELU           6
//---
#define BarrierLoc      barrier(CLK_LOCAL_MEM_FENCE);
//+------------------------------------------------------------------+
//| Sanitizes the input, replacing NaN or infinite values with the   |
//| fallback value.                                                  |
//+------------------------------------------------------------------+
inline float IsNaNOrInf(const float value, const float def_value)
  {
   if(isnan(value) || isinf(value))
      return def_value;
   return value;
  }
//+------------------------------------------------------------------+
//| Sanitizes the input, replacing NaN or infinite values with the   |
//| fallback value.                                                  |
//+------------------------------------------------------------------+
inline float2 IsNaNOrInf2(const float2 value, const float2 def_value)
  {
   if(isnan(value.x) || isinf(value.x) ||
      isnan(value.y) || isinf(value.y))
      return def_value;
   return value;
  }
//+------------------------------------------------------------------+
//| Sanitizes the input, replacing NaN or infinite values with the   |
//| fallback value.                                                  |
//+------------------------------------------------------------------+
inline float4 IsNaNOrInf4(const float4 value, const float def_value)
  {
   float4 result;
   result.s0 = IsNaNOrInf(value.s0, def_value);
   result.s1 = IsNaNOrInf(value.s1, def_value);
   result.s2 = IsNaNOrInf(value.s2, def_value);
   result.s3 = IsNaNOrInf(value.s3, def_value);
   return result;
  }
//+------------------------------------------------------------------+
//| Applies the selected activation function to a scalar value.      |
//+------------------------------------------------------------------+
float fActivation(const float value, const int function)
  {
   float result = IsNaNOrInf(value, 0);
   switch(function)
     {
      case ActFunc_TANH:
         result = tanh(clamp(result, -20.0f, 20.0f));
         break;
      case ActFunc_SIGMOID:  //Sigmoid
         result = 1 / (1 + exp(clamp(-result, -20.0f, 20.0f)));
         break;
      case ActFunc_LReLU:  //LReLU
         if(result < 0)
            result *= 0.01f;
         break;
      case ActFunc_SoftPlus:  //SoftPlus
         result = (result >= 20.0f ? result : IsNaNOrInf(log(1 + exp(result)), 0));
         break;
      case ActFunc_GELU:  //GELU
         result = result / (1 + exp(clamp(-1.702f * result, -20.0f, 20.0f)));
         break;
      case ActFunc_MinusSoftPlus:  // -SoftPlus
         result = -fActivation(result, 3);
         break;
      case ActFunc_ELU:  //ELU
         if(result < 0)
            result = IsNaNOrInf(exp(result), 0) - 1;
         break;
      default:
         break;
     }
//---
   return IsNaNOrInf(result, 0);
  }
//+------------------------------------------------------------------+
//| Applies the derivative of the activation function to an output   |
//| gradient.                                                        |
//+------------------------------------------------------------------+
float Deactivation(const float grad, const float inp_value, const int function)
  {
   float result = IsNaNOrInf(grad, 0);
//---
   if(isnan(inp_value) || isinf(inp_value))
      result = 0;
   else
      switch(function)
        {
         case ActFunc_TANH: //TANH
            result = clamp(grad + inp_value, -1.0f, 1.0f) - inp_value;
            result *= 1.0f - inp_value * inp_value;
            break;
         case ActFunc_SIGMOID:  //Sigmoid
            result = clamp(grad + inp_value, 0.0f, 1.0f) - inp_value;
            result *= inp_value * (1.0f - inp_value);
            break;
         case ActFunc_LReLU: //LReLU
            if(inp_value < 0)
               result *= 0.01f;
            break;
         case ActFunc_SoftPlus:  //SoftPlus
            result *= (1.0f - exp(-inp_value));
            break;
         case ActFunc_GELU:  //GELU
            if(inp_value < 0.9f)
               result *= fActivation(5 * inp_value, 1);
            break;
         case ActFunc_MinusSoftPlus:  // -SoftPlus
            result = Deactivation(-result, -inp_value, ActFunc_SoftPlus);
            break;
         case ActFunc_ELU: //ELU
            if(inp_value < 0)
               result *= inp_value + 1;
            break;
         default:
            break;
        }
//---
   return clamp(IsNaNOrInf(result, 0), -MAX_GRAD, MAX_GRAD);
  }
//+------------------------------------------------------------------+
//| Maps the row/column indices of a tensor to its flat buffer       |
//| offset.                                                          |
//+------------------------------------------------------------------+
inline int RCtoFlat(const int row,
                    const int col,
                    const int total_rows,
                    const int total_cols,
                    const int variable)
  {
   return (variable * total_rows + row) * total_cols + col;
  }
//+------------------------------------------------------------------+
//| Implements the vget4 helper on the GPU.                          |
//+------------------------------------------------------------------+
inline float vget4(float4 v, int idx)
  {
   float val = 0.0f;
   switch(idx)
     {
      case 0:
         val = v.s0;
         break;
      case 1:
         val = v.s1;
         break;
      case 2:
         val = v.s2;
         break;
      case 3:
         val = v.s3;
         break;
     }
   return val;
  }
//+------------------------------------------------------------------+
//| Performs a local reduction (local max) over the work-group items.|
//+------------------------------------------------------------------+
float LocalMax(const float value, const int loc, __local float* Temp)
  {
   const size_t id = get_local_id(loc);
   const size_t total = get_local_size(loc);
//---
   const uint ls = min((uint)total, (uint)LOCAL_ARRAY_SIZE);
   float val = IsNaNOrInf(value, MIN_VALUE);
//--- Look Max
   if(id < ls)
      Temp[id] = val;
   BarrierLoc
   for(int d = ls; d < total; d += ls)
     {
      if(id >= d && id < (d + ls) &&
         (Temp[id - d] < val))
         Temp[id - d] = val;
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(id < count && (id + count) < ls && Temp[id] < Temp[id + count])
         Temp[id] = Temp[id + count];
      BarrierLoc
     }
   while(count > 1);
//---
   return Temp[0];
  }
//+------------------------------------------------------------------+
//| Performs a local reduction (local min) over the work-group items.|
//+------------------------------------------------------------------+
float LocalMin(const float value, const int loc, __local float* Temp)
  {
   const size_t id = get_local_id(loc);
   const size_t total = get_local_size(loc);
//---
   const uint ls = min((uint)total, (uint)LOCAL_ARRAY_SIZE);
   float val = IsNaNOrInf(value, MAX_VALUE);
//--- Look Min
   if(id  < ls)
      Temp[id] = val;
   BarrierLoc
   for(int d = ls; d < total; d += ls)
     {
      if(id >= d && id < (d + ls) &&
         (Temp[id - d] > val))
         Temp[id - d] = val;
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(id < count && (id + count) < ls && Temp[id] > Temp[id + count])
         Temp[id] = Temp[id + count];
      BarrierLoc
     }
   while(count > 1);
//---
   return Temp[0];
  }
//+------------------------------------------------------------------+
//| Performs a local reduction (local sum) over the work-group items.|
//+------------------------------------------------------------------+
float LocalSum(const float value, const int loc, __local float* Temp)
  {
   const size_t id = get_local_id(loc);
   const size_t total = get_local_size(loc);
//---
   if(total <= 1)
      return IsNaNOrInf(value, 0.0f);
//---
   const uint ls = min((uint)total, (uint)LOCAL_ARRAY_SIZE);
//--- Sum
   float result = IsNaNOrInf(value, 0);
   if(id  < ls)
      Temp[id] = result;
   BarrierLoc
   for(int d = ls; d < total; d += ls)
     {
      if(id >= d && id < (d + ls))
         Temp[id - d] = Temp[id - d] + result;
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(id < count && (id + count) < ls)
        {
         Temp[id] += Temp[id + count];
         Temp[id + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
   result = IsNaNOrInf(Temp[0], 0);
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Performs a local reduction (local sum4) over the work-group      |
//| items.                                                           |
//+------------------------------------------------------------------+
float4 LocalSum4(const float4 value, const int loc, __local float4* Temp)
  {
   const size_t id = get_local_id(loc);
   const size_t total = get_local_size(loc);
//---
   if(total <= 1)
      return IsNaNOrInf4(value, 0.0f);
//---
   const uint ls = min((uint)total, (uint)LOCAL_ARRAY_SIZE);
//---
   float4 result = IsNaNOrInf4(value, 0.0f);
   if(id  < ls)
      Temp[id] = result;
   BarrierLoc
   for(int d = ls; d < total; d += ls)
     {
      if(id >= d && id < (d + ls))
         Temp[id - d] = Temp[id - d] + result;
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(id < count && (id + count) < ls)
        {
         Temp[id] += Temp[id + count];
         Temp[id + count] = (float4)0.0f;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   result = IsNaNOrInf4(Temp[0], 0.0f);
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Computes the numerically stable softmax over the local work-group|
//| items.                                                           |
//+------------------------------------------------------------------+
float LocalSoftMax(const float value, const int loc, __local float* Temp)
  {
//--- Look Max
   float max = LocalMax(value, loc, Temp);
   if(max == MIN_VALUE)
      return 0.0f;
//--- SoftMax
   float result = (value == MIN_VALUE ? 0.0f : IsNaNOrInf(exp(value - max), 0.0f));
   const float sum = LocalSum(result, loc, Temp);
   if(sum == 0.0f)
      result = 0;
   else
      result = IsNaNOrInf(result / sum, 0.0f);
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Computes the local softmax gradient contribution for one item.   |
//+------------------------------------------------------------------+
float LocalSoftMaxGrad(const float value, const float grad, const int loc, __local float* Temp)
  {
   const float y = IsNaNOrInf(value, 0.0f);
   const float g = IsNaNOrInf(grad, 0.0f);
   const float s = LocalSum(y * g, loc, Temp);
//--- d_i = y_i * (g_i - sum_j(y_j * g_j))
   return IsNaNOrInf(y * (g - s), 0.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_ff Feed forward process kernel
/// Describes the forward path process for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para41">the link.</A>
//+------------------------------------------------------------------+
__kernel void FeedForward(__global const float *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - number
                          ///< of neurons in layer and n - number of outputs
                          ///< (neurons in next layer)
                          __global const float *matrix_i, ///<[in] Inputs tensor
                           __global float *matrix_o, ///<[out] Output tensor
                           const int inputs,               ///< Number of inputs
                           const int activation,           ///< Activation type (#ENUM_ACTIVATION)
                           __global float *numeric_flags,  ///<[out] One status per output
                           const int flags_enabled
                          )
  {
   const int i = get_global_id(0);
   const int total_out = get_global_size(0);
   const int loc = get_local_id(1);
    const int total_loc = get_local_size(1);
//---
    __local float Temp[LOCAL_ARRAY_SIZE];
    __local float FlagTemp[LOCAL_ARRAY_SIZE];
//---
    float sum = 0;
    float inp;
    float invalid = 0;
    int shift = RCtoFlat(i, 0, total_out, (inputs + 1), 0);
    for(int k = loc; k < inputs; k += total_loc)
      {
       if(flags_enabled)
         {
          const float raw_inp = matrix_i[k];
          const float raw_weight = matrix_w[shift + k];
          const float raw_product = raw_inp * raw_weight;
          invalid += (isnan(raw_inp) || isinf(raw_inp) || isnan(raw_weight) || isinf(raw_weight) ||
                      isnan(raw_product) || isinf(raw_product));
         }
       inp = IsNaNOrInf(matrix_i[k], 0.0f);
       if(inp == 0.0f)
          continue;
       sum += IsNaNOrInf(inp * matrix_w[shift + k], 0.0f);
      }
    if(loc == 0)
      {
       if(flags_enabled)
         {
          const float raw_bias = matrix_w[shift + inputs];
          invalid += isnan(raw_bias) || isinf(raw_bias);
         }
       sum += IsNaNOrInf(matrix_w[shift + inputs], 0.0f);
      }
    if(flags_enabled)
       invalid += isnan(sum) || isinf(sum);
    if(total_loc > 1)
       sum = LocalSum(sum, 1, Temp);
    if(flags_enabled)
      {
       if(total_loc > 1)
          invalid += isnan(Temp[0]) || isinf(Temp[0]);
       invalid = LocalSum(invalid, 1, FlagTemp);
      }
//---
    if(loc == 0)
      {
       if(flags_enabled)
          invalid += isnan(sum) || isinf(sum);
//--- Preserve the activation result for the numerical flag before storing it.
       const float activation_value = fActivation(sum, activation);
       if(flags_enabled)
         {
          invalid += isnan(activation_value) || isinf(activation_value);
          numeric_flags[i] = (invalid > 0.0f ? 1.0f : 0.0f);
         }
       matrix_o[i] = activation_value;
      }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Output Gradients Calculation kernel
/// Describes the process of output gradients calculation for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcOutputGradient(__global float *matrix_t,  ///<[in] Target tensor
                                 __global float *matrix_o,  ///<[in] Output tensor
                                 __global float *matrix_ig, ///<[out] Tensor of gradients
                                 int activation, ///< Activation type (#ENUM_ACTIVATION)
                                 float error)
  {
   int i = get_global_id(0);
   float out = matrix_o[i];
   float temp = 0;
   if(!isnan(out) && !isinf(out))
      temp = Deactivation(matrix_t[i] - out, out, activation);
   matrix_ig[i] = temp;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Hidden Gradients Calculation kernel
/// Describes the process of hidden gradients calculation for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradient(__global float *matrix_w,  ///<[in] Weights matrix (m+1)*n, where m - number
                                 ///< of neurons in previous layer and n - number
                                 ///< of neurons in current layer
                                 __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                 __global float *matrix_o,  ///<[in] Previous layer Output tensor
                                 __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                 int outputs,               ///< Number of outputs
                                 int activation             ///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   const int i = get_global_id(0);
   const int inputs = get_global_size(0);
   const int loc = get_local_id(1);
   const int total_loc = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   float sum = 0;
   float out = matrix_o[i];
   float4 grad, weight;
   for(int k = 4 * loc; k < outputs; k += 4 * total_loc)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2],
                            matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i],
                              matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      //---
      weight = IsNaNOrInf4(weight, 0);
      grad = IsNaNOrInf4(grad, 0);
      //---
      sum += dot(grad, weight);
     }
   if(total_loc > 1)
      sum = LocalSum(sum, 1, Temp);
//---
   matrix_ig[i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base SGD Updating Weights Calculation kernel
/// Describes the process of SGD optimization weights for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMomentum(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< number of neurons in previous layer and n -
                                    ///< number of neurons in current layer
                                    __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
                                    int inputs,     ///< Number of inputs
                                    float learning_rates, ///< Learning rates
                                    float momentum        ///< Momentum multiplier
                                   )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   int wi = i * (inputs + 1) + j;
   float grad = clamp(matrix_g[i], -MAX_GRAD, MAX_GRAD);
   float delta = IsNaNOrInf(learning_rates * grad * (j < inputs ? matrix_i[j] : 1), 0) +
                 IsNaNOrInf(momentum * matrix_dw[wi], 0);
   matrix_dw[wi] = delta;
   if(fabs(delta) > 0)
      matrix_w[wi] = IsNaNOrInf(matrix_w[wi] + delta, 0);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Adam Updating Weights Calculation
/// kernel
/// Describes the process of Adam optimization weights for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdam(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                ///< number of neurons in previous layer and n -
                                ///< number of neurons in current layer
                                __global const float
                                *matrix_g, ///<[in] Tensor of gradients at current layer
                                __global const float *matrix_i, ///<[in] Inputs tensor
                                __global float *matrix_m,       ///<[in,out] Matrix of first momentum
                                __global float *matrix_v,       ///<[in,out] Matrix of seconfd momentum
                                const int inputs,               ///< Number of inputs
                                const float l,                  ///< Learning rates
                                const float b1,                 ///< First momentum multiplier
                                const float b2                  ///< Second momentum multiplier
                               )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j;
   float m, v, weight, inp;
   inp = IsNaNOrInf((j == inputs ? 1.0f : matrix_i[j]), 0);
   weight = IsNaNOrInf(matrix_w[wi], 0);
   m = IsNaNOrInf(matrix_m[wi], 0);
   v = IsNaNOrInf(matrix_v[wi], 0);
//---
   float g = clamp(IsNaNOrInf(matrix_g[i] * inp, 0), -MAX_GRAD, MAX_GRAD);
   float mt = IsNaNOrInf(b1 * m + (1 - b1) * g, 0);
   float vt = IsNaNOrInf(b2 * v + (1 - b2) * (g * g), 0);
   float delta =
      IsNaNOrInf(l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight)), 0);
   if(fabs(delta) > 0)
      matrix_w[wi] = IsNaNOrInf(matrix_w[wi] + delta, 0);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Least Squares Updating Weights
/// Calculation kernel
/// Describes the process of Least Squares optimization weights for the Neuron
/// Base (#CNeuronBaseOCL).
//\details Detailed description on <A
// HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights ls step using|
//| the optimizer rule.                                              |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsLS(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                              ///< number of neurons in previous layer and n -
                              ///< number of neurons in current layer
                              __global const float
                              *matrix_g, ///<[in] Tensor of gradients at current layer
                              __global const float *matrix_i, ///<[in] Inputs tensor
                              __global float *matrix_xg,      ///<[in,out] Matrix of summ x*g
                              __global float *matrix_xx,      ///<[in,out] Matrix of summ x*x
                              const int inputs,               ///< Number of inputs
                              const float l,                  ///< Learning rates
                              const int update                ///< Update flag
                             )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j * 4;
   float4 xg, xx, weight, inp;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[j * 4], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         break;
      case 3:
         inp =
            (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2],
                        matrix_i[j * 4 + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         break;
     }
   xg = (float4)(matrix_g[i]) * inp;
   xx = inp * inp;
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(update)
           {
            matrix_w[wi + 3] =
               matrix_w[wi + 3] + l * (matrix_xg[wi + 3] + xg.s3) /
               (matrix_xx[wi + 3] + xx.s3 + 1.0e-37f);
            matrix_xg[wi + 3] = 0;
            matrix_xx[wi + 3] = 0;
           }
         else
           {
            matrix_xg[wi + 3] += xg.s3;
            matrix_xx[wi + 3] += xx.s3;
           }
      case 2:
         if(update)
           {
            matrix_w[wi + 2] =
               matrix_w[wi + 2] + l * (matrix_xg[wi + 2] + xg.s2) /
               (matrix_xx[wi + 2] + xx.s2 + 1.0e-37f);
            matrix_xg[wi + 2] = 0;
            matrix_xx[wi + 2] = 0;
           }
         else
           {
            matrix_xg[wi + 2] += xg.s2;
            matrix_xx[wi + 2] += xx.s2;
           }
      case 1:
         if(update)
           {
            matrix_w[wi + 1] =
               matrix_w[wi + 1] + l * (matrix_xg[wi + 1] + xg.s1) /
               (matrix_xx[wi + 1] + xx.s1 + 1.0e-37f);
            matrix_xg[wi + 1] = 0;
            matrix_xx[wi + 1] = 0;
           }
         else
           {
            matrix_xg[wi + 1] += xg.s1;
            matrix_xx[wi + 1] += xx.s1;
           }
      case 0:
         if(update)
           {
            matrix_w[wi] = matrix_w[wi] + l * (matrix_xg[wi] + xg.s0) /
                           (matrix_xx[wi] + xx.s0 + 1.0e-37f);
            matrix_xg[wi] = 0;
            matrix_xx[wi] = 0;
           }
         else
           {
            matrix_xg[wi] += xg.s0;
            matrix_xx[wi] += xx.s0;
           }
         break;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_ff
/// Kernel of the Pooling neuron for Feed forward process (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardProof(__global float *matrix_i, ///<[in] Inputs tensor
                               __global float *matrix_o, ///<[out] Output tensor
                               int inputs,               ///< Number of inputs
                               int window, ///< Size of input window
                               int step    ///< Step size
                              )
  {
   int i = get_global_id(0);
   int pos = i * step;
   float result = matrix_i[pos];
//---
   for(int k = 1; k < window; k++)
     {
      int shift = k + pos;
      if(shift >= inputs)
         break;
      result = max(result, matrix_i[shift]);
     }
   matrix_o[i] = result;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_gr
/// Kernel of the Pooling neuron to transfer gradient to previous layer
/// (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void CalcInputGradientProof(__global float *matrix_i,  ///<[in] Inputs tensor
                                     __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,  ///<[in] Output tensor
                                     __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                     int outputs,               ///< Number of outputs
                                     int window,                ///< Size of input window
                                     int step                   ///< Step size
                                    )
  {
   int i = get_global_id(0);
   float prev_gradient = 0;
   float value = matrix_i[i];
   int start = i - window + step;
   start = (start - start % step) / step;
   int stop = (i - i % step) / step + 1;
   for(int out = max(0, start); out < min(outputs, stop); out++)
     {
      if(value == matrix_o[out])
         prev_gradient += matrix_g[out];
     }
   matrix_ig[i] = prev_gradient;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_ff
/// Kernel of the Convolution neuron for Feed forward process (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardConv(__global const float *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - input
                              ///< window and n - output window
                              __global const float *matrix_i, ///<[in] Inputs tensor
                              __global float *matrix_o, ///<[out] Output tensor
                              const int inputs,         ///< Number of inputs
                              const int step,           ///< Step size
                              const int window_in,      ///< Size of input window
                              const int window_out,     ///< Size of output window
                              const int activation      ///< Activation type (#ENUM_ACTIVATION)
                             )
  {
   const size_t i = get_global_id(0);
   const int out = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t outputs = get_global_size(0);
//---
   const int shift_out = window_out * i;
   const int shift_in = step * i;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * window_out * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
//---
   float sum = 0;
   float inp;
//---
   int shift = (window_in + 1) * out;
   int stop = (window_in <= (inputs - shift_in) ? window_in : (inputs - shift_in));
   for(int k = 0; k < stop; k ++)
     {
      inp = IsNaNOrInf(matrix_i[shift_var_in + shift_in + k], 0.0f);
      if(inp == 0.0f)
         continue;
      sum += IsNaNOrInf(inp * matrix_w[shift_var_w + shift + k], 0.0f);
     }
   sum += IsNaNOrInf(matrix_w[shift_var_w + shift + window_in], 0.0f);
//---
   matrix_o[shift_var_out + out + shift_out] = fActivation(sum, activation);;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_gr
/// Kernel of the Convolution neuron to transfer gradient
/// to previous layer (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientConv(__global float *matrix_w,       ///<[in] Weights matrix (m+1)*n, where m - input
                                     ///<     window and n - output window
                                     __global float *matrix_g,       ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,       ///<[in] Output tensor
                                     __global float *matrix_ig,      ///<[out] Tensor of gradients at previous layer
                                     const int outputs,              ///< Number of outputs
                                     const int step,                 ///< Step size
                                     const int window_in,            ///< Size of input window
                                     const int window_out,           ///< Size of output window
                                     const int activation,           ///< Activation type (#ENUM_ACTIVATION)
                                     const int shift_out             ///< Shift in output and gradient buffer
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t inputs = get_global_size(0);
   const size_t v = get_global_id(1);
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
//---
   float sum = 0;
   float out = matrix_o[shift_var_in + i];
   const int w_start = i % step;
   const int start = max((int)((i - window_in + step) / step), 0);
   int stop = (w_start + step - 1) / step;
   stop = min((int)((i + step - 1) / step + 1), stop) + start;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
   for(int h = 0; h < window_out; h ++)
     {
      for(int k = start; k < stop; k++)
        {
         int shift_g = k * window_out + h;
         int shift_w = (stop - k - 1) * step + i % step + h * (window_in + 1);
         if(shift_g >= outputs || shift_w >= (window_in + 1) * window_out)
            break;
         float grad = IsNaNOrInf(matrix_g[shift_out + shift_g + shift_var_out], 0.0f);
         if(fabs(grad) > 0.0f)
            sum += IsNaNOrInf(grad * matrix_w[shift_w + shift_var_w], 0.0f);
        }
     }
//---
   matrix_ig[shift_var_in + i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron SGD optimization Updating Weights
/// Calculation kernel
/// Describes the process of SGD optimization weights for the Convolution Neuron
/// (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvMomentum(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                        ///< input window and n - output window
                                        __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                        __global float *matrix_i, ///<[in] Inputs tensor
                                        __global float
                                        *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
                                        int inputs,     ///< Number of inputs
                                        float learning_rates, ///< Learning rates
                                        float momentum,       ///< Momentum multiplier
                                        int window_in,        ///< Size of input window
                                        int window_out,       ///< Size of output window
                                        int step              ///< Step size
                                       )
  {
   const size_t i = get_global_id(0);
//---
   const int v = i / ((window_in + 1) * window_out);
   const int shift = i % (window_in + 1);
   const int shift_out = i / (window_in + 1) - v;
   const int total = (inputs - window_in + step - 1) / step;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
//---
   float grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += IsNaNOrInf(matrix_g[t * window_out + shift_out + shift_var_out] *
                         (shift == window_in ? 1 : matrix_i[shift + t * step + shift_var_in]),
                         0.0f);
     }
   float delta = IsNaNOrInf(learning_rates * grad, 0) + momentum * matrix_dw[i];
   if(!isnan(delta))
     {
      matrix_dw[i] = delta;
      if(fabs(delta) > 0)
         matrix_w[i] = IsNaNOrInf(matrix_w[i] + delta, 0);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Adam optimization Updating
/// Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Convolution
/// Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvAdam(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< input window and n - output window
                                    __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_m,       ///<[in] Matrix of first momentum
                                    __global float *matrix_v,       ///<[in] Matrix of seconfd momentum
                                    const int inputs,               ///< Number of inputs
                                    const float l,                  ///< Learning rates
                                    const float b1,                 ///< First momentum multiplier
                                    const float b2,                 ///< Second momentum multiplier
                                    int window_in,                  ///< Size of input window
                                    int window_out,                 ///< Size of output window
                                    int step                        ///< Step size
                                   )
  {
   const size_t i = get_global_id(0);
//---
   const int v = i / ((window_in + 1) * window_out);
   const int shift = i % (window_in + 1);
   const int shift_out = i / (window_in + 1) - v * window_out;
   const int total = (inputs - (window_in - step) + (step - 1)) / step;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
//---
   float grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += IsNaNOrInf(matrix_g[t * window_out + shift_out + shift_var_out] *
                         (shift == window_in ? 1 : matrix_i[shift + t * step + shift_var_in]), 0);
     }
   grad = clamp(IsNaNOrInf(grad, 0), -MAX_GRAD, MAX_GRAD);
   float mt = IsNaNOrInf(b1 * matrix_m[i] + (1 - b1) * grad, 0);
   float vt = IsNaNOrInf(b2 * matrix_v[i] + (1 - b2) * (grad * grad), 1.0f);
   float weight = IsNaNOrInf(matrix_w[i] + IsNaNOrInf(l * mt / sqrt(vt), 0), 0);
   matrix_w[i] = weight;
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Least Squares optimization
/// Updating Weights Calculation kernel
/// Describes the process of Least Squares optimization weights for the
/// Convolution Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvLS(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                  ///< input window and n - output window
                                  __global const float
                                  *matrix_g, ///<[in] Tensor of gradients at current layer
                                  __global const float *matrix_i, ///<[in] Inputs tensor
                                  __global float *matrix_xg,      ///<[in] Matrix of summ x*g
                                  __global float *matrix_xx,      ///<[in] Matrix of summ x*x
                                  const int inputs,               ///< Number of inputs
                                  const float l,                  ///< Learning rates
                                  const int update,               ///< Update flag
                                  int window_in,                  ///< Size of input window
                                  int window_out,                 ///< Size of output window
                                  int step                        ///< Step size
                                 )
  {
   const int i = get_global_id(0);
   if(i > window_in)
      return;
//---
   int total = (inputs - (window_in - step)) % step;
   total = (inputs - (window_in - step) - total) / step + (total > 0 ? 1 : 0);
//---
   for(int out = 0; out < window_out; out++)
     {
      if((window_out - out) > 4)
        {
         float4 xg = {0, 0, 0, 0};
         float x2 = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += (float4)(matrix_g[t * window_out + out],
                           matrix_g[t * window_out + out + 1],
                           matrix_g[t * window_out + out + 2],
                           matrix_g[t * window_out + out + 3]) *
                  (i == window_in ? 1 : matrix_i[i + t * step]);
            x2 += (i == window_in ? 1 : matrix_i[i + t * step] * matrix_i[i + t * step]);
           }
         if(update)
           {
            xg = (float4)(matrix_xg[shift_w], matrix_xg[shift_w + window_in + 1],
                          matrix_xg[shift_w + 2 * (window_in + 1)],
                          matrix_xg[shift_w + 3 * (window_in + 1)]) +
                 xg;
            float4 xx =
               (float4)(matrix_xx[shift_w], matrix_xx[shift_w + window_in + 1],
                        matrix_xx[shift_w + 2 * (window_in + 1)],
                        matrix_xx[shift_w + 3 * (window_in + 1)]) +
               x2;
            float4 delta = l * xg / (xx + 1.0e-37f);
            float4 weight =
               (float4)(matrix_w[shift_w], matrix_w[shift_w + (window_in + 1)],
                        matrix_w[shift_w + 2 * (window_in + 1)],
                        matrix_w[shift_w + 3 * (window_in + 1)]) +
               delta;
            matrix_w[shift_w] = weight.s0;
            matrix_w[shift_w + (window_in + 1)] = weight.s1;
            matrix_w[shift_w + 2 * (window_in + 1)] = weight.s2;
            matrix_w[shift_w + 3 * (window_in + 1)] = weight.s3;
            matrix_xg[shift_w] = 0;
            matrix_xg[shift_w + (window_in + 1)] = 0;
            matrix_xg[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xg[shift_w + 3 * (window_in + 1)] = 0;
            matrix_xx[shift_w] = 0;
            matrix_xx[shift_w + (window_in + 1)] = 0;
            matrix_xx[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xx[shift_w + 3 * (window_in + 1)] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg.s0;
            matrix_xg[shift_w + (window_in + 1)] += xg.s1;
            matrix_xg[shift_w + 2 * (window_in + 1)] += xg.s2;
            matrix_xg[shift_w + 3 * (window_in + 1)] += xg.s3;
            matrix_xx[shift_w] = matrix_xx[shift_w + (window_in + 1)] =
                                    matrix_xx[shift_w + 2 * (window_in + 1)] =
                                       matrix_xx[shift_w + 3 * (window_in + 1)] += x2;
           }
         out += 3;
        }
      else
        {
         float xg = 0;
         float xx = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += matrix_g[t * window_out + out] *
                  (i == window_in ? 1 : matrix_i[i + t * step]);
            xx += (i == window_in ? 1 : matrix_i[i + t * step] * matrix_i[i + t * step]);
           }
         if(update)
           {
            xg = matrix_xg[shift_w] + xg;
            xx = matrix_xx[shift_w] + xx;
            float delta = l * xg / (xx + 1.0e-37f);
            matrix_w[shift_w] = matrix_w[shift_w] + delta;
            matrix_xg[shift_w] = 0;
            matrix_xx[shift_w] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg;
            matrix_xx[shift_w] += xx;
           }
        }
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Score calculation kernel |
/// Describes the Score calculation process for the Neuron of attention layer
/// (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionScore(__global float *querys, ///<[in] Matrix of Querys
                             __global float *keys,   ///<[in] Matrix of Keys
                             __global float *score,  ///<[out] Matrix of Scores
                             int dimension,          ///< Dimension of Key
                             int mask ///< 1 - calc only previous units, 0 - calc all
                            )
  {
   int q = get_global_id(0);
   int shift_q = q * dimension;
   int units = get_global_size(0);
   int shift_s = q * units;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
//---
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = k * dimension;
      for(int i = 0; i < dimension; i++)
         result += (querys[shift_q + i] * keys[shift_k + i]);
      result = IsNaNOrInf(exp(result / koef), 0);
      score[shift_s + k] = result;
      sum += result;
     }
//---
   for(int k = 0; (k < units && sum > 0); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Out calculation kernel
/// Describes the Attention out calculation process for the Neuron of attention
/// layer (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionOut(__global float *scores, ///<[in] Matrix of Scores
                           __global float *values, ///<[in] Matrix of Values
                           __global float *inputs, ///<[in] Inputs tensor
                           __global float *out     ///<[out] Output tensor
                          )
  {
   int units = get_global_size(0);
   int u = get_global_id(0);
   int d = get_global_id(1);
   int dimension = get_global_size(1);
   int shift = u * dimension + d;
   float result = 0;
//---
   for(int i = 0; i < units; i++)
      result += IsNaNOrInf(scores[u * units + i], 0) * IsNaNOrInf(values[i * dimension + d], 0);
   out[shift] = IsNaNOrInf(result, 0) + inputs[shift];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 2 matrixs with
/// multiplyer.
/// Describes the calculation Sum of 2 matrixs.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void SumMatrix(__global float *matrix1,    ///<[in] First matrix
                        __global float *matrix2,    ///<[in] Second matrix
                        __global float *matrix_out, ///<[out] Output matrix
                        int dimension,              ///< Dimension of matrix
                        float multiplyer,           ///< Multiplyer for output
                        int shift_in1,              ///< Shift for input 1
                        int shift_in2,              ///< Shift for input 2
                         int shift_out,              ///< Shift for output
                         __global float *numeric_flags,
                         const int flags_enabled
                       )
  {
   const int i = get_global_id(0);
   const int step = get_global_size(0);
//---
    for(int k = 0; k < dimension; k++)
      {
       int index = i * dimension + k;
       if(flags_enabled)
         {
          const float value1 = matrix1[i * shift_in1 + index];
          const float value2 = matrix2[i * shift_in2 + index];
          const float raw_sum = value1 + value2;
          const float raw_result = raw_sum * multiplyer;
          const int invalid = isnan(value1) || isinf(value1) || isnan(value2) || isinf(value2) ||
                              isnan(raw_sum) || isinf(raw_sum) || isnan(raw_result) || isinf(raw_result);
          numeric_flags[index] = (invalid ? 1.0f : 0.0f);
         }
       matrix_out[i * shift_out + index] =
          IsNaNOrInf((matrix1[i * shift_in1 + index] + matrix2[i * shift_in2 + index]) * multiplyer, 0);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 4 matrixs with
/// multiplyer.
/// Describes the calculation Sum of 4 matrixs.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
//+------------------------------------------------------------------+
__kernel void Sum5Matrix(__global float *matrix1,    ///<[in] First matrix
                         __global float *matrix2,    ///<[in] Second matrix
                         __global float *matrix3,    ///<[in] Third matrix
                         __global float *matrix4,    ///<[in] Fourth matrix
                         __global float *matrix5,    ///<[in] Fifth matrix
                         __global float *matrix_out, ///<[out] Output matrix
                         int dimension,              ///< Dimension of matrix
                         float multiplyer            ///< Multiplyer for output
                        )
  {
   const int i = get_global_id(0) * dimension;
//---
   for(int k = 0; k < dimension; k++)
      matrix_out[i + k] = (matrix1[i + k] + matrix2[i + k] + matrix3[i + k] +
                           matrix4[i + k] + matrix5[i + k]) *
                          multiplyer;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation
/// kernel
/// Describes the gradients calculation process for the Neuron of attention
/// layer (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para44">the link.</A>
/// @param[in] querys Matrix of Querys
/// @param[out] querys_g Matrix of Querys' Gradients
/// @param[in] keys Matrix of Keys
/// @param[out] keys_g Matrix of Keys' Gradients
/// @param[in] values Matrix of Values
/// @param[out] values_g Matrix of Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] gradient Matrix of Gradients from previous iteration
//+------------------------------------------------------------------+
__kernel void AttentionInsideGradients(__global float *querys, __global float *querys_g,
                                       __global float *keys, __global float *keys_g,
                                       __global float *values, __global float *values_g,
                                       __global float *scores, __global float *gradient)
  {
   int u = get_global_id(0);
   int d = get_global_id(1);
   int units = get_global_size(0);
   int dimension = get_global_size(1);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float vg = 0;
   float qg = 0;
   float kg = 0;
//---
   for(int iu = 0; iu < units; iu++)
     {
      float g = gradient[iu * dimension + d];
      float sc = scores[iu * units + u];
      vg += sc * g;
      //---
      float sqg = 0;
      float skg = 0;
      for(int id = 0; id < dimension; id++)
        {
         sqg += values[iu * dimension + id] * gradient[u * dimension + id];
         skg += values[u * dimension + id] * gradient[iu * dimension + id];
        }
      qg += (scores[u * units + iu] == 0 || scores[u * units + iu] == 1
             ? 0.0001f
             : scores[u * units + iu] * (1 - scores[u * units + iu])) *
            sqg * keys[iu * dimension + d] / koef;
      //---
      kg += (scores[iu * units + u] == 0 || scores[iu * units + u] == 1
             ? 0.0001f
             : scores[iu * units + u] * (1 - scores[iu * units + u])) *
            skg * querys[iu * dimension + d] / koef;
     }
   int shift = u * dimension + d;
   values_g[shift] = clamp(IsNaNOrInf(vg, 0.0f), -1.0f, 1.0f);
   querys_g[shift] = clamp(IsNaNOrInf(qg, 0.0f), -1.0f, 1.0f);
   keys_g[shift] = clamp(IsNaNOrInf(kg, 0.0f), -1.0f, 1.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of matrix normalization process
/// Describes the process of matrix normalization.
///\details Detailed description on <A
/// HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void Normalize(__global float *buffer, int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   if(dimension < 1)
      return;
//---
   float mean = 0;
   float M2 = 0;
   float variance = 0;
//---
   for(int i = 0; i < dimension; i++)
     {
      float val = IsNaNOrInf(buffer[shift + i], 0);
      double delta = val - mean;
      mean += delta / (i + 1);
      M2 += delta * (val - mean);
     }
   variance = M2 / (dimension - 1);
//---
   for(int i = 0; i < dimension; i++)
      if(variance > 1)
         buffer[shift + i] =
            IsNaNOrInf((buffer[shift + i] - mean) / variance, 0);
      else
         buffer[shift + i] =
            IsNaNOrInf(buffer[shift + i] - mean, 0);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of weights matrix normalization process
/// Describes the process of weights matrix normalization.
///\details Detailed description on <A
/// HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void NormalizeWeights(__global float *buffer, int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   float sum = 0;
   float k = 1;
//---
   do
     {
      for(int i = 0; (i < dimension && !(isnan(sum) || !isinf(sum))); i++)
        {
         float normalized = IsNaNOrInf(buffer[shift + i], 0) / k;
         sum = normalized * normalized / dimension;
        }
      if(isnan(sum) || isinf(sum))
         k *= 10;
     }
   while(isnan(sum) || isinf(sum));
   sum = sqrt(sum);
   if(k * sum > 1)
      for(int i = 0; i < dimension; i++)
         buffer[shift + i] = IsNaNOrInf(buffer[shift + i], 0) / (k * sum);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff
/// Describes the process of concatenate 4 matrices.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para52">the link.</A>
/// @param[in] input1, input2, input3, input4 Input buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] output Output buffer
//+------------------------------------------------------------------+
__kernel void ConcatenateBuffers(__global float *input1, int window1,
                                 __global float *input2, int window2,
                                 __global float *input3, int window3,
                                 __global float *input4, int window4,
                                 __global float *output)
  {
   int n = get_global_id(0);
   int w = get_global_id(1);
   int total_w = get_global_size(1);
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_in = n * window1;
   for(int i = w; i < window1; i += total_w)
      output[shift + i] = IsNaNOrInf(input1[shift_in + i], 0);
//---
   shift += window1;
   shift_in = n * window2;
//---
   for(int i = w; i < window2; i += total_w)
      output[shift + i] = IsNaNOrInf(input2[shift_in + i], 0);
//---
   shift += window2;
   shift_in = n * window3;
//---
   for(int i = w; i < window3; i += total_w)
      output[shift + i] = IsNaNOrInf(input3[shift_in + i], 0);
//---
   shift += window3;
   shift_in = n * window4;
//---
   for(int i = w; i < window4; i += total_w)
      output[shift + i] = IsNaNOrInf(input4[shift_in + i], 0);
  }
//+------------------------------------------------------------------+
//| Copies raw 32-bit buffer payloads without float sanitization.    |
//+------------------------------------------------------------------+
__kernel void CopyBufferRaw(__global const uint *source,
                            __global uint *destination,
                            const int total)
  {
   const int index = get_global_id(0);
   if(index < total)
      destination[index] = source[index];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr
/// Describes the process of deconcatenate matrix.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
/// @param[in] output1, output2, output3, output4 Output buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] inputs Input buffer
//+------------------------------------------------------------------+
__kernel void DeconcatenateBuffers(__global float *output1, int window1,
                                   __global float *output2, int window2,
                                   __global float *output3, int window3,
                                   __global float *output4, int window4,
                                   __global float *inputs)
  {
   int n = get_global_id(0);
   int w = get_global_id(1);
   int total_w = get_global_size(1);
//--- Head 1
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_out = n * window1;
//---
   for(int i = w; i < window1; i += total_w)
      output1[shift_out + i] = IsNaNOrInf(inputs[shift + i], 0);
//--- Head 2
   shift += window1;
   shift_out = n * window2;
//---
   for(int i = w; i < window2; i += total_w)
      output2[shift_out + i] = IsNaNOrInf(inputs[shift + i], 0);
//--- Head 3
   shift += window2;
   if(window3 > 0)
     {
      shift_out = n * window3;
      //---
      for(int i = w; i < window3; i += total_w)
         output3[shift_out + i] = IsNaNOrInf(inputs[shift + i], 0);
     }
//--- Head 4
   shift += window3;
   if(window4 > 0)
     {
      shift_out = n * window4;
      //---
      for(int i = w; i < window4; i += total_w)
         output4[shift_out + i] = IsNaNOrInf(inputs[shift + i], 0);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-Heads Attention Neuron Score calculation
/// kernel
/// Describes the Score calculation process for the Neuron of multi-heads
/// attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionScore(__global float *qkv,   ///<[in] Matrix of Querys, Keys, Values
                               __global float *score, ///<[out] Matrix of Scores
                               int dimension,         ///< Dimension of Key
                               int mask ///< 1 - calc only previous units, 0 - calc all
                              )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
//---
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot(IsNaNOrInf4((float4)(qkv[shift_q + i], qkv[shift_q + i + 1],
                                               qkv[shift_q + i + 2], qkv[shift_q + i + 3]), 0),
                          IsNaNOrInf4((float4)(qkv[shift_k + i], qkv[shift_k + i + 1],
                                               qkv[shift_k + i + 2], qkv[shift_k + i + 3]), 0));
            i += 3;
           }
         else
            result += IsNaNOrInf(qkv[shift_q + i] * qkv[shift_k + i], 0);
        }
      result = exp(clamp(result / koef, -100.0f, 100.0f));
      if(isnan(result))
         result = 0;
      score[shift_s + k] = result;
      sum += result;
     }
//---
   for(int k = 0; (k < units && sum > 1); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-heads Attention Neuron Out calculation kernel
/// Describes the Multi-heads Attention out calculation process for the Neuron
/// of multi-heads attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                             __global float *qkv,    ///<[in] Matrix of Values
                             __global float *out,    ///<[out] Output tensor
                             int dimension           ///< Dimension of Value
                            )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
   int layer = 3 * dimension * heads;
//---
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v++)
        {
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += scores[shift_s + v] * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation
/// kernel
/// Describes the gradients calculation process for the Neuron of attention
/// layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para33">the link.</A>
/// @param[in] qkv Matrix of Querys, Keys and Values
/// @param[out] qkv_g Matrix of Querys', Keys' and Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] scores_g Matrix of Scores' Gradients
/// @param[in] gradient Matrix of Gradients from previous iteration
/// @param[in] dimension Dimension of Key vector
//+------------------------------------------------------------------+
__kernel void MHAttentionInsideGradients(__global float *qkv, __global float *qkv_g,
      __global float *scores,
      __global float *gradient)
  {
   size_t u = get_global_id(0);
   size_t h = get_global_id(1);
   size_t d = get_global_id(2);
   size_t units = get_global_size(0);
   size_t heads = get_global_size(1);
   size_t dimension = get_global_size(2);
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- init
   const int shift_q = dimension * (heads * 3 * u + h);
   const int shift_k = dimension * (heads * (3 * u + 1) + h);
   const int shift_v = dimension * (heads * (3 * u + 2) + h);
   const int shift_g = dimension * (heads * u + h);
   int shift_score = h * units;
   int step_score = units * heads;
//--- Calculating Value's gradients
   float sum = 0;
//---
   for(int i = 0; i < units; i++)
      sum += gradient[(h + i * heads) * dimension + d] * scores[shift_score + u + i * step_score];
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   shift_score = h * units + u * step_score;
   float grad = 0;
   float grad_out = gradient[shift_g + d];
//---
   for(int k = 0; k < units; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k];
      for(int v = 0; v < units; v++)
         sc_g += scores[shift_score + v] * qkv[dimension * (heads * (3 * v + 2) + h)] *
                 grad_out * ((k == v) - sc);
      grad += sc_g / koef * qkv[dimension * (heads * (3 * k + 1) + h) + d];
     }
   qkv_g[shift_q + d] = grad;
//--- Calculating Key's gradients
   grad = 0;
//---
   for(int q = 0; q < units; q++)
     {
      shift_score = h * units + q * step_score;
      float sc_g = 0;
      float sc = scores[shift_score + u];
      float grad_out = gradient[dimension * (heads * q + h) + d];
      for(int v = 0; v < units; v++)
         sc_g += scores[shift_score + v] * qkv[dimension * (heads * (3 * v + 2) + h)] *
                 grad_out * ((u == v) - sc);
      grad += sc_g / koef * qkv[dimension * (heads * 3 * q + h) + d];
     }
   qkv_g[shift_k + d] = grad;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_dropout Kernel for Dropout.
/// Describes the dropout method.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9112#para32">the link.</A>
//+------------------------------------------------------------------+
__kernel void Dropout(__global const float *inputs, ///<[in] Input matrix
                      __global const float *map,    ///<[in] Dropout map matrix
                      __global float *out,    ///<[out] Output matrix
                      const int dimension           ///< Dimension of matrix
                     )
  {
   const int i = get_global_id(0) * 4;
   float m = 0, inp = 0;
//---
   for(int k = i; k < min(dimension, i + 4); k++)
     {
      m = IsNaNOrInf(map[i + k], 0.0f);
      if(m == 0)
        {
         out[i + k] = 0;
         continue;
        }
      inp = IsNaNOrInf(inputs[i + k], 0.0f);
      if(inp == 0)
        {
         out[i + k] = 0;
         continue;
        }
      out[i + k] = IsNaNOrInf(inp * m, 0);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of Batch normalization process
/// Describes the process of Batch normalization. (#CNeuronBatchNormOCL)
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para42">the link.</A>
/// @param[in] inputs Input data tenzor
/// @param[in,out] options Tenzor of variables
/// @param[out] output Tenzor of output data
/// @param[in] batch Batch size
/// @param[in] optimization Optimization type
/// @param[in] activation Activation type
//+------------------------------------------------------------------+
__kernel void BatchFeedForward(__global float *inputs, __global float *options,
                               __global float *output, int batch,
                               int optimization, int activation, int training)
  {
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   float inp = IsNaNOrInf(inputs[n], 0);
   if(training == 0)
     {
      const float mean = IsNaNOrInf(options[shift], 0);
      float variance = IsNaNOrInf(options[shift + 1], 1);
      if(variance <= 0)
         variance = 1;
      float gamma = IsNaNOrInf(options[shift + 3], 1);
      if(gamma == 0)
         gamma = 1;
      const float betta = IsNaNOrInf(options[shift + 4], 0);
      const float nx = (inp - mean) / sqrt(variance);
      output[n] = fActivation(gamma * nx + betta, activation);
      return;
     }
   if(batch <= 1)
      return;
   float mean = (IsNaNOrInf(options[shift], 0) * max((float)batch - 1.0f, 0.0f) + inp) / max((float)batch, 1.0f);
   float delt = inp - mean;
   float variance = (IsNaNOrInf(options[shift + 1] * max((float)batch - 1.0f, 0.0f), 0) + delt * delt) / max((float)batch, 1.0f);
   if(variance <= 0)
      variance = 1;
   float nx = delt / sqrt(variance);
//---
   float gamma = IsNaNOrInf(options[shift + 3], 1);
   if(gamma == 0)
     {
      options[shift + 3] = 1;
      gamma = 1;
     }
   float betta = IsNaNOrInf(options[shift + 4], 0);
//---
   options[shift] = mean;
   options[shift + 1] = variance;
   options[shift + 2] = nx;
   output[n] = fActivation(gamma * nx + betta, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_gr
/// Kernel of the Batch neuron to transfer gradient to previous layer
/// (#CNeuronBatchNormOCL)
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientBatch(__global float *options,      ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
                                      __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                      __global float *matrix_i,  ///<[in] Tensor of previous layer output
                                      __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                      int activation,            ///< Activation type (#ENUM_ACTIVATION)
                                      int batch,                 ///< Batch size
                                      int optimization           ///< Optimization type
                                     )
  {
   if(batch <= 1)
      return;
//---
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   float variance = IsNaNOrInf(options[shift + 1], 1);
//---
   float inp = IsNaNOrInf(matrix_i[n], 0);
   float gnx = IsNaNOrInf(matrix_g[n], 0) * IsNaNOrInf(options[shift + 3], 1);
   float temp = (variance > 0 ? 1.0f / sqrt(variance) : 0);
   float gmu = (-temp) * gnx;
   float gvar =
      (variance > 0
       ? (IsNaNOrInf(options[shift], 0) * inp) / (2 * pow(variance, 3.0f / 2.0f)) * gnx
       : 0);
   float batch_ratio = max((float)(batch - 1), 0.0f) / max((float)batch, 1.0f);
   float gx = temp * gnx + gmu / max(batch, 1) +
              gvar * 2 * inp / max(batch, 1) * batch_ratio * batch_ratio;
//---
   matrix_ig[n] = Deactivation(gx, inp, activation);;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron SGD optimization Updating
/// options kernel
/// Describes the process of SGD optimization options for the Batch
/// normalization Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsMomentum(__global float *options,     ///<[in,out] Options matrix m*7, where m - Number of neurons in previous layer
      __global float *matrix_g, ///<[in] Tensor of gradients at current layer
      float learning_rates,     ///< Learning rates
      float momentum            ///< Momentum multiplier
                                        )
  {
   const int n = get_global_id(0);
   const int inputs = get_global_size(0);
   const int shift = n * 7;
   float grad = clamp(IsNaNOrInf(matrix_g[n], 0), -MAX_GRAD, MAX_GRAD);
//---
   float2 delta = learning_rates * grad * (float2)(IsNaNOrInf(options[shift + 2], 0), 1) +
                  momentum * (float2)(IsNaNOrInf(options[shift + 5], 0), IsNaNOrInf(options[shift + 6], 0));
//---
   delta.s0 = IsNaNOrInf(delta.s0, 0);
   delta.s1 = IsNaNOrInf(delta.s1, 0);
   options[shift + 5] = delta.s0;
   float value = IsNaNOrInf(options[shift + 3], 1);
   options[shift + 3] = value + delta.s0 - learning_rates * (l1 * sign(value) +
                        l2 * value / inputs);
//---
   options[shift + 6] = delta.s1;
   value = IsNaNOrInf(options[shift + 4], 0);
   options[shift + 4] = value + delta.s1 - learning_rates * (l1 * sign(value) +
                        l2 * value / inputs);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron Adam optimization Updating
/// options kernel
/// Describes the process of Adam optimization options for the Batch
/// normalization  Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsAdam(__global float *options,     ///<[in,out] Options matrix m*9, where m - Number of neurons in previous layer
                                     __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                     const float l,            ///< Learning rates
                                     const float b1,           ///< First momentum multiplier
                                     const float b2            ///< Second momentum multiplier
                                    )
  {
   const int n = get_global_id(0);
   int inputs = get_global_size(0);
   const int shift = n * 9;
   float grad = clamp(IsNaNOrInf(matrix_g[n], 0), -MAX_GRAD, MAX_GRAD);
//---
   float nx = IsNaNOrInf(options[shift + 2], 0);
   float gamma = IsNaNOrInf(options[shift + 3], 1);
   if(gamma == 0)
      gamma = 1;
   float betta = IsNaNOrInf(options[shift + 4], 0);
//---
   float gamma_m1 = IsNaNOrInf(options[shift + 5], 0);
   float betta_m1 = IsNaNOrInf(options[shift + 6], 0);
   float gamma_m2 = IsNaNOrInf(options[shift + 7], 0);
   float betta_m2 = IsNaNOrInf(options[shift + 8], 0);
//---
   float2 mt = b1 * (float2)(gamma_m1, betta_m1) +
               (1 - b1) * (float2)(grad * nx, grad);
   float2 grad2 = (float2)(grad * nx, grad);
   float2 vt = b2 * (float2)(gamma_m2, betta_m2) +
               (1 - b2) * (grad2 * grad2);
   vt.s0 = IsNaNOrInf(vt.s0, 1);
   vt.s1 = IsNaNOrInf(vt.s1, 1);
   float2 delta = l * mt / sqrt(vt);
   delta.s0 = IsNaNOrInf(delta.s0, 0);
   delta.s1 = IsNaNOrInf(delta.s0, 0);
   float2 weight = delta -
                   (l1 * sign((float2)(gamma, betta)) +
                    l2 * (float2)(gamma, betta) / inputs);
//---
   options[shift + 3] = IsNaNOrInf(gamma + weight.s0, 1);
   options[shift + 4] = IsNaNOrInf(betta + weight.s1, 0);
   options[shift + 5] = IsNaNOrInf(mt.s0, 0);
   options[shift + 6] = IsNaNOrInf(mt.s1, 0);
   options[shift + 7] = vt.s0;
   options[shift + 8] = vt.s1;
  }
//+------------------------------------------------------------------+
//| Runs the vae feed forward feed-forward pass on the GPU.          |
//+------------------------------------------------------------------+
__kernel void VAE_FeedForward(__global float *inputs, __global float *random,
                              __global float *outputs, __global float *numeric_flags,
                              const int flags_enabled)
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
    if(flags_enabled)
      {
//--- Read the raw VAE operands before this work-item updates a potentially aliased output.
       const float mean = inputs[i];
       const float log_var = inputs[i + total];
       const float epsilon = random[i];
       const float half_log_var = 0.5f * log_var;
       const float sigma = exp(half_log_var);
       const float product = IsNaNOrInf(sigma, 0) * epsilon;
       const float raw_result = IsNaNOrInf(mean, 0) + product;
       const int invalid = isnan(mean) || isinf(mean) || isnan(log_var) || isinf(log_var) ||
                           isnan(epsilon) || isinf(epsilon) || isnan(half_log_var) || isinf(half_log_var) ||
                           isnan(sigma) || isinf(sigma) || isnan(product) || isinf(product) ||
                           isnan(raw_result) || isinf(raw_result);
       numeric_flags[i] = (invalid ? 1.0f : 0.0f);
      }
   outputs[i] = IsNaNOrInf(inputs[i], 0) + IsNaNOrInf(exp(0.5f * inputs[i + total]), 0) * random[i];
  }
//+------------------------------------------------------------------+
//| Computes the vae calc hidden gradient gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void VAE_CalcHiddenGradient(__global float *inputs,
                                     __global float *inp_grad,
                                     __global float *random,
                                     __global float *gradient,
                                     const float kld_mult)
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   float log_var = IsNaNOrInf(inputs[i + total], 0);
   float mean = IsNaNOrInf(inputs[i], 0);
   float sigma = exp(0.5f * log_var);
   float variance = exp(log_var);
   float grad = clamp(IsNaNOrInf(gradient[i], 0), -MAX_GRAD, MAX_GRAD);
//--- gradient stores the negative downstream derivative used by additive updates.
//--- Add the negative derivative of kld_mult * KL(N(mean,variance)||N(0,1)).
   inp_grad[i] = IsNaNOrInf(grad - kld_mult * mean, 0);
   inp_grad[i + total] = IsNaNOrInf(0.5f * grad * sigma * random[i] -
                                    0.5f * kld_mult * (variance - 1.0f), 0);
  }
//+------------------------------------------------------------------+
//| Runs the lstm feed forward feed-forward pass on the GPU.         |
//+------------------------------------------------------------------+
__kernel void LSTM_FeedForward(__global const float *inputs, int inputs_size,
                               __global const float *weights,
                               __global float *concatenated,
                               __global float *memory, __global float *output)
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   uint id2 = (uint)get_local_id(1);
   uint idv = (uint)get_global_id(2);
   uint total_v = (uint)get_global_size(2);
//---
   __local float Temp[4];
//---
   float sum = 0;
   uint shift_in = idv * inputs_size;
   uint shift_out = idv * total;
   uint shift = (inputs_size + total + 1) * (id2 + id);
//---
   for(uint i = 0; i < total; i += 4)
     {
      if(total - i > 4)
         sum += IsNaNOrInf(
                   dot((float4)(output[shift_out + i], output[shift_out + i + 1], output[shift_out + i + 2], output[shift_out + i + 3]),
                       (float4)(weights[shift + i], weights[shift + i + 1],
                                weights[shift + i + 2], weights[shift + i + 3])), 0);
      else
         for(uint k = i; k < total; k++)
            sum += IsNaNOrInf(output[shift_out + k] * weights[shift + k], 0);
     }
//---
   shift += total;
//---
   for(uint i = 0; i < inputs_size; i += 4)
     {
      if(total - i > 4)
         sum += IsNaNOrInf(
                   dot((float4)(inputs[shift_in + i], inputs[shift_in + i + 1], inputs[shift_in + i + 2], inputs[shift_in + i + 3]),
                       (float4)(weights[shift + i], weights[shift + i + 1],
                                weights[shift + i + 2], weights[shift + i + 3])), 0);
      else
         for(uint k = i; k < total; k++)
            sum += IsNaNOrInf(inputs[shift_in + k] * weights[shift + k], 0);
     }
   sum += IsNaNOrInf(weights[shift + inputs_size], 0);
   if(id2 < 3)
      sum = fActivation(sum, 1);
   else
      sum = fActivation(sum, 0);
   Temp[id2] = sum;
   concatenated[4 * shift_out + id2 * total + id] = sum;
//---
   BarrierLoc
   if(id2 == 0)
     {
      float mem = memory[shift_out + id + total_v * total] = memory[shift_out + id];
      float fg = Temp[0];
      float ig = Temp[1];
      float og = Temp[2];
      float nc = Temp[3];
      //---
      memory[shift_out + id] = mem = IsNaNOrInf(mem * fg + ig * nc, 0);
      output[shift_out + id] = IsNaNOrInf(og * fActivation(mem, 0), 0);
     }
  }
//+------------------------------------------------------------------+
//| Computes the lstm concatenated gradient gradient contributions on|
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void LSTM_ConcatenatedGradient(__global float *gradient,
                                        __global float *concatenated_gradient,
                                        __global float *memory,
                                        __global float *concatenated)
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   uint idv = (uint)get_global_id(1);
   uint total_v = (uint)get_global_size(1);
//---
   uint shift_out = idv * total;
   float t = tanh(memory[shift_out + id]);
//---
   concatenated_gradient[4 * shift_out + id + 2 * total] = gradient[shift_out + id] * t; // output gate
//---
   float memory_gradient = gradient[shift_out + id] * concatenated[4 * shift_out + id + 2 * total];
   memory_gradient *= 1 - t * t;
//---
   concatenated_gradient[4 * shift_out + id + 3 * total] =
      memory_gradient * concatenated[4 * shift_out + id + total]; // new content
//---
   concatenated_gradient[4 * shift_out + id + total] =
      memory_gradient * concatenated[4 * shift_out + id + 3 * total]; // input gate
//---
   concatenated_gradient[4 * shift_out + id] =
      memory_gradient * memory[shift_out + id + total_v * total]; // forgat gate
  }
//+------------------------------------------------------------------+
//| Computes the lstm hidden gradient gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void LSTM_HiddenGradient(__global float *concatenated_gradient, __global float *inputs_gradient,
                                  __global float *weights_gradient, __global float *hidden_state,
                                  __global float *inputs, __global float *weights, __global float *output,
                                  const int hidden_size, const int inputs_size)
  {
   uint id = get_global_id(0);
   uint total = get_global_size(0);
   uint idv = (uint)get_global_id(1);
   uint total_v = (uint)get_global_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   uint ls = min(total_v, (uint)LOCAL_ARRAY_SIZE);
//---
   uint shift_in = idv * inputs_size;
   uint shift_out = idv * total;
   uint weights_step = hidden_size + inputs_size + 1;
//---
//---
   for(int i = id; i < (hidden_size + inputs_size); i += total)
     {
      float inp = 0;
      if(i < hidden_size)
        {
         inp = hidden_state[shift_out + i];
         hidden_state[shift_out + i] = output[shift_out + i];
        }
      else
        {
         inp = inputs[shift_in + i - hidden_size];
         float grad = 0;
         for(uint g = 0; g < 3 * hidden_size; g++)
           {
            float temp = concatenated_gradient[4 * shift_out + g];
            grad += temp * (1 - temp) * weights[i + g * weights_step];
           }
         for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
           {
            float temp = concatenated_gradient[4 * shift_out + g];
            grad += temp * (1 - temp * temp) * weights[i + g * weights_step];
           }
         inputs_gradient[shift_in + i - hidden_size] = grad;
        }
      //---
      for(uint g = 0; g < 3 * hidden_size; g++)
        {
         float temp = concatenated_gradient[4 * shift_out + g];
         if(idv < ls)
            Temp[idv % ls] = 0;
         BarrierLoc
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - temp) * inp;
            BarrierLoc
           }
         if(idv == 0)
           {
            temp = Temp[0];
            for(int v = 1; v < ls; v++)
               temp += Temp[v];
            weights_gradient[i + g * weights_step] = temp;
           }
         BarrierLoc
        }
      for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
        {
         float temp = concatenated_gradient[4 * shift_out + g];
         if(idv < ls)
            Temp[idv % ls] = 0;
         BarrierLoc
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - temp * temp) * inp;
            BarrierLoc
           }
         if(idv == 0)
           {
            temp = Temp[0];
            for(int v = 1; v < ls; v++)
               temp += Temp[v];
            weights_gradient[i + g * weights_step] = temp;
           }
         BarrierLoc
        }
     }
//---
   for(int i = id; i < 4 * hidden_size; i += total)
     {
      if(idv < ls)
         Temp[idv % ls] = 0;
      BarrierLoc
      float temp = concatenated_gradient[4 * shift_out + i];
      if(i < 3 * hidden_size)
        {
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - temp);
            BarrierLoc
           }
        }
      else
        {
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += 1 - temp * temp;
            BarrierLoc
           }
        }
      if(idv == 0)
        {
         temp = Temp[0];
         for(int v = 1; v < ls; v++)
            temp += Temp[v];
         weights_gradient[(i + 1) * weights_step] = temp;
        }
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
///\ingroup LSTM_opt  LSTM Adam Updating Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Neuron LSTM
/// (#CNeuronLSTMOCL).
//+------------------------------------------------------------------+
__kernel void LSTM_UpdateWeightsAdam(__global float *weights,   ///<[in,out] Weights matrix (m+1)*n, where m -
                                     ///< number of neurons in previous layer and n -
                                     ///< number of neurons in current layer
                                     __global float
                                     *weights_gradient,    ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_m, ///<[in,out] Matrix of first momentum
                                     __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                                     const float l,            ///< Learning rates
                                     const float b1,           ///< First momentum multiplier
                                     const float b2            ///< Second momentum multiplier
                                    )
  {
   const uint id = get_global_id(0);
   const uint total = get_global_size(0);
   const uint id1 = get_global_id(1);
   const uint wi = id1 * total + id;
   float g = clamp(IsNaNOrInf(weights_gradient[wi], 0), -MAX_GRAD, MAX_GRAD);
   float mt = b1 * IsNaNOrInf(matrix_m[wi], 0) + (1 - b1) * g;
   float vt = b2 * IsNaNOrInf(matrix_v[wi], 1) + (1 - b2) * (g * g);
   float weight = IsNaNOrInf(weights[wi], 0);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) -
                      (l1 * sign(weight) + l2 * weight / total));
   weights[wi] = IsNaNOrInf(weight + delta, 0);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//| Runs the soft max feed forward feed-forward pass on the GPU.     |
//+------------------------------------------------------------------+
__kernel void SoftMax_FeedForward(__global float *inputs,
                                  __global float *outputs)
  {
   const uint total = (uint)get_local_size(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   uint shift_head = h * total;
//---
   outputs[shift_head + l] = LocalSoftMax(IsNaNOrInf(inputs[shift_head + l], MIN_VALUE), 0, Temp);
  }
//+------------------------------------------------------------------+
//| Computes the soft max hidden gradient gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void SoftMax_HiddenGradient(__global float *outputs,
                                     __global float *output_gr,
                                     __global float *input_gr)
  {
   size_t i = get_local_id(0);
   size_t outputs_total = get_local_size(0);
   size_t h = get_global_id(1);
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   uint shift = h * outputs_total;
   float output = IsNaNOrInf(outputs[shift + i], 0);
   float grad = IsNaNOrInf(output_gr[shift + i], 0);
   input_gr[shift + i] = LocalSoftMaxGrad(output, grad, 0, Temp);
  }
//+------------------------------------------------------------------+
//| Computes the soft max output gradient gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void SoftMax_OutputGradient(__global float *outputs,
                                     __global float *targets,
                                     __global float *output_gr)
  {
   size_t i = get_global_id(0);
   output_gr[i] = (outputs[i] == 0 ? 0 : targets[i] / outputs[i]);
  }
//+------------------------------------------------------------------+
//| Runs the fqf cosine kernel on the GPU.                           |
//+------------------------------------------------------------------+
__kernel void FQF_Cosine(__global float *softmax, __global float *output)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float result = 0;
//---
   for(int it = 0; it < i; it++)
      result += softmax[shift + it];
   result += softmax[shift + i] / 2.0f;
   output[shift + i] = cos(i * M_PI_F * result);
  }
//+------------------------------------------------------------------+
//| Runs the fqf output kernel on the GPU.                           |
//+------------------------------------------------------------------+
__kernel void FQF_Output(__global float *quantiles, __global float *delta_taus,
                         __global float *output, int total)
  {
   size_t action = get_global_id(0);
   int shift = action * total;
//---
   float result = 0;
//---
   for(int i = 0; i < total; i++)
      result += quantiles[shift + i] * delta_taus[shift + i];
   output[action] = result;
  }
//+------------------------------------------------------------------+
//| Computes the fqf output gradient gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void FQF_OutputGradient(__global float *quantiles,
                                 __global float *delta_taus,
                                 __global float *output_gr,
                                 __global float *quantiles_gr,
                                 __global float *taus_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = output_gr[action];
   quantiles_gr[shift + i] = gradient * delta_taus[shift + i];
   taus_gr[shift + i] = gradient * quantiles[shift + i];
  }
//+------------------------------------------------------------------+
//| Computes the fqf quantile gradient gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void FQF_QuantileGradient(__global float *state_embeding,
                                   __global float *taus_embeding,
                                   __global float *quantiles_gr,
                                   __global float *state_gr,
                                   __global float *taus_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = quantiles_gr[shift + i];
   state_gr[shift + i] = gradient * taus_embeding[shift + i];
   taus_gr[shift + i] = gradient * state_embeding[shift + i];
  }
//+------------------------------------------------------------------+
//| Computes the fqf cosine gradient gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void FQF_CosineGradient(__global float *softmax,
                                 __global float *output_gr,
                                 __global float *softmax_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float cumul = 0;
//---
   for(int it = 0; it < i; it++)
      cumul += softmax[shift + it];
   float result = -M_PI_F * i *
                  sin(M_PI_F * i * (cumul + softmax[shift + i] / 2)) *
                  output_gr[shift + i];
//---
   for(int it = i + 1; it < total; it++)
     {
      cumul += softmax[shift + it - 1];
      float temp = cumul + softmax[shift + it] / 2;
      result += -M_PI_F * it * sin(M_PI_F * it * temp) * output_gr[shift + it] *
                softmax[shift + it] / temp;
     }
   softmax_gr[shift + i] += result;
  }
//+------------------------------------------------------------------+
//| Sparse Attention                                                 |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionScore(__global float *qkv,     ///<[in] Matrix of Querys, Keys, Values
                                     __global float *score, ///<[out] Matrix of Scores
                                     int dimension,         ///< Dimension of Key
                                     float sparse           ///< less than 1.0 coefficient of sparse
                                    )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
   int active_units = (int)max((float)(units * sparse), min((float)units, 3.0f));
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0.0f;
   float min_s = 0.0f;
   float max_s = 0.0f;
//---
   for(int k = 0; k < units; k++)
     {
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot((float4)(qkv[shift_q + i], qkv[shift_q + i + 1],
                                   qkv[shift_q + i + 2], qkv[shift_q + i + 3]),
                          (float4)(qkv[shift_k + i], qkv[shift_k + i + 1],
                                   qkv[shift_k + i + 2], qkv[shift_k + i + 3]));
            i += 3;
           }
         else
            result += (qkv[shift_q + i] * qkv[shift_k + i]);
        }
      score[shift_s + k] = result;
      if(k == 0)
         min_s = max_s = result;
      else
        {
         max_s = max(max_s, result);
         min_s = min(min_s, result);
        }
     }
//---
   int count = units;
//---
   while(count > active_units && min_s < max_s)
     {
      count = 0;
      float temp = max_s;
      for(int k = 0; k < units; k++)
        {
         float value = score[shift_s + k];
         if(value < min_s)
            continue;
         count++;
         if(value < temp && value > min_s)
            temp = value;
        }
      if(count > active_units)
         min_s = temp;
     }
//---
   if(max_s == 0.0f)
      max_s = 1.0f;
//---
   for(int k = 0; k < units; k++)
     {
      float value = score[shift_s + k];
      if(value < min_s)
        {
         score[shift_s + k] = 0.0f;
         continue;
        }
      value = exp(value / max_s / koef);
      score[shift_s + k] = value;
      sum += value;
     }
//---
   for(int k = 0; (k < units && sum > 1); k++)
     {
      float temp = score[shift_s + k];
      if(temp == 0.0f)
         continue;
      score[shift_s + k] = temp / sum;
     }
  }
//+------------------------------------------------------------------+
//| Implements the mh sparse attention out kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                                   __global float *qkv,   ///<[in] Matrix of Values
                                   __global float *out,   ///<[out] Output tensor
                                   int dimension           ///< Dimension of Value
                                  )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v++)
        {
         float cur_score = scores[shift_s + v];
         if(cur_score == 0)
            continue;
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += cur_score * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
//| Runs the feed forward multi models feed-forward pass on the GPU. |
//+------------------------------------------------------------------+
__kernel void FeedForwardMultiModels(__global float *matrix_w,   ///<[in] Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
                                     __global float *matrix_i, ///<[in] Inputs tensor
                                     __global float *matrix_o, ///<[out] Output tensor
                                     int inputs,               ///< Number of inputs
                                     int activation            ///< Activation type (#ENUM_ACTIVATION)
                                    )
  {
   int i = get_global_id(0);
   int outputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs + 1) * (i + outputs * m);
   int shift_in = inputs * m;
   int shift_out = outputs * m;
//---
   for(int k = 0; k <= inputs; k = k + 4)
     {
      switch(inputs - k)
        {
         case 0:
            inp = (float4)(1, 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 1:
            inp = (float4)(matrix_i[shift_in + k], 1, 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], 1, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         case 3:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1],
                           matrix_i[shift_in + k + 2], 1);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
         default:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1],
                           matrix_i[shift_in + k + 2], matrix_i[shift_in + k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[shift_out + i] = fActivation(sum, activation);
  }
//+------------------------------------------------------------------+
//| Computes the calc hidden gradient multi models gradient          |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMultiModels(__global float *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - number
      ///< of neurons in previous layer and n - number
      ///< of neurons in current layer
      __global float *matrix_g, ///<[in] Tensor of gradients at current layer
      __global float *matrix_o, ///<[in] Previous layer Output tensor
      __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
      int outputs,               ///< Number of outputs
      int activation,            ///< Activation type (#ENUM_ACTIVATION),
      int model)
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   int shift_in = inputs * m;
   if(model >= 0 && model != m)
     {
      matrix_ig[shift_in + i] = 0;
      return;
     }
//---
   int shift_out = outputs * m;
   int shift_w = (inputs + 1) * outputs * m;
   float sum = 0;
   float out = matrix_o[shift_in + i];
   float4 grad, weight;
//---
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[shift_out + k], 0, 0, 0);
            break;
         case 2:
            grad =
               (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1], 0, 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1],
                            matrix_g[shift_out + k + 2], 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1],
                            matrix_g[shift_out + k + 2], matrix_g[shift_out + k + 3]);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 2) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
//---
   matrix_ig[shift_in + i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights adam multi   |
//| models step using the optimizer rule.                            |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdamMultiModels(
   __global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
   ///< number of neurons in previous layer and n -
   ///< number of neurons in current layer
   __global const float
   *matrix_g, ///<[in] Tensor of gradients at current layer
   __global const float *matrix_i, ///<[in] Inputs tensor
   __global float *matrix_m,      ///<[in,out] Matrix of first momentum
   __global float *matrix_v,      ///<[in,out] Matrix of seconfd momentum
   const int inputs,               ///< Number of inputs
   const float l,                  ///< Learning rates
   const float b1,                 ///< First momentum multiplier
   const float b2,                 ///< Second momentum multiplier
   const int model)
  {
   const int outputs = get_global_size(0);
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = (i + outputs * model) * (inputs + 1) + j * 4;
   float4 m, v, weight, inp;
   int shift_in = j * 4 + inputs * model;
   if((inputs + 1 - j * 4) < 0)
      return;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         m = (float4)(matrix_m[wi], 0, 0, 0);
         v = (float4)(matrix_v[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[shift_in], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], 0, 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], 0);
         break;
      case 3:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1],
                        matrix_i[shift_in + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2],
                      matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2],
                      matrix_v[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1],
                        matrix_i[shift_in + 2], matrix_i[shift_in + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2],
                      matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2],
                      matrix_v[wi + 3]);
         break;
     }
   float4 g = (float4)(matrix_g[(outputs + 1) * model + i]) * inp;
   float4 mt = b1 * m + (1 - b1) * g;
   float4 vt = b2 * v + (1 - b2) * (g * g);
   float4 delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(fabs(delta.s3) > 0)
            matrix_w[wi + 3] = matrix_w[wi + 3] + delta.s3;
         matrix_m[wi + 3] = mt.s3;
         matrix_v[wi + 3] = vt.s3;
      case 2:
         if(fabs(delta.s2) > 0)
            matrix_w[wi + 2] = matrix_w[wi + 2] + delta.s2;
         matrix_m[wi + 2] = mt.s2;
         matrix_v[wi + 2] = vt.s2;
      case 1:
         if(fabs(delta.s1) > 0)
            matrix_w[wi + 1] = matrix_w[wi + 1] + delta.s1;
         matrix_m[wi + 1] = mt.s1;
         matrix_v[wi + 1] = vt.s1;
      case 0:
         if(fabs(delta.s0) > 0)
            matrix_w[wi] = matrix_w[wi] + delta.s0;
         matrix_m[wi] = mt.s0;
         matrix_v[wi] = vt.s0;
         break;
     }
  }
//+------------------------------------------------------------------+
//| Runs the concat feed forward feed-forward pass on the GPU.       |
//+------------------------------------------------------------------+
__kernel void Concat_FeedForward(__global float *matrix_w,   ///<[in] Weights matrix (m+1)*n, where m - number
                                 ///< of neurons in layer and n - number of outputs
                                 ///< (neurons in next layer)
                                 __global float *matrix_i1, ///<[in] Inputs 1 tensor
                                 __global float *matrix_i2, ///<[in] Inputs 2 tensor
                                 __global float *matrix_o, ///<[out] Output tensor
                                 int inputs1,               ///< Number of inputs
                                 int inputs2,               ///< Number of inputs
                                 int activation             ///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs1 + inputs2 + 1) * i;
//---
//---
   for(int k = 0; k < inputs1; k += 4)
     {
      switch(inputs1 - k)
        {
         case 1:
            inp = (float4)(matrix_i1[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2],
                           matrix_i1[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
//---
   shift += inputs1;
//---
   for(int k = 0; k < inputs2; k += 4)
     {
      switch(inputs2 - k)
        {
         case 1:
            inp = (float4)(matrix_i2[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2],
                           matrix_i2[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   sum += matrix_w[shift + inputs2];
//---
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[i] = fActivation(sum, activation);;
  }
//+------------------------------------------------------------------+
//| Computes the concat hidden gradient gradient contributions on the|
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void Concat_HiddenGradient(__global float *matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                    __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global float *matrix_o1, ///<[in] Previous layer Output tensor
                                    __global float *matrix_o2, ///<[in] Previous layer Output tensor
                                    __global float *matrix_ig1, ///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_ig2, ///<[out] Tensor of gradients at previous layer
                                    int outputs,                ///< Number of outputs
                                    int inputs1, int inputs2,
                                    int activation1, ///< Activation type (#ENUM_ACTIVATION)
                                    int activation2  ///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
   if(i >= (inputs1 + inputs2))
      return;
   int inputs = inputs1 + inputs2;
   float sum = 0;
   float out = (i < inputs1 ? matrix_o1[i] : matrix_o2[i - inputs1]);
   float4 grad, weight;
//---
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2],
                            matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i],
                              matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
   if(isnan(sum))
      sum = 0;
   if(i < inputs1)
      matrix_ig1[i] = Deactivation(sum, out, activation1);
   else
      matrix_ig2[i - inputs1] = Deactivation(sum, out, activation2);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the concat update weights       |
//| momentum step using the optimizer rule.                          |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsMomentum(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global float *matrix_g, ///<[in] Tensor of gradients at current layer
      __global float *matrix_i1, ///<[in] Inputs tensor
      __global float *matrix_i2, ///<[in] Inputs tensor
      __global float
      *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
      int inputs1,    ///< Number of inputs
      int inputs2,    ///< Number of inputs
      float learning_rates, ///< Learning rates
      float momentum        ///< Momentum multiplier
                                          )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp = (j < inputs1 ? matrix_i1[j] : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float delta = learning_rates * matrix_g[i] * inp + momentum * matrix_dw[wi];
   if(!isnan(delta))
     {
      matrix_dw[wi] = delta;
      if(fabs(delta) > 0)
         matrix_w[wi] = matrix_w[wi] + delta;
     }
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the concat update weights adam  |
//| step using the optimizer rule.                                   |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsAdam(__global float *matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
                                       ///< number of neurons in previous layer and n -
                                       ///< number of neurons in current layer
                                       __global const float
                                       *matrix_g, ///<[in] Tensor of gradients at current layer
                                       __global const float *matrix_i1, ///<[in] Inputs tensor
                                       __global const float *matrix_i2, ///<[in] Inputs tensor
                                       __global float *matrix_m,       ///<[in,out] Matrix of first momentum
                                       __global float *matrix_v,       ///<[in,out] Matrix of seconfd momentum
                                       const int inputs1,               ///< Number of inputs
                                       const int inputs2,               ///< Number of inputs
                                       const float l,                   ///< Learning rates
                                       const float b1,                  ///< First momentum multiplier
                                       const float b2                   ///< Second momentum multiplier
                                      )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   const int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp =
      (j < inputs1 ? matrix_i1[j]
       : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float weight = matrix_w[wi];
   float g = matrix_g[i] * inp;
   float mt = b1 * matrix_m[wi] + (1 - b1) * g;
   float vt = b2 * matrix_v[wi] + (1 - b2) * (g * g);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      matrix_w[wi] = matrix_w[wi] + delta;
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the soft update kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void SoftUpdate(__global float *target, ///<[in,out] Target matrix
                         __global const float *source, ///<[in] Source matrix
                         const float tau ///<[in] Multiplicator Tau
                        )
  {
   const int i = get_global_id(0);
   target[i] = source[i] * tau + (1.0f - tau) * target[i];
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the soft update adam step using |
//| the optimizer rule.                                              |
//+------------------------------------------------------------------+
__kernel void SoftUpdateAdam(__global float *target, __global const float *source,
                             __global float *matrix_m, ///<[in,out] Matrix of first momentum
                             __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                             const float tau,          ///<[in] Multiplicator Tau
                             const float b1,           ///< First momentum multiplier
                             const float b2            ///< Second momentum multiplier
                            )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = source[i] - weight;
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * (g * g);
   float delta = (1 - tau) * m / (v != 0.0f ? sqrt(v) : 1.0f);
   if(fabs(delta) > 0)
      target[i] = weight + delta;
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
//| Implements the sac alpha log probs kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaLogProbs(__global float *outputs,
                                __global float *quantiles,
                                __global float *probs,
                                __global float *alphas,
                                __global float *log_probs,
                                __global float *random,
                                const int count_quants,
                                const int activation)
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float prob = 0;
   float value = 0;
   float sum = 0;
   float rnd = random[i];
//---
   for(int r = 0; r < count_quants; r++)
     {
      prob = probs[shift + r];
      sum += prob;
      if(sum >= rnd || r == (count_quants - 1))
        {
         value = quantiles[shift + r];
         break;
        }
     }
//---
   outputs[i] = fActivation(value, activation);
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//| Computes the sac alpha gradients gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaGradients(__global float *outputs,
                                 __global float *gradient,
                                 __global float *log_probs,
                                 __global float *alphas_grad,
                                 const int activation)
  {
   const int i = get_global_id(0);
   float out = outputs[i];
//---
   float grad = -gradient[i] * log_probs[i];
//---
   alphas_grad[i] = Deactivation(grad, out, activation);
  }
//+------------------------------------------------------------------+
//| Computes the sac output gradient gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SAC_OutputGradient(__global float *quantiles, __global float *delta_taus,
                                 __global float *output_gr, __global float *quantiles_gr,
                                 __global float *taus_gr, __global float *output,
                                 const int count_quants, const int activation)
  {
   size_t action = get_global_id(0);
   int shift = action * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   int pos1 = -1;
   int pos2 = -1;
   float value = output[action];
//---
   for(int i = 0; i < count_quants; i++)
     {
      float quant = fActivation(quantiles[shift + i], activation);
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         pos1 = shift + i;
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         pos2 = shift + i;
        }
      quantiles_gr[shift + i] = 0.0f;
      taus_gr[shift + i] = 0.0f;
     }
   float gradient = output_gr[action];
   if(quant1 > -1e37f)
     {
      quantiles_gr[pos1] = gradient * delta_taus[pos1];
      taus_gr[pos1] = gradient * quant1;
     }
   if(quant2 < 1e37f)
     {
      quantiles_gr[pos2] = gradient * delta_taus[pos2];
      taus_gr[pos2] = gradient * quant2;
     }
  }
//+------------------------------------------------------------------+
//| Implements the sac calc log probs kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void SAC_CalcLogProbs(__global float *outputs,
                               __global float *quantiles, __global float *probs,
                               __global float *alphas,
                               __global float *log_probs,
                               const int count_quants, const int activation)
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   float prob1 = 0;
   float prob2 = 0;
   float value = outputs[i];
//---
   for(int q = 0; q < count_quants; q++)
     {
      float quant = fActivation(quantiles[shift + q], activation);
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         prob1 = probs[shift + q];
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         prob2 = probs[shift + q];
        }
     }
//---
   float prob = fabs(value - quant1) / fabs(quant2 - quant1);
   prob = clamp((1 - prob) * prob1 + prob * prob2, 1.0e-3f, 1.0f);
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//| Implements the embedding kernel on the GPU.                      |
//+------------------------------------------------------------------+
__kernel void Embedding(__global float *inputs, __global float *outputs,
                        __global float *weights, __global int *windows,
                        __global float *std, const int stack_size)
  {
   const int window_out = get_global_size(0);
   const int pos = get_global_id(0);
   const int emb = get_global_id(1);
   const int emb_total = get_global_size(1);
   const int shift_out = emb * window_out + pos;
   const int step = emb_total * window_out;
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
//---
   for(int i = stack_size - 1; i > 0; i--)
      outputs[i * step + shift_out] = outputs[(i - 1) * step + shift_out];
   int shift_in = 0;
//---
   for(int i = 0; i < emb; i++)
      shift_in += windows[i];
   const int window_in = windows[emb];
   const int shift_weights = (shift_in + emb) * window_out + (window_in + 1) * pos;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   if(pos < LOCAL_ARRAY_SIZE)
      temp[pos] = 0;
   BarrierLoc
//---
   float value = weights[shift_weights + window_in];
//---
   for(int i = 0; i < window_in; i++)
      value += inputs[shift_in + i] * weights[shift_weights + i];
//---
   for(int i = 0; i < window_out; i += ls)
     {
      if(pos >= i && pos < (i + ls))
         temp[pos % ls] += value;
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos + count < ls)
        {
         if(pos < count)
            temp[pos] += temp[pos + count];
         temp[pos + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   value -= temp[0] / (float)window_out;
   BarrierLoc
//---
   if(pos < LOCAL_ARRAY_SIZE)
      temp[pos] = 0;
   BarrierLoc
//---
   for(int i = 0; i < window_out; i += ls)
     {
      if(pos >= i && pos < (i + ls))
         temp[pos % ls] += (value * value) / (float)window_out;
      BarrierLoc
     }
//---
   count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos + count < ls)
        {
         if(pos < count)
            temp[pos] += temp[pos + count];
         temp[pos + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   if(temp[0] > 0)
      value /= sqrt(temp[0]);
//---
   outputs[shift_out] = value;
   if(pos == 0)
      std[emb] = sqrt(temp[0]);
  }
//+------------------------------------------------------------------+
//| Computes the embedding hidden gradient gradient contributions on |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void EmbeddingHiddenGradient(__global float *inputs_gradient,
                                      __global float *outputs_gradient,
                                      __global float *weights,
                                      __global int *windows,
                                      __global float *std,
                                      const int window_out)
  {
   const int pos = get_global_id(0);
   int emb = -1;
   int count = 0;
   do
     {
      emb++;
      count += windows[emb];
     }
   while(count < pos);
   const int shift_out = emb * window_out;
   const int shift_weights = pos + (count - windows[emb] + emb) * window_out;
//---
   float value = 0;
//---
   for(int i = 0; i < window_out; i++)
      value += outputs_gradient[shift_out + i] * weights[shift_weights + i * window_out];
   float s = std[emb];
   if(s > 0)
      value /= s;
//---
   inputs_gradient[pos] = value;
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the embedding update weights    |
//| adam step using the optimizer rule.                              |
//+------------------------------------------------------------------+
__kernel void EmbeddingUpdateWeightsAdam(__global float *weights,   ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< number of neurons in previous layer and n -
      ///< number of neurons in current layer
      __global const float *gradient,                ///<[in] Tensor of gradients at current layer
      __global const float *inputs, ///<[in] Inputs tensor
      __global float *matrix_m,    ///<[in,out] Matrix of first momentum
      __global float *matrix_v,    ///<[in,out] Matrix of seconfd momentum
      __global int *windows, __global float *std, const int window_out,
      const float l,  ///< Learning rates
      const float b1, ///< First momentum multiplier
      const float b2  ///< Second momentum multiplier
                                        )
  {
   const int i = get_global_id(0);
   int emb = -1;
   int count = 0;
   int shift = 0;
   int window_in = 0;
   do
     {
      emb++;
      shift = count;
      window_in = windows[emb];
      count += (window_in + 1) * window_out;
     }
   while(count <= i);
   const int shift_out = emb * window_out;
   int shift_in = shift / window_out - emb;
   shift = (i - shift) % (window_in + 1);
   float inp = 1.0f;
   if(shift < window_in)
      inp = inputs[shift_in + shift];
//---
   float weight = weights[i];
   float g = gradient[shift_out] * inp / std[emb];
   float mt = b1 * matrix_m[i] + (1 - b1) * g;
   float vt = b2 * matrix_v[i] + (1 - b2) * (g * g);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      weights[i] = weights[i] + delta;
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
//| Performs the transpose data-layout operation on the GPU.         |
//+------------------------------------------------------------------+
__kernel void Transpose(__global float *matrix_in, ///<[in] Input matrix
                        __global float *matrix_out ///<[out] Output matrix
                       )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
//---
   matrix_out[c * rows + r] = matrix_in[r * cols + c];
  }
//+------------------------------------------------------------------+
//| Implements the mh2 attention out kernel on the GPU.              |
//+------------------------------------------------------------------+
__kernel void MH2AttentionOut(__global float *q,    ///<[in] Matrix of Querys
                              __global float *kv,   ///<[in] Matrix of Keys
                              __global float *score, ///<[out] Matrix of Scores
                              __global float *out, ///<[out] Matrix of attention
                              int dimension,        ///< Dimension of Key
                              int heads_kv,
                              int mask ///< 1 - calc only previous units, 0 - calc all
                             )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   float sum = MIN_VALUE;
   if(mask == 0 || q_id >= k)
     {
      sum = 0;
      for(int d = 0; d < dimension; d++)
         sum += q[shift_q + d] * kv[shift_k + d];
     }
   float sc = LocalSoftMax(IsNaNOrInf(sum / koef, MIN_VALUE), 1, temp);
   score[shift_s] = sc;
//--- out
   for(int d = 0; d < dimension; d++)
     {
      BarrierLoc
      sum = LocalSum(IsNaNOrInf(kv[shift_v + d ] * sc, 0), 1, temp);
      //---
      if(k == 0)
         out[shift_q + d] = sum;
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh2 attention inside gradients gradient             |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void MH2AttentionInsideGradients(__global float *q, __global float *q_g,
      __global float *kv, __global float *kv_g,
      __global float *scores, __global float *gradient,
      int kunits, int heads_kv)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = q_id * kunits * heads + h * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v = q_id; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
//---
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + 2 * v * heads_kv * dimension] *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + 2 * k * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += sc_g * q[shift_q + scr * dimension];
              }
           }
         kv_g[shift_k] = grad / koef;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the cg conv hidden gradient gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void CGConv_HiddenGradient(__global const float *matrix_g,    ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_f, ///<[in] Previous layer Output tensor
                                    __global const float *matrix_s, ///<[in] Previous layer Output tensor
                                    __global float *matrix_fg, ///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_sg, ///<[out] Tensor of gradients at previous layer
                                    const int activationf,           ///< Activation type (#ENUM_ACTIVATION)
                                    const int activations            ///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
//---
   float grad = matrix_g[i];
   float f = matrix_f[i];
   float s = matrix_s[i];
//---
   float sg = grad * f;
   float fg = grad * s;
//---
   matrix_fg[i] = Deactivation(fg, f, activationf);
   matrix_sg[i] = Deactivation(sg, s, activations);
  }
//+------------------------------------------------------------------+
//| Runs the x ci t feed forward feed-forward pass on the GPU.       |
//+------------------------------------------------------------------+
__kernel void XCiTFeedForward(__global float *qkv, __global float *score,
                              __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_local_id(1);
   const size_t units = get_local_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   const uint ls_u = min((uint)units, (uint)LOCAL_ARRAY_SIZE);
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float q[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
   __local float k[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
//--- Normalize Query and Key
   for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
     {
      float q_val = 0;
      float k_val = 0;
      //---
      if(d < ls_d && (cur_d + d) < dimension && u < ls_u)
        {
         for(int count = u; count < units; count += ls_u)
           {
            int shift = count * dimension * heads * 3 + dimension * h + cur_d + d;
            q_val += qkv[shift] * qkv[shift];
            k_val += qkv[shift + dimension * heads] * qkv[shift + dimension * heads];
           }
         q[u][d] = q_val;
         k[u][d] = k_val;
        }
      BarrierLoc
      //---
      uint count = ls_u;
      do
        {
         count = (count + 1) / 2;
         if(d < ls_d)
           {
            if(u < ls_u && u < count && (u + count) < units)
              {
               float q_val = q[u][d] + q[u + count][d];
               float k_val = k[u][d] + k[u + count][d];
               q[u + count][d] = 0;
               k[u + count][d] = 0;
               q[u][d] = q_val;
               k[u][d] = k_val;
              }
           }
         BarrierLoc
        }
      while(count > 1);
      //---
      int shift = u * dimension * heads * 3 + dimension * h + cur_d;
      qkv[shift] = qkv[shift] / sqrt(q[0][d]);
      qkv[shift + dimension * heads] =
         qkv[shift + dimension * heads] / sqrt(k[0][d]);
      BarrierLoc
     }
//--- Score
   int step = dimension * heads * 3;
//---
   for(int cur_r = 0; cur_r < dimension; cur_r += ls_u)
     {
      for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
        {
         if(u < ls_d && d < ls_d)
            q[u][d] = 0;
         BarrierLoc
         //---
         if((cur_r + u) < ls_d && (cur_d + d) < ls_d)
           {
            int shift_q = dimension * h + cur_d + d;
            int shift_k = dimension * (heads + h) + cur_r + u;
            float scr = 0;
            for(int i = 0; i < units; i++)
               scr += qkv[shift_q + i * step] * qkv[shift_k + i * step];
            scr = exp(scr / sqrt((float)units));
            score[(cur_r + u) * dimension * heads + dimension * h + cur_d + d] =
               scr;
            q[u][d] += scr;
           }
        }
      BarrierLoc
      //---
      int count = ls_d;
      do
        {
         count = (count + 1) / 2;
         if(u < ls_d)
           {
            if(d < ls_d && d < count && (d + count) < dimension)
               q[u][d] += q[u][d + count];
            if(d + count < ls_d)
               q[u][d + count] = 0;
           }
         BarrierLoc
        }
      while(count > 1);
      //---
      if((cur_r + u) < ls_d)
         score[(cur_r + u) * dimension * heads + dimension * h + d] /= q[u][0];
      BarrierLoc
     }
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_s = dimension * (heads * d + h);
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
//---
   for(int i = 0; i < dimension; i++)
      sum += qkv[shift_v + i] * score[shift_s + i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//| Computes the x ci t inside gradients gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void XCiTInsideGradients(__global float *qkv, __global float *qkv_g,
                                  __global float *scores,
                                  __global float *gradient)
  {
//--- init
   const int q = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int units = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (heads * 3 * q + h);
   const int shift_k = dimension * (heads * (3 * q + 1) + h);
   const int shift_v = dimension * (heads * (3 * q + 2) + h);
   const int shift_g = dimension * (heads * q + h);
   int shift_score = dimension * h;
   int step_score = dimension * heads;
//--- Calculating Value's gradients
   float sum = 0;
//---
   for(int i = 0; i < dimension; i++)
      sum += gradient[shift_g + i] * scores[shift_score + d + i * step_score];
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   float val = qkv[shift_v + d];
//---
   for(int k = 0; k < dimension; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_score + v] * val *
                 gradient[shift_g + v * dimension] * ((float)(k == v) - sc);
      grad += sc_g * qkv[shift_k + k];
     }
   qkv_g[shift_q + d] = grad / sqrt((float)units);
//--- Calculating Key's gradients
   grad = 0;
   float out_g = gradient[shift_g];
//---
   for(int scr = 0; scr < dimension; scr++)
     {
      float sc_g = 0;
      int shift_sc = scr * dimension * heads;
      float sc = scores[shift_sc + d];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_sc + v] * out_g * qkv[shift_v + v] *
                 ((float)(d == v) - sc);
      grad += sc_g * qkv[shift_q + scr];
     }
   qkv_g[shift_k + d] = grad / sqrt((float)units);
  }
//+------------------------------------------------------------------+
//| Runs the dot feed forward feed-forward pass on the GPU.          |
//+------------------------------------------------------------------+
__kernel void DOTFeedForward(__global float *qkv, __global float *score,
                             __global float *rpb, __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_global_id(1);
   const size_t units = get_global_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   uint shift_q = u * step + h * dimension;
   uint shift_k = start * step + dimension * (heads + h);
   uint shift_score = u * 3 * heads;
//---
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float temp[LOCAL_ARRAY_SIZE][3];
//--- Score
   if(d < ls_d)
     {
      //---
      for(uint pos = start; pos <= stop; pos++)
         temp[d][pos - start] = 0;
      //---
      for(uint dim = d; dim < dimension; dim += ls_d)
        {
         float q = qkv[shift_q + dim];
         for(uint pos = start; pos <= stop; pos++)
           {
            uint i = pos - start;
            temp[d][i] = temp[d][i] + q * qkv[shift_k + i * step + dim];
           }
        }
      BarrierLoc
      //---
      int count = ls_d;
      //---
      do
        {
         count = (count + 1) / 2;
         if(d < count && (d + count) < dimension)
            for(uint i = 0; i <= (stop - start); i++)
              {
               temp[d][i] += temp[d + count][i];
               temp[d + count][i] = 0;
              }
         BarrierLoc
        }
      while(count > 1);
     }
//---
   if(d == 0)
     {
      float sum = 0;
      //---
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = exp(temp[0][i] + rpb[shift_score + i]);
         sum += temp[0][i];
        }
      //---
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = temp[0][i] / sum;
         score[shift_score + i] = temp[0][i];
        }
     }
   BarrierLoc
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
//---
   for(uint i = 0; i <= (stop - start); i++)
      sum += qkv[shift_v + i] * temp[0][i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//| Computes the dot inside gradients gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void DOTInsideGradients(__global float *qkv, __global float *qkv_g,
                                 __global float *scores, __global float *rpb,
                                 __global float *rpb_g,
                                 __global float *gradient)
  {
//--- init
   const uint u = get_global_id(0);
   const uint d = get_global_id(1);
   const uint h = get_global_id(2);
   const uint units = get_global_size(0);
   const uint dimension = get_global_size(1);
   const uint heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   const uint shift_q = u * step + dimension * h + d;
   const uint shift_k = u * step + dimension * (heads + h) + d;
   const uint shift_v = u * step + dimension * (2 * heads + h) + d;
//--- Calculating Value's gradients
   float sum = 0;
//---
   for(uint i = start; i <= stop; i++)
     {
      int shift_score = i * 3 * heads;
      if(u == i)
        {
         shift_score += (uint)(u > 0);
        }
      else
        {
         if(u > i)
            shift_score += (uint)(start > 0) + 1;
        }
      uint shift_g = dimension * (i * heads + h) + d;
      sum += gradient[shift_g] * scores[shift_score];
     }
   qkv_g[shift_v] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   uint shift_score = u * heads * 3;
//---
   for(int k = start; k <= stop; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k - start];
      for(int v = start; v <= stop; v++)
         for(int dim = 0; dim < dimension; dim++)
            sc_g += scores[shift_score + v - start] *
                    qkv[v * step + dimension * (2 * heads + h) + dim] *
                    gradient[dimension * (u * heads + h) + dim] *
                    ((float)(k == v) - sc);
      grad += sc_g * qkv[k * step + dimension * (heads + h) + d];
      if(d == 0)
         rpb_g[shift_score + k - start] = sc_g;
     }
   qkv_g[shift_q] = grad;
//--- Calculating Key's gradients
   grad = 0;
//---
   for(int q = start; q <= stop; q++)
     {
      float sc_g = 0;
      shift_score = q * heads * 3;
      if(u == q)
         shift_score += (uint)(u > 0);
      else
        {
         if(u > q)
            shift_score += (uint)(start > 0) + 1;
        }
      float sc = scores[shift_score];
      for(int v = start; v <= stop; v++)
        {
         shift_score = v * heads * 3;
         if(u == v)
            shift_score += (uint)(u > 0);
         else
           {
            if(u > v)
               shift_score += (uint)(start > 0) + 1;
           }
         for(int dim = 0; dim < dimension; dim++)
            sc_g += scores[shift_score] * qkv[shift_v - d + dim] *
                    gradient[dimension * (v * heads + h) + d] *
                    ((float)(d == v) - sc);
        }
      grad += sc_g * qkv[q * step + dimension * h + d];
     }
   qkv_g[shift_k] = grad;
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the rpb update adam step using  |
//| the optimizer rule.                                              |
//+------------------------------------------------------------------+
__kernel void RPBUpdateAdam(__global float *target, __global const float *gradient,
                            __global float *matrix_m, ///<[in,out] Matrix of first momentum
                            __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                            const float b1,           ///< First momentum multiplier
                            const float b2            ///< Second momentum multiplier
                           )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = gradient[i];
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * (g * g);
   float delta = m / (v != 0.0f ? sqrt(v) : 1.0f);
   target[i] = weight + delta;
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
//| Runs the gte feed forward feed-forward pass on the GPU.          |
//+------------------------------------------------------------------+
__kernel void GTEFeedForward(__global float *qkv, __global float *score,
                             __global float *out, int dimension)
  {
   const size_t cur_q = get_global_id(0);
   const size_t units_q = get_global_size(0);
   const size_t cur_k = get_local_id(1);
   const size_t units_k = get_local_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   int shift_q = dimension * (cur_q + h * units_q);
   int shift_k = (cur_k + h * units_k + heads * units_q);
   int shift_v = dimension * (h * units_k + heads * (units_q + units_k));
   int shift_score_con = units_k * (cur_q * 2 * heads + h) + cur_k;
   int shift_score_notcon = units_k * (cur_q * 2 * heads + heads + h) + cur_k;
   int shift_out_con = dimension * (cur_q + h * units_q);
   int shift_out_notcon = dimension * (cur_q + units_q * (h + heads));
//---
   const uint ls_score = min((uint)units_k, (uint)LOCAL_ARRAY_SIZE);
   __local float local_score[LOCAL_ARRAY_SIZE][2];
//--- Score
   float scr = 0;
//---
   for(int d = 0; d < dimension; d++)
      scr += qkv[shift_q + d] * qkv[shift_k + d];
   scr = exp(min(scr / sqrt((float)dimension), 30.0f));
   if(cur_q == cur_k)
     {
      score[shift_score_con] = scr;
      score[shift_score_notcon] = scr;
      if(cur_k < ls_score)
        {
         local_score[cur_k][0] = scr;
         local_score[cur_k][1] = scr;
        }
     }
   else
     {
      if(abs(cur_q - cur_k) == 1)
        {
         score[shift_score_con] = scr;
         score[shift_score_notcon] = 0;
         if(cur_k < ls_score)
           {
            local_score[cur_k][0] = scr;
            local_score[cur_k][1] = 0;
           }
        }
      else
        {
         score[shift_score_con] = 0;
         score[shift_score_notcon] = scr;
         if(cur_k < ls_score)
           {
            local_score[cur_k][0] = 0;
            local_score[cur_k][1] = scr;
           }
        }
     }
   BarrierLoc
//---
   for(int k = ls_score; k < units_k; k += ls_score)
     {
      if((cur_k + k) < units_k)
        {
         local_score[cur_k][0] += score[shift_score_con + k];
         local_score[cur_k][1] += score[shift_score_notcon + k];
        }
     }
   BarrierLoc
//---
   int count = ls_score;
   do
     {
      count = (count + 1) / 2;
      if(cur_k < count)
        {
         if((cur_k + count) < units_k)
           {
            local_score[cur_k][0] += local_score[cur_k + count][0];
            local_score[cur_k][1] += local_score[cur_k + count][1];
            local_score[cur_k + count][0] = 0;
            local_score[cur_k + count][1] = 0;
           }
        }
      BarrierLoc
     }
   while(count > 1);
   BarrierLoc
//---
   score[shift_score_con] /= local_score[0][0];
   score[shift_score_notcon] /= local_score[0][1];
   BarrierLoc
//---
   shift_score_con -= cur_k;
   shift_score_notcon -= cur_k;
//---
   for(int d = 0; d < dimension; d += ls_score)
     {
      if((cur_k + d) < dimension)
        {
         float sum_con = 0;
         float sum_notcon = 0;
         for(int v = 0; v < units_k; v++)
           {
            sum_con += qkv[shift_v + v * dimension + cur_k + d] *
                       score[shift_score_con + v];
            sum_notcon += qkv[shift_v + v * dimension + cur_k + d] *
                          score[shift_score_notcon + v];
           }
         out[shift_out_con + cur_k + d] = sum_con;
         out[shift_out_notcon + cur_k + d] = sum_notcon;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the gte inside gradients gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void GTEInsideGradients(__global float *qkv, __global float *qkv_g,
                                 __global float *scores,
                                 __global float *gradient)
  {
//--- init
   const uint u = get_global_id(0);
   const uint d = get_global_id(1);
   const uint h = get_global_id(2);
   const uint units = get_global_size(0);
   const uint dimension = get_global_size(1);
   const uint heads = get_global_size(2);
//--- Calculating Value's gradients
     {
      int shift_out_con = dimension * h * units + d;
      int shift_out_notcon = dimension * units * (h + heads) + d;
      int shift_score_con = units * h + u;
      int shift_score_notcon = units * (heads + h) + u;
      int step_score = units * 2 * heads;
      int shift_v = dimension * (h * units + 2 * heads * units + u) + d;
      //---
      float sum = 0;
      //---
      for(uint i = 0; i <= units; i++)
        {
         sum += gradient[shift_out_con + i * dimension] *
                scores[shift_score_con + i * step_score];
         sum += gradient[shift_out_notcon + i * dimension] *
                scores[shift_score_notcon + i * step_score];
        }
      qkv_g[shift_v] = sum;
     }
//--- Calculating Query's gradients
     {
      int shift_q = dimension * (u + h * units) + d;
      int shift_out_con = dimension * (h * units + u) + d;
      int shift_out_notcon = dimension * (u + units * (h + heads)) + d;
      int shift_score_con = units * h;
      int shift_score_notcon = units * (heads + h);
      int shift_v = dimension * (h * units + 2 * heads * units);
      float grad = 0;
      //---
      for(int k = 0; k < units; k++)
        {
         int shift_k = (k + h * units + heads * units) + d;
         float sc_g = 0;
         float sc_con = scores[shift_score_con + k];
         float sc_notcon = scores[shift_score_notcon + k];
         for(int v = 0; v < units; v++)
            for(int dim = 0; dim < dimension; dim++)
              {
               sc_g += scores[shift_score_con + v] *
                       qkv[shift_v + v * dimension + dim] *
                       gradient[shift_out_con + dim] * ((float)(k == v) - sc_con);
               sc_g += scores[shift_score_notcon + v] *
                       qkv[shift_v + v * dimension + dim] *
                       gradient[shift_out_notcon + dim] *
                       ((float)(k == v) - sc_notcon);
              }
         grad += sc_g * qkv[shift_k];
        }
      qkv_g[shift_q] = grad;
     }
//--- Calculating Key's gradients
     {
      int shift_k = (u + (h + heads) * units) + d;
      int shift_out_con = dimension * h * units + d;
      int shift_out_notcon = dimension * units * (h + heads) + d;
      int shift_score_con = units * h + u;
      int shift_score_notcon = units * (heads + h) + u;
      int step_score = units * 2 * heads;
      int shift_v = dimension * (h * units + 2 * heads * units);
      float grad = 0;
      //---
      for(int q = 0; q < units; q++)
        {
         int shift_q = dimension * (q + h * units) + d;
         float sc_g = 0;
         float sc_con = scores[shift_score_con + u + q * step_score];
         float sc_notcon = scores[shift_score_notcon + u + q * step_score];
         for(int g = 0; g < units; g++)
           {
            for(int dim = 0; dim < dimension; dim++)
              {
               sc_g += scores[shift_score_con + g] *
                       qkv[shift_v + u * dimension + dim] *
                       gradient[shift_out_con + g * dimension + dim] *
                       ((float)(u == g) - sc_con);
               sc_g += scores[shift_score_notcon + g] *
                       qkv[shift_v + u * dimension + dim] *
                       gradient[shift_out_notcon + g * dimension + dim] *
                       ((float)(u == g) - sc_notcon);
              }
           }
         grad += sc_g * qkv[shift_q];
        }
      qkv_g[shift_k] = grad;
     }
  }
//+------------------------------------------------------------------+
//| Runs the feed forward nodef feed-forward pass on the GPU.        |
//+------------------------------------------------------------------+
__kernel void FeedForwardNODEF(__global float *matrix_w,   ///<[in] Weights matrix (m+1)*n, where m - input
                               ///< window and n - output window
                               __global float *matrix_i, ///<[in] Inputs tensor
                               __global float *matrix_o, ///<[out] Output tensor
                               int dimension,            ///< input dimension
                               float step,               ///< h
                               int activation            ///< Activation type (#ENUM_ACTIVATION)
                              )
  {
   int d = get_global_id(0);
   int dimension_out = get_global_size(0);
   int v = get_global_id(1);
   int variables = get_global_size(1);
   int i = get_global_id(2);
   int lenth = get_global_size(2);
//---
   int shift = variables * i + v;
   int input_shift = shift * dimension;
   int output_shift = shift * dimension_out + d;
   int weight_shift = (v * dimension_out + d) * (dimension + 2);
//---
   float sum = matrix_w[dimension + 1 + weight_shift] +
               matrix_w[dimension + weight_shift] * step;
//---
   for(int w = 0; w < dimension; w++)
      sum += matrix_w[w + weight_shift] * matrix_i[input_shift + w];
//---
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[output_shift] = fActivation(sum, activation);
  }
//+------------------------------------------------------------------+
//| Runs the feed forward node inp k feed-forward pass on the GPU.   |
//+------------------------------------------------------------------+
__kernel void FeedForwardNODEInpK(__global float *matrix_i,   ///<[in] Inputs tensor
                                  __global float *matrix_k1,  ///<[in] K1 tensor
                                  __global float *matrix_k2,  ///<[in] K2 tensor
                                  __global float *matrix_k3,  ///<[in] K3 tensor
                                  __global float *matrix_k4,  ///<[in] K4 tensor
                                  __global float *matrix_k5,  ///<[in] K5 tensor
                                  __global float *matrix_k6,  ///<[in] K6 tensor
                                  __global float *matrix_beta, ///<[in] beta tensor
                                  __global float *matrix_o    ///<[out] Output tensor
                                 )
  {
   int i = get_global_id(0);
//---
   float sum = matrix_i[i];
//---
   for(int b = 0; b < 6; b++)
     {
      float beta = matrix_beta[b];
      if(beta == 0.0f || isnan(beta))
         continue;
      //---
      float val = 0.0f;
      switch(b)
        {
         case 0:
            val = matrix_k1[i];
            break;
         case 1:
            val = matrix_k2[i];
            break;
         case 2:
            val = matrix_k3[i];
            break;
         case 3:
            val = matrix_k4[i];
            break;
         case 4:
            val = matrix_k5[i];
            break;
         case 5:
            val = matrix_k6[i];
            break;
        }
      if(val == 0.0f || isnan(val))
         continue;
      //---
      sum += val * beta;
     }
//---
   matrix_o[i] = sum;
  }
//+------------------------------------------------------------------+
//| Computes the hidden gradient node inp k gradient contributions on|
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void HiddenGradientNODEInpK(__global float *matrix_ig,  ///<[in] Inputs tensor
                                     __global float *matrix_k1g, ///<[in] K1 tensor
                                     __global float *matrix_k2g, ///<[in] K2 tensor
                                     __global float *matrix_k3g, ///<[in] K3 tensor
                                     __global float *matrix_k4g, ///<[in] K4 tensor
                                     __global float *matrix_k5g, ///<[in] K5 tensor
                                     __global float *matrix_k6g, ///<[in] K6 tensor
                                     __global float *matrix_beta, ///<[in] beta tensor
                                     __global float *matrix_og   ///<[out] Output tensor
                                    )
  {
   int i = get_global_id(0);
//---
   float grad = IsNaNOrInf(matrix_og[i], 0);
   matrix_ig[i] = grad;
//---
   for(int b = 0; b < 6; b++)
     {
      float beta = IsNaNOrInf(matrix_beta[b], 0.0f);
      //---
      float val = IsNaNOrInf(beta * grad, 0.0f);
      switch(b)
        {
         case 0:
            matrix_k1g[i] = val;
            break;
         case 1:
            matrix_k2g[i] = val;
            break;
         case 2:
            matrix_k3g[i] = val;
            break;
         case 3:
            matrix_k4g[i] = val;
            break;
         case 4:
            matrix_k5g[i] = val;
            break;
         case 5:
            matrix_k6g[i] = val;
            break;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the hidden gradient nodef gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void HiddenGradientNODEF(__global float *matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - input
                                  ///< window and n - output window
                                  __global float *matrix_g, ///<[in] Gradient tensor
                                  __global float *matrix_i, ///<[in] Inputs tensor
                                  __global float *matrix_ig, ///<[out] Inputs Gradient tensor
                                  int dimension_out,         ///< output dimension
                                  int activation             ///< Input Activation type (#ENUM_ACTIVATION)
                                 )
  {
   int d = get_global_id(0);
   int dimension = get_global_size(0);
   int v = get_global_id(1);
   int variables = get_global_size(1);
   int i = get_global_id(2);
   int lenth = get_global_size(2);
//---
   int shift = variables * i + v;
   int input_shift = shift * dimension + d;
   int output_shift = shift * dimension_out;
   int weight_step = (dimension + 2);
   int weight_shift = (v * dimension_out) * weight_step + d;
//---
   float sum = 0;
//---
   for(int k = 0; k < dimension_out; k++)
      sum +=
         matrix_g[output_shift + k] * matrix_w[weight_shift + k * weight_step];
   if(isnan(sum))
      sum = 0;
//---
   float out = matrix_i[input_shift];
//---
   matrix_ig[input_shift] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the nodef update weights adam   |
//| step using the optimizer rule.                                   |
//+------------------------------------------------------------------+
__kernel void NODEF_UpdateWeightsAdam(__global float *matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
                                      ///< number of neurons in previous layer and n -
                                      ///< number of neurons in current layer
                                      __global const float *matrix_gk1, ///<[in] Tensor of gradients at k1
                                      __global const float *matrix_gk2, ///<[in] Tensor of gradients at k2
                                      __global const float *matrix_gk3, ///<[in] Tensor of gradients at k3
                                      __global const float *matrix_gk4, ///<[in] Tensor of gradients at k4
                                      __global const float *matrix_gk5, ///<[in] Tensor of gradients at k5
                                      __global const float *matrix_gk6, ///<[in] Tensor of gradients at k6
                                      __global const float *matrix_ik1, ///<[in] Inputs tensor
                                      __global const float *matrix_ik2, ///<[in] Inputs tensor
                                      __global const float *matrix_ik3, ///<[in] Inputs tensor
                                      __global const float *matrix_ik4, ///<[in] Inputs tensor
                                      __global const float *matrix_ik5, ///<[in] Inputs tensor
                                      __global const float *matrix_ik6, ///<[in] Inputs tensor
                                      __global float *matrix_m,        ///<[in,out] Matrix of first momentum
                                      __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
                                      __global const float *alpha,     ///< h
                                      const int lenth,                  ///< Number of inputs
                                      const float l,                    ///< Learning rates
                                      const float b1,                   ///< First momentum multiplier
                                      const float b2                    ///< Second momentum multiplier
                                     )
  {
   const int d_in = get_global_id(0);
   const int dimension_in = get_global_size(0);
   const int d_out = get_global_id(1);
   const int dimension_out = get_global_size(1);
   const int v = get_global_id(2);
   const int variables = get_global_id(2);
//---
   const int weight_shift = (v * dimension_out + d_out) * dimension_in;
   const int input_step = variables * (dimension_in - 2);
   const int input_shift = v * (dimension_in - 2) + d_in;
   const int output_step = variables * dimension_out;
   const int output_shift = v * dimension_out + d_out;
//---
   float weight = matrix_w[weight_shift];
   float g = 0;
//---
   for(int i = 0; i < lenth; i++)
     {
      int shift_g = i * output_step + output_shift;
      int shift_i = i * input_step + input_shift;
      switch(dimension_in - d_in)
        {
         case 1:
            g += matrix_gk1[shift_g] + matrix_gk2[shift_g] + matrix_gk3[shift_g] +
                 matrix_gk4[shift_g] + matrix_gk5[shift_g] + matrix_gk6[shift_g];
            break;
         case 2:
            g += matrix_gk1[shift_g] * alpha[0] + matrix_gk2[shift_g] * alpha[1] +
                 matrix_gk3[shift_g] * alpha[2] + matrix_gk4[shift_g] * alpha[3] +
                 matrix_gk5[shift_g] * alpha[4] + matrix_gk6[shift_g] * alpha[5];
            break;
         default:
            g += matrix_gk1[shift_g] * matrix_ik1[shift_i] +
                 matrix_gk2[shift_g] * matrix_ik2[shift_i] +
                 matrix_gk3[shift_g] * matrix_ik3[shift_i] +
                 matrix_gk4[shift_g] * matrix_ik4[shift_i] +
                 matrix_gk5[shift_g] * matrix_ik5[shift_i] +
                 matrix_gk6[shift_g] * matrix_ik6[shift_i];
            break;
        }
     }
//---
   float mt = b1 * matrix_m[weight_shift] + (1 - b1) * g;
   float vt = b2 * matrix_v[weight_shift] + (1 - b2) * (g * g);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      matrix_w[weight_shift] =
         matrix_w[weight_shift] + delta;
   matrix_m[weight_shift] = mt;
   matrix_v[weight_shift] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the time derivative kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void TimeDerivative(__global float *qkv, __global float *dqkv,
                             int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//---
   const int shift = 3 * heads * variables * dimension;
   const int shift_query =
      pos * shift + (3 * variable * heads + head) * dimension;
   const int shift_key = shift_query + heads * dimension;
//---
//---
   for(int i = 0; i < dimension; i++)
     {
      //--- dQ/dt
        {
         int count = 0;
         float delta = 0;
         float value = qkv[shift_query + i];
         if(pos > 0)
           {
            delta = value - qkv[shift_query + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            delta += qkv[shift_query + i + shift] - value;
            count++;
           }
         if(count > 0)
            dqkv[shift_query + i] = delta / count;
        }
      //--- dK/dt
        {
         int count = 0;
         float delta = 0;
         float value = qkv[shift_key + i];
         if(pos > 0)
           {
            delta = value - qkv[shift_key + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            delta += qkv[shift_key + i + shift] - value;
            count++;
           }
         if(count > 0)
            dqkv[shift_key + i] = delta / count;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the hidden gradient time derivative gradient            |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void HiddenGradientTimeDerivative(__global float *qkv_g,
      __global float *dqkv_g,
      int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//---
   const int shift = 3 * heads * variables * dimension;
   const int shift_query =
      pos * shift + (3 * variable * heads + head) * dimension;
   const int shift_key = shift_query + heads * dimension;
//---
//---
   for(int i = 0; i < dimension; i++)
     {
      //--- dQ/dt
        {
         int count = 0;
         float grad = 0;
         float current = dqkv_g[shift_query + i];
         if(pos > 0)
           {
            grad += current - dqkv_g[shift_query + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            grad += dqkv_g[shift_query + i + shift] - current;
            count++;
           }
         if(count > 0)
            grad /= count;
         qkv_g[shift_query + i] += grad;
        }
      //--- dK/dt
        {
         int count = 0;
         float grad = 0;
         float current = dqkv_g[shift_key + i];
         if(pos > 0)
           {
            grad += current - dqkv_g[shift_key + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            grad += dqkv_g[shift_key + i + shift] - current;
            count++;
           }
         if(count > 0)
            grad /= count;
         qkv_g[shift_key + i] += dqkv_g[shift_key + i] + grad;
        }
     }
  }
//+------------------------------------------------------------------+
//| Runs the feed forward cont att feed-forward pass on the GPU.     |
//+------------------------------------------------------------------+
__kernel void FeedForwardContAtt(__global float *qkv, __global float *dqkv,
                                 __global float *score, __global float *out,
                                 int dimension,
                                 int heads)
  {
   const size_t query = get_global_id(0);
   const size_t key = get_global_id(1);
   const size_t variable = get_global_id(2);
   const size_t queris = get_global_size(0);
   const size_t keis = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const uint ls_score = min((uint)keis, (uint)LOCAL_ARRAY_SIZE);
   __local float local_score[LOCAL_ARRAY_SIZE];
//---
//---
   for(int head = 0; head < heads; head++)
     {
      const int shift = 3 * heads * variables * dimension;
      const int shift_query =
         query * shift + (3 * variable * heads + head) * dimension;
      const int shift_key =
         key * shift + (3 * variable * heads + heads + head) * dimension;
      const int shift_out =
         dimension * (heads * (query * variables + variable) + head);
      int shift_score = keis * (heads * (query * variables + variable) + head) + key;
      //--- Score
      float scr = 0;
      for(int d = 0; d < dimension; d++)
         scr += qkv[shift_query + d] * dqkv[shift_key + d] +
                qkv[shift_key + d] * dqkv[shift_query + d];
      scr = exp(min(scr / sqrt((float)dimension), 30.0f));
      score[shift_score] = scr;
      BarrierLoc
      //---
      if(key < ls_score)
        {
         local_score[key] = scr;
         for(int k = ls_score + key; k < keis; k += ls_score)
            local_score[key] += score[shift_score + k];
        }
      BarrierLoc
      //---
      int count = ls_score;
      do
        {
         count = (count + 1) / 2;
         if(key < count)
           {
            if((key + count) < keis)
              {
               local_score[key] += local_score[key + count];
               local_score[key + count] = 0;
              }
           }
         BarrierLoc
        }
      while(count > 1);
      //---
      score[shift_score] /= local_score[0];
      BarrierLoc
      //---
      shift_score -= key;
      for(int d = key; d < dimension; d += keis)
        {
         float sum = 0;
         int shift_value = (3 * variable * heads + 2 * heads + head) * dimension + d;
         for(int v = 0; v < keis; v++)
            sum += qkv[shift_value + v * shift] * score[shift_score + v];
         out[shift_out + d] = sum;
        }
      BarrierLoc
     }
//---
  }
//+------------------------------------------------------------------+
//| Computes the hidden gradient cont att gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void HiddenGradientContAtt(__global float *qkv, __global float *qkv_g,
                                    __global float *dqkv,
                                    __global float *dqkv_g,
                                    __global float *score,
                                    __global float *out_g, const int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//--- Value gradient
     {
      const int shift_value =
         dimension * (heads * (3 * variables * pos + 3 * variable + 2) + head);
      const int shift_out = dimension * (head + variable * heads);
      const int shift_score = total * (variable * heads + head);
      const int step_out = variables * heads * dimension;
      const int step_score = variables * heads * total;
      //---
      //---
      for(int d = 0; d < dimension; d++)
        {
         float sum = 0;
         for(int g = 0; g < total; g++)
            sum += out_g[shift_out + g * step_out + d] *
                   score[shift_score + g * step_score];
         qkv_g[shift_value + d] = sum;
        }
     }
//--- Query gradient
     {
      const int shift_out =
         dimension * (heads * (pos * variables + variable) + head);
      const int step = 3 * variables * heads * dimension;
      const int shift_query =
         dimension * (3 * heads * variable + head) + pos * step;
      const int shift_key = dimension * (heads * (3 * variable + 1) + head);
      const int shift_value = dimension * (heads * (3 * variable + 2) + head);
      const int shift_score =
         total * (heads * (pos * variables + variable) + head);
      //--- Score gradient
      //---
      for(int k = 0; k < total; k++)
        {
         float score_grad = 0;
         float scr = score[shift_score + k];
         for(int v = 0; v < total; v++)
           {
            float grad = 0;
            for(int d = 0; d < dimension; d++)
               grad += qkv[shift_value + v * step + d] * out_g[shift_out + d];
            score_grad += score[shift_score + v] * grad * ((float)(pos == v) - scr);
           }
         score_grad /= sqrt((float)dimension);
         //--- Query gradient
         for(int d = 0; d < dimension; d++)
           {
            if(k == 0)
              {
               dqkv_g[shift_query + d] = score_grad * qkv[shift_key + k * step + d];
               qkv_g[shift_query + d] = score_grad * dqkv[shift_key + k * step + d];
              }
            else
              {
               dqkv_g[shift_query + d] += score_grad * qkv[shift_key + k * step + d];
               qkv_g[shift_query + d] += score_grad * dqkv[shift_key + k * step + d];
              }
           }
        }
     }
//--- Key gradient
     {
      const int shift_key =
         dimension * (heads * (3 * variables * pos + 3 * variable + 1) + head);
      const int shift_out = dimension * (head + variable * heads);
      const int step_out = variables * heads * dimension;
      const int step = 3 * variables * heads * dimension;
      const int shift_query = dimension * (3 * heads * variable + head);
      const int shift_value =
         dimension * (heads * (3 * variable + 2) + head) + pos * step;
      const int shift_score = total * (heads * variable + head);
      const int step_score = variables * heads * total;
      //--- Score gradient
      //---
      for(int q = 0; q < total; q++)
        {
         float score_grad = 0;
         float scr = score[shift_score + q * step_score];
         for(int g = 0; g < total; g++)
           {
            float grad = 0;
            for(int d = 0; d < dimension; d++)
               grad += qkv[shift_value + d] * out_g[shift_out + d + g * step_out] / sqrt((float)dimension);
            score_grad += score[shift_score + q * step_score + g] * grad * ((float)(q == pos) - scr);
           }
         //--- Key gradient
         for(int d = 0; d < dimension; d++)
           {
            if(q == 0)
              {
               dqkv_g[shift_key + d] = qkv[shift_query + q * step + d] * score_grad;
               qkv_g[shift_key + d] = score_grad * dqkv[shift_query + q * step + d];
              }
            else
              {
               qkv_g[shift_key + d] += score_grad * dqkv[shift_query + q * step + d];
               dqkv_g[shift_key + d] += score_grad * qkv[shift_query + q * step + d];
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Runs the rev in feed forward feed-forward pass on the GPU.       |
//+------------------------------------------------------------------+
__kernel void RevInFeedForward(__global float *inputs, __global float *options,
                               __global float *output, int options_size,
                               int optimization)
  {
   int n = get_global_id(0);
   int shift = (n * (optimization == 0 ? 7 : 9)) % options_size;
//---
   float mean = options[shift];
   float variance = options[shift + 1];
   float k = options[shift + 3];
//---
   float res = 0;
   res = sqrt(variance) * (inputs[n] - options[shift + 4]) / fmax(k, 0.001f) + mean;
   if(isnan(res))
      res = 0;
//---
   output[n] = res;
  }
//+------------------------------------------------------------------+
//| Implements the rev in hidden graddient kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void RevInHiddenGraddient(__global float *inputs, __global float *inputs_gr,
                                   __global float *options, __global float *output_gr,
                                   int options_size,
                                   int optimization,
                                   int activation)
  {
   int n = get_global_id(0);
   int shift = (n * (optimization == 0 ? 7 : 9)) % options_size;
//---
   float variance = options[shift + 1];
   float inp = inputs[n];
   float k = options[shift + 3];
//---
   float res = sqrt(variance) * output_gr[n];
   if(fabs(k) > 1)
      res /= k;
   if(isnan(res))
      res = 0;
//---
   inputs_gr[n] = Deactivation(res, inp, activation);
  }
//+------------------------------------------------------------------+
//| Implements the activation kernel on the GPU.                     |
//+------------------------------------------------------------------+
__kernel void Activation(__global const float *inputs,
                         __global float *outputs,
                         const int activation)
  {
   int n = get_global_id(0);
//---
   float res = IsNaNOrInf(inputs[n], 0);
//---
   outputs[n] = fActivation(res, activation);
  }
//+------------------------------------------------------------------+
//| Implements the de activation kernel on the GPU.                  |
//+------------------------------------------------------------------+
__kernel void DeActivation(__global const float *inputs, __global float *inputs_gr,
                           __global const float *output_gr, const int activation)
  {
   int n = get_global_id(0);
//---
   float inp = inputs[n];
   float res = IsNaNOrInf(output_gr[n], 0);
//---
   inputs_gr[n] = Deactivation(res, inp, activation);
  }
//+------------------------------------------------------------------+
//| Implements the patch create kernel on the GPU.                   |
//+------------------------------------------------------------------+
__kernel void PatchCreate(__global float *inputs,
                          __global float *weights,
                          __global float *outputs,
                          int inputs_total,
                          int window_in,
                          int step,
                          int activation
                         )
  {
   const int i = get_global_id(0);
   const int w = get_global_id(1);
   const int v = get_global_id(2);
   const int window_out = get_global_size(1);
   const int variables = get_global_size(2);
//---
   const int shift_in = i * step * variables + v;
   const int shift_out = (i * variables + v) * window_out + w;
   const int shift_weights = (window_in + 1) * (v * window_out + w);
//---
   float res = weights[shift_weights + window_in];
//---
   for(int p = 0; p < window_in; p++)
      if((shift_in + p * variables) < inputs_total)
         res += inputs[shift_in + p * variables] * weights[shift_weights + p];
   if(isnan(res))
      res = 0;
//---
   outputs[shift_out] = fActivation(res, activation);
  }
//+------------------------------------------------------------------+
//| Computes the patch hidden gradient gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void PatchHiddenGradient(__global float *inputs,
                                  __global float *inputs_gr,
                                  __global float *weights,
                                  __global float *outputs_gr,
                                  int window_in,
                                  int step,
                                  int window_out,
                                  int outputs_total,
                                  int activation
                                 )
  {
   const int i = get_global_id(0);
   const int v = get_global_id(1);
   const int variables = get_global_size(1);
//---
   const int w_start = i % step;
   const int r_start = max((i - window_in + step) / step, 0);
   int total = (window_in - w_start + step - 1) / step;
   total = min((i + step) / step, total);
//---
   float grad = 0;
//---
   for(int p = 0; p < total; p ++)
     {
      int row = r_start + p;
      if(row >= outputs_total)
         break;
      for(int wo = 0; wo < window_out; wo++)
        {
         int shift_g = (row * variables + v) * window_out + wo;
         int shift_w = v * (window_in + 1) * window_out + w_start + (total - p - 1) * step + wo * (window_in + 1);
         grad += outputs_gr[shift_g] * weights[shift_w];
        }
     }
//---
   float inp = inputs[i * variables + v];
//---
   inputs_gr[i * variables + v] = Deactivation(grad, inp, activation);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the patch update weights adam   |
//| step using the optimizer rule.                                   |
//+------------------------------------------------------------------+
__kernel void PatchUpdateWeightsAdam(__global float *weights,
                                     __global const float *outputs_gr,
                                     __global const float *inputs,
                                     __global float *weights_m,
                                     __global float *weights_v,
                                     const int inputs_total,
                                     const float l,
                                     const float b1,
                                     const float b2,
                                     int step
                                    )
  {
   const int c = get_global_id(0);
   const int r = get_global_id(1);
   const int v = get_global_id(2);
   const int window_in = get_global_size(0) - 1;
   const int window_out = get_global_size(1);
   const int variables = get_global_size(2);
//---
   const int start_input = c * variables + v;
   const int step_input = step * variables;
   const int start_out = v * window_out + r;
   const int step_out = variables * window_out;
   const int total = inputs_total / (variables * step);
//---
   float grad = 0;
//---
   for(int p = 0; p < total; p++)
     {
      int i = start_input + i * step_input;
      int o = start_out + i * step_out;
      grad += (c == window_in ? 1 : inputs[i]) * outputs_gr[0];
     }
   if(isnan(grad))
      grad = 0;
//---
   const int shift_weights = (window_in + 1) * (window_out * v + r) + c;
//---
   float weight = weights[shift_weights];
   float mt = b1 * weights_m[shift_weights] + (1 - b1) * grad;
   float vt = b2 * weights_v[shift_weights] + (1 - b2) * (grad * grad);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      weights[shift_weights] = weight + delta;
   weights_m[shift_weights] = mt;
   weights_v[shift_weights] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the mat mult kernel on the GPU.                       |
//+------------------------------------------------------------------+
__kernel void MatMult(__global const float *matr1,
                      __global const float *matr2,
                      __global float *result,
                      int dimension,
                       int multvarsecond,
                       int offset_matr1,
                       int offset_matr2,
                       int offset_result,
                       __global float *numeric_flags,
                       const int flags_enabled)
  {
   size_t row = get_global_id(0);
   size_t col = get_global_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_global_size(1);
//---
   int shift1 = offset_matr1 + RCtoFlat(row, 0, rows, dimension, var);
   int shift2 = offset_matr2 + RCtoFlat(0, col, dimension, cols, multvarsecond * var);
   int shift_out = offset_result + RCtoFlat(row, col, rows, cols, var);
//---
    float res = 0;
    int invalid = 0;
//---
    for(int i = 0; i < dimension; i++)
      {
       if(flags_enabled)
         {
          const float value1 = matr1[shift1 + i];
          const float value2 = matr2[shift2 + i * cols];
          const float product = value1 * value2;
          invalid |= isnan(value1) || isinf(value1) || isnan(value2) || isinf(value2) ||
                     isnan(product) || isinf(product);
         }
//--- Preserve the legacy sanitized accumulation after checking its raw operands.
       res += IsNaNOrInf(matr1[shift1 + i] * matr2[shift2 + i * cols], 0);
       if(flags_enabled)
          invalid |= isnan(res) || isinf(res);
      }
//---
    if(flags_enabled)
      {
       invalid |= isnan(res) || isinf(res);
       numeric_flags[RCtoFlat(row, col, rows, cols, var)] = (invalid ? 1.0f : 0.0f);
      }
    result[shift_out] = res;
  }
//+------------------------------------------------------------------+
//| Computes the mat mult grad gradient contributions on the GPU.    |
//+------------------------------------------------------------------+
__kernel void MatMultGrad(__global const float *matr1,
                          __global float *matr1_gr,
                          __global const float *matr2,
                          __global float *matr2_gr,
                          __global const float *result_gr,
                          int dimension,
                          int multvarsecond,
                          int offset_matr1,
                          int offset_matr1_gr,
                          int offset_matr2,
                          int offset_matr2_gr,
                          int offset_result_gr)
  {
   size_t row = get_global_id(0);
   size_t col = get_global_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_global_size(1);
//--- Tensor offsets select the current row and variable views.
   int shift1 = offset_matr1 + (row  + var * rows) * dimension;
   int shift1_gr = offset_matr1_gr + (row + var * rows) * dimension;
   int shift2 = offset_matr2 + var * dimension * cols * multvarsecond;
   int shift2_gr = offset_matr2_gr + var * dimension * cols * multvarsecond;
   int shift_out = offset_result_gr + (row + var * rows) * cols;
//--- First-matrix gradients use disjoint full-feature lanes.
   for(int k = 0; k * cols < dimension; k++)
     {
      int feature = col + k * cols;
      if(feature >= dimension)
         continue;
      float grad = 0;
      for(int i = 0; i < cols; i++)
         grad += IsNaNOrInf(result_gr[shift_out + i] * matr2[shift2 + feature * cols + i], 0);
      matr1_gr[shift1_gr + feature] = IsNaNOrInf(grad, 0);
     }
//--- Second-matrix gradients reduce rows and all shared variables per entry.
   for(int k = 0; k * rows < dimension; k++)
     {
      int feature = row + k * rows;
      if(feature >= dimension)
         continue;
      if(multvarsecond || var == 0)
        {
         float grad = 0;
         if(multvarsecond)
           {
            shift_out = offset_result_gr + var * rows * cols + col;
            shift1 = offset_matr1 + var * rows * dimension + feature;
            for(int i = 0; i < rows; i++)
               grad += IsNaNOrInf(result_gr[shift_out + i * cols] * matr1[shift1 + i * dimension], 0);
           }
         else
           {
            for(int v = 0; v < get_global_size(2); v++)
              {
               shift_out = offset_result_gr + v * rows * cols + col;
               shift1 = offset_matr1 + v * rows * dimension + feature;
               for(int i = 0; i < rows; i++)
                  grad += IsNaNOrInf(result_gr[shift_out + i * cols] * matr1[shift1 + i * dimension], 0);
              }
           }
         matr2_gr[shift2_gr + col + feature * cols] = IsNaNOrInf(grad, 0);
        }
     }
  }
//+------------------------------------------------------------------+
//| Runs the fft kernel on the GPU.                                  |
//+------------------------------------------------------------------+
__kernel void FFT(__global float *inputs_re,
                  __global float *inputs_im,
                  __global float *outputs_re,
                  __global float *outputs_im,
                  const int input_window,
                  const int input_complex,
                  const int output_window,
                  const int reverse
                 )
  {
   size_t variable = get_global_id(0);
//---
   const ulong N = output_window;
   const ulong N2 = N / 2;
   const ulong inp_shift = input_window * variable;
   const ulong out_shift = output_window * variable;
//---
   uint target = 0;
//---
   for(uint position = 0; position < N; position++)
     {
      if(target > position)
        {
         outputs_re[out_shift + position] = (target < input_window ? inputs_re[inp_shift + target] : 0);
         outputs_im[out_shift + position] = ((target < input_window && input_complex) ? inputs_im[inp_shift + target] : 0);
         outputs_re[out_shift + target] = inputs_re[inp_shift + position];
         outputs_im[out_shift + target] = (input_complex ? inputs_im[inp_shift + position] : 0);
        }
      else
        {
         outputs_re[out_shift + position] = inputs_re[inp_shift + position];
         outputs_im[out_shift + position] = (input_complex ? inputs_im[inp_shift + position] : 0);
        }
      unsigned int mask = N;
      while(target & (mask >>= 1))
         target &= ~mask;
      target |= mask;
     }
   float real = 0, imag = 0;
//---
   for(int len = 2; len <= (int)N; len <<= 1)
     {
      float w_real = (float)cos(2 * M_PI_F / len);
      float w_imag = (float)sin(2 * M_PI_F / len);
      for(int i = 0; i < (int)N; i += len)
        {
         float cur_w_real = 1;
         float cur_w_imag = 0;
         for(int j = 0; j < len / 2; j++)
           {
            real = cur_w_real * outputs_re[out_shift + i + j + len / 2] - cur_w_imag * outputs_im[out_shift + i + j + len / 2];
            imag = cur_w_imag * outputs_re[out_shift + i + j + len / 2] + cur_w_real * outputs_im[out_shift + i + j + len / 2];
            outputs_re[out_shift + i + j + len / 2] = outputs_re[out_shift + i + j] - real;
            outputs_im[out_shift + i + j + len / 2] = outputs_im[out_shift + i + j] - imag;
            outputs_re[out_shift + i + j] += real;
            outputs_im[out_shift + i + j] += imag;
            real = cur_w_real * w_real - cur_w_imag * w_imag;
            cur_w_imag = cur_w_imag * w_real + cur_w_real * w_imag;
            cur_w_real = real;
           }
        }
     }
//---
   if(reverse)
     {
      outputs_re[0] /= N;
      outputs_im[0] /= N;
      outputs_re[N2] /= N;
      outputs_im[N2] /= N;
      //---
      for(int i = 1; i < N2; i++)
        {
         real = outputs_re[i] / N;
         imag = outputs_im[i] / N;
         outputs_re[i] = outputs_re[N - i] / N;
         outputs_im[i] = outputs_im[N - i] / N;
         outputs_re[N - i] = real;
         outputs_im[N - i] = imag;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the complex layer kernel on the GPU.                  |
//+------------------------------------------------------------------+
__kernel void ComplexLayer(__global float *inputs_re,
                           __global float *inputs_im,
                           __global float *outputs_re,
                           __global float *outputs_im
                          )
  {
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);
   size_t total_i = get_global_size(0);
   size_t total_j = get_global_size(1);
   uint shift = i * total_j + j;
//---
   outputs_re[shift] = inputs_re[shift] - inputs_im[shift];
   outputs_im[shift] = inputs_im[shift] + inputs_re[shift];
  }
//+------------------------------------------------------------------+
//| Computes the complex layer gradient gradient contributions on the|
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void ComplexLayerGradient(__global float *inputs_re,
                                   __global float *inputs_im,
                                   __global float *outputs_re,
                                   __global float *outputs_im
                                  )
  {
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);
   size_t total_i = get_global_size(0);
   size_t total_j = get_global_size(1);
   uint shift = i * total_j + j;
//---
   inputs_re[shift] = outputs_re[shift] + outputs_im[shift];
   inputs_im[shift] = outputs_im[shift] - outputs_re[shift];
  }
//+------------------------------------------------------------------+
//| Computes the gradient msa gradient contributions on the GPU.     |
//+------------------------------------------------------------------+
__kernel void GradientMSA(__global float *matrix_t, ///<[in] Target tensor
                          __global float *matrix_o, ///<[in] Forecast tensor
                          __global float *matrix_g ///<[out] Tensor of gradients
                         )
  {
   int i = get_global_id(0);
   matrix_g[i] = matrix_t[i] - matrix_o[i];
  }
//+------------------------------------------------------------------+
//| Computes the cumulative gradient gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void CumulativeGradient(__global float *gradient1,
                                 __global float *gradient2,
                                 __global float *gradient_out,
                                 float alpha
                                )
  {
   int i = get_global_id(0);
   gradient_out[i] = alpha * gradient1[i] + (1 - alpha) * gradient2[i];
  }
//+------------------------------------------------------------------+
//| Implements the complex mul helper on the GPU.                    |
//+------------------------------------------------------------------+
inline float2 ComplexMul(const float2 a, const float2 b)
  {
   float2 result = 0;
   result.x = IsNaNOrInf(a.x * b.x - a.y * b.y, 0);
   result.y = IsNaNOrInf(a.x * b.y + a.y * b.x, 0);
   return result;
  }
//+------------------------------------------------------------------+
//| Implements the complex div helper on the GPU.                    |
//+------------------------------------------------------------------+
inline float2 ComplexDiv(const float2 a, const float2 b)
  {
   float2 result = 0;
   float z = IsNaNOrInf(b.x * b.x + b.y * b.y, 1);
   if(z > 0)
     {
      result.x = IsNaNOrInf(a.x * b.x + a.y * b.y, 0) / z;
      result.y = IsNaNOrInf(a.y * b.x - a.x * b.y, 0) / z;
     }
   return result;
  }
//+------------------------------------------------------------------+
//| Implements the complex abs helper on the GPU.                    |
//+------------------------------------------------------------------+
inline float ComplexAbs(float2 a)
  {
   return sqrt(IsNaNOrInf(a.x * a.x + a.y * a.y, 0));
  }
//+------------------------------------------------------------------+
//| Implements the complex sqrt helper on the GPU.                   |
//+------------------------------------------------------------------+
inline float2 ComplexSqrt(float2 a)
  {
   float2 result = 0;
   float z = ComplexAbs(a);
   result.x = sqrt((z + IsNaNOrInf(a.x, 0)) / 2);
   result.y = sqrt((z - IsNaNOrInf(a.x, 0)) / 2);
   if(a.y < 0)
      result.y *= (-1);
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Implements the complex exp helper on the GPU.                    |
//+------------------------------------------------------------------+
inline float2 ComplexExp(float2 a)
  {
   float2 result = exp(clamp(IsNaNOrInf(a.x, 0), -20.0f, 20.0f));
   result.x *= IsNaNOrInf(cos(a.y), 0);
   result.y *= IsNaNOrInf(sin(a.y), 0);
   return result;
  }
//+------------------------------------------------------------------+
//| Implements the complex tanh helper on the GPU.                   |
//+------------------------------------------------------------------+
inline float2 ComplexTanh(float2 a)
  {
   float sinh_re = sinh(a.x);
   float cosh_re = cosh(a.x);
   float sin_im = sin(a.y);
   float cos_im = cos(a.y);
//---
   float2 sinh_a = 0;
   float2 cosh_a = 0;
   sinh_a.x = sinh_re * cos_im;
   sinh_a.y = cosh_re * sin_im;
   cosh_a.x = cosh_re * cos_im;
   cosh_a.y = sinh_re * sin_im;
//---
   return ComplexDiv(sinh_a, cosh_a);
  }
//+------------------------------------------------------------------+
//| Runs the feed forward complex conv feed-forward pass on the GPU. |
//+------------------------------------------------------------------+
__kernel void FeedForwardComplexConv(__global const float2* __attribute__((aligned(8))) matrix_w, ///<[in] Weights matrix (m+1)*n, where m - input
                                     ///< window and n - output window
                                     __global const float2* __attribute__((aligned(8))) matrix_i, ///<[in] Inputs tensor
                                     __global float2* __attribute__((aligned(8))) matrix_o, ///<[out] Output tensor
                                     const int inputs,               ///< Number of inputs
                                     const int step,                 ///< Step size
                                     const int window_in,            ///< Size of input window
                                     const int activation            ///< Activation type (#ENUM_ACTIVATION)
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t units = get_global_size(0);
   const size_t out = get_global_id(1);
   const size_t w_out = get_global_size(1);
   const size_t var = get_global_id(2);
   const size_t variables = get_global_size(2);
//---
   int w_in = window_in;
   int shift_out = w_out * (i + units * var);
   int shift_in = step * i;
   int shift = (w_in + 1) * (out + var * w_out);
   int stop = (w_in <= (inputs - shift_in) ? w_in : (inputs - shift_in));
   shift_in += + inputs * var;
//---
   float2 sum = ComplexMul((float2)(1, 0), matrix_w[shift + w_in]);
//---
   for(int k = 0; k < stop; k ++)
      sum += IsNaNOrInf2(ComplexMul(matrix_i[shift_in + k], matrix_w[shift + k]), (float2)0);
//---
   switch(activation)
     {
      case 0:
         sum = ComplexTanh(sum);
         break;
      case 1:
         sum = ComplexDiv((float2)(1, 0), (float2)(1, 0) + ComplexExp(-sum));
         break;
      case 2:
         if(sum.x < 0)
           {
            sum.x *= 0.01f;
            sum.y *= 0.01f;
           }
         break;
      default:
         break;
     }
   matrix_o[out + shift_out] = sum;
  }
//+------------------------------------------------------------------+
//| Computes the calc hidden gradient complex conv gradient          |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientComplexConv(__global const float2* __attribute__((aligned(8))) matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - input
      ///< window and n - output window
      __global const float2* __attribute__((aligned(8))) matrix_g, ///<[in] Tensor of gradients at current layer
      __global const float2* __attribute__((aligned(8))) matrix_o, ///<[in] Output tensor
      __global float2* __attribute__((aligned(8))) matrix_ig, ///<[out] Tensor of gradients at previous layer
      const int outputs,               ///< Number of outputs
      const int step,                  ///< Step size
      const int window_in,             ///< Size of input window
      const int window_out,            ///< Size of output window
      const int activation,            ///< Activation type (#ENUM_ACTIVATION)
      const int shift_out              ///< Shift in output and gradient buffer
                                           )
  {
   const size_t i = get_global_id(0);
   const size_t inputs = get_global_size(0);
   const size_t var = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   float2 sum = (float2)0;
   float2 out = matrix_o[i];
   int start = i - window_in + step;
   start = max((start - start % step) / step, 0) + var * inputs;
   int stop = (i + step - 1) / step;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
   stop += var * outputs;
//---
   for(int h = 0; h < window_out; h ++)
     {
      for(int k = start; k < stop; k++)
        {
         int shift_g = k * window_out + h;
         int shift_w = (stop - k - 1) * step + i % step + h * (window_in + 1);
         if(shift_g >= outputs || shift_w >= (window_in + 1) * window_out)
            break;
         sum += ComplexMul(matrix_g[shift_out + shift_g], matrix_w[shift_w]);
        }
     }
   sum = IsNaNOrInf2(sum, (float2)0);
//---
   switch(activation)
     {
      case 0:
         sum = ComplexMul(sum, (float2)1.0f - ComplexMul(out, out));
         break;
      case 1:
         sum = ComplexMul(sum, ComplexMul(out, (float2)1.0f - out));
         break;
      case 2:
         if(out.x < 0.0f)
           {
            sum.x *= 0.01f;
            sum.y *= 0.01f;
           }
         break;
      default:
         break;
     }
   matrix_ig[i] = clamp(sum, (float2)(-MAX_GRAD), (float2)MAX_GRAD);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights complex conv |
//| momentum step using the optimizer rule.                          |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsComplexConvMomentum(__global float2* __attribute__((aligned(8))) matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< input window and n - output window
      __global float2* __attribute__((aligned(8))) matrix_g, ///<[in] Tensor of gradients at current layer
      __global float2* __attribute__((aligned(8))) matrix_i, ///<[in] Inputs tensor
      __global float2* __attribute__((aligned(8))) matrix_dw, ///<[in,out] Matrix of delta weights in last correction
      int inputs,     ///< Number of inputs
      float learning_rates, ///< Learning rates
      float momentum,       ///< Momentum multiplier
      int window_in,        ///< Size of input window
      int window_out,       ///< Size of output window
      int step              ///< Step size
                                              )
  {
   const size_t i = get_global_id(0);
   const size_t total_w = get_global_size(0);
   const size_t var = get_global_id(1);
   const size_t variables = get_global_size(1);
   const int shift = i % (window_in + 1);
   int shift_out = (i - shift) / (window_in + 1);
   int total = (inputs - window_in) % step;
   total = (inputs - window_in - total) / step + (total > 0 ? 1 : 0);
   shift_out += total * window_out * var;
   float2 grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += ComplexMul(matrix_g[t * window_out + shift_out],
                         (shift == window_in ? (float2)(1, 0) : matrix_i[inputs * var + shift + t * step]));
     }
   float2 delta = ComplexMul((float2)(learning_rates, 0), clamp(grad, (float2) - MAX_GRAD, (float2)MAX_GRAD)) + ComplexMul((float2)(momentum, 0), matrix_dw[i + total_w * var]);
   if(!(isnan(delta.x) || isnan(delta.y) || isinf(delta.x) || isinf(delta.y)))
     {
      matrix_dw[i + total_w * var] = delta;
      matrix_w[i + total_w * var] = matrix_w[i + total_w * var] + delta;
     }
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights complex conv |
//| adam step using the optimizer rule.                              |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsComplexConvAdam(__global float2* __attribute__((aligned(8))) matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< input window and n - output window
      __global const float2* __attribute__((aligned(8))) matrix_g, ///<[in] Tensor of gradients at current layer
      __global const float2* __attribute__((aligned(8))) matrix_i, ///<[in] Inputs tensor
      __global float2* __attribute__((aligned(8))) matrix_m,      ///<[in] Matrix of first momentum
      __global float2* __attribute__((aligned(8))) matrix_v,      ///<[in] Matrix of seconfd momentum
      const int inputs,               ///< Number of inputs
      const float l,                  ///< Learning rates
      const float b1,                 ///< First momentum multiplier
      const float b2,                 ///< Second momentum multiplier
      int window_in,                  ///< Size of input window
      int window_out,                 ///< Size of output window
      int step                        ///< Step size
                                          )
  {
   const size_t i = get_global_id(0);
   const size_t total_w = get_global_size(0);
   const size_t var = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   const int shift = i % (window_in + 1);
   int shift_out = (i - shift) / (window_in + 1);
   int total = (inputs - window_in + step - 1) / step;
   shift_out += total * window_out * var;
   const int shift_var_in = var * inputs;
   const int shift_var_out = var * total * window_out;
//---
   float2 grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += IsNaNOrInf2(ComplexMul(matrix_g[t * window_out + shift_out + shift_var_out],
                                     (shift == window_in ? (float2)(1, 0) : matrix_i[shift + t * step + shift_var_in])), (float2)0);
     }
   grad = clamp(grad, (float2) - MAX_GRAD, (float2)MAX_GRAD);
   float2 mt = IsNaNOrInf2(b1 * matrix_m[i + total_w * var] + (1 - b1) * grad, (float2)0);
   float2 vt = IsNaNOrInf2(b2 * matrix_v[i + total_w * var] + (1 - b2) * ComplexMul(grad, grad), (float2)(1.0e-6f, 0));
   float2 weight = matrix_w[i + total_w * var] + IsNaNOrInf2(l * ComplexDiv(mt, ComplexSqrt(vt)), (float2)0);
   matrix_w[i + total_w * var] = weight;
   matrix_m[i + total_w * var] = mt;
   matrix_v[i + total_w * var] = vt;
  }
//+------------------------------------------------------------------+
//| Runs the complex soft max feed forward feed-forward pass on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_FeedForward(__global float2* __attribute__((aligned(8))) inputs,
      __global float2* __attribute__((aligned(8))) outputs, const int total)
  {
   const uint i = (uint)get_global_id(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
   uint shift_head = h * total;
//---
   __local float2 temp[LOCAL_ARRAY_SIZE];
   uint count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            temp[l].x = (count > 0 ? fmax(ComplexAbs(inputs[shift]), temp[l].x)
                         : ComplexAbs(inputs[shift]));
         count++;
        }
      while((count * ls + l) < total);
   BarrierLoc
   float max_value = temp[0].x;
//---
   for(int i = 1; i < ls; i++)
      max_value = fmax(max_value, temp[i].x);
//---
   count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         temp[l] = (count > 0 ? temp[l] : (float2)0) +
                   (shift < ((h + 1) * total) ? ComplexExp(ComplexDiv(inputs[shift], (float2)(max_value, 0))) : (float2)0);
         count++;
        }
      while((count * ls + l) < total);
   BarrierLoc
   count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] += (l < count && (l + count) < total ? temp[l + count] : (float2)0);
      if(l + count < ls)
         temp[l + count] = (float2)0;
      BarrierLoc
     }
   while(count > 1);
//---
   float2 sum = temp[0];
   if(ComplexAbs(sum) > 0)
     {
      count = 0;
      while((count * ls + l) < total)
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            outputs[shift] = ComplexDiv(ComplexExp(ComplexDiv(inputs[shift], (float2)(max_value, 0))), sum);
         count++;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the complex soft max hidden gradient gradient           |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_HiddenGradient(__global float2* __attribute__((aligned(8))) outputs,
      __global float2* __attribute__((aligned(8))) output_gr,
      __global float2* __attribute__((aligned(8))) input_gr)
  {
   size_t i = get_global_id(0);
   size_t outputs_total = get_global_size(0);
   size_t h = get_global_id(1);
   uint shift = h * outputs_total;
   float2 output = outputs[shift + i];
   float2 result = 0;
//---
   for(int j = 0; j < outputs_total; j++)
      result += ComplexMul(ComplexMul(outputs[shift + j], output_gr[shift + j]), ((i == j ? (float2)(1, 0) : (float2)0) - output));
   input_gr[shift + i] = result;
  }
//+------------------------------------------------------------------+
//| Computes the complex soft max output gradient gradient           |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_OutputGradient(__global float2* __attribute__((aligned(8))) outputs,
      __global float2* __attribute__((aligned(8))) targets,
      __global float2* __attribute__((aligned(8))) output_gr)
  {
   size_t i = get_global_id(0);
   if(ComplexAbs(outputs[i]) == 0)
      output_gr[i] = (float2)0;
   else
      output_gr[i] = ComplexDiv(targets[i], outputs[i]);
  }
//+------------------------------------------------------------------+
//| Implements the complex mh attention score kernel on the GPU.     |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionScore(__global float2* __attribute__((aligned(8))) qkv,  ///<[in] Matrix of Querys, Keys, Values
                                      __global float2* __attribute__((aligned(8))) score, ///<[out] Matrix of Scores
                                      int dimension,         ///< Dimension of Key
                                      int mask ///< 1 - calc only previous units, 0 - calc all
                                     )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
//---
   float2 koef = (float2)(sqrt((float)dimension), 0);
   if(koef.x < 1)
      koef.x = 1;
   float2 sum = 0;
//---
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = (float2)0;
         continue;
        }
      float2 result = (float2)0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
         result += ComplexMul(qkv[shift_q + i], qkv[shift_k + i]);
      result = ComplexExp(ComplexDiv(result, koef));
      if(isnan(result.x) || isnan(result.y) || isinf(result.x) || isinf(result.y))
         result = (float2)0;
      score[shift_s + k] = result;
      sum += result;
     }
   if(ComplexAbs(sum) > 0)
     {
      //---
      for(int k = 0; k < units; k++)
         score[shift_s + k] = ComplexDiv(score[shift_s + k], sum);
     }
  }
//+------------------------------------------------------------------+
//| Implements the complex mh attention out kernel on the GPU.       |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionOut(__global float2* __attribute__((aligned(8))) scores, ///<[in] Matrix of Scores
                                    __global float2* __attribute__((aligned(8))) qkv,   ///<[in] Matrix of Values
                                    __global float2* __attribute__((aligned(8))) out,   ///<[out] Output tensor
                                    int dimension           ///< Dimension of Value
                                   )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
//---
//---
   for(int d = 0; d < dimension; d++)
     {
      float2 result = (float2)0;
      for(int v = 0; v < units; v++)
        {
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += ComplexMul(scores[shift_s + v], qkv[shift_v]);
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
//| Computes the complex mh attention gradients gradient             |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionGradients(__global float2* __attribute__((aligned(8))) qkv,
      __global float2* __attribute__((aligned(8))) qkv_g,
      __global float2* __attribute__((aligned(8))) scores,
      __global float2* __attribute__((aligned(8))) gradient
                                         )
  {
   size_t u = get_global_id(0);
   size_t h = get_global_id(1);
   size_t d = get_global_id(2);
   size_t units = get_global_size(0);
   size_t heads = get_global_size(1);
   size_t dimension = get_global_size(2);
//---
   float2 koef = (float2)(sqrt((float)dimension), 0);
   if(koef.x < 1)
      koef.x = 1;
//--- init
   const int shift_q = dimension * (heads * 3 * u + h);
   const int shift_k = dimension * (heads * (3 * u + 1) + h);
   const int shift_v = dimension * (heads * (3 * u + 2) + h);
   const int shift_g = dimension * (heads * u + h);
   int shift_score = h * units;
   int step_score = units * heads;
//--- Calculating Value's gradients
   float2 sum = (float2)0;
//---
   for(int i = 0; i < units; i++)
      sum += ComplexMul(gradient[(h + i * heads) * dimension + d], scores[shift_score + u + i * step_score]);
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   shift_score = h * units + u * step_score;
   float2 grad = 0;
   float2 grad_out = gradient[shift_g + d];
//---
   for(int k = 0; k < units; k++)
     {
      float2 sc_g = (float2)0;
      float2 sc = scores[shift_score + k];
      for(int v = 0; v < units; v++)
         sc_g += ComplexMul(
                    ComplexMul(scores[shift_score + v],
                               ComplexMul(qkv[dimension * (heads * (3 * v + 2) + h)],
                                          grad_out)),
                    ((float2)(k == v, 0) - sc)
                 );
      grad += ComplexMul(ComplexDiv(sc_g, koef), qkv[dimension * (heads * (3 * k + 1) + h) + d]);
     }
   qkv_g[shift_q + d] = grad;
//--- Calculating Key's gradients
   grad = 0;
//---
   for(int q = 0; q < units; q++)
     {
      shift_score = h * units + q * step_score;
      float2 sc_g = (float2)0;
      float2 sc = scores[shift_score + u];
      float2 grad_out = gradient[dimension * (heads * q + h) + d];
      for(int v = 0; v < units; v++)
         sc_g += ComplexMul(
                    ComplexMul(scores[shift_score + v],
                               ComplexMul(qkv[dimension * (heads * (3 * v + 2) + h)],
                                          grad_out)),
                    ((float2)(u == v, 0) - sc)
                 );
      grad += ComplexMul(ComplexDiv(sc_g, koef), qkv[dimension * (heads * 3 * q + h) + d]);
     }
   qkv_g[shift_k + d] = grad;
  }
//+------------------------------------------------------------------+
//| Implements the complex normalize kernel on the GPU.              |
//+------------------------------------------------------------------+
__kernel void ComplexNormalize(__global float2* __attribute__((aligned(8))) inputs,
                               __global float2* __attribute__((aligned(8))) outputs,
                               __global float2* __attribute__((aligned(8))) means,
                               __global float *vars,
                               int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
   const float2 dim = (float2)(dimension, 0);
//---
   float2 mean = 0;
//---
   for(int i = 0; i < dimension; i++)
      mean = IsNaNOrInf2(inputs[shift + i], (float2)0);
   means[n] = mean = ComplexDiv(mean, dim);
   float variance = 0;
//---
   for(int i = 0; i < dimension; i++)
     {
      float abs_delta = ComplexAbs(inputs[shift + i] - mean);
      variance += abs_delta * abs_delta;
     }
   vars[n] = variance = sqrt(IsNaNOrInf(variance / dimension, 1));
   float2 v = (float2)(variance, 0);
//---
   for(int i = 0; i < dimension; i++)
     {
      float2 val = IsNaNOrInf2(ComplexDiv((inputs[shift + i] - mean), v), (float2)0);
      outputs[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//| Computes the complex normalize gradient gradient contributions on|
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void ComplexNormalizeGradient(__global float2* __attribute__((aligned(8))) inputs_gr,
                                       __global float2* __attribute__((aligned(8))) outputs_gr,
                                       __global float *vars,
                                       int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
//---
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexDiv(outputs_gr[shift + i], variance);
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      inputs_gr[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//| Implements the complex un normalize kernel on the GPU.           |
//+------------------------------------------------------------------+
__kernel void ComplexUnNormalize(__global float2* __attribute__((aligned(8))) inputs,
                                 __global float2* __attribute__((aligned(8))) outputs,
                                 __global float2* __attribute__((aligned(8))) means,
                                 __global float *vars,
                                 int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
   float2 mean = means[n];
//---
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexMul(inputs[shift + i], variance) + mean;
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      outputs[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//| Computes the complex un normalize gradient gradient contributions|
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void ComplexUnNormalizeGradient(__global float2* __attribute__((aligned(8))) inputs_gr,
      __global float2* __attribute__((aligned(8))) outputs_gr,
      __global float *vars,
      int dimension
                                        )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
//---
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexMul(outputs_gr[shift + i], variance);
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      inputs_gr[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//| Implements the main freq weight kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void MainFreqWeight(__global float2* __attribute__((aligned(8))) freq,
                             __global float *weight,
                             int dimension
                            )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float max_f = 0;
   float total = 0;
   float energy;
//---
   for(int i = 0; i < dimension; i++)
     {
      energy = ComplexAbs(freq[shift + i]);
      total += energy;
      max_f = fmax(max_f, energy);
     }
   weight[n] = max_f / (total > 0 ? total : 1);
  }
//+------------------------------------------------------------------+
//| Implements the weighted sum kernel on the GPU.                   |
//+------------------------------------------------------------------+
__kernel void WeightedSum(__global float *inputs1,
                          __global float *inputs2,
                          __global float *outputs,
                          __global float *weight,
                          int dimension
                         )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float w = weight[n];
//---
   for(int i = 0; i < dimension; i++)
      outputs[shift + i] = inputs1[shift + i] * w + inputs2[shift + i] * (1 - w);
  }
//+------------------------------------------------------------------+
//| Computes the weighted sum gradient gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void WeightedSumGradient(__global float *inputs_gr1,
                                  __global float *inputs_gr2,
                                  __global float *outputs_gr,
                                  __global float *weight,
                                  int dimension
                                 )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float w = weight[n];
   float w1 = 1 - weight[n];
//---
   for(int i = 0; i < dimension; i++)
     {
      float grad = outputs_gr[shift + i];
      inputs_gr1[shift + i] = grad * w;
      inputs_gr2[shift + i] = grad * w1;
     }
  }
//+------------------------------------------------------------------+
//| Runs the feed forward s3 feed-forward pass on the GPU.           |
//+------------------------------------------------------------------+
__kernel void FeedForwardS3(__global float *inputs,
                            __global float *probability,
                            __global float *weights,
                            __global float *outputs,
                            __global float *positions,
                            const int window,
                            const int total
                           )
  {
   int pos = get_global_id(0);
   int segments = get_global_size(0);
//---
   if((segments * window) > total)
      segments--;
//---
   int segment = 0;
   if(pos < segments)
     {
      const float prob = probability[pos];
      //---
      for(int i = 0; i < pos; i++)
        {
         if(probability[i] <= prob)
            segment++;
        }
      //---
      for(int i = pos + 1; i < segments; i++)
        {
         if(probability[i] < prob)
            segment++;
        }
     }
   else
      segment = pos;
//---
   const int shift_in = segment * window;
   const int shift_out = pos * window;
   const float w1 = weights[0];
   const float w2 = weights[1];
   positions[pos] = (float)segment;
//---
   for(int i = 0; i < window; i++)
     {
      if((shift_in + i) >= total || (shift_out + i) >= total)
         break;
      outputs[shift_out + i] = w1 * inputs[shift_in + i] + w2 * inputs[shift_out + i];
     }
  }
//+------------------------------------------------------------------+
//| Computes the inside gradient s3 gradient contributions on the    |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void InsideGradientS3(__global float *inputs,
                               __global float *inputs_gr,
                               __global float *probability,
                               __global float *probability_gr,
                               __global float *weights,
                               __global float *outputs_gr,
                               __global float *positions,
                               const int window,
                               const int total
                              )
  {
   size_t pos = get_global_id(0);
//---
   int segment = (int)positions[pos];
   float prob = probability[pos];
   const float w1 = weights[0];
   const float w2 = weights[1];
   const int shift_in = segment * window;
   const int shift_out = pos * window;
//---
   float grad = 0;
   float temp = 0;
//---
   for(int i = 0; i < window; i++)
     {
      if((shift_out + i) >= total)
         break;
      temp = outputs_gr[shift_out + i] * w1;
      grad += temp * inputs[shift_in + i];
      inputs_gr[shift_in + i] = temp + outputs_gr[shift_in + i] * w2;
     }
   probability_gr[segment] = grad / prob;
  }
//+------------------------------------------------------------------+
//| Computes the weight gradient s3 gradient contributions on the    |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void WeightGradientS3(__global float *inputs,
                               __global float *positions,
                               __global float *outputs_gr,
                               __global float *weights_gr,
                               const int window,
                               const int total
                              )
  {
   size_t l = get_local_id(0);
   size_t w = get_global_id(1);
   size_t ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   if(l < ls)
     {
      float val = 0;
      //---
      for(int i = l; i < total; i += ls)
        {
         int shift_in = i;
         if(w == 0)
           {
            int pos = i / window;
            shift_in = positions[pos] * window + i % window;
           }
         val += outputs_gr[i] * inputs[shift_in];
        }
      temp[l] = val;
     }
   BarrierLoc
//---
   int t = ls;
   do
     {
      t = (t + 1) / 2;
      if(l < t && (l + t) < ls)
        {
         temp[l] += temp[l + t];
         temp[l + t] = 0;
        }
      BarrierLoc
     }
   while(t > 1);
//---
   if(l == 0)
      weights_gr[w] = temp[0];
  }
//+------------------------------------------------------------------+
//| Implements the mh2 pyr attention out kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void MH2PyrAttentionOut(__global float *q,
                                 __global float *kv,
                                 __global float *score,
                                 __global float *out,
                                 const int dimension,
                                 const int heads_kv,
                                 const int window
                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   const int delta_win = (window + 1) / 2;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   float sum = 0;
   for(int d = 0; d < dimension; d++)
      sum = q[shift_q + d] * kv[shift_k + d];
   float sc = LocalSoftMax(IsNaNOrInf(sum / koef, MIN_VALUE), 1, temp);
   score[shift_s] = sc;
   BarrierLoc
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               sum = 0;
               if(abs(count * ls + k - q_id) <= delta_win)
                 {
                  int sh_v = 2 * dimension * heads_kv * count * ls;
                  sum = kv[shift_v + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
                  if(isnan(sum))
                     sum = 0;
                 }
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      BarrierLoc
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         BarrierLoc
        }
      while(count > 1);
      //---
      if(k == 0)
         out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//| Implements the plr kernel on the GPU.                            |
//+------------------------------------------------------------------+
__kernel void PLR(__global const float *inputs,
                  __global float *outputs,
                  __global int *isttp,
                  const int transpose,
                  const float min_step
                 )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
//--- look for ttp
   float value = inputs[shift_in];
   bool bttp = false;
   if(i == 0 || i == lenth - 1)
      bttp = true;
   else
     {
      float prev = value;
      int prev_pos = i;
      float max_v = value;
      float max_pos = i;
      float min_v = value;
      float min_pos = i;
      while(fmax(fabs(prev - max_v), fabs(prev - min_v)) < min_step && prev_pos > 0)
        {
         prev_pos--;
         prev = inputs[shift_in - (i - prev_pos) * step_in];
         if(prev >= max_v && (prev - min_v) < min_step)
           {
            max_v = prev;
            max_pos = prev_pos;
           }
         if(prev <= min_v && (max_v - prev) < min_step)
           {
            min_v = prev;
            min_pos = prev_pos;
           }
        }
      //---
      float next = value;
      int next_pos = i;
      while(fmax(fabs(next - max_v), fabs(next - min_v)) < min_step && next_pos < (lenth - 1))
        {
         next_pos++;
         next = inputs[shift_in + (next_pos - i) * step_in];
         if(next > max_v && (next - min_v) < min_step)
           {
            max_v = next;
            max_pos = next_pos;
           }
         if(next < min_v && (max_v - next) < min_step)
           {
            min_v = next;
            min_pos = next_pos;
           }
        }
      //---
      if(
         (value >= prev && value > next) ||
         (value > prev && value == next) ||
         (value <= prev && value < next) ||
         (value < prev && value == next)
      )
         if(max_pos == i || min_pos == i)
            bttp = true;
     }
//---
   isttp[shift_in] = (int)bttp;
   outputs[shift_in] = 0;
   BarrierLoc
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int prev_ttp = 0;
   if(bttp)
     {
      pos = 0;
      //---
      for(int p = 0; p < i; p++)
        {
         int current_in = ((bool)transpose ? (p * variables + v) : (v * lenth + p));
         if((bool)isttp[current_in])
           {
            pos++;
            prev_ttp = p;
            prev_in = current_in;
           }
        }
     }
//--- cacl tendency
   if(pos > 0 && pos < (lenth / 3))
     {
      float sum_x = 0;
      float sum_y = 0;
      float sum_xy = 0;
      float sum_xx = 0;
      int dist = i - prev_ttp;
      //---
      for(int p = 0; p < dist; p++)
        {
         float x = (float)(p);
         float y = inputs[prev_in + p * step_in];
         sum_x += x;
         sum_y += y;
         sum_xy += x * y;
         sum_xx += x * x;
        }
      float slope = (dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1);
      float intercept = (sum_y - slope * sum_x) / dist;
      int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3));
      outputs[shift_out] = slope;
      outputs[shift_out + step_in] = intercept;
      outputs[shift_out + 2 * step_in] = ((float)dist) / lenth;
     }
   else
     {
      if(pos == (lenth / 3))
        {
         float sum_x = 0;
         float sum_y = 0;
         float sum_xy = 0;
         float sum_xx = 0;
         int dist = lenth - prev_ttp;
         //---
         for(int p = 0; p < dist; p++)
           {
            float x = (float)(p);
            float y = inputs[prev_in + p * step_in];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
           }
         float slope = (dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1);
         float intercept = (sum_y - slope * sum_x) / dist;
         int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3));
         outputs[shift_out] = slope;
         outputs[shift_out + step_in] = intercept;
         outputs[shift_out + 2 * step_in] = ((float)dist) / lenth;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the plr gradient gradient contributions on the GPU.     |
//+------------------------------------------------------------------+
__kernel void PLRGradient(__global float *inputs_gr,
                          __global const float *outputs,
                          __global const float *outputs_gr,
                          const int transpose
                         )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
   const int shift_out = ((bool)transpose ? v : (v * lenth));
   const int step_out = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(outputs[shift_out + pos * step_out + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- calc constants
   float sum_x = 0;
   float sum_xx = 0;
//---
   for(int p = 0; p < dist; p++)
     {
      float x = (float)(p);
      sum_x += x;
      sum_xx += x * x;
     }
//--- get output gradient
   float grad_slope = outputs_gr[shift_out + pos * step_out];
   float grad_intercept = outputs_gr[shift_out + pos * step_out + step_in];
//--- calc gradient
   grad_slope -= sum_x / dist * grad_intercept;
   grad_slope /= fmax(dist * sum_xx - sum_x * sum_x, 1);
   float grad = grad_intercept / dist;
   grad += (dist * (i - prev_in) - sum_x) * grad_slope;
   if(isnan(grad) || isinf(grad))
      grad = 0;
//--- save result
   inputs_gr[shift_in] = grad;
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights adam mini    |
//| step using the optimizer rule.                                   |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdamMini(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< number of neurons in previous layer and n -
                                    ///< number of neurons in current layer
                                    __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_m,      ///<[in,out] Matrix of first momentum
                                    __global float *matrix_v,      ///<[in,out] Matrix of seconfd momentum
                                    const float l,                  ///< Learning rates
                                    const float b1,                 ///< First momentum multiplier
                                    const float b2                  ///< Second momentum multiplier
                                   )
  {
//--- inputs
   const size_t i = get_local_id(0);
   const size_t inputs = get_local_size(0) - 1;
//--- outputs
   const size_t o = get_global_id(1);
   const size_t outputs = get_global_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((uint)LOCAL_ARRAY_SIZE, (uint)inputs);
   const float inp = (i < inputs ? matrix_i[i] : 1.0f);
   int count = 0;
   do
     {
      if(count == (i / ls))
        {
         int shift = i % ls;
         temp[shift] = (count == 0 ? 0 : temp[shift]) + ((isnan(inp) || isinf(inp)) ? 0 : inp * inp) / inputs;
        }
      count++;
      BarrierLoc
     }
   while(count * ls < inputs);
//--- sum
   count = (ls + 1) / 2;
   do
     {
      if(i < count && (i + count) < ls)
        {
         temp[i] += temp[i + count];
         temp[i + count] = 0;
        }
      count = (count + 1) / 2;
      BarrierLoc
     }
   while(count > 1);
//--- calc v
   if(i == 0)
     {
      temp[1] = matrix_g[o];
      if(isnan(temp[1]) || isinf(temp[1]))
         temp[1] = 0;
      if(isnan(temp[0]) || isinf(temp[0]))
         temp[0] = 1;
      float v = matrix_v[o];
      if(isnan(v) || isinf(v))
         v = 1;
      temp[0] = b2 * v + (1 - b2) * (temp[1] * temp[1]) * temp[0];
      matrix_v[o] = temp[0];
     }
   BarrierLoc
//---
   const int wi = o * (inputs + 1) + i;
   float weight = matrix_w[wi];
   if(isnan(weight) || isinf(weight))
      weight = 0;
//---
   float m = matrix_m[wi];
   if(isnan(m) || isinf(m))
      m = 0;
//--- calc m
   m = b1 * m + (1 - b1) * temp[1] * inp;
   if(isnan(m) || isinf(m))
      m = 0;
//---
   float delta = l * (m / (sqrt(temp[0]) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(isnan(delta) || isinf(delta))
      delta = 0;
   if(delta > 0)
      matrix_w[wi] = weight + delta;
   matrix_m[wi] = m;
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights conv adam    |
//| mini step using the optimizer rule.                              |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvAdamMini(__global float *matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
                                        ///< input window and n - output window
                                        __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                        __global const float *matrix_i, ///<[in] Inputs tensor
                                        __global float *matrix_m,      ///<[in] Matrix of first momentum
                                        __global float *matrix_v,      ///<[in] Matrix of seconfd momentum
                                        const int inputs,               ///< Number of inputs
                                        const float l,                  ///< Learning rates
                                        const float b1,                 ///< First momentum multiplier
                                        const float b2,                 ///< Second momentum multiplier
                                        int step                        ///< Step size
                                       )
  {
//--- window in
   const size_t i = get_global_id(0);
   const size_t window_in = get_global_size(0) - 1;
//--- window out
   const size_t f = get_global_id(1);
   const size_t window_out = get_global_size(1);
//--- head window out
   const size_t f_h = get_local_id(1);
   const size_t window_out_h = get_local_size(1);
//--- variable
   const size_t v = get_global_id(2);
   const size_t variables = get_global_size(2);
//--- constants
   const int total = (inputs - window_in + step - 1) / step;
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
   const int shift_w = (f + v * window_out) * (window_in + 1) + i;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((uint)window_in, (uint)LOCAL_ARRAY_SIZE);
//--- calc gradient
   float grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(i != window_in && (i + t * window_in) >= inputs)
         break;
      float gt = matrix_g[t * window_out + f + shift_var_out] *
                 (i == window_in ? 1 : matrix_i[i + t * step + shift_var_in]);
      if(!(isnan(gt) || isinf(gt)))
         grad += gt;
     }
//--- calc sum grad
   int count;
//---
   for(int h = 0; h < window_out_h; h++)
     {
      count = 0;
      do
        {
         if(h == f_h)
           {
            if(count == (i / ls))
              {
               int shift = i % ls;
               temp[shift] = ((count == 0 && h == 0) ? 0 : temp[shift]) + ((isnan(grad) || isinf(grad)) ? 0 : grad * grad) / (window_in * window_out_h);
              }
           }
         count++;
         BarrierLoc
        }
      while((count * ls) < window_in);
     }
   count = (ls + 1) / 2;
   do
     {
      if(i < count && (i + count) < ls && f_h == 0)
        {
         temp[i] += temp[i + count];
         temp[i + count] = 0;
        }
      count = (count + 1) / 2;
      BarrierLoc
     }
   while(count > 1);
//--- calc v
   if(i == 0 && f_h == 0)
     {
      if(isnan(temp[0]) || isinf(temp[0]))
         temp[0] = 1;
      int head = f / window_out_h;
      float v = matrix_v[head];
      if(isnan(v) || isinf(v))
         v = 1;
      temp[0] = clamp(b2 * v + (1 - b2) * temp[0], 1.0e-6f, 1.0e6f);
      matrix_v[head] = temp[0];
     }
   BarrierLoc
//--- calc m
   float mt = clamp(b1 * matrix_m[shift_w] + (1 - b1) * grad, -1.0e5f, 1.0e5f);
   if(isnan(mt) || isinf(mt))
      mt = 0;
   float weight = matrix_w[shift_w] + l * mt / sqrt(temp[0]);
   if(!(isnan(weight) || isinf(weight)))
      matrix_w[shift_w] = weight;
   matrix_m[shift_w] = mt;
  }
//+------------------------------------------------------------------+
//| Implements the cut trend and other kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void CutTrendAndOther(__global const float *inputs,
                               __global const float *plr,
                               __global float *trend,
                               __global float *other
                              )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = i * variables + v;
   const int step_in = variables;
   const int shift_plr = v;
   const int step_plr = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(plr[shift_plr + pos * step_plr + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- calc trend
   float sloat = plr[shift_plr + pos * step_plr];
   float intercept = plr[shift_plr + pos * step_plr + step_in];
   pos = i - prev_in;
   float trend_i = sloat * pos + intercept;
   float other_i = inputs[shift_in] - trend_i;
//--- save result
   trend[shift_in] = trend_i;
   other[shift_in] = other_i;
  }
//+------------------------------------------------------------------+
//| Computes the cut trend and other gradient gradient contributions |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void CutTrendAndOtherGradient(__global float *inputs_gr,
                                       __global const float *plr,
                                       __global float *plr_gr,
                                       __global const float *trend_gr,
                                       __global const float *other_gr
                                      )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = i * variables + v;
   const int step_in = variables;
   const int shift_plr = v;
   const int step_plr = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(plr[shift_plr + pos * step_plr + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- get gradient
   float other_i_gr = other_gr[shift_in];
   float trend_i_gr = trend_gr[shift_in] - other_i_gr;
//--- calc plr gradient
   pos = i - prev_in;
   float sloat_gr = trend_i_gr * pos;
   float intercept_gr = trend_i_gr;
//--- save result
   plr_gr[shift_plr + pos * step_plr] += sloat_gr;
   plr_gr[shift_plr + pos * step_plr + step_in] += intercept_gr;
   inputs_gr[shift_in] = other_i_gr;
  }
//+------------------------------------------------------------------+
//| Implements the cut one from another kernel on the GPU.           |
//+------------------------------------------------------------------+
__kernel void CutOneFromAnother(__global const float *inputs,
                                __global const float *cut,
                                __global float *other
                               )
  {
   const size_t i = get_global_id(0);
//--- save result
   other[i] = inputs[i] - cut[i];
  }
//+------------------------------------------------------------------+
//| Computes the cut one from another gradient gradient contributions|
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void CutOneFromAnotherGradient(__global float *inputs_gr,
                                        __global float *cut_gr,
                                        __global const float *other_gr
                                       )
  {
   const size_t i = get_global_id(0);
   float gr = other_gr[i];
//--- save result
   inputs_gr[i] = gr;
   cut_gr[i] = (-gr);
  }
//+------------------------------------------------------------------+
//| Implements the uni traj prepare kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void UniTrajPrepare(__global const float *history,
                             __global const float *h_mask,
                             __global const float *future,
                             __global const float *f_mask,
                             __global float *output,
                             const int h_total,
                             const int f_total
                            )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   const int shift_in = i * variables + v;
   const int shift_out = 3 * shift_in;
   const int shift_f_out = 3 * (h_total * variables + v);
//--- history
   if(i < h_total)
     {
      float mask = h_mask[shift_in];
      float h = history[shift_in];
      float v = (i < (h_total - 1) && mask != 0 ? (history[shift_in + variables] - h) * mask : 0);
      if(isnan(v) || isinf(v))
         v = h = mask = 0;
      output[shift_out] = h * mask;
      output[shift_out + 1] = v;
      output[shift_out + 2] = mask;
     }
//--- future
   if(i < f_total)
     {
      float mask = f_mask[shift_in];
      float f = future[shift_in];
      float v = (i < (f_total - 1) && mask != 0 ? (future[shift_in + variables] - f) * mask : 0);
      if(isnan(v) || isinf(v))
         v = f = mask = 0;
      output[shift_f_out + shift_out] = f * mask;
      output[shift_f_out + shift_out + 1] = v;
      output[shift_f_out + shift_out + 2] = mask;
     }
  }
//+------------------------------------------------------------------+
//| Computes the uni traj prepare grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void UniTrajPrepareGrad(__global float *history_gr,
                                 __global float *future_gr,
                                 __global const float *output,
                                 __global const float *output_gr,
                                 const int h_total,
                                 const int f_total
                                )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   const int shift_in = i * variables + v;
   const int shift_out = 3 * shift_in;
   const int shift_f_out = 3 * (h_total * variables + v);
//--- history
   if(i < h_total)
     {
      float mask = output[shift_out + 2];
      float grad = 0;
      if(mask > 0)
        {
         grad = output_gr[shift_out] * mask;
         grad -= (i < (h_total - 1) && mask != 0 ? (output_gr[shift_out + 1]) * mask : 0);
         grad += (i > 0 ? output[shift_out + 1 - 3 * variables] * output[shift_out + 2 - 3 * variables] : 0);
         if(isnan(grad) || isinf(grad))
            grad = 0;
         //---
        }
      history_gr[shift_in] = grad;
     }
//--- future
   if(i < f_total)
     {
      float mask = output[shift_f_out + shift_out + 2];
      float grad = 0;
      if(mask > 0)
        {
         grad = output_gr[shift_f_out + shift_out] * mask;
         grad -= (i < (h_total - 1) && mask != 0 ? (output_gr[shift_f_out + shift_out + 1]) * mask : 0);
         grad += (i > 0 ? output[shift_f_out + shift_out + 1 - 3 * variables] * output[shift_f_out + shift_out + 2 - 3 * variables] : 0);
         if(isnan(grad) || isinf(grad))
            grad = 0;
         //---
        }
      future_gr[shift_in] = grad;
     }
  }
//+------------------------------------------------------------------+
//| Implements the uni traj bts kernel on the GPU.                   |
//+------------------------------------------------------------------+
__kernel void UniTrajBTS(__global const float *concat_inp,
                         __global float *d_forw,
                         __global float *d_bakw,
                         const int total
                        )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   if(i == 0)
     {
      const int step = variables * 3;
      const int start = v * 3 + 2;
      float last = 0;
      d_forw[v] = 0;
      //---
      for(int p = 1; p < total; p++)
        {
         float m = concat_inp[start + p * step];
         d_forw[p * variables + v] = last = 1 + (1 - m) * last;
        }
     }
   else
     {
      const int step = -(variables * 3);
      const int start = (total - 1) * variables + v * 3 + 2;
      float last = 0;
      d_bakw[(total - 1) + v] = 0;
      //---
      for(int p = 1; p < total; p++)
        {
         float m = concat_inp[start + p * step];
         d_bakw[(total - 1 - p) * variables + v] = last = 1 + (1 - m) * last;
        }
     }
  }
//+------------------------------------------------------------------+
//| Performs the rotate data-layout operation on the GPU.            |
//+------------------------------------------------------------------+
float2 Rotate(const float x, const float cos_theta, const float sin_theta)
  {
   float2 result = 0;
   result.s0 = cos_theta + x * sin_theta;
   result.s1 = x * cos_theta - sin_theta;
   return result;
  }
//+------------------------------------------------------------------+
//| Implements the hi vt prepare kernel on the GPU.                  |
//+------------------------------------------------------------------+
__kernel void HiVTPrepare(__global const float *data,
                          __global float2* __attribute__((aligned(8))) output
                         )
  {
   const size_t t = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t total_v = get_global_size(1);
//---
   const int shift_data = t * total_v;
   const int shift_out = shift_data * total_v;
//---
   float value = data[shift_data + v + total_v] - data[shift_data + v];
   const float theta = atan(value);
   const float cos_theta = cos(theta);
   const float sin_theta = sin(theta);
   const float2 main = Rotate(value, cos_theta, sin_theta);
//---
//---
   for(int a = 0; a < total_v; a++)
     {
      float2 o = main;
      if(a != v)
         o -= Rotate(data[shift_data + a + total_v] - data[shift_data + a], cos_theta, sin_theta);
      output[shift_out + a] = o;
     }
  }
//+------------------------------------------------------------------+
//| Implements the gate element mul kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void GateElementMul(__global const float *inputs1,
                              __global const float *inputs2,
                              __global const float *gate,
                              __global float *out,
                              __global float *numeric_flags,
                              const int flags_enabled
                            )
  {
   const int i = get_global_id(0);
//---
    int invalid = 0;
    if(flags_enabled)
      {
//--- Evaluate raw gate products before their operands are sanitized for legacy output.
       const float raw_gate = gate[i];
       const float raw_input1 = inputs1[i];
       const float raw_input2 = inputs2[i];
       const float raw_product1 = raw_gate * raw_input1;
       const float raw_product2 = (1.0f - raw_gate) * raw_input2;
       invalid = isnan(raw_gate) || isinf(raw_gate) || isnan(raw_input1) || isinf(raw_input1) ||
                 isnan(raw_input2) || isinf(raw_input2) || isnan(raw_product1) || isinf(raw_product1) ||
                 isnan(raw_product2) || isinf(raw_product2);
      }
    const float g = IsNaNOrInf(gate[i], 0.5f);
    float result = 0;
    float inp = IsNaNOrInf(inputs1[i], 0.0f);
    if(inp != 0.0f && g != 0.0f)
       result += g * inp;
    inp = IsNaNOrInf(inputs2[i], 0.0f);
    if(inp != 0.0f && (1 - g) != 0.0f)
       result += (1 - g) * inp;
//---
    if(flags_enabled)
      {
       invalid |= isnan(result) || isinf(result);
       numeric_flags[i] = (invalid ? 1.0f : 0.0f);
      }
    out[i] = IsNaNOrInf(result, 0);
  }
//+------------------------------------------------------------------+
//| Computes the gate element mul grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void GateElementMulGrad(__global const float *inputs1,
                                 __global float *inputs1_gr,
                                 __global const float *inputs2,
                                 __global float *inputs2_gr,
                                 __global const float *gate,
                                 __global float *gate_gr,
                                 __global const float *out_gr,
                                 const int activ1,
                                 const int activ2,
                                 const int activ_gate
                                )
  {
   const int i = get_global_id(0);
//---
   const float g = IsNaNOrInf(gate[i], 0.5f);
   const float i1 = IsNaNOrInf(inputs1[i], 0);
   const float i2 = IsNaNOrInf(inputs2[i], 0);
   const float grad = IsNaNOrInf(out_gr[i], 0);
//---
   float i1_gr = IsNaNOrInf(grad * g, 0);
   float i2_gr = IsNaNOrInf(grad * (1 - g), 0);
   float g_gr = IsNaNOrInf(grad * (i1 - i2), 0);
//---
   inputs1_gr[i] = Deactivation(i1_gr, i1, activ1);
   inputs2_gr[i] = Deactivation(i2_gr, i2, activ2);
   gate_gr[i] = Deactivation(g_gr, g, activ_gate);
  }
//+------------------------------------------------------------------+
//| Performs the transpose rcd data-layout operation on the GPU.     |
//+------------------------------------------------------------------+
__kernel void TransposeRCD(__global const float *matrix_in, ///<[in] Input matrix
                           __global float *matrix_out ///<[out] Output matrix
                          )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int d = get_global_id(2);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
   const int dimension = get_global_size(2);
//---
   matrix_out[(c * rows + r)*dimension + d] = matrix_in[(r * cols + c) * dimension + d];
  }
//+------------------------------------------------------------------+
//| Implements the orthoganal loss kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void OrthoganalLoss(__global const float *data,
                             __global float *grad,
                             const int add
                            )
  {
   const size_t r = get_global_id(0);
   const size_t c = get_local_id(1);
   const size_t cols = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   uint ls = min((uint)cols, (uint)LOCAL_ARRAY_SIZE);
//---
   const int shift1 = r * cols + c;
   const int shift2 = c * cols + r;
   float value1 = IsNaNOrInf(data[shift1], 0);
   float value2 = (shift1 == shift2 ? value1 : IsNaNOrInf(data[shift2], 0));
   float v2 = IsNaNOrInf(value1 * value2, 0);
//---
   for(int i = 0; i < cols; i += ls)
     {
      //---
      if(i <= c && (i + ls) > c)
         Temp[c - i] = (i == 0 ? 0 : Temp[c - i]) + v2;
      BarrierLoc
     }
//---
   uint count = min(ls, (uint)cols);
   do
     {
      count = (count + 1) / 2;
      if(c < ls)
         Temp[c] += (c < count && (c + count) < cols ? Temp[c + count] : 0);
      if(c + count < ls)
         Temp[c + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//---
   const float sum = Temp[0];
   float diff = (float)(r == c) - sum;
   float loss = -(diff * diff);
   float g = (2 * (sum - (float)(r == c))) * loss;
   g = 2 * value2 * g;
   if(isinf(g) || isnan(g))
      g = 0;
   if(add == 1)
      grad[shift1] += g;
   else
      grad[shift1] = g;
  }
//+------------------------------------------------------------------+
//| Calculates calc distance on the GPU.                             |
//+------------------------------------------------------------------+
__kernel void CalcDistance(__global const float *data,
                           __global float *distance,
                            const int dimension,
                            const int groups,
                            const int normalize,
                            __global float *numeric_flags,
                            const int flags_enabled
                          )
  {
   const size_t main = get_global_id(0);
   const size_t slave = get_local_id(1);
   const size_t group = get_global_id(2);
   const int total = (int)get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   if(group >= (size_t)groups)
      return;
   const int shift_group = (int)group * total;
   const int shift_main = (shift_group + main) * dimension;
   const int shift_slave = (shift_group + slave) * dimension;
   const int shift_dist = shift_group * total + main * total + slave;
//--- calc distance
    float dist = 0;
    int invalid = 0;
    if(flags_enabled)
       for(int d = 0; d < dimension; d++)
         {
          const float value_main = data[shift_main + d];
          const float value_slave = data[shift_slave + d];
          invalid |= isnan(value_main) || isinf(value_main) || isnan(value_slave) || isinf(value_slave);
         }
    if(main != slave)
     {
      //---
      for(int d = 0; d < dimension; d++)
        {
          float delta = data[shift_main + d] - data[shift_slave + d];
          dist += delta * delta;
          if(flags_enabled)
            {
             const float square = delta * delta;
             invalid |= isnan(delta) || isinf(delta) || isnan(square) || isinf(square);
            }
        }
     }
//--- Look Max
    float max_dist = LocalMax(dist, 1, Temp);
    if(flags_enabled)
       invalid |= isnan(dist) || isinf(dist) || isnan(max_dist) || isinf(max_dist);
//--- Normalize
    if(normalize && max_dist > 0)
       dist /= Temp[0];
    if(flags_enabled)
       invalid |= isnan(dist) || isinf(dist);
    if(flags_enabled)
       numeric_flags[shift_dist] = (invalid ? 1.0f : 0.0f);
    dist = IsNaNOrInf(dist, 1);
//--- result
    distance[shift_dist] = dist;
  }
//+------------------------------------------------------------------+
//| Runs the feed forward local max feed-forward pass on the GPU.    |
//+------------------------------------------------------------------+
__kernel void FeedForwardLocalMax(__global const float *matrix_i,
                                  __global const float *distance,
                                  __global float *matrix_o,
                                  const float radius
                                 )
  {
   const size_t i = get_global_id(0);
   const size_t total = get_global_size(0);
   const size_t d = get_global_id(1);
   const size_t dimension = get_global_size(1);
//---
   const int shift_dist = i * total;
   const int shift_out = i * dimension + d;
//---
   float result = -3.402823466e+38f;
//---
   for(int k = 0; k < total; k++)
     {
      if(distance[shift_dist + k] > radius)
         continue;
      int shift = k * dimension + d;
      result = max(result, matrix_i[shift]);
     }
   matrix_o[shift_out] = result;
  }
//+------------------------------------------------------------------+
//| Computes the calc input gradient local max gradient contributions|
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void CalcInputGradientLocalMax(__global const float *matrix_i,
                                        __global float *matrix_ig,
                                        __global const float *distance,
                                        __global const float *matrix_o,
                                        __global const float *matrix_g,
                                        const float radius
                                       )
  {
   const size_t i = get_global_id(0);
   const size_t total = get_global_size(0);
   const size_t d = get_global_id(1);
   const size_t dimension = get_global_size(1);
//---
   float result = 0;
   float value = matrix_i[i * dimension + d];
//---
   for(int k = 0; k < total; k++)
     {
      if(distance[k * total + i] > radius)
         continue;
      int shift = k * dimension + d;
      if(fabs(matrix_o[shift] - value) <= 1.192092896e-07f)
         result += matrix_g[shift];
     }
   matrix_ig[i * dimension + d] = result;
  }
//+------------------------------------------------------------------+
//| Implements the mh mask attention out kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void MHMaskAttentionOut(__global const float *q,     ///<[in] Matrix of Querys
                                 __global const float *kv,    ///<[in] Matrix of Keys
                                 __global float *score,       ///<[out] Matrix of Scores
                                 __global const float *mask,  ///<[in] Mask Matrix
                                 __global float *out,         ///<[out] Matrix of attention
                                 const int dimension,          ///< Dimension of Key
                                 const int heads_kv,
                                 const float mask_level
                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const bool b_mask = (mask[shift_s] < mask_level);
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   float sum = 0;
   if(b_mask)
      sum = MIN_VALUE;
   else
      for(int d = 0; d < dimension; d++)
         sum += q[shift_q + d] * kv[shift_k + d];
   float sc = LocalSoftMax(IsNaNOrInf(sum / koef, MIN_VALUE), 1, temp);
   score[shift_s] = sc;
   BarrierLoc
//--- out
   for(int d = 0; d < dimension; d++)
     {
      BarrierLoc
      sum = LocalSum(IsNaNOrInf(kv[shift_v + d] * sc, 0), 1, temp);
      if(k == 0)
         out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh mask attention inside gradients gradient         |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void MHMaskAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *kv, __global float *kv_g,
      __global const float *mask, __global float *mask_g,
      __global const float *scores, __global const float *gradient,
      const int kunits, const int heads_kv, const float mask_level)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v = q_id; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
//---
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + 2 * v * heads_kv * dimension] *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + 2 * k * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += sc_g * q[shift_q + scr * dimension];
              }
           }
         kv_g[shift_k] = grad / koef;
        }
     }
//--- Mask's gradient
   for(int k = q_id; k < kunits; k += qunits)
     {
      float m = mask[shift_s + k];
      if(m < mask_level)
         mask_g[shift_s + k] = 0;
      else
         mask_g[shift_s + k] = 1 - m;
     }
  }
//+------------------------------------------------------------------+
//| Calculates calc position bias on the GPU.                        |
//+------------------------------------------------------------------+
__kernel void CalcPositionBias(__global const float *data1,
                               __global const float *data2,
                               __global float *result,
                               const int dimension
                              )
  {
   const size_t idx1 = get_global_id(0);
   const size_t idx2 = get_global_id(1);
   const size_t total1 = get_global_size(0);
   const size_t total2 = get_global_size(1);
//---
   const int shift1 = idx1 * dimension;
   const int shift2 = idx2 * dimension;
   const int shift_out = idx1 * total2 + idx2;
//---
   float res = 0;
//---
   for(int i = 0; i < dimension; i++)
     {
      float delta = data1[shift1 + i] - data2[shift2 + i];
      res = delta * delta;
     }
   res = sqrt(res);
   res = exp(-res);
   if(isnan(res) || isinf(res))
      res = 0;
//---
   result[shift_out] = res;
  }
//+------------------------------------------------------------------+
//| Implements the mh pos bias attention out kernel on the GPU.      |
//+------------------------------------------------------------------+
__kernel void MHPosBiasAttentionOut(__global const float *q,        ///<[in] Matrix of Querys
                                    __global const float *k,        ///<[in] Matrix of Keys
                                    __global const float *v,        ///<[in] Matrix of Values
                                    __global float *score,          ///<[out] Matrix of Scores
                                    __global const float *pos_bias, ///<[in] Position Bias
                                    __global float *out,            ///<[out] Matrix of attention
                                    const int dimension,             ///< Dimension of Key
                                    const int heads_kv,
                                    const int use_pos_bias
                                   )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads_kv * k_id + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k_id < ls)
     {
      temp[k_id] = 0;
      //---
      do
        {
         if(q_id >= (count * ls + k_id))
            if((count * ls) < (kunits - k_id))
              {
               float sum = 0;
               int sh_k = dimension * heads_kv * count * ls;
               for(int d = 0; d < dimension; d++)
                  sum = q[shift_q + d] * k[shift_kv + d + sh_k];
               sum = exp(sum / koef);
               if(isnan(sum))
                  sum = 0;
               temp[k_id] = temp[k_id] + sum + (use_pos_bias > 0 ? pos_bias[shift_pb + count * ls] : 0);
              }
         count++;
        }
      while((count * ls + k_id) < kunits);
     }
   BarrierLoc
   count = min(ls, (uint)kunits);
//---
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(q_id >= (count * ls + k_id))
      if(sum != 0)
        {
         //---
         for(int d = 0; d < dimension; d++)
            sc = q[shift_q + d] * k[shift_kv + d];
         sc = (exp(sc / koef) + (use_pos_bias > 0 ? pos_bias[shift_pb] : 0)) / sum;
         if(isnan(sc))
            sc = 0;
        }
   score[shift_s] = sc;
   BarrierLoc
//--- out
//---
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k_id < ls)
         do
           {
            if((count * ls) < (kunits - k_id))
              {
               int sh_v = 2 * dimension * heads_kv * count * ls;
               float sum =
                  v[shift_kv + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k_id] = (count > 0 ? temp[k_id] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k_id) < kunits);
      BarrierLoc
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < ls)
            temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         BarrierLoc
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh pos bias attention inside gradients gradient     |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void MHPosBiasAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *k, __global float *k_g,
      __global const float *v, __global float *v_g,
      __global const float *scores, __global const float *gradient,
      const int kunits, const int heads_kv)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v_id = q_id; v_id < kunits; v_id += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v_id;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (heads_kv * v_id + h) + d;
         v_g[shift_v] = grad;
        }
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = h_kv * dimension + d;
   int shift_key = h_kv * dimension + d;
//---
   for(int k_id = 0; k_id < kunits; k_id++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k_id];
      if(sc == 0)
         continue;
      for(int v_id = 0; v_id < kunits; v_id++)
         sc_g += scores[shift_s + v_id] * out_g * v[shift_val + v_id * heads_kv * dimension] *
                 ((float)(k_id == v_id) - sc);
      grad += sc_g * k[shift_key + k_id * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k_id = q_id; k_id < kunits; k_id += qunits)
        {
         int shift_k = dimension * (heads_kv * k_id + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + k_id;
            float val = v[shift_k];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k_id];
               if(sc == 0)
                  continue;
               for(int v_id = 0; v_id < kunits; v_id++)
                  sc_g += scores[shift_sc + v_id] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k_id == v_id) - sc);
               grad += sc_g * q[shift_g + scr * heads * dimension];
              }
           }
         k_g[shift_k] = grad / koef;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the diversity loss kernel on the GPU.                 |
//+------------------------------------------------------------------+
__kernel void DiversityLoss(__global const float *data,
                            __global float *grad,
                            const int dimension,
                            const int activation,
                            const int add
                           )
  {
   const size_t main = get_global_id(0);
   const size_t loc = get_local_id(1);
   const size_t total = get_local_size(0);
   const size_t total_loc = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   float delts = 0;
//---
   for(int d = 0; d < dimension; d++)
     {
      const float value_main = IsNaNOrInf(data[main * dimension + d], 0);
      for(int i = loc; i < total; i += total_loc)
        {
         float value_slave = IsNaNOrInf(data[i * dimension + d], 0);
         float delta = value_main - value_slave;
         delts += IsNaNOrInf(delta * delta / total, 0);
        }
     }
//---
   float loss = exp(LocalSum(delts, 1, Temp));
   float gr = 0;
//---
   for(int d = 0; d < dimension; d++)
     {
      const float value_main = IsNaNOrInf(data[main * dimension + d], 0);
      for(int i = loc; i < total; i += total_loc)
        {
         float value_slave = IsNaNOrInf(data[i * dimension + d], 0);
         gr += IsNaNOrInf(2 * loss * (value_main - value_slave) / total, 0);
        }
      //---
      gr = LocalSum(gr, 1, Temp);
      if(loc == 0)
        {
         if(add > 0)
            grad[main * dimension + d] += Deactivation(gr, value_main, activation);
         else
            grad[main * dimension + d] = Deactivation(gr, value_main, activation);
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the mh relative attention out kernel on the GPU.      |
//+------------------------------------------------------------------+
__kernel void MHRelativeAttentionOut(__global const float *q,        ///<[in] Matrix of Querys
                                     __global const float *k,        ///<[in] Matrix of Keys
                                     __global const float *v,        ///<[in] Matrix of Values
                                     __global const float *bk,       ///<[in] Matrix of Positional Bias Keys
                                     __global const float *bv,       ///<[in] Matrix of Positional Bias Values
                                     __global const float *gc,       ///<[in] Global content bias vector
                                     __global const float *gp,       ///<[in] Global positional bias vector
                                     __global float *score,          ///<[out] Matrix of Scores
                                     __global float *out,            ///<[out] Matrix of attention
                                     const int dimension              ///< Dimension of Key
                                    )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads * k_id + h);
   const int shift_gc = dimension * h;
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- score
   float sc = 0;
//---
   for(int d = 0; d < dimension; d++)
     {
      float val_q = q[shift_q + d];
      float val_k = k[shift_kv + d];
      float val_bk = bk[shift_kv + d];
      sc += val_q * val_k + val_q * val_bk + val_k * val_bk + gc[shift_q + d] * val_k + gp[shift_q + d] * val_bk;
     }
   sc = sc / koef;
//--- max value
//---
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? sc : fmax(temp[shift_local], sc));
        }
      BarrierLoc
     }
   uint count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] = (k_id < count && (k_id + count) < kunits ? fmax(temp[k_id + count], temp[k_id]) : temp[k_id]);
      BarrierLoc
     }
   while(count > 1);
   sc = IsNaNOrInf(exp(fmax(sc - temp[0], -120)), 0);
   BarrierLoc
//--- sum of exp
//---
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? 0 : temp[shift_local]) + sc;
        }
      BarrierLoc
     }
//---
   count = min(ls, (uint)kunits);
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//--- score
   float sum = IsNaNOrInf(temp[0], 1);
   if(fabs(sum) <= 1.2e-7f)
      sum = 1;
   sc /= sum;
   score[shift_s] = sc;
   BarrierLoc
//--- out
//---
   for(int d = 0; d < dimension; d++)
     {
      float val_v = v[shift_kv + d];
      float val_bv = bv[shift_kv + d];
      float val = IsNaNOrInf(sc * (val_v + val_bv), 0);
      //--- sum of value
      for(int cur_v = 0; cur_v < kunits; cur_v += ls)
        {
         if(k_id >= cur_v && k_id < (cur_v + ls))
           {
            int shift_local = k_id % ls;
            temp[shift_local] = (cur_v == 0 ? 0 : temp[shift_local]) + val;
           }
         BarrierLoc
        }
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < count && (k_id + count) < kunits)
            temp[k_id] += temp[k_id + count];
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         BarrierLoc
        }
      while(count > 1);
      //---
      if(k_id == 0)
         out[shift_q + d] = IsNaNOrInf(temp[0], 0);
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh relative attention inside gradients gradient     |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void MHRelativeAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *k, __global float *k_g,
      __global const float *v, __global float *v_g,
      __global const float *bk, __global float *bk_g,
      __global const float *bv, __global float *bv_g,
      __global const float *gc, __global float *gc_g,
      __global const float *gp, __global float *gp_g,
      __global const float *scores,
      __global const float *gradient,
      const int kunits
                                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
//---
   for(int v_id = q_id; v_id < kunits; v_id += qunits)
     {
      float grad = 0;
      int shift_score = h * kunits + v_id;
      for(int g = 0; g < qunits; g++)
         grad += gradient[shift_g + dimension * (g  * heads)] *
                 scores[shift_score + g * step_score];
      int shift_v = dimension * (heads * v_id + h) + d;
      grad = IsNaNOrInf(grad, 0);
      v_g[shift_v] = grad;
      bv_g[shift_v] = grad;
     }
//--- Calculating Query's gradients
   float grad_gc = 0;
   float grad_gp = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = h * dimension + d;
   int shift_key = h * dimension + d;
//---
   for(int k_id = 0; k_id < kunits; k_id++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k_id];
      if(sc == 0)
         continue;
      for(int v_id = 0; v_id < kunits; v_id++)
         sc_g += scores[shift_s + v_id] * out_g *
                 (v[shift_val + v_id * heads * dimension] + bv[shift_val + v_id * heads * dimension]) *
                 ((float)(k_id == v_id) - sc);
      grad_gc += IsNaNOrInf(sc_g * k[shift_key + k_id * heads * dimension], 0);
      grad_gp += IsNaNOrInf(sc_g * bk[shift_key + k_id * heads * dimension], 0);
     }
//---
   q_g[shift_q] = (grad_gc + grad_gp) / koef;
   gc_g[shift_q] = grad_gc / koef;
   gp_g[shift_q] = grad_gp / koef;
//--- Calculating Key's gradients
//---
   for(int k_id = q_id; k_id < kunits; k_id += qunits)
     {
      int shift_k = dimension * (heads * k_id + h) + d;
      float grad = 0;
      float grad_bk = 0;
      int shift_score = h * kunits + k_id;
      float val = (v[shift_k] + bv[shift_k]);
      for(int scr = 0; scr < qunits; scr++)
        {
         float sc_g = 0;
         int shift_sc = scr * kunits * heads;
         float sc = scores[shift_sc + k_id];
         if(sc == 0)
            continue;
         for(int v_id = 0; v_id < kunits; v_id++)
            sc_g += scores[shift_sc + v_id] * gradient[shift_g + scr * dimension] *
                    val * ((float)(k_id == v_id) - sc);
         float _q = q[shift_g + scr * heads * dimension];
         grad += sc_g * (_q + bk[shift_k] + gc[shift_g + scr * heads * dimension]);
         grad_bk += sc_g * (_q + k[shift_k] + gp[shift_g + scr * heads * dimension]);
        }
      k_g[shift_k] = grad / koef;
      bk_g[shift_k] = grad_bk / koef;
     }
  }
//+------------------------------------------------------------------+
//| Computes the calc alignment gradient gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void CalcAlignmentGradient(__global const float *matrix_o1,
                                    __global const float *matrix_o2,
                                    __global float *matrix_g1,
                                    __global float *matrix_g2,
                                    const int activation,
                                    const int add)
  {
   int i = get_global_id(0);
   const float out1 = IsNaNOrInf(matrix_o1[i], 0);
   const float out2 = IsNaNOrInf(matrix_o2[i], 0);
   float grad1 = Deactivation(out2 - out1, out1, activation);
   float grad2 = Deactivation(out1 - out2, out2, activation);
//---
   if(add > 0)
     {
      matrix_g1[i] += grad1;
      matrix_g2[i] += grad2;
     }
   else
     {
      matrix_g1[i] = grad1;
      matrix_g2[i] = grad2;
     }
  }
//+------------------------------------------------------------------+
//| Implements the feature smoothing kernel on the GPU.              |
//+------------------------------------------------------------------+
__kernel void FeatureSmoothing(__global const float *feature,
                               __global float *outputs,
                               const int smoothing
                              )
  {
   const size_t pos = get_global_id(0);
   const size_t d = get_global_id(1);
   const size_t total = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   const int shift_input = pos * dimension + d;
   const int shift_output = dimension * pos * smoothing + d;
//---
   float value = IsNaNOrInf(feature[shift_input], 0);
   outputs[shift_output] = value;
//---
   for(int s = 1; s <= smoothing; s++)
     {
      if(((int)pos - s) >= 0)
         value += IsNaNOrInf(feature[shift_input - s * dimension], 0);
      if(((int)pos + s) < total)
         value += IsNaNOrInf(feature[shift_input + s * dimension], 0);
      float factor = IsNaNOrInf(1.0f / (min((int)total, (int)(pos + s)) - max((int)(pos - s), 0) + 1), 0);
      float out = IsNaNOrInf(value * factor, 0);
      outputs[shift_output + s * dimension] = out;
     }
  }
//+------------------------------------------------------------------+
//| Computes the feature smoothing gradient gradient contributions on|
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void FeatureSmoothingGradient(__global float *feature_gr,
                                       __global const float *outputs_gr,
                                       const int smoothing
                                      )
  {
   const size_t pos = get_global_id(0);
   const size_t d = get_global_id(1);
   const size_t total = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   const int shift_input = pos * dimension + d;
   const int shift_output = dimension * pos * smoothing + d;
   const int step_output = dimension * smoothing;
//---
   float grad = IsNaNOrInf(outputs_gr[shift_output], 0);
//---
   for(int s = 1; s <= smoothing; s++)
     {
      int shift = shift_output + s * dimension;
      float factor = 1.0f / (min((int)total, (int)(pos + s)) - max((int)(pos - s), 0) + 1);
      //---
      float value = IsNaNOrInf(outputs_gr[shift] * factor, 0);
      //---
      if(((int)pos - s) >= 0)
         grad += IsNaNOrInf(outputs_gr[shift - s * step_output] * factor, 0);
      //---
      if(((int)pos + s) < total)
         grad += IsNaNOrInf(outputs_gr[shift + s * step_output] * factor, 0);
     }
//---
   feature_gr[shift_input] = IsNaNOrInf(grad, 0);
  }
//+------------------------------------------------------------------+
//| Runs the batch feed forward add noise feed-forward pass on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void BatchFeedForwardAddNoise(__global const float *inputs, __global float *options,
                                       __global const float *noise, __global float *output,
                                       const int batch, const int optimization,
                                       const int activation, const float alpha)
  {
   if(batch <= 1)
      return;
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   float inp = inputs[n];
   float mean = (batch > 1 ? (IsNaNOrInf(options[shift], 0) * ((float)batch - 1.0f) + inp) / ((float)batch) : inp);
   float delt = inp - mean;
   float variance = IsNaNOrInf(options[shift + 1], 0) * ((float)batch - 1.0f) + delt * delt;
   if(batch > 0)
      variance /= (float)batch;
   float nx = (variance > 0 ? delt / sqrt(variance) : 0);
   float noisex = sqrt(alpha) * nx + sqrt(1 - alpha) * fabs(noise[n]) * sign(nx);
//---
   float gamma = IsNaNOrInf(options[shift + 3], 0);
   if(gamma == 0)
     {
      options[shift + 3] = 1;
      gamma = 1;
     }
   float betta = IsNaNOrInf(options[shift + 4], 0);
//---
   options[shift] = mean;
   options[shift + 1] = variance;
   options[shift + 2] = nx;
   output[n] = fActivation(gamma * noisex + betta, activation);
  }
//+------------------------------------------------------------------+
//| Implements the hyper projection kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void HyperProjection(__global const float *inputs,
                              __global float *outputs
                             )
  {
   const size_t pos = get_global_id(0);
   const size_t d = get_local_id(1);
   const size_t total = get_global_size(0);
   const size_t dimension = get_local_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((int)dimension, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_in = pos * dimension + d;
   const int shift_out = pos * (dimension + 1) + d + 1;
//---
   float v = IsNaNOrInf(inputs[shift_in], 0);
//---
   float v2 = IsNaNOrInf(v * v, 0);
//---
   if(d < ls)
      temp[d] = v2;
   BarrierLoc
//---
   for(int i = ls; i < (int)dimension; i += ls)
     {
      if(d >= i && d < (i + ls))
         temp[d % ls] += v2;
      BarrierLoc
     }
//---
   int count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//---
   outputs[shift_out] = v;
   if(d == 0)
     {
      v = IsNaNOrInf(((float)pos) / ((float)total), 0);
      outputs[shift_out - 1] = sqrt(fmax(temp[0] - v * v, 1.2e-07f));
     }
  }
//+------------------------------------------------------------------+
//| Computes the hyper projection grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void HyperProjectionGrad(__global const float *inputs,
                                  __global float *inputs_gr,
                                  __global const float *outputs_gr
                                 )
  {
   const size_t pos = get_global_id(0);
   const size_t d = get_global_id(1);
   const size_t total = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   const int shift_in = pos * dimension + d;
   const int shift_start_out = pos * (dimension + 1);
   const int shift_out = shift_start_out + d + 1;
//---
   float v = IsNaNOrInf(inputs[shift_in], 0);
   float grad = IsNaNOrInf(outputs_gr[shift_out], 0);
//---
   v = IsNaNOrInf(v * outputs_gr[shift_start_out], 0);
//---
   inputs_gr[shift_in] = v + grad;
  }
//+------------------------------------------------------------------+
//| Implements the log map kernel on the GPU.                        |
//+------------------------------------------------------------------+
__kernel void LogMap(__global const float *features,
                     __global const float *centroids,
                     __global const float *curvatures,
                     __global float *outputs,
                     __global float *product,
                     __global float *distance,
                     __global float *norma
                    )
  {
//--- identify
   const size_t f = get_global_id(0);
   const size_t cent = get_global_id(1);
   const size_t d = get_local_id(2);
   const size_t total_f = get_global_size(0);
   const size_t total_cent = get_global_size(1);
   const size_t dimension = get_local_size(2);
//--- create local array
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((int)dimension, (int)LOCAL_ARRAY_SIZE);
//--- calc shifts
   const int shift_f = f * dimension + d;
   const int shift_out = (f * total_cent + cent) * dimension + d;
   const int shift_cent = cent * dimension + d;
   const int shift_temporal = f * total_cent + cent;
//--- load inputs
   float feature = IsNaNOrInf(features[shift_f], 0);
   float centroid = IsNaNOrInf(centroids[shift_cent], 0);
   float curv = IsNaNOrInf(curvatures[cent], 1.2e-7f);
//--- dot(features, centroids)
   float fc = IsNaNOrInf(feature * centroid, 0);
//---
   if(d < ls)
      temp[d] = (d > 0 ? fc : -fc);
   BarrierLoc
//---
   for(int i = ls; i < (int)dimension; i += ls)
     {
      if(d >= i && d < (i + ls))
         temp[d % ls] += fc;
      BarrierLoc
     }
//---
   int count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
   float prod = IsNaNOrInf(temp[0], 0);
   product[shift_temporal] = prod;
//--- project
   float u = IsNaNOrInf(feature + prod * centroid * curv, 0);
//--- norm(u)
   float u2 = IsNaNOrInf(u * u, 0);
//---
   if(d < ls)
      temp[d] = (d > 0 ? u2 : -u2);
   BarrierLoc
//---
   for(int i = ls; i < (int)dimension; i += ls)
     {
      if(d >= i && d < (i + ls))
         temp[d % ls] += u2;
      BarrierLoc
     }
//---
   count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
   float normu = IsNaNOrInf(temp[0], 0);
   if(normu <= 0)
      normu = 1.0e-7f;
   normu = sqrt(normu);
   norma[shift_temporal] = normu;
//--- distance features to centroid
   float theta = IsNaNOrInf(-prod * curv, 0);
   theta = fmax(theta, 1.0f + 1.2e-07f);
   float acosh_theta = acosh(theta);
   float dist = IsNaNOrInf(sqrt(clamp((acosh_theta * acosh_theta) / curv, 0.0f, 50.0f)), 0);
   distance[shift_temporal] = dist;
   float proj_u = IsNaNOrInf(dist * u / normu, 0);
//---
   if(d < ls)
      temp[d] = (d > 0 ? proj_u * centroid : 0);
   BarrierLoc
//---
   for(int i = ls; i < (int)dimension; i += ls)
     {
      if(d >= i && d < (i + ls))
         temp[d % ls] += proj_u * centroid;
      BarrierLoc
     }
//---
   count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//---
   if(d == 0)
     {
      proj_u = IsNaNOrInf(temp[0] / centroid, 0);
      proj_u = fmax(u, 1.2e-7f);
     }
//---
   outputs[shift_out] = proj_u;
  }
//+------------------------------------------------------------------+
//| Computes the log map grad gradient contributions on the GPU.     |
//+------------------------------------------------------------------+
__kernel void LogMapGrad(__global const float *features, __global float *features_gr,
                         __global const float *centroids, __global float *centroids_gr,
                         __global const float *curvatures, __global float *curvatures_gr,
                         __global const float *outputs, __global const float *outputs_gr,
                         __global const float *product,
                         __global const float *distance,
                         __global const float *norma
                        )
  {
//--- identify
   const size_t f = get_global_id(0);
   const size_t cent = get_global_id(1);
   const size_t d = get_local_id(2);
   const size_t total_f = get_global_size(0);
   const size_t total_cent = get_global_size(1);
   const size_t dimension = get_local_size(2);
//--- create local array
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((int)dimension, (int)LOCAL_ARRAY_SIZE);
//--- calc shifts
   const int shift_f = f * dimension + d;
   const int shift_out = (f * total_cent + cent) * dimension + d;
   const int shift_cent = cent * dimension + d;
   const int shift_temporal = f * total_cent + cent;
//--- load inputs
   float feature = features[shift_f];
   if(isinf(feature) || isnan(feature))
      feature = 0;
   float centroid = centroids[shift_cent];
   if(isinf(centroid) || isnan(centroid))
      centroid = 0;
   float centroid0 = (d > 0 ? centroids[shift_cent - d] : centroid);
   if(isinf(centroid0) || isnan(centroid0) || centroid0 == 0)
      centroid0 = 1.2e-7f;
   float curv = curvatures[cent];
   if(isinf(curv) || isnan(curv))
      curv = 1.2e-7f;
   float prod = product[shift_temporal];
   float dist = distance[shift_temporal];
   float normu = norma[shift_temporal];
   float u = feature + prod * centroid * curv;
   if(isinf(u) || isnan(u))
      u = 0;
//---
   float grad = outputs_gr[shift_out];
   if(isinf(grad) || isnan(grad))
      grad = 0;
   float grad0 = (d > 0 ? outputs_gr[shift_out - d] : grad);
   if(isinf(grad0) || isnan(grad0))
      grad0 = 0;
//---
   float feature_gr = 0;
   float centroid_gr = 0;
   float curv_gr = 0;
   float prod_gr = 0;
   float normu_gr = 0;
   float dist_gr = 0;
//---
   float proj_u_gr = (d > 0 ? grad + grad0 / centroid0 * centroid : 0);
   if(d == 0)
      centroid_gr += outputs[shift_out] / centroid * grad;
   else
      centroid_gr += grad0 / centroid0 * outputs[shift_out];
   if(isnan(centroid_gr) || isinf(centroid_gr))
      centroid_gr = 0;
//---
   dist_gr = u / normu * proj_u_gr;
   float u_gr = dist / normu * proj_u_gr;
   normu_gr = dist * u / (normu * normu) * proj_u_gr;
//---
   if(d < ls)
      temp[d] = dist_gr;
   BarrierLoc
//---
   for(int id = ls; id < (int)dimension; id += ls)
     {
      if(d >= id && d < (id + ls))
         temp[d % ls] += dist_gr;
      BarrierLoc
     }
//---
   int count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
   if(isinf(temp[0]) || isnan(temp[0]))
      temp[0] = 0;
   dist_gr = temp[0];
//---
   if(d == 0)
     {
      float theta = -prod * curv;
      float theta_gr = 1.0f / sqrt(curv * (theta * theta - 1)) * dist_gr;
      if(isinf(theta_gr) || isnan(theta_gr))
         theta_gr = 0;
      float acosh_theta = acosh(theta);
      curv_gr += -(acosh_theta * acosh_theta) / (2 * sqrt(curv * curv * curv)) * dist_gr;
      if(isinf(curv_gr) || isnan(curv_gr))
         curv_gr = 0;
      temp[0] = -curv * theta_gr;
      if(isinf(temp[0]) || isnan(temp[0]))
         temp[0] = 0;
      curv_gr += -prod * theta_gr;
      if(isinf(curv_gr) || isnan(curv_gr))
         curv_gr = 0;
     }
   BarrierLoc
//---
   prod_gr += temp[0];
   BarrierLoc
//---
   if(d < ls)
      temp[d] = normu_gr;
   BarrierLoc
//---
   for(int id = ls; id < (int)dimension; id += ls)
     {
      if(d >= id && d < (id + ls))
         temp[d % ls] += normu_gr;
      BarrierLoc
     }
//---
   count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
   normu_gr = temp[0];
   if(isinf(normu_gr) || isnan(normu_gr))
      normu_gr = 1.2e-7f;
   u_gr += u / normu * normu_gr;
   if(isnan(u_gr) || isinf(u_gr))
      u_gr = 0;
//---
   feature_gr += u_gr;
   centroid_gr += prod * curv * u_gr;
   BarrierLoc
//--- dot (u_gr * centroid)
   if(d < ls)
      temp[d] = u_gr * centroid;
   BarrierLoc
//---
   for(int id = ls; id < (int)dimension; id += ls)
     {
      if(d >= id && d < (id + ls))
         temp[d % ls] += u_gr * centroid;
      BarrierLoc
     }
//---
   count = min(ls, (int)dimension);
//---
   do
     {
      count = (count + 1) / 2;
      if(d < count)
         temp[d] += ((d + count) < dimension ? temp[d + count] : 0);
      if(d + count < dimension)
         temp[d + count] = 0;
      BarrierLoc
     }
   while(count > 1);
   if(d == 0)
     {
      if(isinf(temp[0]) || isnan(temp[0]))
         temp[0] = 0;
      prod_gr += temp[0] * curv;
      if(isinf(prod_gr) || isnan(prod_gr))
         prod_gr = 0;
      curv_gr += temp[0] * prod;
      if(isinf(curv_gr) || isnan(curv_gr))
         curv_gr = 0;
      temp[0] = prod_gr;
     }
   BarrierLoc
//---
   prod_gr = temp[0];
   feature_gr += prod_gr * centroid * (d > 0 ? 1 : -1);
   centroid_gr += prod_gr * feature * (d > 0 ? 1 : -1);
//--- result
   features_gr[shift_f] += feature_gr;
   centroids_gr[shift_cent] += centroid_gr;
   if(f == 0 && d == 0)
      curvatures_gr[cent] += curv;
  }
//+------------------------------------------------------------------+
//| Calculates calc epsilon weights on the GPU.                      |
//+------------------------------------------------------------------+
__kernel void CalcEpsilonWeights(__global const float *matrix_w,
                                 __global const float *matrix_g,
                                 __global const float *matrix_i,
                                 __global float *matrix_epsw,
                                 const float rho
                                )
  {
   const size_t inp = get_local_id(0);
   const size_t inputs = get_local_size(0) - 1;
   const size_t out = get_global_id(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((int)inputs, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_w = out * (inputs + 1) + inp;
   const float w = IsNaNOrInf(matrix_w[shift_w], 0);
   float grad = fabs(w) * IsNaNOrInf(matrix_g[out], 0) * (inputs == inp ? 1.0f : IsNaNOrInf(matrix_i[inp], 0));
//---
   const int local_shift = inp % ls;
//---
   for(int i = 0; i <= inputs; i += ls)
     {
      if(i <= inp && inp < (i + ls))
         temp[local_shift] = (i == 0 ? 0 : temp[local_shift]) + IsNaNOrInf(grad * grad, 0);
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(inp < count)
         temp[inp] += ((inp + count) < inputs ? IsNaNOrInf(temp[inp + count], 0) : 0);
      if(inp + count < inputs)
         temp[inp + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//---
   float norm = sqrt(IsNaNOrInf(temp[0], 0));
   float epsw = IsNaNOrInf(w * w * grad * rho / (norm + 1.2e-7f), w);
//---
   matrix_epsw[shift_w] = epsw;
  }
//+------------------------------------------------------------------+
//| Calculates calc epsilon weights conv on the GPU.                 |
//+------------------------------------------------------------------+
__kernel void CalcEpsilonWeightsConv(__global const float *matrix_w,
                                     __global const float *matrix_g,
                                     __global const float *matrix_i,
                                     __global float *matrix_epsw,
                                     const int inputs,
                                     const float rho,
                                     const int step
                                    )
  {
//---
   const size_t inp = get_local_id(0);
   const size_t window_in = get_local_size(0) - 1;
   const size_t out = get_global_id(1);
   const size_t window_out = get_global_size(1);
   const size_t v = get_global_id(2);
   const size_t variables = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((int)(window_in + 1), (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_w = (out + v * window_out) * (window_in + 1) + inp;
   const int total = (inputs - window_in + step - 1) / step;
   const int shift_out = v * total * window_out + out;
   const int shift_in = v * inputs + inp;
   const float w = IsNaNOrInf(matrix_w[shift_w], 0);
//---
   float grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(inp != window_in && (inp + t * step) >= inputs)
         break;
      float g = IsNaNOrInf(matrix_g[t * window_out + shift_out], 0);
      float i = IsNaNOrInf(inp == window_in ? 1.0f : matrix_i[t * step + shift_in], 0);
      grad += IsNaNOrInf(g * i, 0);
     }
   grad *= fabs(w);
//---
   const int local_shift = inp % ls;
//---
   for(int i = 0; i <= inputs; i += ls)
     {
      if(i <= inp && inp < (i + ls))
         temp[local_shift] = (i == 0 ? 0 : temp[local_shift]) + IsNaNOrInf(grad * grad, 0);
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(inp < count && (inp + count) < inputs)
        {
         temp[inp] += IsNaNOrInf(temp[inp + count], 0);
         temp[inp + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   float norm = sqrt(IsNaNOrInf(temp[0], 0));
   float epsw = IsNaNOrInf(w * w * grad * rho / (norm + 1.2e-7f), w);
//---
   matrix_epsw[shift_w] = epsw;
  }
//+------------------------------------------------------------------+
//| Implements the plr multi agents kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void PLRMultiAgents(__global const float *inputs,
                             __global float *outputs,
                             __global int *isttp,
                             const int transpose,
                             __global const float *min_step
                            )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
   const size_t a = get_global_id(2);
   const size_t agents = get_global_size(2);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
   const int shift_ag = a * lenth * variables;
//--- look for ttp
   float value = IsNaNOrInf(inputs[shift_in], 0);
   bool bttp = false;
   if(i == 0 || i == lenth - 1)
      bttp = true;
   else
     {
      float prev = value;
      int prev_pos = i;
      float max_v = value;
      float max_pos = i;
      float min_v = value;
      float min_pos = i;
      while(fmax(fabs(prev - max_v), fabs(prev - min_v)) < min_step[a] && prev_pos > 0)
        {
         prev_pos--;
         prev = IsNaNOrInf(inputs[shift_in - (i - prev_pos) * step_in], 0);
         if(prev >= max_v && (prev - min_v) < min_step[a])
           {
            max_v = prev;
            max_pos = prev_pos;
           }
         if(prev <= min_v && (max_v - prev) < min_step[a])
           {
            min_v = prev;
            min_pos = prev_pos;
           }
        }
      //---
      float next = value;
      int next_pos = i;
      while(fmax(fabs(next - max_v), fabs(next - min_v)) < min_step[a] && next_pos < (lenth - 1))
        {
         next_pos++;
         next = IsNaNOrInf(inputs[shift_in + (next_pos - i) * step_in], 0);
         if(next > max_v && (next - min_v) < min_step[a])
           {
            max_v = next;
            max_pos = next_pos;
           }
         if(next < min_v && (max_v - next) < min_step[a])
           {
            min_v = next;
            min_pos = next_pos;
           }
        }
      //---
      if(
         (value >= prev && value > next) ||
         (value > prev && value == next) ||
         (value <= prev && value < next) ||
         (value < prev && value == next)
      )
         if(max_pos == i || min_pos == i)
            bttp = true;
     }
//---
   isttp[shift_in + shift_ag] = (int)bttp;
   outputs[shift_in + shift_ag] = 0;
   BarrierLoc
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int prev_ttp = 0;
   if(bttp)
     {
      pos = 0;
      //---
      for(int p = 0; p < i; p++)
        {
         int current_in = ((bool)transpose ? (p * variables + v) : (v * lenth + p));
         if((bool)isttp[current_in + shift_ag])
           {
            pos++;
            prev_ttp = p;
            prev_in = current_in;
           }
        }
     }
//--- cacl tendency
   if(pos > 0 && pos < (lenth / 3))
     {
      float sum_x = 0;
      float sum_y = 0;
      float sum_xy = 0;
      float sum_xx = 0;
      int dist = i - prev_ttp;
      //---
      for(int p = 0; p < dist; p++)
        {
         float x = (float)(p);
         float y = IsNaNOrInf(inputs[prev_in + p * step_in], 0);
         sum_x += x;
         sum_y += y;
         sum_xy += x * y;
         sum_xx += x * x;
        }
      float slope = IsNaNOrInf((dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1), 0);
      float intercept = IsNaNOrInf((sum_y - slope * sum_x) / dist, 0);
      int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3)) + shift_ag;
      outputs[shift_out] = slope;
      outputs[shift_out + step_in] = intercept;
      outputs[shift_out + 2 * step_in] = ((float)dist) / lenth;
     }
   else
     {
      if(pos == (lenth / 3))
        {
         float sum_x = 0;
         float sum_y = 0;
         float sum_xy = 0;
         float sum_xx = 0;
         int dist = lenth - prev_ttp;
         //---
         for(int p = 0; p < dist; p++)
           {
            float x = (float)(p);
            float y = IsNaNOrInf(inputs[prev_in + p * step_in], 0);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
           }
         float slope = IsNaNOrInf((dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1), 0);
         float intercept = IsNaNOrInf((sum_y - slope * sum_x) / dist, 0);
         int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3)) + shift_ag;
         outputs[shift_out] = slope;
         outputs[shift_out + step_in] = intercept;
         outputs[shift_out + 2 * step_in] = IsNaNOrInf((float)dist / lenth, 0);
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the plr multi agents gradient gradient contributions on |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void PLRMultiAgentsGradient(__global float *inputs_gr,
                                     __global const float *outputs,
                                     __global const float *outputs_gr,
                                     const int transpose,
                                     const int agents
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
   const int shift_out = ((bool)transpose ? v : (v * lenth));
   const int step_out = 3 * step_in;
   const int shift_ag = lenth * variables;
//--- Sum gradient
   float grad = 0;
//---
   for(int a = 0; a < agents; a++)
     {
      //--- calc position
      int pos = -1;
      int prev_in = 0;
      int dist = 0;
      do
        {
         pos++;
         prev_in += dist;
         dist = (int)fmax(outputs[shift_out + pos * step_out + 2 * step_in + a * shift_ag] * lenth, 1);
        }
      while(!(prev_in <= i && (prev_in + dist) > i));
      //--- calc constants
      float sum_x = 0;
      float sum_xx = 0;
      for(int p = 0; p < dist; p++)
        {
         float x = (float)(p);
         sum_x += x;
         sum_xx += x * x;
        }
      //--- get output gradient
      float grad_slope = IsNaNOrInf(outputs_gr[shift_out + pos * step_out + a * shift_ag], 0);
      float grad_intercept = IsNaNOrInf(outputs_gr[shift_out + pos * step_out + step_in + a * shift_ag], 0);
      //--- calc gradient
      grad_slope -= IsNaNOrInf(sum_x / dist * grad_intercept, 0);
      grad_slope /= fmax(IsNaNOrInf(dist * sum_xx - sum_x * sum_x, 0), 1);
      grad += IsNaNOrInf(grad_intercept / dist, 0);
      grad += IsNaNOrInf((dist * (i - prev_in) - sum_x) * grad_slope, 0);
     }
   grad = clamp(IsNaNOrInf(grad / agents, 0), -MAX_GRAD, MAX_GRAD);
//--- save result
   inputs_gr[shift_in] = grad;
  }
//+------------------------------------------------------------------+
//| Runs the feed forward mh conv feed-forward pass on the GPU.      |
//+------------------------------------------------------------------+
__kernel void FeedForwardMHConv(__global float *matrix_w,
                                __global float *matrix_i,
                                __global float *matrix_o,
                                const int inputs,
                                const int step,
                                const int window_in,
                                const int window_out,
                                const int activation
                               )
  {
   const size_t i = get_global_id(0);
   const size_t h = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t heads = get_global_size(1);
//---
   const int window_in_h = (window_in + heads - 1) / heads;
   const int window_out_h = (window_out + heads - 1) / heads;
   const int shift_out = window_out * i + window_out_h * h;
   const int shift_in = step * i + window_in_h * h;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * window_out * total;
   const int shift_var_w = v * window_out * (window_in_h + 1);
   const int shift_w_h = h * window_out_h * (window_in_h + 1);
//---
   float sum = 0;
   float4 inp, weight;
   int stop = (window_in_h <= (inputs - shift_in) ? window_in_h : (inputs - shift_in));
   stop = min(stop, (int)(window_in - h * window_in_h));
//---
//---
   for(int out = 0; (out < window_out_h && (window_out_h * h + out) < window_out); out++)
     {
      int shift = (window_in_h + 1) * out + shift_w_h;
      for(int k = 0; k <= stop; k += 4)
        {
         switch(stop - k)
           {
            case 0:
               inp = (float4)(1, 0, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + window_in_h], 0, 0, 0);
               break;
            case 1:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], 1, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + window_in_h], 0, 0);
               break;
            case 2:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k],
                              matrix_i[shift_var_in + shift_in + k + 1], 1, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + window_in_h], 0);
               break;
            case 3:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], 1);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + shift_w_h]);
               break;
            default:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], matrix_i[shift_var_in + shift_in + k + 3]);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + k + 3]);
               break;
           }
         sum += IsNaNOrInf(dot(inp, weight), 0);
        }
      sum = IsNaNOrInf(sum, 0);
      //---
      matrix_o[shift_var_out + out + shift_out] = fActivation(sum, activation);;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_gr
/// Kernel of the Convolution neuron to transfer gradient
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMHConv(__global float *matrix_w,
                                       __global float *matrix_g,
                                       __global float *matrix_o,
                                       __global float *matrix_ig,
                                       const int outputs,
                                       const int step,
                                       const int window_in,
                                       const int window_out,
                                       const int activation,
                                       const int shift_out,
                                       const int heads
                                      )
  {
   const size_t i = get_global_id(0);
   const size_t inputs = get_global_size(0);
   const size_t v = get_global_id(1);
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
   const int window_in_h = (window_in + heads - 1) / heads;
   const int window_out_h = (window_out + heads - 1) / heads;
//---
   float sum = 0;
   float out = matrix_o[shift_var_in + i];
   const int w_start = i % step;
   const int start = max((int)((i - window_in + step) / step), 0);
   int stop = (w_start + step - 1) / step;
   stop = min((int)((i + step - 1) / step + 1), stop) + start;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
//---
   for(int k = start; k < stop; k++)
     {
      int head = (k % window_out) / window_out_h;
      for(int h = 0; h < window_out_h; h ++)
        {
         if((head * window_out_h + h) >= window_out)
            break;
         int shift_g = k * window_out + head * window_out_h + h;
         int shift_w = (stop - k - 1) * step + (i % step) / window_in_h +
                       head * (window_in_h + 1) + h * (window_in_h + 1);
         if(shift_g >= outputs || shift_w >= (window_in_h + 1) * window_out)
            break;
         sum += IsNaNOrInf(matrix_g[shift_out + shift_g + shift_var_out] * matrix_w[shift_w + shift_var_w], 0);
        }
     }
//---
   matrix_ig[shift_var_in + i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Adam optimization Updating
/// Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Convolution
/// Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMHConvAdam(__global float *matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m -
                                      ///< input window and n - output window
                                      __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                      __global const float *matrix_i, ///<[in] Inputs tensor
                                      __global float *matrix_m,      ///<[in] Matrix of first momentum
                                      __global float *matrix_v,      ///<[in] Matrix of seconfd momentum
                                      const int inputs,               ///< Number of inputs
                                      const float l,                  ///< Learning rates
                                      const float b1,                 ///< First momentum multiplier
                                      const float b2,                 ///< Second momentum multiplier
                                      const int window_in,                  ///< Size of input window
                                      const int window_out,                 ///< Size of output window
                                      const int step,                        ///< Step size
                                      const int heads
                                     )
  {
   const size_t i = get_global_id(0);
//---
   const int window_in_h = (window_in + heads - 1) / heads;
   const int v = i / ((window_in_h + 1) * window_out);
   const int shift = i % window_out;
   const int shift_out = i / (window_in_h + 1) - v;
   const int total = (inputs - window_in + step - 1) / step;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
//---
   float grad = 0;
//---
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in_h && (shift + t * window_in) >= inputs)
         break;
      grad += IsNaNOrInf(matrix_g[t * window_out + shift_out + shift_var_out] *
                         (shift == window_in_h ? 1 : matrix_i[shift + t * step + shift_var_in]), 0);
     }
   float mt = IsNaNOrInf(clamp(b1 * matrix_m[i] + (1 - b1) * grad, -1.0e5f, 1.0e5f), 0);
   float vt = IsNaNOrInf(clamp(b2 * matrix_v[i] + (1 - b2) * (grad * grad), 1.0e-6f, 1.0e6f), 1.0e-6f);
   float weight = matrix_w[i] + l * mt / sqrt(vt);
   matrix_w[i] = weight;
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the more less equal kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void MoreLessEqual(__global const float *input,
                            __global float *output)
  {
   const size_t i = get_global_id(0);
   const float value = IsNaNOrInf(input[i], 0);
   float result = 0;
   if(fabs(value) > 1.2e-7f)
     {
      if(value > 0)
         result = 1;
      else
         result = -1;
     }
   output[i] = result;
  }
//+------------------------------------------------------------------+
//| Implements the multi scale relative attention out kernel on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void MultiScaleRelativeAttentionOut(__global const float *q,        ///<[in] Matrix of Querys
      __global const float *k,        ///<[in] Matrix of Keys
      __global const float *v,        ///<[in] Matrix of Values
      __global const float *bk,       ///<[in] Matrix of Positional Bias Keys
      __global const float *bv,       ///<[in] Matrix of Positional Bias Values
      __global const float *gc,       ///<[in] Global content bias vector
      __global const float *gp,       ///<[in] Global positional bias vector
      __global float *score,          ///<[out] Matrix of Scores
      __global float *out,            ///<[out] Matrix of attention
      const int dimension              ///< Dimension of Key
                                            )
  {
//--- init
   const uint q_id = get_global_id(0);
   const uint k_id = get_local_id(1);
   const uint h = get_global_id(2);
   const uint qunits = get_global_size(0);
   const uint kunits = get_local_size(1);
   const uint heads = get_global_size(2);
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads * k_id + h);
   const int shift_gc = dimension * h;
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   const uint window = max((uint)((kunits + h) / (h + 1)), min((uint)3, kunits));
   float koef = sqrt((float)dimension);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- score
   float sc = 0;
   if(k_id < window)
     {
      //---
      for(int d = 0; d < dimension; d++)
        {
         float val_q = q[shift_q + d];
         float val_k = k[shift_kv + d];
         float val_bk = bk[shift_kv + d];
         sc += val_q * val_k + val_q * val_bk + val_k * val_bk + gc[shift_q + d] * val_k + gp[shift_q + d] * val_bk;
        }
      sc = sc / koef;
     }
//--- max value
//---
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id < window)
         if(k_id >= cur_k && k_id < (cur_k + ls))
           {
            int shift_local = k_id % ls;
            temp[shift_local] = (cur_k == 0 ? sc : fmax(temp[shift_local], sc));
           }
      BarrierLoc
     }
   uint count = min(ls, kunits);
//---
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < (window + 1) / 2)
         if(k_id < ls)
            temp[k_id] = (k_id < count && (k_id + count) < kunits ? fmax(temp[k_id + count], temp[k_id]) : temp[k_id]);
      BarrierLoc
     }
   while(count > 1);
   if(k_id < window)
      sc = IsNaNOrInf(exp(fmax(sc - temp[0], -120)), 0);
   BarrierLoc
//--- sum of exp
//---
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? 0 : temp[shift_local]) + sc;
        }
      BarrierLoc
     }
//---
   count = min(ls, (uint)kunits);
   do
     {
      count = (count + 1) / 2;
      if(k_id < count && k_id < (window + 1) / 2)
         temp[k_id] += ((k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      BarrierLoc
     }
   while(count > 1);
//--- score
   float sum = IsNaNOrInf(temp[0], 1);
   if(sum <= 1.2e-7f)
      sum = 1;
   sc /= sum;
   score[shift_s] = sc;
   BarrierLoc
//--- out
   int shift_local = k_id % ls;
//---
   for(int d = 0; d < dimension; d++)
     {
      float val_v = v[shift_kv + d];
      float val_bv = bv[shift_kv + d];
      float val = IsNaNOrInf(sc * (val_v + val_bv), 0);
      //--- sum of value
      for(int cur_v = 0; cur_v < kunits; cur_v += ls)
        {
         if(k_id >= cur_v && k_id < (cur_v + ls))
            temp[shift_local] = (cur_v == 0 ? 0 : temp[shift_local]) + val;
         BarrierLoc
        }
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < count && (k_id + count) < kunits)
            temp[k_id] += temp[k_id + count];
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         BarrierLoc
        }
      while(count > 1);
      //---
      if(k_id == 0)
         out[shift_q + d] = IsNaNOrInf(temp[0], 0);
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
//| Runs the ssm2 d feed forward feed-forward pass on the GPU.       |
//+------------------------------------------------------------------+
__kernel void SSM2D_FeedForward(__global const float *ah,
                                __global const float *b_time,
                                __global const float *b_var,
                                __global const float *px_time,
                                __global const float *px_var,
                                __global const float *c_time,
                                __global const float *c_var,
                                __global const float *delta_time,
                                __global const float *delta_var,
                                __global       float *hidden,
                                __global       float *y
                               )
  {
   const size_t n = get_local_id(0);
   const size_t d = get_global_id(1);
   const size_t n_total = get_local_size(0);
   const size_t d_total = get_global_size(1);
//--- Hidden state
//---
   for(int h = 0; h < 2; h++)
     {
      float new_h = ah[(2 * n + h) * d_total + d] + ah[(2 * n_total + 2 * n + h) * d_total + d];
      if(h == 0)
         new_h += b_time[n] * px_time[n * d_total + d];
      else
         new_h += b_var[n] * px_var[n * d_total + d];
      hidden[(h * n_total + n)*d_total + d] = IsNaNOrInf(new_h, 0);
     }
   BarrierLoc
//--- Output
   uint shift_c = n;
   uint shift_h1 = d;
   uint shift_h2 = shift_h1 + n_total * d_total;
   float value = 0;
//---
   for(int i = 0; i < n_total; i++)
     {
      value += IsNaNOrInf(c_time[shift_c] * delta_time[shift_c] * hidden[shift_h1], 0);
      value += IsNaNOrInf(c_var[shift_c] * delta_var[shift_c] * hidden[shift_h2], 0);
      shift_c += n_total;
      shift_h1 += d_total;
      shift_h2 += d_total;
     }
//---
   y[n * d_total + d] = IsNaNOrInf(value, 0);
  }
//+------------------------------------------------------------------+
//| Computes the ssm2 d calc hidden gradient gradient contributions  |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void SSM2D_CalcHiddenGradient(__global const float *ah,
                                       __global       float *grad_ah,         // Gradient with respect to ah
                                       __global const float *b_time,
                                       __global       float *grad_b_time,     // Gradient with respect to b_time
                                       __global const float *b_var,
                                       __global       float *grad_b_var,      // Gradient with respect to b_var
                                       __global const float *px_time,
                                       __global       float *grad_px_time,    // Gradient with respect to px_time
                                       __global const float *px_var,
                                       __global       float *grad_px_var,     // Gradient with respect to px_var
                                       __global const float *c_time,
                                       __global       float *grad_c_time,     // Gradient with respect to c_time
                                       __global const float *c_var,
                                       __global       float *grad_c_var,      // Gradient with respect to c_var
                                       __global const float *delta_time,
                                       __global       float *grad_delta_time, // Gradient with respect to delta_time
                                       __global const float *delta_var,
                                       __global       float *grad_delta_var,  // Gradient with respect to delta_var
                                       __global const float *hidden,
                                       __global const float *grad_y           // Gradient of loss with respect to y
                                      )
  {
//---
   const size_t n = get_global_id(0);
   const size_t d = get_local_id(1);
   const size_t n_total = get_global_size(0);
   const size_t d_total = get_local_size(1);
//--- Initialize indices for data access
   uint shift_c = n;
   uint shift_h1 = d;
   uint shift_h2 = shift_h1 + n_total * d_total;
   float grad_hidden1 = 0;
   float grad_hidden2 = 0;
//--- Backpropagation: compute hidden gradients from y
   for(int i = 0; i < n_total; i++)
     {
      float grad = grad_y[i * d_total + d];
      float c_t = c_time[shift_c];
      float c_v = c_var[shift_c];
      float delta_t = delta_time[shift_c];
      float delta_v = delta_var[shift_c];
      float h1 = hidden[shift_h1];
      float h2 = hidden[shift_h2];
      //-- Accumulate gradients for hidden states
      grad_hidden1 += IsNaNOrInf(grad * c_t * delta_t, 0);
      grad_hidden2 += IsNaNOrInf(grad * c_v * delta_v, 0);
      //--- Compute gradients for c_time, c_var, delta_time, delta_var
      grad_c_time[shift_c] += grad * delta_t * h1;
      grad_c_var[shift_c]  += grad * delta_v * h2;
      grad_delta_time[shift_c] += grad * c_t * h1;
      grad_delta_var[shift_c]  += grad * c_v * h2;
      //--- Update indices for the next element
      shift_c += n_total;
      shift_h1 += d_total;
      shift_h2 += d_total;
     }
//--- Backpropagate through hidden -> ah, b_time, px_time
   for(int h = 0; h < 2; h++)
     {
      float grad_h = (h == 0) ? grad_hidden1 : grad_hidden2;
      //--- Store gradients in ah (considering its influence on two elements)
      grad_ah[(2 * n + h) * d_total + d] = grad_h;
      grad_ah[(2 * (n_total + n) + h) * d_total + d] = grad_h;
     }
//--- Backpropagate through px_time and px_var (influenced by b_time and b_var)
   grad_px_time[n * d_total + d] = grad_hidden1 * b_time[n];
   grad_px_var[n * d_total + d] = grad_hidden2 * b_var[n];
   if(d == 0)
     {
      grad_b_time[n] = 0;
      grad_b_var[n] = 0;
     }
   BarrierLoc
//--- Sum gradients over all d for b_time and b_var
   grad_b_time[n] += grad_hidden1 * px_time[n * d_total + d];
   grad_b_var[n] += grad_hidden2 * px_var[n * d_total + d];
  }
//+------------------------------------------------------------------+
//| Implements the p scan kernel on the GPU.                         |
//+------------------------------------------------------------------+
__kernel void PScan(__global const float* A,
                    __global const float* X,
                    __global const float* H,
                    __global float* X_out)
  {
   const size_t idx = get_local_id(0);
   const size_t dim = get_global_id(1);
   const size_t L = get_local_size(0);
   const size_t D = get_global_size(1);
   const int num_steps = (int)log2((float)L);
//---
   __local float local_A[1024];
   __local float local_X[1024];
   __local float local_H[1024];
//--- Load data to local memory
   int offset = dim + idx * D;
   local_A[idx] = A[offset];
   local_X[idx] = X[offset];
   local_H[idx] = H[offset];
   BarrierLoc
//--- Scan
//---
   for(int step = 0; step < num_steps; step++)
     {
      int halfT = L >> (step + 1);
      if(idx < halfT)
        {
         int base = idx * 2;
         local_X[base + 1] += local_A[base + 1] * local_X[base];
         local_X[base + 1] *= local_H[base + 1];
         local_A[base + 1] *= local_A[base];
        }
      BarrierLoc
     }
//--- Save result
   X_out[offset] = local_X[idx];
  }
//+------------------------------------------------------------------+
//| Computes the p scan calc hidden gradient gradient contributions  |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void PScan_CalcHiddenGradient(__global const float* A,
                                       __global float*  grad_A,
                                       __global const float* X,
                                       __global float*  grad_X,
                                       __global const float* H,
                                       __global float*  grad_H,
                                       __global const float* grad_X_out)
  {
   const size_t idx = get_local_id(0);
   const size_t dim = get_global_id(1);
   const size_t L = get_local_size(0);
   const size_t D = get_global_size(1);
   const int num_steps = (int)log2((float)L);
//---
   __local float local_A[1024];
   __local float local_X[1024];
   __local float local_H[1024];
   __local float local_grad_X[1024];
   __local float local_grad_A[1024];
   __local float local_grad_H[1024];
//--- Load data to local memory
   int offset = idx * D + dim;
   local_A[idx] = A[offset];
   local_X[idx] = X[offset];
   local_H[idx] = H[offset];
   local_grad_X[idx] = grad_X_out[offset];
   local_grad_A[idx] = 0.0f;
   local_grad_H[idx] = 0.0f;
   BarrierLoc
//--- Reverse Scan (Backward)
//---
   for(int step = num_steps - 1; step >= 0; step--)
     {
      int halfT = L >> (step + 1);
      if(idx < halfT)
        {
         int base = idx * 2;
         // Compute gradients
         float grad_next = local_grad_X[base + 1] * local_H[base + 1];
         local_grad_H[base + 1] = local_grad_X[base + 1] * local_X[base];
         local_grad_A[base + 1] = local_grad_X[base + 1] * local_X[base];
         local_grad_X[base] += local_A[base + 1] * grad_next;
        }
      BarrierLoc
     }
//--- Save gradients
   grad_A[offset] = local_grad_A[idx];
   grad_X[offset] = local_grad_X[idx];
   grad_H[offset] = local_grad_H[idx];
  }
//+------------------------------------------------------------------+
//| Implements the diag mat mult kernel on the GPU.                  |
//+------------------------------------------------------------------+
__kernel void DiagMatMult(__global const float *diag,
                          __global const float *matr,
                          __global float *result,
                          int activation)
  {
   size_t row = get_global_id(0);
   size_t col = get_local_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_local_size(1);
//---
   __local float local_diag[1];
   if(cols == 0)
      local_diag[0] = diag[row + var * rows];
   BarrierLoc
//---
   int shift = RCtoFlat(row, col, rows, cols, var);
//---
   float res = local_diag[0] * matr[shift];
//---
   result[shift] = fActivation(res, activation);
  }
//+------------------------------------------------------------------+
//| Computes the diag mat mult grad gradient contributions on the    |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void DiagMatMultGrad(__global const float *diag,
                              __global float *grad_diag,
                              __global const float *matr,
                              __global float *grad_matr,
                              __global const float *grad_result)
  {
   size_t row = get_global_id(0);
   size_t col = get_local_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_local_size(1);
   size_t vars = get_global_size(2);
//---
   __local float local_diag[LOCAL_ARRAY_SIZE];
   if(cols == 0)
      local_diag[0] = diag[row + var * rows];
   BarrierLoc
//---
   int shift = (row  + var * rows) * cols + col;
//---
   float grad = grad_result[shift];
   float inp = matr[shift];
//---
   grad_matr[shift] = IsNaNOrInf(local_diag[0] * grad, 0);
   BarrierLoc
//---
   int loc = col % LOCAL_ARRAY_SIZE;
//---
   for(int c = 0; c < cols; c += LOCAL_ARRAY_SIZE)
     {
      if(c <= col && (c + LOCAL_ARRAY_SIZE) > col)
        {
         if(c == 0)
            local_diag[loc] = IsNaNOrInf(grad * inp, 0);
         else
            local_diag[loc] += IsNaNOrInf(grad * inp, 0);
        }
      BarrierLoc
     }
//---
   int count = min(LOCAL_ARRAY_SIZE, (int)cols);
   int ls = count;
//---
   do
     {
      count = (count + 1) / 2;
      if((col + count) < ls)
        {
         local_diag[col] += local_diag[col + count];
         local_diag[col + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   if(col == 0)
      grad_diag[row + var * rows] = IsNaNOrInf(local_diag[0], 0);
  }
//+------------------------------------------------------------------+
//| Implements the top kgates kernel on the GPU.                     |
//+------------------------------------------------------------------+
__kernel void TopKgates(__global const float *inputs,
                        __global const float *noises,
                        __global float *gates,
                        const uint k)
  {
   size_t idx = get_local_id(0);
   size_t var = get_global_id(1);
   size_t window = get_local_size(0);
   size_t vars = get_global_size(1);
//---
   const int shift_logit = var * 2 * window + idx;
   const int shift_std = shift_logit + window;
   const int shift_gate = var * window + idx;
//---
   float logit = IsNaNOrInf(inputs[shift_logit], MIN_VALUE);
   float noise = IsNaNOrInf(noises[shift_gate], 0);
   if(noise != 0)
     {
      noise *= fActivation(inputs[shift_std], 3);
      logit += IsNaNOrInf(noise, 0);
     }
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   const uint ls = min((uint)window, (uint)LOCAL_ARRAY_SIZE);
   uint bigger = 0;
   float max_logit = logit;
//--- Top K
   for(int i = 0; i < window; i += ls)
     {
      if(idx >= i && idx < (i + ls))
         temp[idx % ls] = logit;
      BarrierLoc
      for(int i1 = 0; (i1 < min((int)ls, (int)(window - i)) && bigger <= k); i1++)
        {
         if(temp[i1] > logit)
            bigger++;
         if(temp[i1] > max_logit)
            max_logit = temp[i1];
        }
      BarrierLoc
     }
//---
   if(bigger <= k)
      gates[shift_gate] = logit - max_logit;
   else
      gates[shift_gate] = MIN_VALUE;
  }
//+------------------------------------------------------------------+
//| Computes the top kgates grad gradient contributions on the GPU.  |
//+------------------------------------------------------------------+
__kernel void TopKgatesGrad(__global const float *inputs,
                            __global float *grad_inputs,
                            __global const float *noises,
                            __global const float *gates,
                            __global float *grad_gates)
  {
   size_t idx = get_global_id(0);
   size_t var = get_global_id(1);
   size_t window = get_global_size(0);
   size_t vars = get_global_size(1);
//---
   const int shift_logit = var * 2 * window + idx;
   const int shift_std = shift_logit + window;
   const int shift_gate = var * window + idx;
//---
   float grad = IsNaNOrInf(grad_gates[shift_gate], 0);
   grad_inputs[shift_logit] = grad;
//---
   float noise = IsNaNOrInf(noises[shift_gate], 0);
   if(noise == 0)
     {
      grad_inputs[shift_std] = 0;
      return;
     }
//---
   grad *= noise;
   grad_inputs[shift_std] = Deactivation(grad, fActivation(inputs[shift_std], 3), 3);
  }
//+------------------------------------------------------------------+
//| Applies the mask by distance operation to the GPU buffers.       |
//+------------------------------------------------------------------+
__kernel void MaskByDistance(__global const float *buf_real,
                             __global const float *buf_imag,
                             __global float *mask,
                             const int dimension
                            )
  {
   const size_t main = get_global_id(0);
   const size_t slave = get_local_id(1);
   const int total = (int)get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   int ls = min((int)total, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_main = main * dimension;
   const int shift_slave = slave * dimension;
   const int shift_mask = main * total + slave;
//--- calc distance
   float dist = 0;
   if(main != slave)
     {
      //---
      for(int d = 0; d < dimension; d++)
        {
         float delta = ComplexAbs((float2)(buf_real[shift_main + d], buf_imag[shift_main + d])) -
                       ComplexAbs((float2)(buf_real[shift_slave + d], buf_imag[shift_slave + d]));
         dist += delta * delta;
        }
      dist = sqrt(dist);
     }
//--- Look Max
//---
   for(int i = 0; i < total; i += ls)
     {
      if(i <= slave && (i + ls) > slave)
         Temp[slave % ls] = fmax((i == 0 ? 0 : Temp[slave % ls]), IsNaNOrInf(dist, 0));
      BarrierLoc
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(slave < count && (slave + count) < ls)
        {
         if(Temp[slave] < Temp[slave + count])
            Temp[slave] = Temp[slave + count];
         Temp[slave + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//--- Normalize
   if(Temp[0] > 0)
      dist /= Temp[0];
//--- result
   mask[shift_mask] = 1 - IsNaNOrInf(dist, 1);
  }
//+------------------------------------------------------------------+
//| Applies the mask attention operation to the GPU buffers.         |
//+------------------------------------------------------------------+
__kernel void MaskAttention(__global const float *q,     ///<[in] Matrix of Querys
                            __global const float *kv,    ///<[in] Matrix of Keys
                            __global float *scores,       ///<[out] Matrix of Scores
                            __global const float *masks,  ///<[in] Mask Matrix
                            __global float *out,         ///<[out] Matrix of attention
                            const int dimension,          ///< Dimension of Key
                            const int heads_kv
                           )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const float mask = IsNaNOrInf(masks[q_id * kunits + k], 0);
   const uint ls = min((uint)kunits, (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   float score = 0;
   if(mask != 0)
     {
      for(int d = 0; d < dimension; d++)
         score += IsNaNOrInf(q[shift_q + d] * kv[shift_k + d], 0);
      score = IsNaNOrInf(exp(score / koef * mask), 0);
     }
//--- sum of exp
   float sum = LocalSum(score, 1, temp);
//--- score
   if(sum > 0)
      score /= sum;
   scores[shift_s] = score;
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val = LocalSum(kv[shift_v + d] * score, 1, temp);
      if(k == 0)
         out[shift_q + d] = val;
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
//| Computes the mask attention gradients gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void MaskAttentionGradients(__global const float *q, __global float *q_g,
                                     __global const float *kv, __global float *kv_g,
                                     __global const float *scores,
                                     __global const float *gradient,
                                     const int kunits, const int heads_kv
                                    )
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v = q_id; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = IsNaNOrInf(gradient[shift_g + q_id * dimension], 0);
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
//---
   for(int k = 0; (k < kunits && out_g != 0); k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + 2 * v * heads_kv * dimension] *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + 2 * k * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += sc_g * q[dimension * (h + scr * heads) + d];
              }
           }
         kv_g[shift_k] = grad / koef;
        }
     }
  }
//+------------------------------------------------------------------+
//| Runs the feed forward mult win conv feed-forward pass on the GPU.|
//+------------------------------------------------------------------+
__kernel void FeedForwardMultWinConv(__global const float *matrix_w,
                                     __global const float *matrix_i,
                                     __global float *matrix_o,
                                     __global const int *windows_in,
                                     const int inputs,
                                     const int windows_total,
                                     const int window_out,
                                     const int activation
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t outputs = get_global_size(0);
//---
   const int block = window_out * windows_total;
   const int id = (int)(i % block);
   const int win_id = id / window_out;
   const int out_id = id % window_out;
//---
   int step = 0;
   int window_shift = 0;
   int shift_weight = 0;
   for(int w = 0; w < windows_total; w++)
     {
      int win = windows_in[w];
      if(w < win_id)
        {
         window_shift += win;
         shift_weight += (win + 1) * window_out;
        }
      step += win;
     }
   int window_in = windows_in[win_id];
   int steps = (int)(i / block);
   int shift_in = steps * step + window_shift;
   shift_weight += out_id * (window_in + 1);
   float sum = matrix_w[shift_weight + window_in];
   float inp = 0.0f;
//---
   for(int w = 0; w < window_in; w++)
      if((shift_in + w) < inputs)
        {
         inp = IsNaNOrInf(matrix_i[v * inputs + shift_in + w], 0.0f);
         if(inp == 0.0f)
            continue;
         sum += IsNaNOrInf(inp * matrix_w[shift_weight + w], 0.0f);
        }
//---
   matrix_o[v * outputs + i] = fActivation(sum, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_gr
/// Kernel of the Convolution neuron to transfer gradient
/// to previous layer (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMultWinConv(__global const float *matrix_w,
      __global const float *matrix_i,
      __global float *matrix_ig,
      __global const float *matrix_og,
      __global const int *windows_in,
      const int outputs,
      const int windows_total,
      const int window_out,
      const int activation
                                           )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t inputs = get_global_size(0);
//---
   int step = 0;
//---
   for(int w = 0; w < windows_total; w++)
      step += windows_in[w];
//---
   int steps = (int)(i / step);
   int id = i % step;
   int window = 0;
   int before = 0;
   int window_in = 0;
//---
   for(int w = 0; w < windows_total; w++)
     {
      window_in = windows_in[w];
      if(id < (before + window_in))
         break;
      window = w + 1;
      before += window_in;
     }
//---
   int shift_weight = (before + window) * window_out + id - before;
   int shift_out = (steps * windows_total + window) * window_out + v * outputs;
   float sum = 0;
//---
   for(int w = 0; w < window_out; w++)
     {
      float grad = IsNaNOrInf(matrix_og[shift_out + w], 0.0f);
      if(grad == 0.0f)
         continue;
      sum +=  IsNaNOrInf(grad * matrix_w[shift_weight + w * (window_in + 1)], 0);
     }
//---
   matrix_ig[v * inputs + i] = Deactivation(sum, matrix_i[v * inputs + i], activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Adam optimization Updating
/// Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Convolution
/// Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMultWinConvAdam(__global float *matrix_w,
      __global const float *matrix_og,
      __global const float *matrix_i,
      __global float *matrix_m,
      __global float *matrix_v,
      __global const int *windows_in,
      const int windows_total,
      const int window_out,
      const int inputs,
      const int outputs,
      const float l,
      const float b1,
      const float b2
                                          )
  {
   const size_t i = get_global_id(0);  // weight shift
   const size_t v = get_local_id(1);   // variable
   const size_t variables = get_local_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   int step_out = window_out * windows_total;
//---
   int step_in = 0;
   for(int w = 0; w < windows_total; w++)
      step_in += windows_in[w];
//---
   int window_shift = 0;
   int shift_in = 0;
   int shift_out = 0;
   int window = 0;
   int weight_shift = 0;
   int local_weight = 0;
//---
   for(int w = 0; w < windows_total; w++)
     {
      int win = windows_in[w];
      int block = (win + 1) * window_out;
      if(i < weight_shift + block)
        {
         shift_in = window_shift;
         shift_out = w * window_out;
         window = win;
         local_weight = i - weight_shift;
         break;
        }
      window_shift += win;
      weight_shift += block;
     }
   bool bias = ((local_weight % (window + 1)) == window);
   int t = local_weight / (window + 1);
   int input_offset = local_weight % (window + 1);
   shift_out += t;
   shift_in += input_offset;
//---
   float grad = 0;
   int total = (inputs + step_in - 1) / step_in;
//---
   for(int t = 0; t < total; t++)
     {
      int sh_out = t * step_out + shift_out;
      if(sh_out >= outputs)
         break;
      if(bias)
        {
         grad += IsNaNOrInf(matrix_og[v * outputs + sh_out], 0);
         continue;
        }
      //---
      int sh_in = t * step_in + shift_in;
      if(sh_in >= inputs)
         break;
      float grad_out = IsNaNOrInf(matrix_og[v * outputs + sh_out], 0.0f);
      if(grad_out == 0.0f)
         continue;
      float inp = IsNaNOrInf(matrix_i[v * inputs + sh_in], 0.0f);
      if(inp == 0.0f)
         continue;
      grad += IsNaNOrInf(grad_out * inp, 0);
     }
//--- sum
   grad = LocalSum(grad, 1, temp);
//---
   if(v == 0)
     {
      float mt = IsNaNOrInf(clamp(b1 * matrix_m[i] + (1 - b1) * grad, -1.0e5f, 1.0e5f), 0);
      float vt = IsNaNOrInf(clamp(b2 * matrix_v[i] + (1 - b2) * (grad * grad), 1.0e-6f, 1.0e6f), 1.0e-6f);
      float weight = matrix_w[i] + IsNaNOrInf(l * mt / sqrt(vt), 0);
      matrix_w[i] = weight;
      matrix_m[i] = mt;
      matrix_v[i] = vt;
     }
  }
//+------------------------------------------------------------------+
//| Applies the mask attention complex operation to the GPU buffers. |
//+------------------------------------------------------------------+
__kernel void MaskAttentionComplex(__global const float2* __attribute__((aligned(8)))q,     ///<[in] Matrix of Querys
                                   __global const float2* __attribute__((aligned(8)))kv,    ///<[in] Matrix of Keys
                                   __global float *scores,       ///<[out] Matrix of Scores
                                   __global const float *masks,  ///<[in] Mask Matrix
                                   __global float2* __attribute__((aligned(8)))out,         ///<[out] Matrix of attention
                                   const int dimension,          ///< Dimension of Key
                                   const int heads_kv
                                  )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
//---
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const float mask = IsNaNOrInf(masks[shift_s], 0);
   const uint ls = min((uint)kunits, (uint)LOCAL_ARRAY_SIZE);
   float2 koef = (float2)(fmax((float)sqrt((float)dimension), (float)1), 0);
   __local float2 temp[LOCAL_ARRAY_SIZE];
//--- Score
   float score = 0;
   float2 score2 = (float2)0;
   if(ComplexAbs(mask) >= 0.01f)
     {
      for(int d = 0; d < dimension; d++)
         score2 += IsNaNOrInf2(ComplexMul(q[shift_q + d], kv[shift_k + d]), (float2)0);
      score = IsNaNOrInf(ComplexAbs(ComplexExp(ComplexDiv(score, koef))) * mask, 0);
     }
//--- sum of exp
//---
   for(int i = 0; i < kunits; i += ls)
     {
      if(k >= i && k < (i + ls))
         temp[k % ls].x = (i == 0 ? 0 : temp[k % ls].x) + score;
      BarrierLoc
     }
//---
   uint count = ls;
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k].x += (k < count && (k + count) < kunits ? temp[k + count].x : 0);
      if(k + count < ls)
         temp[k + count].x = 0;
      BarrierLoc
     }
   while(count > 1);
//--- score
   if(temp[0].x > 0)
      score = score / temp[0].x;
   scores[shift_s] = score;
//--- out
//---
   for(int d = 0; d < dimension; d++)
     {
      float2 val = (score > 0 ? ComplexMul(kv[shift_v + d], (float2)(score, 0)) : (float2)0);
      //---
      for(int i = 0; i < kunits; i += ls)
        {
         if(k >= i && k < (i + ls))
            temp[k % ls] = (i == 0 ? (float2)0 : temp[k % ls]) + val;
         BarrierLoc
        }
      //---
      uint count = ls;
      //---
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : (float2)0);
         if((k + count) < ls)
            temp[k + count] = (float2)0;
         BarrierLoc
        }
      while(count > 1);
      //---
      if(k == 0)
         out[shift_q + d] = temp[0];
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
//| Computes the mask attention gradients complex gradient           |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void MaskAttentionGradientsComplex(__global const float2* __attribute__((aligned(8)))q, __global float2* __attribute__((aligned(8)))q_g,
      __global const float2* __attribute__((aligned(8)))kv, __global float2* __attribute__((aligned(8)))kv_g,
      __global const float *scores,
      __global const float *mask, __global float *mask_g,
      __global const float2* __attribute__((aligned(8)))gradient,
      const int kunits, const int heads_kv
                                           )
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float2 koef = (float2)(fmax(sqrt((float)dimension), (float)1), 0);
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v = q_id; v < kunits; v += qunits)
        {
         float2 grad = (float2)0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
              {
               float sc = IsNaNOrInf(scores[shift_score + g * step_score], 0);
               if(sc > 0)
                  grad += ComplexMul(gradient[shift_g + dimension * (hq - h + g  * heads)],
                                     (float2)(sc, 0));
              }
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
     }
//--- Calculating Query's gradients
   float2 grad = 0;
   float2 out_g = IsNaNOrInf2(gradient[shift_g + q_id * dimension], (float2)0);
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
//---
   for(int k = 0; (k < kunits && ComplexAbs(out_g) != 0); k++)
     {
      float2 sc_g = 0;
      float2 sc = (float2)(scores[shift_s + k], 0);
      for(int v = 0; v < kunits; v++)
         sc_g += IsNaNOrInf2(ComplexMul(
                                ComplexMul((float2)(scores[shift_s + v], 0),
                                           out_g * kv[shift_val + 2 * v * heads_kv * dimension]),
                                ((float2)(k == v, 0) - sc)), (float2)0);
      float m = mask[shift_s + k];
      mask_g[shift_s + k] = IsNaNOrInf(sc.x / m * sc_g.x + sc.y / m * sc_g.y, 0);
      grad += IsNaNOrInf2(ComplexMul(sc_g, kv[shift_key + 2 * k * heads_kv * dimension]), (float2)0);
     }
   q_g[shift_q] = IsNaNOrInf2(ComplexDiv(grad, koef), (float2)0);
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float2 val = IsNaNOrInf2(kv[shift_k + heads_kv * dimension], (float2)0);
            for(int scr = 0; scr < qunits; scr++)
              {
               float2 sc_g = (float2)0;
               int shift_sc = scr * kunits * heads;
               float2 sc = (float2)(IsNaNOrInf(scores[shift_sc + k], 0), 0);
               if(ComplexAbs(sc) == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += IsNaNOrInf2(
                             ComplexMul(
                                ComplexMul((float2)(scores[shift_sc + v], 0),
                                           gradient[shift_g + scr * dimension]),
                                ComplexMul(val, ((float2)(k == v, 0) - sc))),
                             (float2)0);
               grad += IsNaNOrInf2(ComplexMul(sc_g, q[(h + scr * heads) * dimension + d]), (float2)0);
              }
           }
         kv_g[shift_k] = IsNaNOrInf2(ComplexDiv(grad, koef), (float2)0);
        }
     }
  }
//+------------------------------------------------------------------+
//| Runs the cslstm feed forward feed-forward pass on the GPU.       |
//+------------------------------------------------------------------+
__kernel void CSLSTM_FeedForward(__global const float4* __attribute__((aligned(16))) concatenated,
                                 __global float *memory,
                                 __global float *output)
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);       // hidden size
   uint idv = (uint)get_global_id(1);
   uint total_v = (uint)get_global_size(1);     // variables
//---
   uint shift = id + total * idv;
   float4 concat = concatenated[shift];
//---
   float fg_s = fActivation(concat.s0, ActFunc_SIGMOID);
   float fg = 1 - fActivation(1 - 1 / (fg_s * fg_s), ActFunc_TANH);
   float ig = fActivation(fActivation(concat.s1, ActFunc_SIGMOID), ActFunc_TANH);
   float nc = fActivation(concat.s2, ActFunc_TANH);
   float og = fActivation(concat.s3, ActFunc_SIGMOID);
   float mem = IsNaNOrInf(memory[shift] * fg + ig * nc, 0);
   float out = IsNaNOrInf(og * fActivation(mem, ActFunc_TANH), 0);
//---
   memory[shift] = mem;
   output[shift] = out;
  }
//+------------------------------------------------------------------+
//| Computes the cslstm calc hidden gradient gradient contributions  |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void CSLSTM_CalcHiddenGradient(__global const float4* __attribute__((aligned(16))) concatenated,    // Input from forward pass (W*x + U*h + b)
                                        __global float4* __attribute__((aligned(16))) grad_concat,           // Output: gradients w.r.t. gate pre-activations
                                        __global const float* memory,                                        // Updated memory (after forward pass)
                                        __global const float* grad_output                                    // dL/dOutput from the next layer
                                       )
  {
   uint id = get_global_id(0);              // Index within sequences
   uint total = get_global_size(0);         // Total size of sequences
   uint idv = get_global_id(1);             // Index over independent univariate sequences (e.g., features or channels) in a multivariate time series
   uint shift = id + total * idv;           // Flattened index
//---
   float4 concat = concatenated[shift];     // Pre-activation values for all 4 gates
// --- Forward reconstruction of gates ---
   float fg_s = fActivation(concat.s0, ActFunc_SIGMOID);
   float fg = 1.0f - fActivation(1.0f - 1.0f / (fg_s * fg_s), ActFunc_TANH);  // Forget gate (ft)
   float ig_s = fActivation(concat.s1, ActFunc_SIGMOID);
   float ig = fActivation(ig_s, ActFunc_TANH);            // Input gate (it)
   float nc = fActivation(concat.s2, ActFunc_TANH);         // Candidate (ct~)
   float og = fActivation(concat.s3, ActFunc_SIGMOID);      // Output gate (ot)
   float mem = memory[shift];                              // New memory state (ct)
   float mem_t = fActivation(mem, ActFunc_TANH);         // tanh(ct)
// --- Reconstruct previous memory state (t-1) ---
   float prev_mem = IsNaNOrInf((mem - ig * nc) / fg, 0);
// --- Gradients computation ---
   float out_g = grad_output[shift];
   float og_g = Deactivation(out_g * mem_t, og, ActFunc_SIGMOID);
   float mem_g = Deactivation(out_g * og, mem_t, ActFunc_TANH);
   float nc_g = Deactivation(mem_g * ig, nc, ActFunc_TANH);
   float ig_g = Deactivation(Deactivation(mem_g * nc, ig, ActFunc_TANH), ig_s, ActFunc_SIGMOID);
// ∂L/∂fg = ∂L/∂ct * mem_(t-1)
   float fg_g = mem_g * prev_mem;
// Derivative of the complex forget gate:
// f(z) = 1 - tanh(1 - 1 / σ(z)^2)
   float fg_s_g = 2 / (fg_s * fg_s * fg_s) * Deactivation(-fg_g, fg, ActFunc_TANH);
   fg_g = Deactivation(fg_s_g, fg_s, ActFunc_SIGMOID);
// --- Write back gradients ---
   grad_concat[shift] = (float4)(fg_g, ig_g, nc_g, og_g);
  }
//+------------------------------------------------------------------+
//| Implements the prob attention qeury imp kernel on the GPU.       |
//+------------------------------------------------------------------+
__kernel void ProbAttentionQeuryImp(__global const float* querys,
                                    __global const float2* __attribute__((aligned(8))) keys_values,
                                    __global const float* index_keys,
                                    __global float* querys_imp,
                                    const int dimension
                                   )
  {
   const size_t id_q = get_global_id(0);
   const size_t total_q = get_global_size(0);
   const size_t ind_k = get_local_id(1);
   const size_t total_ind = get_local_size(1);
   const size_t id_h = get_global_id(2);
   const size_t total_h = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE][2];
   const int ls = min((int)total_ind, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_q = dimension * (id_q * total_h + id_h);
   const int id_k = index_keys[ind_k * total_h + id_h];
   const int shift_k = dimension * (id_k * total_h + id_h);
//---
   float sum = 0;
//---
   for(int d = 0; d < dimension; d++)
      sum += IsNaNOrInf(querys[shift_q + d] * keys_values[shift_k + d].s0, 0);
//---
   int id_t = ind_k % ls;
//---
   for(int i = 0; i < total_ind; i += ls)
     {
      if(i <= ind_k || (i + ls) > ind_k)
        {
         temp[id_t][0] = IsNaNOrInf((i == 0 ? 0 : temp[id_t][0]) + sum, 0);
         temp[id_t][1] = (i == 0 ? IsNaNOrInf(sum, MIN_VALUE) : fmax(temp[id_t][1], IsNaNOrInf(sum, MIN_VALUE)));
         BarrierLoc
        }
     }
   int count = ls;
//---
   do
     {
      count = (count + 1) / 2;
      if(ind_k < count && (ind_k + count) < ls)
        {
         temp[ind_k][0] += temp[ind_k + count][0];
         temp[ind_k + count][0] = 0;
         temp[ind_k][1] = fmax(temp[ind_k + count][1], temp[ind_k][1]);
        }
      BarrierLoc
     }
   while(count > 1);
   if(ind_k == 0)
      querys_imp[id_q * total_h + id_h] = IsNaNOrInf(temp[0][1] - temp[0][0] / total_ind, MIN_VALUE);
  }
//+------------------------------------------------------------------+
//| Implements the top k importance to index kernel on the GPU.      |
//+------------------------------------------------------------------+
__kernel void TopKImportanceToIndex(__global const float* importance,
                                    __global float* indexes,
                                    const int top_k
                                   )
  {
   const size_t id_q = get_global_id(0);
   const size_t total_q = get_global_size(0);
   const size_t id_h = get_global_id(1);
   const size_t total_h = get_global_size(1);
//---
   float imp = importance[id_q * total_h + id_h];
   int pos = 0;
//---
   for(int i = 0; i < total_q; i++)
     {
      if(i == id_q)
         continue;
      float val = importance[i * total_h + id_h];
      if(val > imp || (i < id_q && val >= imp))
         pos++;
      if(pos >= top_k)
         break;
     }
//---
   if(pos < top_k)
      indexes[pos * total_h + id_h] = (float)id_q;
  }
//+------------------------------------------------------------------+
//| Implements the q index attention kernel on the GPU.              |
//+------------------------------------------------------------------+
__kernel void QIndexAttention(__global const float *q,     ///<[in] Matrix of Querys
                              __global const float2* kv,    ///<[in] Matrix of Keys
                              __global float *scores,       ///<[out] Matrix of Scores
                              __global const float *indexes,  ///<[in] Querys Indexes
                              __global float *out,         ///<[out] Matrix of attention
                              const int dimension,          ///< Dimension of Key
                              const int heads_kv
                             )
  {
//--- init
   const int ind_q = get_global_id(0);
   const int k = get_local_id(1);
   const int h = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_k = get_local_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int q_id = (int)(indexes[ind_q * heads + h] + 0.001f);
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads_kv * k + h_kv);
   const int shift_s = total_k * (ind_q *  heads + h) + k;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   float score = 0;
   if(q_id >= 0)
     {
      //---
      for(int d = 0; d < dimension; d++)
         score += IsNaNOrInf(q[shift_q + d] * kv[shift_kv + d].s0, 0);
     }
   else
      score = MIN_VALUE;
//--- norm score
   score = IsNaNOrInf(exp(score - LocalMax(score, 1, temp)), 0);
   score = IsNaNOrInf(score / LocalSum(score, 1, temp), 0);
   scores[shift_s] = score;
   BarrierLoc
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val = LocalSum(kv[shift_kv + d].s1 * score, 1, temp);
      if(k == 0)
         out[dimension * (ind_q * heads + h) + d] = temp[0];
      BarrierLoc
     }
  }
//+------------------------------------------------------------------+
//| Computes the q index attention gradients gradient contributions  |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void QIndexAttentionGradients(__global const float* q,
                                       __global float* q_g,
                                       __global const float2* kv,
                                       __global float2* kv_g,
                                       __global const float* indexes,
                                       __global const float* scores,
                                       __global const float* gradient,
                                       const int kunits, const int heads_kv
                                      )
  {
//--- init
   const int ind_q = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int q_id = (int)(indexes[ind_q * heads + h] + 0.001f);
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (ind_q * heads + h) * kunits;
   const int shift_g = h * dimension + d;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
     {
      //---
      for(int v = ind_q; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += IsNaNOrInf(gradient[shift_g + dimension * (hq - h + g  * heads)], 0) *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (heads_kv * v + h) + d;
         kv_g[shift_v].s1 = IsNaNOrInf(grad, 0);
        }
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = IsNaNOrInf(gradient[shift_g + ind_q * dimension], 0);
   int shift_kv = h_kv * dimension + d;
//---
   for(int k = 0; (k < kunits && out_g != 0); k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_kv + v * heads_kv * dimension].s1 *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_kv + k * heads_kv * dimension].s0;
     }
   q_g[shift_q] = grad;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      //---
      for(int k = ind_q; k < kunits; k += qunits)
        {
         int shift_k = dimension * (heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension].s1;
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += IsNaNOrInf(sc_g * q[(hq + (int)(indexes[scr * heads + hq] + 0.001f) * heads) * dimension + d], 0);
              }
           }
         kv_g[shift_k].s0 = IsNaNOrInf(grad, 0);
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the ts positon encoder kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void TSPositonEncoder(__global const float2* __attribute__((aligned(8))) data,
                               __global const float* time,
                               __global float2* __attribute__((aligned(8))) output,
                               __global const float* period
                              )
  {
   const int id = get_global_id(0);
   const int freq = get_global_id(1);
   const int p = get_global_id(2);
   const int total = get_global_size(0);
   const int freqs = get_global_size(1);
   const int periods = get_global_size(2);
//---
   const int shift = id * freqs + freq;
   const float2 d = data[shift * periods + p];
   const float t = time[id] / period[p];
   float val = M_PI_F * t * pow(2.0f, freq + 1);
//---
   output[shift * periods + p] = (float2)(d.s0 + sin(val), d.s1 + cos(val));
  }
//+------------------------------------------------------------------+
//| Runs the feed forward mult win conv w pad feed-forward pass on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void FeedForwardMultWinConvWPad(__global const float *matrix_w,
      __global const float *matrix_i,
      __global float *matrix_o,
      __global const int *windows_in,
      const int inputs,
      const int step,
      const int window_out,
      const int activation
                                        )
  {
   const size_t id = get_global_id(0);
   const size_t id_w = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t outputs = get_global_size(0);
   const size_t windows_total = get_global_size(1);
//---
   int window_in = windows_in[id_w];
   int mid_win = window_in / 2;
   int shift_in = id * step - mid_win;
   int shift_in_var = v * inputs;
   int shift_weight = 0;
//---
   for(int w = 0; w < id_w; w++)
      shift_weight += (windows_in[w] + 1) * window_out;
//---
   for(int w_out = 0; w_out < window_out; w_out++)
     {
      float sum = matrix_w[shift_weight + window_in];
      //---
      for(int w = 0; w < window_in; w++)
         if((shift_in + w) >= 0 && (shift_in + w) < inputs)
            sum += IsNaNOrInf(matrix_i[shift_in_var + shift_in + w] * matrix_w[shift_weight + w], 0);
      //---
      int shift_out = (v * outputs + id) * window_out + w_out;
      matrix_o[shift_out] = fActivation(sum, activation);
      shift_weight += window_in + 1;
     }
  }
//+------------------------------------------------------------------+
//| Computes the calc hidden gradient mult win conv w pad gradient   |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMultWinConvWPad(__global const float *matrix_w,
      __global const float *matrix_i,
      __global float *matrix_ig,
      __global const float *matrix_og,
      __global const int *windows_in,
      const int outputs,
      const int step,
      const int window_out,
      const int filters,
      const int activation
                                               )
  {
   const size_t id_x = get_global_id(0);
   const size_t loc = get_local_id(1);
   const size_t v = get_global_id(2);
   const size_t inputs = get_global_size(0);
   const size_t size_loc = get_local_size(1);
   const size_t windows_total = filters / window_out;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const uint ls = min((uint)size_loc, (uint)LOCAL_ARRAY_SIZE);
//---
   float grad = 0;
   for(int id_loc = loc; id_loc < filters; id_loc += size_loc)
     {
      const size_t id_win = id_loc / window_out;
      const size_t id_f = id_loc % window_out;
      int window_in = windows_in[id_win];
      int shift_weight = id_f * (window_in + 1);
      for(int w = 0; w < id_win; w++)
         shift_weight += (windows_in[w] + 1) * window_out;
      //---
      int shift_out = max((int)((id_x - window_in) / step), 0);
      //---
      int mid_win = (window_in + 1) / 2;
      for(int out = shift_out; out < outputs; out++)
        {
         int shift_in = out * step - mid_win;
         if(shift_in > id_x)
            break;
         int shift_w = id_x - shift_in;
         if(shift_w >= window_in)
            continue;
         int shift_g = ((v * outputs + out) * windows_total + id_win) * window_out + id_f;
         grad += IsNaNOrInf(matrix_w[shift_w + shift_weight] * matrix_og[shift_g], 0);
        }
     }
//---
   grad = LocalSum(grad, 1, temp);
//---
   if(loc == 0)
      matrix_ig[v * inputs + id_x] = Deactivation(grad, matrix_i[v * inputs + id_x], activation);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights mult win conv|
//| adam w pad step using the optimizer rule.                        |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMultWinConvAdamWPad(__global float *matrix_w,
      __global const float *matrix_og,
      __global const float *matrix_i,
      __global float *matrix_m,
      __global float *matrix_v,
      __global const int *windows_in,
      const int windows_total,
      const int window_out,
      const int inputs,
      const int step,
      const int outputs,
      const float l,
      const float b1,
      const float b2
                                              )
  {
   const size_t i = get_global_id(0);  // weight shift
   const size_t v = get_local_id(1);   // variable
   const size_t variables = get_local_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const uint ls = min((uint)variables, (uint)LOCAL_ARRAY_SIZE);
//---
   int step_out = window_out * windows_total;
//---
   int shift_before = 0;
   int window = 0;
   int number_w = 0;
   for(int w = 0; w < windows_total; w++)
     {
      int win = windows_in[w];
      if(shift_before <= i &&
         (win + 1)*window_out > (i - shift_before))
        {
         window = win;
         number_w = w;
        }
      else
         shift_before += (win + 1) * window_out;
     }
//---
   int shift_in = (i - shift_before) % (window + 1);
   int shift_in_var = v * inputs;
   bool bias = (shift_in == window);
   int mid_win = (window + 1) / 2;
   int id_f = (i - shift_before) / (window + 1);
   int shift_out = number_w * window_out + id_f;
   int shift_out_var = v * outputs * step_out;
//---
   float grad = 0;
   if(!bias)
     {
      for(int out = 0; out < outputs; out++)
        {
         int in = out * step - mid_win + shift_in;
         if(in >= inputs)
            break;
         if(in < 0)
            continue;
         //---
         grad += IsNaNOrInf(matrix_og[shift_out_var + shift_out + out * step_out] * matrix_i[shift_in_var + in], 0);
        }
     }
   else
     {
      for(int out = 0; out < outputs; out++)
         grad += IsNaNOrInf(matrix_og[shift_out_var + shift_out + out * step_out], 0);
     }
//--- sum
   for(int s = 0; s < (int)variables; s += ls)
     {
      if(v >= s && v < (s + ls))
         temp[v % ls] = (s == 0 ? 0 : temp[v % ls]) + grad;
      BarrierLoc
     }
//---
   uint count = ls;
   do
     {
      count = (count + 1) / 2;
      if(v < count && (v + count) < ls)
        {
         temp[v] += temp[v + count];
         temp[v + count] = 0;
        }
      BarrierLoc
     }
   while(count > 1);
//---
   if(v == 0)
     {
      grad = temp[0];
      float mt = IsNaNOrInf(clamp(b1 * matrix_m[i] + (1 - b1) * grad, -1.0e5f, 1.0e5f), 0);
      float vt = IsNaNOrInf(clamp(b2 * matrix_v[i] + (1 - b2) * (grad * grad), 1.0e-6f, 1.0e6f), 1.0e-6f);
      float weight = matrix_w[i] + IsNaNOrInf(l * mt / sqrt(vt), 0);
      matrix_w[i] = weight;
      matrix_m[i] = mt;
      matrix_v[i] = vt;
     }
  }
//+------------------------------------------------------------------+
//| Performs the concat diff data-layout operation on the GPU.       |
//+------------------------------------------------------------------+
__kernel void ConcatDiff(__global const float* data,
                         __global float* output,
                         const int step)
  {
   const size_t i = get_global_id(0);
   const size_t v = get_local_id(1);
   const size_t inputs = get_local_size(0);
   const size_t variables = get_local_size(1);
//---
   const int shift = i * variables;
   const float d = data[shift + v];
   float diff = 0;
   if(step > 0 && (i + step) < inputs)
      diff = IsNaNOrInf(d - data[shift + step * variables + v], 0);
//---
   output[2 * shift + v] = d;
   output[2 * shift + v + variables] = diff;
  }
//+------------------------------------------------------------------+
//| Runs the feed forward mask mult win conv feed-forward pass on the|
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void FeedForwardMaskMultWinConv(__global const float *matrix_w,
      __global const float *matrix_i,
      __global const float *masks,
      __global float *matrix_o,
      const int inputs,
      const int window_in,
      const int windows_total,
      const int activation
                                        )
  {
   const size_t u = get_global_id(0);
   const size_t w = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t window_out = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int shift_in = u * window_in * windows_total;
   const int shift_in_var =  v * units * window_in * windows_total;
   const int shift_out = (u + v * units) * window_out + w;
   const int shift_mask = (u + v * units) * windows_total;
   const int shift_weight = (v * window_out * windows_total + w) * (window_in + 1);
   const int step_weight = window_out * (window_in + 1);
//---
   float sum = 0;
   for(int w_in = 0; w_in < windows_total; w_in++)
     {
      float m = IsNaNOrInf(masks[shift_mask + w_in], 0);
      if(m < FLT_EPSILON)
         continue;
      const int shift_in_loc = shift_in + w_in * window_in;
      const int shift_weight_loc = shift_weight + w_in * step_weight;
      for(int i = 0; i < window_in; i++)
         if((shift_in_loc + i) < (inputs / variables))
            sum += IsNaNOrInf(matrix_i[shift_in_var + shift_in_loc + i], 0) *
                   matrix_w[shift_weight_loc + i] * m;
      sum += matrix_w[shift_weight_loc + window_in] * m;
     }
//---
   matrix_o[shift_out] = fActivation(sum, activation);
  }
//+------------------------------------------------------------------+
//| Computes the calc hidden gradient mask mult win conv gradient    |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMaskMultWinConv(__global const float *matrix_w,
      __global const float *matrix_i,
      __global float *matrix_ig,
      __global const float *matrix_og,
      __global const float *masks,
      __global float *masks_g,
      const int outputs,
      const int window_in,
      const int window_out,
      const int activation
                                               )
  {
   const size_t u = get_global_id(0);
   const size_t w_in = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t windows_total = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int shift_in = (u + v * units) * window_in * windows_total + w_in * window_in;
   const int shift_out = u * window_out;
   const int shift_out_var = v * units * window_out;
   const int shift_mask = (u + v * units) * windows_total + w_in;
   const int shift_weight = (v * window_out * windows_total + w_in * window_out) * (window_in + 1);
//---
   const float m = IsNaNOrInf(masks[shift_mask], 0);
   for(int i = 0; i < window_in; i++)
     {
      float sum = 0;
      if(m >= FLT_EPSILON)
        {
         for(int out = 0; out < window_out; out++)
           {
            if((shift_out + out) >= (outputs / variables))
               continue;
            sum += IsNaNOrInf(matrix_og[shift_out_var + shift_out + out] *
                              matrix_w[shift_weight + out * (window_in + 1) + i] *
                              m, 0);
           }
        }
      matrix_ig[shift_in + i] = Deactivation(sum, matrix_i[shift_in + i], activation);
     }
//---
   float sum = 0;
   for(int out = 0; out < window_out; out++)
     {
      int shift_weight_loc = out * (window_in + 1) + shift_weight;
      float temp = matrix_w[shift_weight_loc + window_in];
      for(int i = 0; i < window_in; i++)
         temp += IsNaNOrInf(matrix_i[shift_in + i], 0) * matrix_w[shift_weight_loc + i];
      sum += IsNaNOrInf(temp * matrix_og[shift_out_var + shift_out + out], 0);
     }
   masks_g[shift_mask] = IsNaNOrInf(sum, 0);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights mask mult win|
//| conv adam step using the optimizer rule.                         |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMaskMultWinConvAdam(__global float *matrix_w,
      __global const float *matrix_og,
      __global const float *matrix_i,
      __global const float *masks,
      __global float *matrix_m,
      __global float *matrix_v,
      const int windows_total,
      const int inputs,
      const int outputs,
      const float l,
      const float b1,
      const float b2
                                              )
  {
   const size_t id_in = get_global_id(0);    // input shift
   const size_t id_out = get_global_id(1);   // filter shift
   const size_t id_v = get_global_id(2);     // variable
   const size_t window_in = get_global_size(0) / windows_total - 1;
   const size_t window_out = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int w_id = id_in / (window_in + 1);
   const int shift_in = id_in - w_id;
   const int step_in = window_in * windows_total;
   const int units = outputs / window_out;
   const int shift_in_var = id_v * inputs;
   const int shift_out_var = id_v * outputs;
   const int shift_mask_var = id_v * units * windows_total;
   const int shift_weight = ((id_v * windows_total + w_id) * window_out + id_out) *
                            (window_in + 1) + id_in % (window_in + 1);
   const bool bias = (id_in % (window_in + 1) == window_in);
//---
   float grad = 0;
   for(int u = 0; u < units; u++)
     {
      const int shift_in_loc = shift_in + u * step_in;
      if(shift_in < inputs)
         continue;
      float m = IsNaNOrInf(masks[shift_mask_var + u * windows_total + w_id], 0);
      if(m < FLT_EPSILON)
         continue;
      float inp = (bias ? 1 : IsNaNOrInf(matrix_i[shift_in_var + shift_in_loc], 0));
      grad += IsNaNOrInf(inp * m * matrix_og[shift_out_var + u * window_out + id_out], 0);
     }
   float mt = IsNaNOrInf(clamp(b1 * matrix_m[shift_weight] + (1 - b1) * grad, -1.0e5f, 1.0e5f), 0);
   float vt = IsNaNOrInf(clamp(b2 * matrix_v[shift_weight] + (1 - b2) * (grad * grad), 1.0e-6f, 1.0e6f), 1.0e-6f);
   float weight = matrix_w[shift_weight] + IsNaNOrInf(l * mt / sqrt(vt), 0);
   matrix_w[shift_weight] = weight;
   matrix_m[shift_weight] = mt;
   matrix_v[shift_weight] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the main freq kernel on the GPU.                      |
//+------------------------------------------------------------------+
__kernel void MainFreq(__global const float* freq_r,
                       __global const float* freq_im,
                       __global float *main_freq,
                       int dimension
                      )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float max_f = 0;
   float max_id = 0;
   float energy;
//---
   for(int i = 1; i < dimension; i++)
     {
      float2 freq = (float2)(freq_r[shift + i], freq_im[shift + i]);
      energy = ComplexAbs(freq);
      if(max_f < energy)
        {
         max_f = energy;
         max_id = i + 1;
        }
     }
   main_freq[n] = max_id;
  }
//+------------------------------------------------------------------+
//| Runs the feed forward adapt conv feed-forward pass on the GPU.   |
//+------------------------------------------------------------------+
__kernel void FeedForwardAdaptConv(__global const float *matrix_w,
                                   __global const float *matrix_i,
                                   __global float *matrix_o,
                                   __global const float *main_freq,
                                   const int inputs,
                                   const int window_in,
                                   const int activation
                                  )
  {
   const size_t u = get_global_id(0);
   const size_t f = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t filters = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int freq = main_freq[v];
   int window = (inputs / variables + freq - 1) / freq;
   const int step = (int)(inputs / variables + units + 1) / (units + 2);
   if(window < step)
      window = (int)((step + window - 1) / window) * window;
   if(window > window_in)
      window = window_in;
//---
   const int shift_in = (u < (units - 1) ? u * step : inputs / variables - window);
   const int shift_in_var =  v * inputs / variables;
   const int shift_out = (u + v * units) * filters + f;
   const int shift_weight = (v * filters + f) * (window_in + 1);
//---
   float sum = matrix_w[shift_weight + window_in];
   for(int i = 0; i < window; i++)
      if((shift_in + i) < (inputs / variables))
         sum += IsNaNOrInf(matrix_i[shift_in_var + shift_in + i], 0) *
                matrix_w[shift_weight + i];
//---
   matrix_o[shift_out] = fActivation(sum, activation);
  }
//+------------------------------------------------------------------+
//| Computes the calc hidden gradient adapt conv gradient            |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientAdaptConv(__global const float *matrix_w,
      __global const float *matrix_i,
      __global float *matrix_ig,
      __global const float *matrix_og,
      __global const float *main_freq,
      const int outputs,
      const int window_in,
      const int window_out,
      const int activation
                                         )
  {
   const size_t inp = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t inputs = get_global_size(0);
   const size_t variables = get_global_size(1);
//---
   const int units = outputs / (window_out * variables);
   const int freq = main_freq[v];
   int window = (inputs / variables + freq - 1) / freq;
   const int step = (int)(inputs + units + 1) / (units + 2);
   if(window < step)
      window = (int)((step + window - 1) / window) * window;
   if(window > window_in)
      window = window_in;
//---
   const int shift_in = v * inputs + inp;
   int u = inp / step;
   int shift_out_var = v * (outputs / variables);
   int shift_weight_var = (v * window_out) * (window_in + 1);
//---
   float sum = 0;
   while(u * step <= inp && u < (units - 1))
     {
      int pos = inp - u * step;
      if(pos >= window)
        {
         u++;
         continue;
        }
      int shift_out = u * window_out;
      int shift_weight = pos + shift_weight_var;
      for(int out = 0; out < window_out; out++)
        {
         if((shift_out + out) >= (outputs / variables))
            continue;
         sum += IsNaNOrInf(matrix_og[shift_out_var + shift_out + out] *
                           matrix_w[shift_weight + out * (window_in + 1)], 0);
        }
      u++;
     }
   if(inp >= (inputs - window))
     {
      int pos = inp + window - inputs;
      int shift_out = (units - 1) * window_out;
      int shift_weight = pos + shift_weight_var;
      for(int out = 0; out < window_out; out++)
        {
         if((shift_out + out) >= (outputs / variables))
            continue;
         sum += IsNaNOrInf(matrix_og[shift_out_var + shift_out + out] *
                           matrix_w[shift_weight + out * (window_in + 1)], 0);
        }
     }
   matrix_ig[shift_in] = Deactivation(sum, matrix_i[shift_in], activation);
  }
//+------------------------------------------------------------------+
//| Updates the trainable weights of the update weights adapt conv   |
//| adam step using the optimizer rule.                              |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdaptConvAdam(__global float *matrix_w,
      __global const float *matrix_og,
      __global const float *matrix_i,
      __global float *matrix_m,
      __global float *matrix_v,
      __global float *main_freq,
      const int inputs,
      const int outputs,
      const float l,
      const float b1,
      const float b2
                                        )
  {
   const size_t id_in = get_global_id(0);    // input shift
   const size_t id_out = get_global_id(1);   // filter shift
   const size_t id_v = get_global_id(2);     // variable
   const size_t window_in = get_global_size(0) - 1;
   const size_t window_out = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int units = outputs / (window_out * variables);
   const int freq = main_freq[id_v];
   int window = (inputs / variables + freq - 1) / freq;
   const int step = (int)(inputs / variables + units + 1) / (units + 2);
   if(window < step)
      window = (int)((step + window - 1) / window) * window;
   if(window > window_in)
      window = window_in;
//---
   if(id_in != window_in &&
      id_in >= window)
      return;
//---
   const int shift_in_var = id_v * inputs / variables;
   const int shift_out_var = id_v * outputs / variables;
   const int shift_weight = (id_v * window_out + id_out) *
                            (window_in + 1) + id_in;
   const bool bias = (id_in == window_in);
//---
   float grad = 0;
   for(int u = 0; u < (units - 1); u++)
     {
      const int shift_in_loc = id_in + u * step;
      if(shift_in_loc >= (inputs / variables))
         continue;
      float inp = (bias ? 1 : IsNaNOrInf(matrix_i[shift_in_var + shift_in_loc], 0));
      grad += IsNaNOrInf(inp * matrix_og[shift_out_var + u * window_out + id_out], 0);
     }
     {
      const int shift_in_loc = id_in + inputs / variables - window;
      if(shift_in_loc < (inputs / variables))
        {
         float inp = (bias ? 1 : IsNaNOrInf(matrix_i[shift_in_var + shift_in_loc], 0));
         grad += IsNaNOrInf(inp * matrix_og[shift_out_var + (units - 1) * window_out + id_out], 0);
        }
     }
   float mt = IsNaNOrInf(clamp(b1 * matrix_m[shift_weight] + (1 - b1) * grad, -1.0e5f, 1.0e5f), 0);
   float vt = IsNaNOrInf(clamp(b2 * matrix_v[shift_weight] + (1 - b2) * (grad * grad), 1.0e-6f, 1.0e6f), 1.0e-6f);
   float weight = matrix_w[shift_weight] + IsNaNOrInf(l * mt / sqrt(vt), 0);
   matrix_w[shift_weight] = weight;
   matrix_m[shift_weight] = mt;
   matrix_v[shift_weight] = vt;
  }
//+------------------------------------------------------------------+
//| Implements the ro pe kernel on the GPU.                          |
//+------------------------------------------------------------------+
__kernel void RoPE(__global const float2* __attribute__((aligned(8))) inputs,
                   __global const float2* __attribute__((aligned(8))) position_emb,
                   __global float2* __attribute__((aligned(8))) outputs
                  )
  {
   const size_t id_d = get_global_id(0);     // dimension
   const size_t id_u = get_global_id(1);     // unit
   const size_t id_v = get_global_id(2);     // variable
   const size_t dimension = get_global_size(0);
   const size_t units = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int shift_in = (id_v * units + id_u) * dimension + id_d;
   const int shift_pos = id_u * dimension + id_d;
   const float2 inp = inputs[shift_in];
   const float2 pe = position_emb[shift_pos];
//---
   float2 result = 0;
   result.s0 = inp.s0 * pe.s0 - inp.s1 * pe.s1;
   result.s1 = inp.s0 * pe.s1 + inp.s1 * pe.s0;
//---
   outputs[shift_in] = result;
  }
//+------------------------------------------------------------------+
//| Calculates calc hidden grad ro pe on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradRoPE(__global float2* __attribute__((aligned(8))) inputs_gr,
                                 __global const float2* __attribute__((aligned(8))) position_emb,
                                 __global const float2* __attribute__((aligned(8))) outputs_gr
                                )
  {
   const size_t id_d = get_global_id(0);     // dimension
   const size_t id_u = get_global_id(1);     // unit
   const size_t id_v = get_global_id(2);     // variable
   const size_t dimension = get_global_size(0);
   const size_t units = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const int shift_in = (id_v * units + id_u) * dimension + id_d;
   const int shift_pos = id_u * dimension + id_d;
   const float2 grad = outputs_gr[shift_in];
   const float2 pe = position_emb[shift_pos];
//---
   float2 grad_x;
   grad_x.s0 = grad.s0 * pe.s0 + grad.s1 * pe.s1;
   grad_x.s1 = grad.s1 * pe.s0 - grad.s0 * pe.s1;
//---
   inputs_gr[shift_in] = grad_x;
  }
//+------------------------------------------------------------------+
//| Implements the dif matrix kernel on the GPU.                     |
//+------------------------------------------------------------------+
__kernel void DifMatrix(__global const float *matrix1,    ///<[in] First matrix
                        __global const float *matrix2,    ///<[in] Second matrix
                        __global float *matrix_out, ///<[out] Output matrix
                        const float multiplyer,           ///< Multiplyer for output
                        const int shift_in1,              ///< Shift for input 1
                        const int shift_in2,              ///< Shift for input 2
                        const int shift_out               ///< Shift for output
                       )
  {
   const int i = get_global_id(0);
   const int d = get_global_id(1);
   const int step = get_global_size(0);
   const int dimension = get_global_size(1);
//---
   int index = i * dimension + d;
   matrix_out[i * shift_out + index] =
      IsNaNOrInf((matrix1[i * shift_in1 + index] - matrix2[i * shift_in2 + index]) * multiplyer, 0);
  }
//+------------------------------------------------------------------+
//| Computes the dif matrix grad gradient contributions on the GPU.  |
//+------------------------------------------------------------------+
__kernel void DifMatrixGrad(__global float *matrix1,    ///<[in] First matrix
                            __global float *matrix2,    ///<[in] Second matrix
                            __global const float *matrix_out, ///<[out] Output matrix
                            const float multiplyer,           ///< Multiplyer for output
                            const int shift_in1,              ///< Shift for input 1
                            const int shift_in2,              ///< Shift for input 2
                             const int shift_out               ///< Shift for output
                           )
  {
   const int i = get_global_id(0);
   const int d = get_global_id(1);
   const int step = get_global_size(0);
   const int dimension = get_global_size(1);
//---
   int index = i * dimension + d;
   float grad = IsNaNOrInf(matrix_out[i * shift_out + index] * multiplyer, 0);
   matrix1[i * shift_in1 + index] = grad;
   matrix2[i * shift_in2 + index] = -grad;
  }
//+------------------------------------------------------------------+
//| Implements the identity sum matrix kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void IdentitySumMatrix(__global const float *matrix_in,
                                __global float *matrix_out,
                                const float multiplyer,
                                const int shift_in,
                                const int shift_out
                               )
  {
   const int i = get_global_id(0);
   const int d = get_global_id(1);
   const int step = get_global_size(0);
   const int dimension = get_global_size(1);
//---
   int index = i * dimension + d;
   matrix_out[i * shift_out + index] =
      IsNaNOrInf(((int)(i == d) + matrix_in[i * shift_in + index]) * multiplyer, 0);
  }
//+------------------------------------------------------------------+
//| Implements the identity dif matrix kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void IdentityDifMatrix(__global const float *matrix_in,
                                __global float *matrix_out,
                                const float multiplyer,
                                const int shift_in,
                                const int shift_out
                               )
  {
   const int i = get_global_id(0);
   const int d = get_global_id(1);
   const int step = get_global_size(0);
   const int dimension = get_global_size(1);
//---
   int index = i * dimension + d;
   matrix_out[i * shift_out + index] =
      IsNaNOrInf(((int)(i == d) - matrix_in[i * shift_in + index]) * multiplyer, 0);
  }
//+------------------------------------------------------------------+
//| Computes the identity dif matrix grad gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void IdentityDifMatrixGrad(__global float *matrix_in,
                                    __global const float *matrix_out,
                                    const float multiplyer,
                                    const int shift_in,
                                    const int shift_out
                                   )
  {
   const int i = get_global_id(0);
   const int d = get_global_id(1);
   const int step = get_global_size(0);
   const int dimension = get_global_size(1);
//---
   int index = i * dimension + d;
   matrix_in[i * shift_in + index] = IsNaNOrInf(-multiplyer * matrix_out[i * shift_out + index], 0);
  }
//+------------------------------------------------------------------+
//| Implements the sum vec matrix kernel on the GPU.                 |
//+------------------------------------------------------------------+
__kernel void SumVecMatrix(__global const float *vector_in,
                           __global const float *matrix_in,
                           __global float *matrix_out,
                           const float multiplyer,           ///< Multiplyer for output
                           const int shift_in1,              ///< Shift for input 1
                           const int shift_in2,              ///< Shift for input 2
                            const int shift_out,              ///< Shift for output
                            __global float *numeric_flags,
                            const int flags_enabled
                          )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int v = get_global_id(2);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
   const int variables = get_global_size(2);
//---
   int flat_in = shift_in2 + RCtoFlat(r, c, rows, cols, v);
   int flat_out = shift_out + RCtoFlat(r, c, rows, cols, v);
   int flat_v = shift_in1 + RCtoFlat(0, c, 1, cols, v);
     if(flags_enabled)
      {
       const float vector_value = vector_in[flat_v];
       const float matrix_value = matrix_in[flat_in];
       const float raw_sum = vector_value + matrix_value;
       const float raw_result = raw_sum * multiplyer;
       const int invalid = isnan(vector_value) || isinf(vector_value) || isnan(matrix_value) || isinf(matrix_value) ||
                           isnan(raw_sum) || isinf(raw_sum) || isnan(raw_result) || isinf(raw_result);
       numeric_flags[RCtoFlat(r, c, rows, cols, v)] = (invalid ? 1.0f : 0.0f);
      }
    matrix_out[flat_out] = IsNaNOrInf((vector_in[flat_v] + matrix_in[flat_in]) * multiplyer, 0);
  }
//+------------------------------------------------------------------+
//| Computes the sum vec matrix grad gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SumVecMatrixGrad(__global float *vector_in,
                               __global float *matrix_in,
                               __global const float *matrix_out,
                               const float multiplyer,           ///< Multiplyer for output
                               const int shift_in1,              ///< Shift for input 1
                               const int shift_in2,              ///< Shift for input 2
                               const int shift_out               ///< Shift for output
                              )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int v = get_global_id(2);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
   const int variables = get_global_size(2);
//---
   int flat_in = shift_in2 + RCtoFlat(r, c, rows, cols, v);
   int flat_out = shift_out + RCtoFlat(r, c, rows, cols, v);
   int flat_v = shift_in1 + RCtoFlat(0, c, 1, cols, v);
//---
   float grad = IsNaNOrInf(matrix_out[flat_out] * multiplyer, 0);
   matrix_in[flat_in] = grad;
//---
   if(r == 0)
     {
      for(int i = 1; i < rows; i++)
        {
         flat_out += cols;
         grad += IsNaNOrInf(matrix_out[flat_out] * multiplyer, 0);
        }
      vector_in[flat_v] = IsNaNOrInf(grad / rows, 0);
     }
  }
//+------------------------------------------------------------------+
//| Implements the interpolation attention kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void InterpolationAttention(__global const float* matrix_in,
                                     __global const float* W,
                                     __global const float* A,
                                     __global const float* GL,
                                     __global float* Adj,
                                     __global float* H,
                                     __global float* Atten,
                                     __global float* matrix_out,
                                     const int dimension
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t j = get_local_id(1);
   const size_t total = get_global_size(0);
   const size_t total_loc = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift_i = i * dimension;
   const int shift_j = j * dimension;
   const int shift_adj = i * total_loc + j;
//---
   float adj = 0;
   for(int d = 0; d < dimension; d++)
      adj += IsNaNOrInf(GL[shift_i + d] * GL[shift_j + d], 0);
   adj = max(IsNaNOrInf(adj, 0), 0.0f);
   adj = LocalSoftMax(adj, 1, Temp);
   Adj[shift_adj] = adj;
   adj += (float)(i == j);
//---
   for(int id_h = 0; id_h < dimension; id_h += total_loc)
     {
      if(j >= (dimension - id_h))
         break;
      float h = 0;
      for(int w = 0; w < dimension; w++)
         h += IsNaNOrInf(matrix_in[shift_i + w] * W[(id_h + j) * dimension + w], 0);
      H[shift_i + id_h + j] = h;
      BarrierLoc
     }
   float e = 1e-12f;
   if(adj > 0)
     {
      e = 0;
      for(int a = 0; a < dimension; a++)
         e += IsNaNOrInf(H[shift_i + a] * A[a], 0) + IsNaNOrInf(H[shift_j + a] * A[dimension + a], 0);
     }
   e = LocalSoftMax(e, 1, Temp);
   Atten[shift_adj] = e;
//--- Scale output by attention
   for(int d = 0; d < dimension; d += total_loc)
     {
      if(j >= (dimension - d))
         break;
      float out = 0;
      int shift_h = d + j;
      int shift_att = i * total_loc;
      int shift_out = i * dimension + shift_h;
      for(int n = 0; n < total_loc; n++)
         out += IsNaNOrInf(H[shift_h + n * dimension] * Atten[shift_att + n], 0);
      matrix_out[shift_out] = fActivation(out, ActFunc_LReLU);
     }
  }
//+------------------------------------------------------------------+
//| Computes the interpolation attention grad gradient contributions |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void InterpolationAttentionGrad(__global const float* matrix_in,
      __global float* matrix_in_gr,
      __global const float* W,
      __global float* W_gr,
      __global const float* A,
      __global float* A_gr,
      __global const float* GL,
      __global float* GL_gr,
      __global float* Adj,
      __global float* H,
      __global float* H_gr,
      __global float* Atten,
      __global float* matrix_out_gr,
      const int dimension
                                        )
  {
   const size_t i = get_global_id(0);
   const size_t j = get_local_id(1);
   const size_t total = get_global_size(0);
   const size_t total_loc = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift_i = i * dimension;
   const int shift_j = j * dimension;
   const int shift_adj = i * total_loc + j;
//--- H Gradient
   for(int d = 0; d < dimension; d += total_loc)
     {
      if(j >= (dimension - d))
         break;
      float h_grad = 0;
      int shift_h = shift_i + d + j;
      int shift_att = i;
      int shift_out = d + j;
      for(int n = 0; n < total_loc; n++)
        {
         float gr = matrix_out_gr[shift_out + n * dimension];
         h_grad += IsNaNOrInf(
                      Deactivation(gr, gr, ActFunc_LReLU) * Atten[shift_att + n * total_loc], 0);
        }
      H_gr[shift_h] = h_grad;
      BarrierLoc
     }
//--- Attention Gradient
   float att_grad = 0;
   for(int d = 0; d < dimension; d++)
     {
      float gr = matrix_out_gr[shift_i + d];
      gr = Deactivation(gr, gr, ActFunc_LReLU);
      att_grad += IsNaNOrInf(gr * H[shift_j + d], 0);
     }
   att_grad = LocalSoftMaxGrad(Atten[shift_adj], att_grad, 1, Temp);
//--- Add H Gradient
   for(int d = 0; d < dimension; d++)
     {
      float h_grad = att_grad * A[d];
      h_grad = LocalSum(h_grad, 1, Temp);
      if(j == 0)
         H_gr[shift_i + d] += h_grad;
      h_grad = att_grad * A[dimension + d];
      h_grad = LocalSum(h_grad, 1, Temp);
      if(j == 0)
         H_gr[shift_j + d] += h_grad;
      float a_grad = att_grad * H[shift_i + d];
      a_grad = LocalSum(a_grad, 1, Temp);
      A_gr[d] += a_grad;
      a_grad = att_grad * H[shift_j + d];
      a_grad = LocalSum(a_grad, 1, Temp);
      A_gr[dimension + d] += a_grad;
     }
//--- Inputs' Gradient
   for(int d = 0; d < dimension; d += total_loc)
     {
      if(j >= (dimension + d))
         break;
      float grad = 0;
      for(int w = 0; w < dimension; w++)
         grad += IsNaNOrInf(H_gr[shift_i + w] * W[(d + j) + dimension * w], 0);
      matrix_in_gr[shift_i + d + j] = grad;
      BarrierLoc
     }
//--- Adj Gradient
   float grad = LocalSoftMaxGrad(Adj[shift_adj], att_grad, 1, Temp);
   for(int d = 0; d < dimension; d++)
     {
      GL_gr[shift_i + d] += IsNaNOrInf(grad * GL[shift_j + d], 0);
      GL_gr[shift_j + d] += IsNaNOrInf(grad * GL[shift_i + d], 0);
     }
  }
//+------------------------------------------------------------------+
//| Implements the period norm kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void PeriodNorm(__global const float* inputs,
                         __global float2* mean_stdevs,
                         __global float* outputs,
                         const int total_inputs
                        )
  {
   const size_t i = get_global_id(0);
   const size_t p = get_local_id(1);
   const size_t v = get_global_id(2);
   const size_t windows = get_global_size(0);
   const size_t period = get_local_size(1);
   const size_t variable = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift_i = i * period + p;
   const int shift_v = v * total_inputs;
   const int shift_ms = v * windows + i;
//---
   float val = 0;
   if((shift_i) < total_inputs)
      val = IsNaNOrInf(inputs[shift_v + shift_i], 0);
   float mean = IsNaNOrInf(LocalSum(val, 1, Temp) / period, 0);
   val -= mean;
   BarrierLoc
   float stdev = LocalSum(val * val, 1, Temp) / period;
   stdev = IsNaNOrInf(sqrt(stdev), 1);
//---
   mean_stdevs[shift_ms] = (float2)(mean, stdev);
   if((shift_i) < total_inputs)
      outputs[shift_v + shift_i] = IsNaNOrInf(val / stdev, 0);
  }
//+------------------------------------------------------------------+
//| Computes the period norm grad gradient contributions on the GPU. |
//+------------------------------------------------------------------+
__kernel void PeriodNormGrad(__global const float* inputs,
                             __global float* inputs_gr,
                             __global const float2* mean_stdevs,
                             __global const float2* mean_stdevs_gr,
                             __global const float* outputs,
                             __global const float* outputs_gr,
                             const int total_inputs
                            )
  {
   const size_t i = get_global_id(0);
   const size_t p = get_local_id(1);
   const size_t v = get_global_id(2);
   const size_t windows = get_global_size(0);
   const size_t period = get_local_size(1);
   const size_t variable = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift_i = i * period + p;
   const int shift_v = v * total_inputs;
   const int shift_ms = v * windows + i;
//---
   float inp = 0;
   float inp_gr = 0;
   float out = 0;
   float out_gr = 0;
   const float2 mean_stdev = mean_stdevs[shift_ms];
   const float2 mean_stdev_gr = mean_stdevs_gr[shift_ms];
   if((shift_i) < total_inputs)
     {
      inp = IsNaNOrInf(inputs[shift_v + shift_i], 0);
      out = IsNaNOrInf(outputs[shift_v + shift_i], 0);
      out_gr = IsNaNOrInf(outputs_gr[shift_v + shift_i], 0);
     }
   float mean_gr = LocalSum(out_gr, 1, Temp) / period + IsNaNOrInf(mean_stdev.x, 0);
   BarrierLoc
   float stdev_gr = out * LocalSum(IsNaNOrInf(out * out_gr, 0), 1, Temp) / period + IsNaNOrInf(mean_stdev.y, 0);
   inp_gr = (out_gr - mean_gr - stdev_gr) / IsNaNOrInf(mean_stdev.y, 1);
//---
   if((shift_i) < total_inputs)
      inputs_gr[shift_v + shift_i] = IsNaNOrInf(inp_gr, 0);
  }
//+------------------------------------------------------------------+
//| Implements the adapt spatial norm kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void AdaptSpatialNorm(__global const float* inputs,
                               __global const float* attention,
                               __global float2* mean_stdevs,
                               __global float* outputs
                              )
  {
   const size_t i = get_global_id(0);
   const size_t a = get_local_id(1);
   const size_t v = get_global_id(2);
   const size_t total_inputs = get_global_size(0);
   const size_t total_local = get_local_size(1);
   const size_t variables = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift_v = v * total_inputs;
   const int shift_out = shift_v + i;
//---
   float mean = 0, stdev = 0;
   for(uint l = 0; l < variables; l += total_local)
     {
      const int shift_at = v * variables + (a + l);
      float val = IsNaNOrInf(inputs[(a + l) * total_inputs + i], 0);
      float att = IsNaNOrInf(attention[shift_at], 0);
      mean += val * att;
      stdev += val * val * att;
     }
   mean = LocalSum(mean, 1, Temp);
   BarrierLoc
   stdev = LocalSum(stdev, 1, Temp);
//---
   if(a == 0)
     {
      stdev -= mean * mean;
      stdev = IsNaNOrInf(sqrt(stdev), 1);
      if(stdev <= 0)
         stdev = 1;
      mean_stdevs[shift_out] = (float2)(mean, stdev);
      outputs[shift_out] = IsNaNOrInf((inputs[shift_out] - mean) / stdev, 0);
     }
  }
//+------------------------------------------------------------------+
//| Computes the adapt spatial norm grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void AdaptSpatialNormGrad(__global const float* inputs,
                                   __global float* inputs_gr,
                                   __global const float* attention,
                                   __global float* attention_gr,
                                   __global const float2* mean_stdevs,
                                   __global const float2* mean_stdevs_gr,
                                   __global const float* outputs_gr,
                                   const uint total_inputs
                                  )
  {
   const size_t i = get_global_id(0);              // main
   const size_t loc = get_local_id(1);             // local to sum
   const size_t v = get_global_id(2);              // variable
   const size_t total_main = get_global_size(0);   // total
   const size_t total_loc = get_local_size(1);     // local dimension
   const size_t variables = get_global_size(2);    // total variables
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//--- Inputs gradient
     {
      int shift_in = v * total_inputs + i;
      float grad = 0;
      if(i < total_inputs)
        {
         float x = IsNaNOrInf(inputs[shift_in], 0);
         for(int l = 0; l < variables; l += total_loc)
           {
            if((l + loc) >= variables)
               break;
            int shift_out = i + (l + loc) * total_inputs;
            float att = IsNaNOrInf(attention[(l + loc) * variables + v], 0);
            float out_gr = IsNaNOrInf(outputs_gr[shift_out], 0);
            float2 ms = mean_stdevs[shift_out];
            float2 ms_gr = mean_stdevs_gr[shift_out];
            float dy = (1 - att) * (1 / ms.y - (x - ms.x) * att * x / (ms.y * ms.y * ms.y));
            float dmean = IsNaNOrInf(ms_gr.x * att, 0);
            float dstd = IsNaNOrInf(ms_gr.y * x * (att - att * att) / ms.y, 0);
            grad += IsNaNOrInf(dy * out_gr + dmean + dstd, 0);
           }
        }
      grad = LocalSum(grad, 1, Temp);
      if(loc == 0 && i < total_inputs)
         inputs_gr[shift_in] = grad;
      BarrierLoc
     }
//--- Attention gradient
     {
      int shift_att = v * variables + i;
      float grad = 0;
      if(i < variables)
        {
         float att = IsNaNOrInf(attention[shift_att], 0);
         for(int l = 0; l < total_inputs; l += total_loc)
           {
            if((l + loc) >= total_inputs)
               break;
            int shift_out = (l + loc) + v * total_inputs;
            int shift_in = (l + loc) + i * total_inputs;
            float x = IsNaNOrInf(inputs[shift_in], 0);
            float out_gr = IsNaNOrInf(outputs_gr[shift_out], 0);
            float2 ms = mean_stdevs[shift_out];
            float2 ms_gr = mean_stdevs_gr[shift_out];
            float dy = -x / ms.y - (x - ms.x) * x * x * (1 - 2 * att) / (2 * ms.y * ms.y * ms.y);
            float dmean = IsNaNOrInf(ms_gr.x * x, 0);
            float dstd = IsNaNOrInf(ms_gr.y * x * x * (1 - 2 * att) / (2 * ms.y), 0);
            grad += IsNaNOrInf(dy * out_gr + dmean + dstd, 0);
           }
        }
      grad = LocalSum(grad, 1, Temp);
      if(loc == 0 && i < variables)
         attention_gr[shift_att] = grad;
     }
  }
//+------------------------------------------------------------------+
//| Implements the attent norm kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void AttentNorm(__global const float* inputs,
                         __global const float* attention,
                         __global float* means,
                         __global float* stdevs,
                         __global float* outputs,
                         const int total_inputs,
                         const int segment_size
                        )
  {
   const size_t s = get_global_id(0);
   const size_t i = get_local_id(1);
   const size_t v = get_global_id(2);
   const size_t total_segments = get_global_size(0);
   const size_t total_local = get_local_size(1);
   const size_t variables = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int shift = v * total_inputs + s * segment_size + i;
//---
   float mean = 0, stdev = 0;
   float val = 0;
   for(uint l = 0; l < segment_size; l += total_local)
     {
      if((l + i) >= segment_size ||
         (s * segment_size + l + i) >= total_inputs)
         break;
      float val_l = IsNaNOrInf(inputs[shift + l], 0);
      if(l == 0)
         val = val_l;
      float att = IsNaNOrInf(attention[v * segment_size + l + i], 0);
      mean += val_l * att;
      stdev += val_l * val_l * att;
     }
   mean = LocalSum(mean, 1, Temp);
   BarrierLoc
   stdev = LocalSum(stdev, 1, Temp);
//---
   stdev -= mean * mean;
   stdev = IsNaNOrInf(sqrt(stdev), 1);
   if(stdev <= 0)
      stdev = 1;
//---
   if(i == 0)
     {
      int shift_ms = v * total_segments + s;
      means[shift_ms] = mean;
      stdevs[shift_ms] = stdev;
     }
   for(uint l = 0; l < segment_size; l += total_local)
     {
      if((l + i) >= segment_size ||
         (s * segment_size + l + i) >= total_inputs)
         break;
      if(l > 0)
         val = inputs[shift + l];
      outputs[shift + l] = IsNaNOrInf((val - mean) / stdev, 0);
     }
  }
//+------------------------------------------------------------------+
//| Computes the attent norm grad gradient contributions on the GPU. |
//+------------------------------------------------------------------+
__kernel void AttentNormGrad(__global const float* inputs,
                             __global float* inputs_gr,
                             __global const float* attention,
                             __global float* attention_gr,
                             __global const float* means,
                             __global const float* stdevs,
                             __global const float* means_gr,
                             __global const float* outputs_gr,
                             const int total_inputs,
                             const int segment_size
                            )
  {
   const size_t i = get_global_id(0);              // main
   const size_t loc = get_local_id(1);             // local to sum
   const size_t v = get_global_id(2);              // variable
   const size_t total_main = get_global_size(0);   // total
   const size_t total_loc = get_local_size(1);     // local dimension
   const size_t variables = get_global_size(2);    // total variables
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//--- Inputs gradient
     {
      const int s = i / segment_size;
      const int shift_in = v * total_inputs + i;
      const int shift_ms = v * segment_size + s;
      float grad = 0;
      if(loc == 0 && i < total_inputs)
        {
         Temp[0] = IsNaNOrInf(inputs[shift_in], 0);
         Temp[1] = IsNaNOrInf(means[shift_ms], 0);
         Temp[2] = IsNaNOrInf(stdevs[shift_ms], 1);
         Temp[3] = IsNaNOrInf(means_gr[shift_ms], 0);
         Temp[4] = IsNaNOrInf(attention[(v - s) * segment_size + i], 0);
        }
      BarrierLoc
      if(i < total_inputs)
        {
         float x = Temp[0];
         float mean = Temp[1];
         float stdev = Temp[2];
         float mean_gr = Temp[3];
         float att = Temp[4];
         for(int l = 0; l < segment_size; l += total_loc)
           {
            if((l + loc) >= segment_size ||
               (i * segment_size + loc + l) >= total_inputs)
               break;
            float out_gr = IsNaNOrInf(outputs_gr[v * total_inputs + s * segment_size + loc + l], 0);
            bool same = (i - s * segment_size) == (loc + l);
            float xl = x;
            if(!same)
               xl = IsNaNOrInf(inputs[v * total_inputs + s * segment_size + loc + l], 0);
            float dy = ((int)same - att) * (1 / stdev - (xl - mean) * att * x / (stdev * stdev * stdev));
            float dmean = (same ? IsNaNOrInf(mean_gr * att, 0) : 0);
            grad += IsNaNOrInf(dy * out_gr + dmean, 0);
           }
        }
      grad = LocalSum(grad, 1, Temp);
      if(loc == 0 && i < total_inputs)
         inputs_gr[shift_in] = grad;
      BarrierLoc
     }
//--- Attention gradient
     {
      float grad = 0;
      int shift_att = v * segment_size + i;
      if(i < segment_size)
        {
         float att = IsNaNOrInf(attention[shift_att], 0);
         for(int l = 0; l < total_inputs; l += total_loc)
           {
            if((l + loc) >= total_inputs)
               break;
            int shift_out = (l + loc) + v * total_inputs;
            int s = (l + loc) / segment_size;
            int shift_in = v * total_inputs + s * segment_size + i;
            float x = IsNaNOrInf(inputs[shift_in], 0);
            float out_gr = IsNaNOrInf(outputs_gr[shift_out], 0);
            float mean = means[v * segment_size + s];
            float stdev = stdevs[v * segment_size + s];
            float mean_gr = means_gr[v * segment_size + s];
            bool same = (i - s * segment_size) == (loc + l);
            float xl = x;
            if(!same)
               xl = IsNaNOrInf(inputs[shift_out], 0);
            float dy = -x / stdev - (xl - mean) * x * x * (1 - 2 * att) / (2 * stdev * stdev * stdev);
            float dmean = IsNaNOrInf(mean_gr * x, 0);
            grad += IsNaNOrInf(dy * out_gr + dmean, 0);
           }
        }
      grad = LocalSum(grad, 1, Temp);
      if(loc == 0 && i < segment_size)
         attention_gr[shift_att] = grad;
     }
  }
//+------------------------------------------------------------------+
//| Implements the cheb step kernel on the GPU.                      |
//+------------------------------------------------------------------+
__kernel void ChebStep(__global const float* support,
                       __global float* outputs,
                       const int step
                      )
  {
   const size_t l = get_local_id(0);
   const size_t r = get_global_id(1);
   const size_t c = get_global_id(2);
   const size_t total_l = get_local_size(0);
   const size_t total_r = get_global_size(1);
   const size_t total_c = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   if(step <= 0 || total_r != total_c)
      return;
//---
   if(step <= 3)
     {
      const float diag = (r == c ? 1.0f : 0.0f);
      if(l == 0)
         outputs[RCtoFlat(r, c, total_r, total_c, 0)] = diag;
      if(step < 2)
         return;
      if(l == 0)
        {
         const float s = IsNaNOrInf(support[RCtoFlat(r, c, total_r, total_c, 0)], 0);
         outputs[RCtoFlat(r, c, total_r, total_c, 1)] = s;
        }
      if(step < 3)
         return;
      float out = 0;
      for(int t = 0; t < total_c; t += total_l)
        {
         const float s1 = IsNaNOrInf(support[RCtoFlat(r, t + l, total_r, total_c, 0)], 0);
         const float s2 = IsNaNOrInf(support[RCtoFlat(t + l, c, total_r, total_c, 0)], 0);
         out += IsNaNOrInf(s1 * s2, 0);
        }
      out = 2 * LocalSum(out, 0, Temp);
      if(l == 0)
        {
         out -= diag;
         outputs[RCtoFlat(r, c, total_r, total_c, 2)] = IsNaNOrInf(out, 0);
        }
      return;
     }
//---
   float out = 0;
   for(int t = 0; t < total_c; t += total_l)
     {
      if((t + l) >= total_c)
         continue;
      const float s1 = IsNaNOrInf(support[RCtoFlat(r, t + l, total_r, total_c, 0)], 0);
      const float s2 = IsNaNOrInf(outputs[RCtoFlat(t + l, c, total_r, total_c, step - 2)], 0);
      out += IsNaNOrInf(s1 * s2, 0);
     }
   out = 2 * LocalSum(out, 0, Temp);
   if(l == 0)
     {
      out -= IsNaNOrInf(outputs[RCtoFlat(r, c, total_r, total_c, step - 3)], 0);
      outputs[RCtoFlat(r, c, total_r, total_c, step - 1)] = IsNaNOrInf(out, 0);
     }
   return;
  }
//+------------------------------------------------------------------+
//| Computes the cheb step grad gradient contributions on the GPU.   |
//+------------------------------------------------------------------+
__kernel void ChebStepGrad(__global const float* support,
                           __global float* support_g,
                           __global const float* outputs,
                           __global float* outputs_g,
                           const int step
                          )
  {
   const size_t l = get_local_id(0);
   const size_t r = get_global_id(1);
   const size_t c = get_global_id(2);
   const size_t total_l = get_local_size(0);
   const size_t total_r = get_global_size(1);
   const size_t total_c = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   if(step < 1 || total_r != total_c)
      return;
//---
   if(step >= 2)
     {
      float grad = IsNaNOrInf(outputs_g[RCtoFlat(r, c, total_r, total_c, step)], 0);
      if(l == 0)
         outputs_g[RCtoFlat(r, c, total_r, total_c, step - 2)] -= grad;
      //--- support grad
      grad = 0;
      for(int t = 0; t < total_c; t += total_l)
        {
         if((t + l) >= total_c)
            continue;
         const float s2 = IsNaNOrInf(outputs[RCtoFlat(c, t + l, total_r, total_c, step - 2)], 0);
         grad += IsNaNOrInf(outputs_g[RCtoFlat(r, t + l, total_r, total_c, step)] * s2, 0);
        }
      grad = LocalSum(grad, 0, Temp);
      if(l == 0)
         outputs_g[RCtoFlat(r, c, total_r, total_c, 1)] += grad;
      BarrierLoc
      //--- T(k-1) grad
      grad = 0;
      for(int t = 0; t < total_c; t += total_l)
        {
         if((t + l) >= total_c)
            continue;
         const float s2 = IsNaNOrInf(support[RCtoFlat(t + l, r, total_r, total_c, 0)], 0);
         grad += IsNaNOrInf(outputs_g[RCtoFlat(t + l, c, total_r, total_c, step)] * s2, 0);
        }
      grad = LocalSum(grad, 0, Temp);
      if(l == 0)
         outputs_g[RCtoFlat(r, c, total_r, total_c, step - 1)] += grad;
     }
//---
   if(step <= 2)
     {
      if(l == 0)
         support_g[RCtoFlat(r, c, total_r, total_c, 0)] = outputs_g[RCtoFlat(r, c, total_r, total_c, 1)];
      return;
     }
  }
//+------------------------------------------------------------------+
//| Implements the significant neighbors sampling kernel on the GPU. |
//+------------------------------------------------------------------+
__kernel void SignificantNeighborsSampling(__global const float *data,
      __global const float *candidates,
      __global const float *random_cands,
      __global float *neighbors,
      const int dimension
                                          )
  {
   const size_t main = get_global_id(0);
   const size_t slave = get_local_id(1);
   const int total_main = (int)get_global_size(0);
   const int total_slave = (int)get_local_size(1);
//---
   __local int Idx[LOCAL_ARRAY_SIZE];
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int ls = min(total_slave, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_main = RCtoFlat(main, 0, total_main, dimension, 0);
   int cand = (int)candidates[slave];
   int rand_cand = (int)random_cands[slave];
//--- duplicate check
   if(rand_cand == cand)
      rand_cand = -1;
//--- Look in candidates
   for(int l = 0; l < total_slave; l += ls)
     {
      if(slave >= l && slave < (l + ls))
         Idx[slave - l] = cand;
      BarrierLoc
      for(int i = 0; i < ls; i++)
        {
         if(i >= (slave - l))
            continue;
         if(cand == Idx[i])
            cand = -1;
         if(rand_cand == Idx[i])
            rand_cand = -1;
        }
      BarrierLoc
     }
//--- Look in random candidates
   for(int l = 0; l < total_slave; l += ls)
     {
      if(slave >= l && slave < (l + ls))
         Idx[slave - l] = rand_cand;
      BarrierLoc
      for(int i = 0; i < ls; i++)
        {
         if(i >= (slave - l))
            continue;
         if(cand == Idx[i])
            cand = -1;
         if(rand_cand == Idx[i])
            rand_cand = -1;
        }
      BarrierLoc
     }
//---
   const int shift_cand = RCtoFlat(cand, 0, total_main, dimension, 0);
   const int shift_rand_cand = RCtoFlat(rand_cand, 0, total_main, dimension, 0);
//--- calc distance
   float dist_cand = 0;
   float dist_rand_cand = 0;
   for(int d = 0; d < dimension; d++)
     {
      float value = IsNaNOrInf(data[shift_main + d], 0);
      if(main != cand && cand >= 0)
        {
         float delta = value - IsNaNOrInf(data[shift_cand + d], 0);
         dist_cand += delta * delta;
        }
      if(main != rand_cand && rand_cand >= 0)
        {
         float delta = value - IsNaNOrInf(data[shift_rand_cand + d], 0);
         dist_rand_cand += delta * delta;
        }
     }
//--- calc position
   int cand_position = 0;
   int rand_position = (int)(dist_cand >= dist_rand_cand);
//--- by candidates
   for(int l = 0; l < total_slave; l += ls)
     {
      if(slave >= l && slave < (l + ls))
         Temp[slave - l] = (cand >= 0 ? IsNaNOrInf(dist_cand, -1) : -1);
      BarrierLoc
      for(int i = 0; i < ls; i++)
        {
         if(i == (slave - l))
            continue;
         if(Temp[i] < 0)
            continue;
         if(cand >= 0)
           {
            if(Temp[i] < dist_cand)
               cand_position++;
            else
               if(Temp[i] < dist_cand && i < (slave - l))
                  cand_position++;
           }
         if(rand_cand >= 0)
           {
            if(Temp[i] < dist_rand_cand)
               rand_position++;
            else
               if(Temp[i] < dist_rand_cand && i < (slave - l))
                  rand_position++;
           }
        }
      BarrierLoc
     }
//--- by random candidates
   for(int l = 0; l < total_slave; l += ls)
     {
      if(slave >= l && slave < (l + ls))
         Temp[slave - l] = (rand_cand >= 0 ? IsNaNOrInf(dist_rand_cand, -1) : -1);
      BarrierLoc
      for(int i = 0; i < ls; i++)
        {
         if(i == (slave - l))
            continue;
         if(Temp[i] < 0)
            continue;
         if(cand >= 0)
           {
            if(Temp[i] < dist_cand)
               cand_position++;
            else
               if(Temp[i] < dist_cand && i < (slave - l))
                  cand_position++;
           }
         if(rand_cand >= 0)
           {
            if(Temp[i] < dist_rand_cand)
               rand_position++;
            else
               if(Temp[i] < dist_rand_cand && i < (slave - l))
                  rand_position++;
           }
        }
      BarrierLoc
     }
//--- result
   if(cand >= 0 && cand_position < total_slave)
     {
      const int shift_dist_cand = RCtoFlat(main, cand_position, total_main, total_slave, 0);
      neighbors[shift_dist_cand] = cand;
     }
   if(rand_cand >= 0 && rand_position < total_slave)
     {
      const int shift_dist_cand = RCtoFlat(main, rand_position, total_main, total_slave, 0);
      neighbors[shift_dist_cand] = rand_cand;
     }
  }
//+------------------------------------------------------------------+
//| Implements the sparse mh scores kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void SparseMHScores(__global const float* data,
                             __global const float* indexes,
                             __global float* scores,
                             const float sparse              ///< [0.0 .. 1.0) coefficient of sparse
                            )
  {
   const int main = (int)get_global_id(0);
   const int slave = (int)get_local_id(1);
   const int head = (int)get_global_id(2);
   const int total_mains = (int)get_global_size(0);
   const int total_slaves = (int)get_local_size(1);
   const int total_heads = (int)get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   float value = IsNaNOrInf(data[RCtoFlat(main, head, total_mains, 2 * total_heads, 0)], 0);
   int slave_id = (int)indexes[RCtoFlat(main, slave, total_mains, total_slaves, 0)];
   if(slave_id < total_mains && slave_id >= 0)
      value += IsNaNOrInf(data[RCtoFlat(slave_id, head + total_heads, total_mains, 2 * total_heads, 0)], 0);
//---
   const float max_value = LocalMax(value, 1, Temp);
   const float min_value = LocalMin(value, 1, Temp);
   const float threshold = (max_value - min_value) * sparse + min_value;
   value = (threshold <= value ? IsNaNOrInf(exp(value - max_value), 0) : 0);
   const float sum = LocalSum(value, 1, Temp);
   value = IsNaNOrInf(value / sum, 0);
//---
   scores[RCtoFlat(slave, head, total_slaves, total_heads, main)] = value;
  }
//+------------------------------------------------------------------+
//| Computes the sparse mh scores grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SparseMHScoresGrad(__global float* data_gr,
                                 __global const float* indexes,
                                 __global const float* scores,
                                 __global const float* scores_gr
                                )
  {
   const int main = (int)get_global_id(0);
   const int slave = (int)get_local_id(1);
   const int head = (int)get_global_id(2);
   const int total_mains = (int)get_global_size(0);
   const int total_slaves = (int)get_local_size(1);
   const int total_heads = (int)get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const uint ls = min((uint)total_slaves, (uint)LOCAL_ARRAY_SIZE);
//--- Calc grad by main
     {
      float value = IsNaNOrInf(scores[RCtoFlat(slave, head, total_slaves, total_heads, main)], 0);
      int slave_id = (int)indexes[RCtoFlat(main, slave, total_mains, total_slaves, 0)];
      const float sc_gr = IsNaNOrInf(scores_gr[RCtoFlat(slave, head, total_slaves, total_heads, main)], 0);
      //---
      float grad = 0;
      for(uint d = 0; d < total_slaves; d += ls)
        {
         if(slave >= d && slave < (d + ls))
            Temp[slave - d] = IsNaNOrInf(sc_gr, 0);
         BarrierLoc
         for(uint l = 0; l < min(ls, (uint)(total_slaves - d)); l++)
            grad += IsNaNOrInf(Temp[l] * ((float)((d + l) == slave && slave_id == main) - value), 0);
         BarrierLoc
        }
      grad = LocalSum(grad, 1, Temp);
      if(slave == 0)
         data_gr[RCtoFlat(main, head, total_mains, 2 * total_heads, 0)] = grad;
     }
//--- Calc grad by slave
     {
      float grad = 0;
      for(uint d = 0; d < total_mains; d++)
        {
         float value = IsNaNOrInf(scores[RCtoFlat(slave, head, total_slaves, total_heads, d)], 0);
         const float sc_gr = IsNaNOrInf(scores_gr[RCtoFlat(slave, head, total_slaves, total_heads, d)], 0);
         int slave_id = (int)indexes[RCtoFlat(d, slave, total_mains, total_slaves, 0)];
         //---
         float gr = IsNaNOrInf(sc_gr * ((float)(slave_id == d) - value), 0);
         gr = LocalSum(gr, 1, Temp);
         if(slave == 0)
            grad += gr;
        }
      if(slave == 0)
         data_gr[RCtoFlat(main, head + total_heads, total_mains, 2 * total_heads, 0)] = IsNaNOrInf(grad, 0);
     }
  }
//+------------------------------------------------------------------+
//| Implements the sparse mat mult kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void SparseMatMult(__global const float *sparse_index,
                            __global const float *sparse_data,
                            __global const float *full,
                            __global float *result,
                            const int full_rows
                           )
  {
   const size_t sparse_row = get_global_id(0);
   const size_t sparse_col = get_local_id(1);
   const size_t full_col = get_global_id(2);
   const size_t sparse_rows = get_global_size(0);
   const size_t sparse_cols = get_local_size(1);
   const size_t full_cols = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   const int shift_sparse = RCtoFlat(sparse_row, sparse_col, sparse_rows, sparse_cols, 0);
   const int full_row = sparse_index[shift_sparse];
   const int shift_full = RCtoFlat(full_row, full_col, full_rows, full_cols, 0);
//---
   float res = (full_row >= 0 && full_row < full_rows ?
                IsNaNOrInf(sparse_data[shift_sparse] * full[shift_full], 0) : 0);
   res = LocalSum(res, 1, Temp);
//---
   if(sparse_col == 0)
     {
      const int shift_result = RCtoFlat(sparse_row, full_col, sparse_rows, full_cols, 0);
      result[shift_result] = res;
     }
  }
//+------------------------------------------------------------------+
//| Computes the sparse mat mult grad gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SparseMatMultGrad(__global const float *sparse_index,
                                __global const float *sparse_data,
                                __global float *sparse_gr,
                                __global const float *full,
                                __global float *full_gr,
                                __global const float *result_gr,
                                const int sparse_rows,
                                const int sparse_cols,
                                const int full_rows,
                                const int full_cols
                               )
  {
   const size_t row_id = get_global_id(0);
   const size_t local_id = get_local_id(1);
   const size_t col_id = get_global_id(2);
   const size_t total_rows = get_global_size(0);
   const size_t total_local = get_local_size(1);
   const size_t total_cols = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//--- Calce sparse gradient
   if(row_id < sparse_rows && col_id < sparse_cols)
     {
      float grad = 0;
      int shift_sparse = 0;
      if(local_id == 0)
        {
         shift_sparse = RCtoFlat(row_id, col_id, sparse_rows, sparse_cols, 0);
         Temp[0] = sparse_index[shift_sparse];
        }
      BarrierLoc
      uint full_row = (uint)Temp[0];
      if(full_row < (uint)full_rows)
         for(int i = local_id; i < full_cols; i += total_local)
           {
            int shift_result = RCtoFlat(row_id, i, sparse_rows, full_cols, 0);
            int shift_full = RCtoFlat(full_row, i, full_rows, full_cols, 0);
            grad += IsNaNOrInf(result_gr[shift_result] * full[shift_full], 0);
           }
      grad = LocalSum(grad, 1, Temp);
      if(local_id == 0)
         sparse_gr[shift_sparse] = grad;
     }
//--- Calce full gradient
   if(row_id < full_rows && col_id < full_cols)
     {
      float grad = 0;
      for(int r = 0; r < sparse_rows; r ++)
        {
         float s = 0;
         for(int c = local_id; c < sparse_cols; c += total_local)
           {
            int shift_sparse = RCtoFlat(r, c, sparse_rows, sparse_cols, 0);
            if((int)sparse_index[shift_sparse] == (int)row_id)
              {
               s = sparse_data[shift_sparse];
               break;
              }
           }
         s = LocalSum(s, 1, Temp);
         if(s != 0 && local_id == 0)
           {
            int shift_result = RCtoFlat(r, col_id, sparse_rows, full_cols, 0);
            grad += IsNaNOrInf(s * result_gr[shift_result], 0);
           }
        }
      if(local_id == 0)
        {
         int shift_full = RCtoFlat(row_id, col_id, full_rows, full_cols, 0);
         full_gr[shift_full] = grad;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the random walk kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void RandomWalk(__global const float *data,
                         __global float *inv_diag,
                         __global float *norm,
                         const int total_cols
                        )
  {
   const size_t row_id = get_global_id(0);
   const size_t local_id = get_local_id(1);
   const size_t total_rows = get_global_size(0);
   const size_t total_local = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   float d = 0;
   for(int c = local_id; c < total_cols; c += total_local)
     {
      int shift = RCtoFlat(row_id, c, total_rows, total_cols, 0);
      d += IsNaNOrInf(data[shift], 0);
     }
   d = IsNaNOrInf(1.0f / (LocalSum(d, 1, Temp) + 1.0f), 1.0f);
   if(local_id == 0)
      inv_diag[row_id] = d;
//---
   for(int c = local_id; c < total_cols; c += total_local)
     {
      int shift = RCtoFlat(row_id, c, total_rows, total_cols, 0);
      norm[shift] = IsNaNOrInf(data[shift] * d, 0);
     }
  }
//+------------------------------------------------------------------+
//| Performs the concat by label data-layout operation on the GPU.   |
//+------------------------------------------------------------------+
__kernel void ConcatByLabel(__global const float* data,
                            __global const float* label,
                            __global const float* embedding1,
                            __global const float* embedding2,
                            __global float *output,
                            const int dimension_data,
                            const int dimension_emb1,
                            const int dimension_emb2,
                            const int frame1,
                            const int frame2,
                            const int period1,
                            const int period2
                           )
  {
   const size_t row_id = get_global_id(0);
   const size_t col_id = get_global_id(1);
   const size_t buffer_id = get_global_id(2);
   const size_t total_rows = get_global_size(0);
   const size_t total_cols = get_global_size(1);
   const size_t total_buffers = get_global_size(2);
//---
   __global const float *buffer;
   int dimension_in, dimension_out;
   int shift_in, shift_out;
//---
   switch(total_buffers)
     {
      case 1:
         dimension_out = dimension_data;
         break;
      case 2:
         dimension_out = dimension_data + dimension_emb1;
         break;
      case 3:
         dimension_out = dimension_data + dimension_emb1 + dimension_emb2;
         break;
      default:
         return;
     }
//---
   switch(buffer_id)
     {
      case 0:
         buffer = data;
         dimension_in = dimension_data;
         shift_in = RCtoFlat(row_id, col_id, total_rows, dimension_in, 0);
         shift_out = RCtoFlat(row_id, col_id, total_rows, dimension_out, 0);
         break;
      case 1:
         buffer = embedding1;
         dimension_in = dimension_emb1;
         shift_in = ((int)IsNaNOrInf(label[row_id] / frame1, 0)) % period1;
         shift_in = RCtoFlat(shift_in, col_id, period1, dimension_in, 0);
         shift_out = RCtoFlat(row_id, dimension_data + col_id, total_rows, dimension_out, 0);
         break;
      case 2:
         buffer = embedding2;
         dimension_in = dimension_emb2;
         shift_in = ((int)IsNaNOrInf(label[row_id] / frame2, 0)) % period2;
         shift_in = RCtoFlat(shift_in, col_id, period2, dimension_in, 0);
         shift_out = RCtoFlat(row_id, dimension_data + dimension_emb1 + col_id, total_rows, dimension_out, 0);
         break;
     }
//---
   if(col_id < dimension_in)
      output[shift_out] = IsNaNOrInf(buffer[shift_in], 0);
  }
//+------------------------------------------------------------------+
//| Computes the concat by label grad gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void ConcatByLabelGrad(__global float* data_gr,
                                __global const float* label,
                                __global float* embedding1_gr,
                                __global float* embedding2_gr,
                                __global float *output_gr,
                                const int dimension_data,
                                const int dimension_emb1,
                                const int dimension_emb2,
                                const int frame1,
                                const int frame2,
                                const int period1,
                                const int period2,
                                const int units
                               )
  {
   const size_t row_id = get_global_id(0);
   const size_t col_id = get_global_id(1);
   const size_t buffer_id = get_global_id(2);
   const size_t total_rows = get_global_size(0);
   const size_t total_cols = get_global_size(1);
   const size_t total_buffers = get_global_size(2);
//---
   __global float *buffer;
   int dimension_in, dimension_out;
   int shift_in, shift_out, shift_col;
   int period, frame, rows;
//---
   switch(total_buffers)
     {
      case 1:
         dimension_out = dimension_data;
         break;
      case 2:
         dimension_out = dimension_data + dimension_emb1;
         break;
      case 3:
         dimension_out = dimension_data + dimension_emb1 + dimension_emb2;
         break;
      default:
         return;
     }
//---
   switch(buffer_id)
     {
      case 0:
         if(col_id < dimension_data && row_id < units)
           {
            shift_in = RCtoFlat(row_id, col_id, total_rows, dimension_in, 0);
            shift_out = RCtoFlat(row_id, col_id, total_rows, dimension_out, 0);
            data_gr[shift_in] = IsNaNOrInf(output_gr[shift_out], 0);
           }
         return;
      case 1:
         rows = period1;
         buffer = embedding1_gr;
         dimension_in = dimension_emb1;
         shift_in = RCtoFlat(row_id, col_id, period1, dimension_in, 0);
         shift_col = dimension_data;
         period = period1;
         frame = frame1;
         break;
      case 2:
         rows = period2;
         buffer = embedding2_gr;
         dimension_in = dimension_emb2;
         shift_in = RCtoFlat(row_id, col_id, period2, dimension_in, 0);
         shift_col = dimension_data + dimension_emb1;
         period = period2;
         frame = frame2;
         break;
     }
//---
   if(row_id >= rows || col_id >= dimension_in)
      return;
   float grad = 0;
   for(uint r = 0; r < total_rows; r ++)
     {
      int row = ((int)IsNaNOrInf(label[r] / frame, 0)) % period;
      if(row != row_id)
         continue;
      shift_out = RCtoFlat(r, shift_col + col_id, total_rows, dimension_out, 0);
      grad += IsNaNOrInf(output_gr[shift_out], 0);
     }
   buffer[shift_in] = IsNaNOrInf(grad, 0);
  }
//+------------------------------------------------------------------+
//| Implements the global local attention kernel on the GPU.         |
//+------------------------------------------------------------------+
__kernel void GlobalLocalAttention(__global const float *q,
                                   __global const float2* kv,
                                   __global float *scores,
                                   __global const float* mask,
                                   __global const float* label,
                                   __global float *out,
                                   const int dimension,
                                   const int total_kv,
                                   const int total_mask
                                  )
  {
//--- init
   const int q_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_local = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Score
   int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
   if(h_id % 2 == 0)
     {
      const int shift_kv = RCtoFlat(h_id, 0, total_heads, dimension, local_id);
      const int shift_s = RCtoFlat(h_id / 2, local_id, total_heads / 2, total_kv + total_mask, q_id);
      float score = 0;
      if(local_id < total_kv)
        {
         for(int d = 0; d < dimension; d++)
            score += IsNaNOrInf(q[shift_q + d] * kv[shift_kv + d].s0, 0);
        }
      else
         score = MIN_VALUE;
      //--- norm score
      score = LocalSoftMax(score, 1, temp);
      if(local_id < total_kv)
         scores[shift_s] = score;
      //--- out
      for(int d = 0; d < dimension; d++)
        {
         float val = (local_id < total_kv ? kv[shift_kv + d].s1 * score : 0);
         val = LocalSum(val, 1, temp);
         if(local_id == 0)
            out[shift_q + d] = val;
        }
     }
   else
     {
      int kv_id = -1;
      float score = 0;
      int shift_kv = -1;
      float m = 0;
      const int shift_s = RCtoFlat(h_id / 2, total_kv + local_id, total_heads / 2, total_kv + total_mask, q_id);
      if(local_id < total_mask)
        {
         const int l = RCtoFlat(q_id, local_id, total_q, total_mask, 0);
         kv_id = IsNaNOrInf(label[l], -1);
         m = IsNaNOrInf(mask[l], 0);
         shift_kv = RCtoFlat(h_id, 0, total_heads, dimension, kv_id);
         if(kv_id >= 0)
            for(int d = 0; d < dimension; d++)
               score += IsNaNOrInf(q[shift_q + d] * kv[shift_kv + d].s0, 0);
         else
            score = MIN_VALUE;
        }
      else
         score = MIN_VALUE;
      //--- norm score
      score = LocalSoftMax(score * m, 1, temp);
      if(local_id < total_mask)
         scores[shift_s] = score;
      //--- out
      for(int d = 0; d < dimension; d++)
        {
         float val = (kv_id >= 0 ? IsNaNOrInf(kv[shift_kv + d].s1, 0) * score : 0);
         val = LocalSum(val, 1, temp);
         if(local_id == 0)
            out[shift_q + d] = val;
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the global local attention grad gradient contributions  |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void GlobalLocalAttentionGrad(__global const float *q,
                                       __global float *q_gr,
                                       __global const float *kv,
                                       __global float *kv_gr,
                                       __global float *scores,
                                       __global const float *mask,
                                       __global float *mask_gr,
                                       __global const float *label,
                                       __global float *out_gr,
                                       const int dimension,
                                       const int total_q,
                                       const int total_kv,
                                       const int total_mask
                                      )
  {
//--- init
   const int global_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_global = get_global_size(0);
   const int total_local = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   if(h_id % 2 == 0)
     {
      //--- Value Gradient global_id -> v_id, local_id -> q_id
      for(int d = 0; d < dimension; d++)
        {
         const int shift_v = RCtoFlat(h_id, 2 * d + 1, total_heads, 2 * dimension, global_id);
         float grad = 0;
         for(int q_id = local_id; q_id < total_q; q_id += total_local)
           {
            int shift_s = RCtoFlat(h_id / 2, global_id, total_heads / 2, total_kv + total_mask, q_id);
            int shift_q = RCtoFlat(h_id, d, total_heads, dimension, q_id);
            grad += IsNaNOrInf(scores[shift_s] * out_gr[shift_q], 0);
           }
         grad = LocalSum(grad, 1, temp);
         kv_gr[shift_v] = grad;
        }
      //--- Query Gradient global_id -> q_id, local_id -> k_id/v_id
      if(global_id < total_q)
        {
         //--- 1. Score grad
         float grad_s = 0;
         const int shift_v = RCtoFlat(h_id, 1, total_heads, 2 * dimension, local_id);
         const int shift_s = RCtoFlat(h_id / 2, local_id, total_heads / 2, total_kv + total_mask, global_id);
         int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, global_id);
         if(local_id < total_kv)
            for(int d = 0; d < dimension; d++)
               grad_s += IsNaNOrInf(kv[shift_v + 2 * d] * out_gr[shift_q + d], 0);
         //--- 2. SoftMax grad
         grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
         //--- 3. Query grad
         const int shift_k = shift_v - 1;
         for(int d = 0; d < dimension; d++)
           {
            float grad = 0;
            if(local_id < total_kv)
               grad = kv[shift_k + 2 * d] * grad_s;
            grad = LocalSum(grad, 1, temp);
            if(local_id == 0)
               q_gr[shift_q + d] = grad;
           }
        }
      //--- Key Gradient global_id -> k_id, local_id -> score_id/v_id/dimension
      if(global_id < total_kv)
        {
         float grad = 0;
         for(int q_id = 0; q_id < total_q; q_id++)
           {
            //--- 1. Score grad local_id -> score_id/v_id
            float grad_s = 0;
            const int shift_v = RCtoFlat(h_id, 1, total_heads, 2 * dimension, local_id);
            const int shift_s = RCtoFlat(h_id / 2, local_id, total_heads / 2, total_kv + total_mask, q_id);
            int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
            if(local_id < total_kv)
               for(int d = 0; d < dimension; d++)
                  grad_s += IsNaNOrInf(kv[shift_v + 2 * d] * out_gr[shift_q + d], 0);
            //--- 2. SoftMax grad
            grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
            BarrierLoc
            if(global_id == local_id)
               temp[0] = grad_s;
            BarrierLoc
            grad_s = temp[0];
            //--- 3. Key grad local_id -> dimension
            shift_q = RCtoFlat(h_id, local_id, total_heads, dimension, q_id);
            if(local_id < dimension)
               grad += IsNaNOrInf(q[shift_q] * grad_s, 0);
           }
         const int shift_k = RCtoFlat(h_id, 2 * local_id, total_heads, 2 * dimension, global_id);
         if(local_id < dimension)
            kv_gr[shift_k] = IsNaNOrInf(grad, 0);
        }
     }
   else
     {
      //--- Value Gradient global_id -> v_id, local_id -> mask_index/dimension
      if(global_id < total_kv)
        {
         float grad = 0;
         for(int q_id = 0; q_id < total_q; q_id++)
           {
            //--- 1. kv_id
            int kv_id = -1;
            float m = 0;
            const int l = RCtoFlat(q_id, local_id, total_q, total_mask, 0);
            const int shift_s = RCtoFlat(h_id / 2, total_kv + local_id, total_heads / 2, total_kv + total_mask, q_id);
            //--- Check for use current Value
            if(local_id < total_mask)
               kv_id = (int)label[l];
            if(local_id == 0)
               temp[0] = 0;
            BarrierLoc
            if(kv_id == global_id)
               temp[0] = scores[shift_s];
            BarrierLoc
            if(temp[0] == 0)
               continue;
            //--- Value grad
            int shift_q = RCtoFlat(h_id, local_id, total_heads, dimension, q_id);
            if(local_id < dimension)
               grad += IsNaNOrInf(temp[0] * out_gr[shift_q], 0);
           }
         const int shift_v = RCtoFlat(h_id, 2 * local_id + 1, total_heads, 2 * dimension, global_id);
         if(local_id < dimension)
            kv_gr[shift_v] = IsNaNOrInf(grad, 0);
        }
      //--- Query Gradient global_id -> q_id, local_id -> mask label
      if(global_id < total_q)
        {
         //--- 1. kv_id;
         int kv_id = -1;
         float m = 0;
         const int l = RCtoFlat(global_id, local_id, total_q, total_mask, 0);
         if(local_id < total_mask)
           {
            kv_id = (int)IsNaNOrInf(label[l], -1);
            m = IsNaNOrInf(mask[l], 0);
           }
         //--- 2. Score grad
         float grad_s = 0;
         const int shift_v = RCtoFlat(h_id, 1, total_heads, 2 * dimension, kv_id);
         const int shift_s = RCtoFlat(h_id / 2, total_kv + local_id, total_heads / 2, total_kv + total_mask, global_id);
         int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, global_id);
         if(local_id < total_mask)
            for(int d = 0; d < dimension; d++)
               grad_s += IsNaNOrInf(kv[shift_v + 2 * d] * out_gr[shift_q + d], 0);
         //--- 3. SoftMax grad
         float score = IsNaNOrInf(scores[shift_s], 0);
         grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
         mask_gr[l] = IsNaNOrInf(grad_s * score, 0);
         grad_s *= m;
         //--- 4. Query grad
         const int shift_k = shift_v - 1;
         for(int d = 0; d < dimension; d++)
           {
            float grad = 0;
            if(local_id < total_mask)
               grad = kv[shift_k + 2 * d] * grad_s;
            grad = LocalSum(grad, 1, temp);
            if(local_id == 0)
               q_gr[shift_q + d] = grad;
           }
        }
      //--- Key Gradient global_id -> k_id, local_id -> score_id/v_id/dimension
      if(global_id < total_kv)
        {
         float grad = 0;
         for(int q_id = 0; q_id < total_q; q_id++)
           {
            //--- 1. kv_id;
            int kv_id = -1;
            float m = 0;
            const int l = RCtoFlat(global_id, local_id, total_q, total_mask, 0);
            if(local_id < total_mask)
              {
               kv_id = (int)label[l];
               if(kv_id == global_id)
                  m = mask[l];
              }
            m = LocalSum(m, 1, temp);
            if(m == 0)
               continue;
            //--- 2. Score grad local_id -> score_id/v_id
            float grad_s = 0;
            const int shift_v = RCtoFlat(h_id, 1, total_heads, 2 * dimension, kv_id);
            const int shift_s = RCtoFlat(h_id / 2, total_kv + local_id, total_heads / 2, total_kv + total_mask, q_id);
            int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
            if(local_id < total_mask)
               for(int d = 0; d < dimension; d++)
                  grad_s += IsNaNOrInf(kv[shift_v + 2 * d] * out_gr[shift_q + d], 0);
            //--- 3. SoftMax grad
            grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
            BarrierLoc
            if(global_id == local_id)
               temp[0] = grad_s * m;
            BarrierLoc
            grad_s = temp[0];
            //--- 4. Key grad local_id -> dimension
            shift_q = RCtoFlat(h_id, local_id, total_heads, dimension, q_id);
            if(local_id < dimension)
               grad += IsNaNOrInf(q[shift_q] * grad_s, 0);
           }
         const int shift_k = RCtoFlat(h_id, 2 * local_id, total_heads, 2 * dimension, global_id);
         if(local_id < dimension)
            kv_gr[shift_k] = IsNaNOrInf(grad, 0);
        }
     }
  }
//+------------------------------------------------------------------+
//| Applies the sparse soft max normalization to the input scores.   |
//+------------------------------------------------------------------+
__kernel void SparseSoftMax(__global const float *data,
                            __global float *outputs,
                            __global float *indexes,
                            const int out_dimension
                           )
  {
   const size_t row = get_global_id(0);
   const size_t col_in = get_local_id(1);
   const int total_rows = (int)get_global_size(0);
   const int total_cols_in = (int)get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int ls = min(total_cols_in, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_in = RCtoFlat(row, col_in, total_rows, total_cols_in, 0);
//--- calc position
   float value = IsNaNOrInf(data[shift_in], MIN_VALUE);
   int position = 0;
   for(int l = 0; l < total_cols_in; l += ls)
     {
      if(col_in >= l && col_in < (l + ls))
         Temp[col_in - l] = value;
      BarrierLoc
      for(int i = 0; i < ls; i++)
        {
         if(i == (col_in - l))
            continue;
         if(Temp[i] > value)
            position++;
         else
            if(Temp[i] == value && i < (col_in - l))
               position++;
        }
      BarrierLoc
     }
//--- SoftMax
   if(position >= out_dimension)
      value = MIN_VALUE;
   value = LocalSoftMax(value, 1, Temp);
//--- result
   const int shift_out = RCtoFlat(row, position, total_rows, out_dimension, 0);
   if(position < out_dimension)
     {
      outputs[shift_out] = value;
      indexes[shift_out] = (float)col_in;
     }
  }
//+------------------------------------------------------------------+
//| Computes the sparse soft max grad gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void SparseSoftMaxGrad(__global float *data_gr,
                                __global const float *outputs,
                                __global const float *outputs_gr,
                                __global const float *indexes,
                                const int out_dimension
                               )
  {
   const size_t row = get_global_id(0);
   const size_t col_in = get_local_id(1);
   const int total_rows = (int)get_global_size(0);
   const int total_cols_in = (int)get_local_size(1);
//---
   __local int Ind[LOCAL_ARRAY_SIZE];
   __local float Temp[LOCAL_ARRAY_SIZE];
   const int ls = min(total_cols_in, (int)LOCAL_ARRAY_SIZE);
//--- look position
   float value = 0;
   float grad = 0;
   int position = -1;
   int idx = -1;
   const int shift_idx = RCtoFlat(row, col_in, total_rows, out_dimension, 0);
   if(col_in < out_dimension)
      idx = (int)IsNaNOrInf(indexes[shift_idx], -1.0f);
   for(int l = 0; l < out_dimension; l += ls)
     {
      if(col_in >= l && col_in < (l + ls))
         Ind[col_in - l] = idx;
      BarrierLoc
      for(int i = 0; (i < ls && position < 0); i++)
        {
         if(Ind[i] == col_in)
            position = l + i;
        }
      BarrierLoc
     }
//--- SoftMax Grad
   if(position < out_dimension && position >= 0)
     {
      const int shift_out = RCtoFlat(row, position, total_rows, out_dimension, 0);
      value = IsNaNOrInf(outputs[shift_out], 0);
      grad = IsNaNOrInf(outputs_gr[shift_out], 0);
     }
   grad = LocalSoftMaxGrad(value, grad, 1, Temp);
//--- result
   const int shift_in = RCtoFlat(row, col_in, total_rows, total_cols_in, 0);
   data_gr[shift_in] = grad;
  }
//+------------------------------------------------------------------+
//| Implements the float to spike kernel on the GPU.                 |
//+------------------------------------------------------------------+
__kernel void FloatToSpike(__global float* values,
                           __global const float* levels,
                           __global float* outputs
                          )
  {
   const size_t id = get_global_id(0);
   float val = IsNaNOrInf(values[id], 0.0f);
   if(val == 0.0f)
      outputs[id] = 0.0f;
   else
     {
      const float lev = IsNaNOrInf(levels[id], 0.0f);
      if(fabs(val) < lev)
         outputs[id] = 0.0f;
      else
        {
         outputs[id] = (float)sign(val);
         values[id] = IsNaNOrInf(sign(val) * (fabs(val) - lev), 0.0f);
        }
     }
  }
//+------------------------------------------------------------------+
//| Computes the float to spike grad gradient contributions on the   |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void FloatToSpikeGrad(__global const float* values,
                               __global float* values_gr,
                               __global float* levels_gr,
                               __global const float* gradients
                              )
  {
   const size_t id = get_global_id(0);
   const float grad = IsNaNOrInf(gradients[id], 0.0f);
   values_gr[id] = grad;
   if(fabs(grad) > 0.0f)
     {
      float val = IsNaNOrInf(values[id], 0.0f);
      levels_gr[id] = (float)(-sign(val) * grad);
     }
   else
      levels_gr[id] = 0.0f;
  }
//+------------------------------------------------------------------+
//| Implements the spike mh attention kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void SpikeMHAttention(__global const float *qkv,
                               __global const float *diag_bias,
                               __global float *scores,
                               __global float *out,
                               const int dimension,
                               const int mask_future
                              )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Shifts
   const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, 3 * q_id);
   const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 3 * k_id + 1);
   const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 3 * k_id + 2);
   const int shift_s = RCtoFlat(h_id, k_id, total_heads, total_q, q_id);
   const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
//--- Score
   float score = 0;
   if(mask_future == 0 || q_id <= k_id)
     {
      for(int d = 0; d < dimension; d++)
        {
         float q = IsNaNOrInf(qkv[shift_q + d], 0);
         if(q == 0)
            continue;
         float k = IsNaNOrInf(qkv[shift_k + d], 0);
         if(k == 0)
            continue;
         score += q * k;
        }
     }
   else
      score = MIN_VALUE;
   if(q_id == k_id)
      score += IsNaNOrInf(diag_bias[q_id], 0);
//--- norm score
   score = LocalSoftMax(score, 1, temp);
   scores[shift_s] = score;
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val = 0;
      if(score > 0)
        {
         float v = IsNaNOrInf(qkv[shift_v + d], 0);
         if(v != 0)
            val = v * score;
        }
      val = LocalSum(val, 1, temp);
      if(k_id == 0)
         out[shift_out + d] = val;
     }
  }
//+------------------------------------------------------------------+
//| Computes the spike mh attention grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void SpikeMHAttentionGrad(__global const float *qkv,
                                   __global float *qkv_gr,
                                   __global const float *diag_bias,
                                   __global float *diag_bias_gr,
                                   __global const float *scores,
                                   __global const float *gradients,
                                   const int dimension,
                                   const int mask_future
                                  )
  {
//--- init
   const int global_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_global = get_global_size(0);
   const int total_local = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Value Gradient global_id -> v_id, local_id -> q_id
     {
      //--- Shifts
      const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 3 * global_id + 2);
      const int shift_s = RCtoFlat(h_id, global_id, total_heads, total_global, local_id);
      const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, local_id);
      for(int d = 0; d < dimension; d++)
        {
         float grad = 0;
         if(mask_future == 0 || local_id <= global_id)
           {
            float score = IsNaNOrInf(scores[shift_s], 0);
            if(score > 0)
               grad = IsNaNOrInf(score * gradients[shift_out + d], 0);
           }
         grad = LocalSum(grad, 1, temp);
         if(local_id == 0)
            qkv_gr[shift_v + d] = grad;
        }
     }
//--- Query Gradient global_id -> q_id, local_id -> k_id/v_id
     {
      //--- Shifts
      const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, 3 * global_id);
      const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 3 * local_id + 1);
      const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 3 * local_id + 2);
      const int shift_s = RCtoFlat(h_id, local_id, total_heads, total_local, global_id);
      const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, global_id);
      //--- 1. Score grad
      float grad_s = 0;
      if(mask_future == 0 || global_id <= local_id)
         for(int d = 0; d < dimension; d++)
           {
            float val = IsNaNOrInf(qkv[shift_v + d], 0);
            if(val == 0)
               continue;
            grad_s += IsNaNOrInf(qkv[shift_v + d] * gradients[shift_out + d], 0);
           }
      //--- 2. SoftMax grad
      grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
      if(global_id == local_id)
         diag_bias_gr[global_id] = grad_s;
      //--- 3. Query grad
      for(int d = 0; d < dimension; d++)
        {
         float grad = 0;
         if(mask_future == 0 || global_id <= local_id)
           {
            float key = IsNaNOrInf(qkv[shift_k + d], 0);
            if(key != 0)
               grad = key * grad_s;
           }
         grad = LocalSum(grad, 1, temp);
         if(local_id == 0)
            qkv_gr[shift_q + d] = grad;
        }
     }
//--- Key Gradient global_id -> k_id, local_id -> score_id/v_id/dimension
     {
      //--- Shifts
      const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 3 * global_id + 1);
      const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 3 * local_id + 2);
      const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, local_id);
      float grad = 0;
      for(int q_id = 0; q_id < total_local; q_id++)
        {
         //--- 1. Score grad local_id -> score_id/v_id
         float grad_s = 0;
         const int shift_s = RCtoFlat(h_id, local_id, total_heads, total_local, q_id);
         int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, 3 * q_id);
         if(mask_future == 0 || q_id <= local_id)
            for(int d = 0; d < dimension; d++)
              {
               float val = IsNaNOrInf(qkv[shift_v + d], 0);
               if(val == 0)
                  continue;
               grad_s += IsNaNOrInf(val * gradients[shift_q + d], 0);
              }
         //--- 2. SoftMax grad
         grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
         BarrierLoc
         if(global_id == local_id)
            temp[0] = grad_s;
         BarrierLoc
         grad_s = temp[0];
         //--- 3. Key grad local_id -> dimension
         if(local_id < dimension)
           {
            float query = IsNaNOrInf(qkv[shift_q + local_id], 0);
            if(query != 0)
               grad += IsNaNOrInf(query * grad_s, 0);
           }
        }
      if(local_id < dimension)
         qkv_gr[shift_k + local_id] = IsNaNOrInf(grad, 0);
     }
  }
//+------------------------------------------------------------------+
//| Implements the stfs kernel on the GPU.                           |
//+------------------------------------------------------------------+
__kernel void STFS(__global const float* inputs,
                   __global const float* mask_time,
                   __global const float* mask_spatial,
                   __global float* outputs
                  )
  {
   const size_t time_id = get_global_id(0);
   const size_t spat_id = get_global_id(1);
   const size_t head = get_local_id(2);
   const size_t total_times = get_global_size(0);
   const size_t total_spats = get_global_size(1);
   const size_t total_heads = get_local_size(2);
//---
   __local float temp[3];
//---
   const int shift_in = RCtoFlat(time_id, spat_id, total_times, total_spats, 1);
   const int shift_out = RCtoFlat(time_id, spat_id, total_times, total_spats, head);
//---
   switch(head)
     {
      case 0:
         temp[0] = IsNaNOrInf(inputs[shift_in], 0);
         break;
      case 1:
         temp[1] = IsNaNOrInf(mask_time[time_id], 0);
         break;
      case 2:
         temp[2] = IsNaNOrInf(mask_spatial[spat_id], 0);
         break;
     }
   BarrierLoc
   float out = temp[0];
   if(out != 0)
      switch(head)
        {
         case 1:
            out *= temp[1];
            break;
         case 2:
            out *= (1 - temp[1]);
            break;
         case 3:
            out *= temp[2];
            break;
         case 4:
            out *= (1 - temp[2]);
            break;
        }
//---
   outputs[shift_out] = IsNaNOrInf(out, 0);
  }
//+------------------------------------------------------------------+
//| Computes the stfs grad gradient contributions on the GPU.        |
//+------------------------------------------------------------------+
__kernel void STFSGrad(__global float* inputs_gr,
                       __global const float* mask_time,
                       __global const float* mask_spatial,
                       __global const float* outputs_gr
                      )
  {
   const size_t time_id = get_global_id(0);
   const size_t spat_id = get_global_id(1);
   const size_t head = get_local_id(2);
   const size_t total_times = get_global_size(0);
   const size_t total_spats = get_global_size(1);
   const size_t total_heads = get_local_size(2);
//---
   __local float temp[5];
//---
   const int shift_in = RCtoFlat(time_id, spat_id, total_times, total_spats, 1);
   const int shift_out = RCtoFlat(time_id, spat_id, total_times, total_spats, head);
//---
   switch(head)
     {
      case 0:
         temp[1] = IsNaNOrInf(mask_time[time_id], 0);
         break;
      case 1:
         temp[2] = IsNaNOrInf(mask_spatial[spat_id], 0);
         break;
     }
   BarrierLoc
   float grad = IsNaNOrInf(outputs_gr[shift_out], 0);
   if(grad != 0)
      switch(head)
        {
         case 1:
            grad *= temp[1];
            break;
         case 2:
            grad *= (1 - temp[1]);
            break;
         case 3:
            grad *= temp[2];
            break;
         case 4:
            grad *= (1 - temp[2]);
            break;
        }
//---
   grad = LocalSum(grad, 2, temp);
   BarrierLoc
   if(head == 0)
      inputs_gr[shift_in] = grad;
  }
//+------------------------------------------------------------------+
//| Applies the add to stack operation to the GPU buffers.           |
//+------------------------------------------------------------------+
__kernel void AddToStack(__global const float* inputs,
                         __global float* stack,
                         const int stack_size)
  {
   const size_t id = get_global_id(0);
   const size_t loc_id = get_local_id(1);
   const size_t var = get_global_id(2);
   const size_t dimension = get_global_size(0);
   const size_t total_loc = get_local_size(1);
   const size_t variables = get_global_size(2);
//---
   const int total = (stack_size + total_loc - 1) / total_loc;
   for(int i = total; i >= 0; i--)
     {
      float inp = 0;
      if(i == 0 && loc_id == 0)
         inp = IsNaNOrInf(inputs[RCtoFlat(var, id, variables, dimension, 0)], 0);
      else
         if((i * total_loc + loc_id) < stack_size)
           {
            int shift = RCtoFlat(i * total_loc + loc_id - 1, id, stack_size, dimension, var);
            inp = IsNaNOrInf(stack[shift], 0);
           }
      BarrierLoc
      if((i * total_loc + loc_id) < stack_size)
        {
         int shift = RCtoFlat(i * total_loc + loc_id, id, stack_size, dimension, var);
         stack[shift] = inp;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the aggregation by time kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void AggregationByTime(__global const float* inputs,
                                __global const float* stack,
                                __global float* outputs,
                                const int stack_size,
                                const int levels
                               )
  {
   const size_t id = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t dimension = get_global_size(0);
   const size_t variables = get_global_size(1);
//---
   float val = IsNaNOrInf(inputs[RCtoFlat(var, id, variables, dimension, 0)], 0);
   outputs[RCtoFlat(var, id, variables, dimension, 0)] = val;
   for(int l = 1; l < levels; l++)
     {
      int total = 1 << l;
      int start = total - 1;
      val /= total;
      for(int s = 0; s < total; s++)
        {
         if(s + start >= stack_size)
            continue;
         val += IsNaNOrInf(stack[RCtoFlat(var, id, variables * levels, dimension, start + s)] / total, 0);
        }
      outputs[RCtoFlat(var, id, variables, dimension, l)] = IsNaNOrInf(val, 0);
     }
  }
//+------------------------------------------------------------------+
//| Computes the aggregation by time grad gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void AggregationByTimeGrad(__global float* inputs_gr,
                                    __global const float* outputs_gr,
                                    const int levels
                                   )
  {
   const size_t id = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t dimension = get_global_size(0);
   const size_t variables = get_global_size(1);
//---
   float grad = 0;
   for(int l = 0; l < levels; l++)
     {
      int total = 1 << l;
      grad += IsNaNOrInf(outputs_gr[RCtoFlat(var, id, variables, dimension, l)] / total, 0);
     }
   inputs_gr[RCtoFlat(var, id, variables, dimension, 0)] = IsNaNOrInf(grad, 0);
  }
//+------------------------------------------------------------------+
//| Runs the gru kernel on the GPU.                                  |
//+------------------------------------------------------------------+
__kernel void GRU(__global const float* XH,
                  __global const float* prev_state,
                  __global float* outputs
                 )
  {
   const size_t id = get_global_id(0);
   const size_t d  = get_global_id(1);
   const size_t units = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   const float xz = IsNaNOrInf(XH[RCtoFlat(0, d, 6, dimension, id)], 0);
   const float xr = IsNaNOrInf(XH[RCtoFlat(1, d, 6, dimension, id)], 0);
   const float xh = IsNaNOrInf(XH[RCtoFlat(2, d, 6, dimension, id)], 0);
   const float hz = IsNaNOrInf(XH[RCtoFlat(3, d, 6, dimension, id)], 0);
   const float hr = IsNaNOrInf(XH[RCtoFlat(4, d, 6, dimension, id)], 0);
   const float hh = IsNaNOrInf(XH[RCtoFlat(5, d, 6, dimension, id)], 0);
   const float prev = IsNaNOrInf(prev_state[RCtoFlat(id, d, units, dimension, 0)], 0);
//---
   float r = fActivation(xr + hr, ActFunc_SIGMOID);
   float z = fActivation(xz + hz, ActFunc_SIGMOID);
   float ht = fActivation(r * hh + xh, ActFunc_TANH);
   float out = (1 - z) * prev + z * ht;
//---
   outputs[RCtoFlat(id, d, units, dimension, 0)] = IsNaNOrInf(out, 0);
  }
//+------------------------------------------------------------------+
//| Computes the gru grad gradient contributions on the GPU.         |
//+------------------------------------------------------------------+
__kernel void GRU_Grad(__global const float* XH,
                       __global float * XH_gr,
                       __global const float* prev_state,
                       __global const float* outputs_gr
                      )
  {
   const size_t id = get_global_id(0);
   const size_t d  = get_global_id(1);
   const size_t units = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   const float xz = IsNaNOrInf(XH[RCtoFlat(0, d, 6, dimension, id)], 0);
   const float xr = IsNaNOrInf(XH[RCtoFlat(1, d, 6, dimension, id)], 0);
   const float xh = IsNaNOrInf(XH[RCtoFlat(2, d, 6, dimension, id)], 0);
   const float hz = IsNaNOrInf(XH[RCtoFlat(3, d, 6, dimension, id)], 0);
   const float hr = IsNaNOrInf(XH[RCtoFlat(4, d, 6, dimension, id)], 0);
   const float hh = IsNaNOrInf(XH[RCtoFlat(5, d, 6, dimension, id)], 0);
   const float prev = IsNaNOrInf(prev_state[RCtoFlat(id, d, units, dimension, 0)], 0);
   const float grad = IsNaNOrInf(outputs_gr[RCtoFlat(id, d, units, dimension, 0)], 0);
//---
   float r = fActivation(xr + hr, ActFunc_SIGMOID);
   float z = fActivation(xz + hz, ActFunc_SIGMOID);
   float ht = fActivation(r * hh + xh, ActFunc_TANH);
//---
   float ht_grad = IsNaNOrInf(grad * z, 0);
   float z_grad = IsNaNOrInf(grad * (ht - prev), 0);
   float xh_grad = Deactivation(ht_grad, ht, ActFunc_TANH);
   float hh_grad = IsNaNOrInf(xh_grad * r, 0);
   float r_grad = IsNaNOrInf(xh_grad * hh, 0);
   float xz_grad = Deactivation(z_grad, z, ActFunc_SIGMOID);
   float hz_grad = xz_grad;
   float xr_grad = Deactivation(r_grad, r, ActFunc_SIGMOID);
   float hr_grad = xr_grad;
//---
   XH_gr[RCtoFlat(0, d, 6, dimension, id)] = IsNaNOrInf(xz_grad, 0);
   XH_gr[RCtoFlat(1, d, 6, dimension, id)] = IsNaNOrInf(xr_grad, 0);
   XH_gr[RCtoFlat(2, d, 6, dimension, id)] = IsNaNOrInf(xh_grad, 0);
   XH_gr[RCtoFlat(3, d, 6, dimension, id)] = IsNaNOrInf(hz_grad, 0);
   XH_gr[RCtoFlat(4, d, 6, dimension, id)] = IsNaNOrInf(hr_grad, 0);
   XH_gr[RCtoFlat(5, d, 6, dimension, id)] = IsNaNOrInf(hh_grad, 0);
  }
//+------------------------------------------------------------------+
//| Implements the scalar to vector kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void ScalarToVector(__global const float* scalar,
                             __global const float* vector_in,
                             __global float* vector_out
                            )
  {
   const size_t vec = get_global_id(0);
   const size_t d  = get_global_id(1);
   const size_t vectors = get_global_size(0);
   const size_t dimension = get_global_size(1);
//---
   float sc = IsNaNOrInf(scalar[vec], 0.0f);
   int shift = RCtoFlat(vec, d, vectors, dimension, 0);
   float v = IsNaNOrInf(vector_in[shift], 0.0f);
//---
   vector_out[shift] = IsNaNOrInf(sc * v, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the scalar to vector grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void ScalarToVectorGrad(__global const float* scalar,
                                 __global float* scalar_gr,
                                 __global const float* vector_in,
                                 __global float* vector_in_gr,
                                 __global float* vector_out_gr,
                                 const int dimension
                                )
  {
   const size_t vec = get_global_id(0);
   const size_t loc  = get_local_id(1);
   const size_t vectors = get_global_size(0);
   const size_t total_loc = get_local_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   if(loc == 0)
      temp[0] = IsNaNOrInf(scalar[vec], 0.0f);
   BarrierLoc
   float sc = temp[0];
   float sc_gr = 0;
   for(int d = loc; d < dimension; d += total_loc)
     {
      int shift = RCtoFlat(vec, d, vectors, dimension, 0);
      float v = IsNaNOrInf(vector_in[shift], 0.0f);
      float grad = IsNaNOrInf(vector_out_gr[shift], 0.0f);
      vector_in_gr[shift] = IsNaNOrInf(grad * sc, 0.0f);
      sc_gr += IsNaNOrInf(v * grad, 0.0f);
     }
//---
   sc_gr = LocalSum(sc_gr, 1, temp);
   if(loc == 0)
      scalar_gr[vec] = sc_gr;
  }
//+------------------------------------------------------------------+
//| Calculates calc flow on the GPU.                                 |
//+------------------------------------------------------------------+
__kernel void CalcFlow(__global const float* value,
                       __global float* prev_value,
                       __global float* flow
                      )
  {
   const size_t id = get_global_id(0);
   const size_t total = get_global_size(0);
//---
   const float v = IsNaNOrInf(value[id], 0);
   const float p = IsNaNOrInf(prev_value[id], 0);
   flow[id] = IsNaNOrInf(v - p, 0);
   prev_value[id] = v;
  }
//+------------------------------------------------------------------+
//| Implements the dilated correlation kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void DilatedCorrelation(__global const float* feature,
                                 __global const int* shifts,
                                 __global float* correlations,
                                 const int dimension
                                )
  {
   const size_t main = get_global_id(0);
   const size_t loc = get_local_id(1);
   const size_t sh = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t total_loc = get_local_size(1);
   const size_t total_corr = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   const int slave = main + shifts[sh >> 1] * ((sh & 1) ? -1 : 1);
   if(slave < 0 || slave >= units)
     {
      if(loc == 0)
         correlations[RCtoFlat(main, sh, units, total_corr, 0)] = 0;
      return;
     }
//---
   float result = 0.0f;
   for(int d = loc; d < dimension; d += total_loc)
     {
      float value_main = IsNaNOrInf(feature[RCtoFlat(main, d, units, dimension, 0)], 0);
      float value_slave = IsNaNOrInf(feature[RCtoFlat(slave, d, units, dimension, 0)], 0);
      result += IsNaNOrInf(value_main * value_slave, 0);
     }
   result = LocalSum(result, 1, temp);
//---
   if(loc == 0)
      correlations[RCtoFlat(main, sh, units, total_corr, 0)] = result;
  }
//+------------------------------------------------------------------+
//| Computes the dilated correlation grad gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void DilatedCorrelationGrad(__global const float* feature,
                                     __global float*       feature_gr,
                                     __global const int*   shifts,
                                     __global const float* corr_gr,
                                     const int total_corr
                                    )
  {
   const size_t id = get_global_id(0);
   const size_t loc = get_local_id(1);
   const size_t d = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t total_loc = get_local_size(1);
   const size_t dimension = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   float result = 0.0f;
   for(int sh = loc; sh < total_corr; sh += total_loc)
     {
      const int offset = shifts[sh >> 1];
      const int sign   = (sh & 1) ? -1 : +1;
      // id — main
      int slave = id + sign * offset;
      if(slave >= 0 && slave < units)
        {
         float g = corr_gr[RCtoFlat(id, sh, units, total_corr, 0)];
         result += IsNaNOrInf(g * feature[RCtoFlat(slave, d, units, dimension, 0)], 0.0f);
        }
      // id — slave
      int main = id - sign * offset;
      if(main >= 0 && main < units)
        {
         float g = corr_gr[RCtoFlat(main, sh, units, total_corr, 0)];
         result += IsNaNOrInf(g * feature[RCtoFlat(main, d, units, dimension, 0)], 0.0f);
        }
     }
   result = LocalSum(result, 1, temp);
   if(loc == 0)
      feature_gr[RCtoFlat(id, d, units, dimension, 0)] = result;
  }
//+------------------------------------------------------------------+
//| Implements the dilated difference kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void DilatedDifference(__global const float* feature,
                                __global const int* shifts,
                                __global float* differences
                               )
  {
   const size_t main = get_global_id(0);
   const size_t sh = get_global_id(1);
   const size_t d = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t total_shifts = get_global_size(1);
   const size_t dimension = get_global_size(2);
//---
   const int slave = main + shifts[sh];
   if(slave < 0 || slave >= units)
     {
      differences[RCtoFlat(main, d, units, dimension, sh)] = 0;
      return;
     }
//---
   float value_main = IsNaNOrInf(feature[RCtoFlat(main, d, units, dimension, 0)], 0);
   float value_slave = IsNaNOrInf(feature[RCtoFlat(slave, d, units, dimension, 0)], 0);
   float result = value_main - value_slave;
//---
   differences[RCtoFlat(main, d, units, dimension, sh)] = result;
  }
//+------------------------------------------------------------------+
//| Computes the dilated difference grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void DilatedDifferenceGrad(__global const float* feature,
                                    __global float*       feature_gr,
                                    __global const int*   shifts,
                                    __global const float* differences_gr,
                                    const int total_shifts
                                   )
  {
   const size_t id = get_global_id(0);
   const size_t loc = get_local_id(1);
   const size_t d = get_global_id(2);
   const size_t units = get_global_size(0);
   const size_t total_loc = get_local_size(1);
   const size_t dimension = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   float result = 0.0f;
   for(int sh = loc; sh < total_shifts; sh += total_loc)
     {
      const int offset = shifts[sh];
      int slave = id + offset;
      if(slave >= 0 && slave < units)
        {
         // id — main
         result += IsNaNOrInf(differences_gr[RCtoFlat(id, d, units, dimension, sh)], 0.0f);
         // id — slave
         result -= IsNaNOrInf(differences_gr[RCtoFlat(slave, d, units, dimension, sh)], 0.0f);
        }
     }
   result = LocalSum(result, 1, temp);
   if(loc == 0)
      feature_gr[RCtoFlat(id, d, units, dimension, 0)] = result;
  }
//+------------------------------------------------------------------+
//| Implements the perturbed matrix kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void PerturbedMatrix(__global const float* inputs,
                              __global const float* perturb,
                              __global float*       output,
                              const float perturb_mult)
  {
   const size_t id = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
//---
   int shift = RCtoFlat(var, id, variables, total, 0);
   output[shift] = IsNaNOrInf(inputs[shift] + perturb[shift] * perturb_mult, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the perturbed matrix grad gradient contributions on the |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void PerturbedMatrixGrad(__global float*       inputs_gr,
                                  __global float*       perturb_gr,
                                  __global const float* output_gr,
                                  const float perturb_mult)
  {
   const size_t id = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
//---
   int shift = RCtoFlat(var, id, variables, total, 0);
   float grad = IsNaNOrInf(output_gr[shift], 0.0f);
   inputs_gr[shift] = grad;
   perturb_gr[shift] = IsNaNOrInf(perturb_mult * grad, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the linear upsample kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void LinearUpsample(__global const float* data,
                             __global float*       upsample)
  {
   const size_t id_ltr = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t id_htr = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t dimension_htr = get_global_size(2);
//---
   const float ltr = IsNaNOrInf(data[RCtoFlat(id_ltr, var, total, variables, 0)], 0.0f);
   const float prev_ltr = (id_ltr > 0 ? IsNaNOrInf(data[RCtoFlat(id_ltr - 1, var, total, variables, 0)], 0.0f) : 0.0f);
   const float htr = (id_htr < (dimension_htr - 1) ?
                      (float)id_htr / (float)(dimension_htr - 1)  * (ltr - prev_ltr) + prev_ltr :
                      ltr);
//---
   upsample[RCtoFlat(id_htr, var, dimension_htr, variables, id_ltr)] = IsNaNOrInf(htr, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the linear upsample grad gradient contributions on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void LinearUpsampleGrad(__global float*         data_gr,
                                 __global const float*   upsample_gr,
                                 const int dimension_htr)
  {
   const size_t id_ltr = get_global_id(0);
   const size_t var = get_global_id(1);
   const size_t id_loc = get_local_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t total_loc = get_local_size(2);
   float grad = 0.0f;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
// --- main ltr
     {
      for(int id_htr = id_loc; id_htr < dimension_htr; id_htr += total_loc)
        {
         const float g =
            upsample_gr[RCtoFlat(id_htr, var, dimension_htr, variables, id_ltr)];
         if(id_htr < dimension_htr - 1)
           {
            const float t = (float)id_htr / (float)(dimension_htr - 1);
            grad += g * t;
           }
         else
            grad += g;
        }
     }
// --- prev ltr
   if(id_ltr + 1 < total)
      for(int id_htr = id_loc; id_htr < dimension_htr; id_htr += total_loc)
         if(id_htr < dimension_htr - 1)
           {
            const float g =
               upsample_gr[RCtoFlat(id_htr, var, dimension_htr, variables, id_ltr + 1)];
            const float t = (float)id_htr / (float)(dimension_htr - 1);
            grad += g * (1.0f - t);
           }
// ---
   grad = LocalSum(grad, 2, temp);
   if(id_loc == 0)
      data_gr[RCtoFlat(id_ltr, var, total, variables, 0)] = grad;
  }
//+------------------------------------------------------------------+
//| Implements the mix experts predict kernel on the GPU.            |
//+------------------------------------------------------------------+
__kernel void MixExpertsPredict(__global const float4* __attribute__((aligned(16))) experts,
                                __global float* outputs
                               )
  {
   const size_t id = get_global_id(0);
//---
   float4 expert = experts[id];
   float mu = IsNaNOrInf(expert.s0, 0.0f);
   float alpha = fActivation(expert.s1, ActFunc_SoftPlus);
   float sigma = fActivation(expert.s2, ActFunc_SoftPlus);
   float txi = fActivation(expert.s3, ActFunc_TANH);
   float out = mu + alpha * sigma * txi;
//---
   outputs[id] = IsNaNOrInf(out, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the mix experts predict grad gradient contributions on  |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void MixExpertsPredictGrad(__global const float4* __attribute__((aligned(16))) experts,
                                    __global float4* __attribute__((aligned(16))) experts_gr,
                                    __global const float* outputs_gr
                                   )
  {
   const size_t id = get_global_id(0);
//---
   float4 expert = experts[id];
   float grad = IsNaNOrInf(outputs_gr[id], 0.0f);
   float4 expert_gr = (float4)0.0f;
//---
   float alpha = fActivation(expert.s1, ActFunc_SoftPlus);
   float sigma = fActivation(expert.s2, ActFunc_SoftPlus);
   float txi = fActivation(expert.s3, ActFunc_TANH);
//---
   float mu_grad = grad;
   float alpha_grad = Deactivation(grad * sigma * txi, alpha, ActFunc_SoftPlus);
   float sigma_grad = Deactivation(grad * alpha * txi, sigma, ActFunc_SoftPlus);
   float txi_grad = Deactivation(grad * sigma * alpha, txi, ActFunc_TANH);
//---
   experts_gr[id] = (float4)(mu_grad, alpha_grad, sigma_grad, txi_grad);
  }
//+------------------------------------------------------------------+
//| Implements the mhfat kernel on the GPU.                          |
//+------------------------------------------------------------------+
__kernel void MHFAT(__global const float *q,
                    __global const float *kv,
                    __global const float *scale,
                    __global float *scores,
                    __global float *out,
                    const int dimension,
                    const int mask_future
                   )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_k = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Shifts
   const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
   const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 2 * k_id);
   const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 2 * k_id + 1);
   const int shift_s = RCtoFlat(h_id, k_id, total_heads, total_k, q_id);
   const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
//--- Score
   float score = 0;
   if(mask_future == 0 || q_id <= k_id)
     {
      float sc = IsNaNOrInf(scale[shift_s], 0.0f);
      if(sc != 0)
        {
         for(int d = 0; d < dimension; d++)
           {
            float q_ = IsNaNOrInf(q[shift_q + d], 0.0f);
            if(q_ == 0)
               continue;
            float k = IsNaNOrInf(kv[shift_k + d], 0.0f);
            if(k == 0)
               continue;
            score += q_ * k;
           }
         score *= sc;
        }
      else
         score = MIN_VALUE;
     }
   else
      score = MIN_VALUE;
//--- norm score
   score = LocalSoftMax(score, 1, temp);
   scores[shift_s] = score;
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val = 0;
      if(score > 0)
        {
         float v = IsNaNOrInf(kv[shift_v + d], 0);
         if(v != 0)
            val = v * score;
        }
      val = LocalSum(val, 1, temp);
      if(k_id == 0)
         out[shift_out + d] = val;
     }
  }
//+------------------------------------------------------------------+
//| Computes the mhfat grad gradient contributions on the GPU.       |
//+------------------------------------------------------------------+
__kernel void MHFATGrad(__global const float *q,
                        __global float *q_gr,
                        __global const float *kv,
                        __global float *kv_gr,
                        __global const float *scale,
                        __global float *scale_gr,
                        __global const float *scores,
                        __global const float *gradients,
                        const int dimension,
                        const int total_k,
                        const int mask_future
                       )
  {
//--- init
   const int global_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_global = get_global_size(0);
   const int total_local = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Value Gradient global_id -> v_id, local_id -> q_id
   for(int d = 0; d < dimension; d++)
     {
      for(int v_id = global_id; v_id < total_k; v_id += total_global)
        {
         float grad = 0;
         //--- Shifts
         const int shift_v = RCtoFlat(h_id, d, total_heads, dimension, 2 * v_id + 1);
         for(int q_id = 0; q_id < total_global; q_id += total_local)
           {
            const int shift_s = RCtoFlat(h_id, v_id, total_heads, total_k, q_id + local_id);
            const int shift_out = RCtoFlat(h_id, d, total_heads, dimension, q_id + local_id);
            if((q_id + local_id) < total_global)
               if(mask_future == 0 || (q_id + local_id) <= v_id)
                 {
                  float score = IsNaNOrInf(scores[shift_s], 0.0f);
                  if(score > 0)
                     grad += IsNaNOrInf(score * gradients[shift_out], 0.0f);
                 }
           }
         grad = LocalSum(grad, 1, temp);
         if(local_id == 0)
            kv_gr[shift_v] = grad;
        }
     }
//--- Query Gradient global_id -> q_id, local_id -> k_id/v_id
   for(int d_q = 0; d_q < dimension; d_q++)
     {
      //--- Shifts
      const int shift_q = RCtoFlat(h_id, d_q, total_heads, dimension, global_id);
      const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, global_id);
      float grad = 0;
      for(int id = 0; id < total_k; id += total_local)
        {
         int k_id = id + local_id;
         const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 2 * k_id);
         const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 2 * k_id + 1);
         const int shift_s = RCtoFlat(h_id, k_id, total_heads, total_k, global_id);
         //--- 1. Score grad
         float grad_s = 0;
         float score = 0;
         float sc = 0;
         if(k_id < total_k)
           {
            if(mask_future == 0 || global_id <= local_id)
               for(int d = 0; d < dimension; d++)
                 {
                  float val = IsNaNOrInf(kv[shift_v + d], 0);
                  if(val == 0.0f)
                     continue;
                  grad_s += IsNaNOrInf(kv[shift_v + d] * gradients[shift_out + d], 0);
                 }
            score = scores[shift_s];
            sc = IsNaNOrInf(scale[shift_s], 0.0f);
           }
         //--- 2. SoftMax grad
         grad_s = LocalSoftMaxGrad(score, grad_s, 1, temp);
         float grad_sc = LocalSum(score * grad_s, 1, temp);
         if(local_id == 0 && k_id < total_k)
           {
            if(sc != 0.0f)
               scale_gr[shift_s] = grad_sc * log(scores[shift_s]) / sc;
            else
               scale_gr[shift_s] = grad_sc;
           }
         grad_s *= sc;
         //--- 3. Query grad
         if(grad_s != 0.0f)
            if(mask_future == 0 || global_id <= k_id)
              {
               float key = IsNaNOrInf(kv[shift_k + d_q], 0.0f);
               if(key != 0.0f)
                  grad += key * grad_s;
              }
        }
      grad = LocalSum(grad, 1, temp);
      if(local_id == 0)
         q_gr[shift_q] = grad;
     }
//--- Key Gradient global_id -> k_id, local_id -> score_id/v_id/dimension
   for(int k_id = global_id; k_id < total_k; k_id += total_global)
     {
      //--- Shifts
      const int shift_k = RCtoFlat(h_id, 0, total_heads, dimension, 2 * k_id);
      const int shift_v = RCtoFlat(h_id, 0, total_heads, dimension, 2 * local_id + 1);
      const int shift_out = RCtoFlat(h_id, 0, total_heads, dimension, local_id);
      float grad = 0;
      for(int q_id = 0; q_id < total_global; q_id++)
        {
         //--- 1. Score grad local_id -> score_id/v_id
         float grad_s = 0;
         const int shift_s = RCtoFlat(h_id, local_id, total_heads, total_k, q_id);
         int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
         float score = 0;
         float sc = 0;
         if(local_id < total_k)
           {
            if(mask_future == 0 || q_id <= local_id)
               for(int d = 0; d < dimension; d++)
                 {
                  float val = IsNaNOrInf(kv[shift_v + d], 0);
                  if(val == 0)
                     continue;
                  grad_s += IsNaNOrInf(val * gradients[shift_q + d], 0);
                 }
            score = scores[shift_s];
            sc = IsNaNOrInf(scale[shift_s], 0.0f);
           }
         //--- 2. SoftMax grad
         grad_s = LocalSoftMaxGrad(scores[shift_s], grad_s, 1, temp);
         grad_s *= sc;
         //--- 3. Key grad local_id -> dimension
         if(local_id < dimension)
           {
            float query = IsNaNOrInf(q[shift_q + local_id], 0);
            if(query != 0)
               grad += IsNaNOrInf(query * grad_s, 0);
           }
        }
      if(local_id < dimension)
         kv_gr[shift_k + local_id] = IsNaNOrInf(grad, 0);
     }
  }
//+------------------------------------------------------------------+
//| Implements the sparse concatenate kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void SparseConcatenate(__global const float *sparse_index,
                                __global const float *sparse_data,
                                __global const float *full,
                                __global float *result,
                                const int full_rows
                               )
  {
   const size_t sparse_row = get_global_id(0);
   const size_t sparse_col = get_global_id(1);
   const size_t full_col = get_global_id(2);
   const size_t sparse_rows = get_global_size(0);
   const size_t sparse_cols = get_global_size(1);
   const size_t full_cols = get_global_size(2);
//---
   const int shift_sparse = RCtoFlat(sparse_row, sparse_col, sparse_rows, sparse_cols, 0);
   const int full_row = sparse_index[shift_sparse];
   const int shift_full = RCtoFlat(full_row, full_col, full_rows, full_cols, 0);
   const int shift_out = RCtoFlat(sparse_col, full_col, sparse_cols, full_cols, sparse_row);
//---
   float res = (full_row >= 0 && full_row < full_rows ?
                IsNaNOrInf(sparse_data[shift_sparse] * full[shift_full], 0) : 0.0f);
   result[shift_out] = res;
  }
//+------------------------------------------------------------------+
//| Computes the sparse concatenate grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void SparseConcatenateGrad(__global const float *sparse_index,
                                    __global const float *sparse_data,
                                    __global float *sparse_gr,
                                    __global const float *full,
                                    __global float *full_gr,
                                    __global const float *result_gr,
                                    const int sparse_rows,
                                    const int sparse_cols,
                                    const int full_rows,
                                    const int full_cols
                                   )
  {
   const size_t row_id = get_global_id(0);
   const size_t local_id = get_local_id(1);
   const size_t col_id = get_global_id(2);
   const size_t total_rows = get_global_size(0);
   const size_t total_local = get_local_size(1);
   const size_t total_cols = get_global_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//--- Calce sparse gradient
   if(row_id < sparse_rows && col_id < sparse_cols)
     {
      float grad = 0;
      int shift_sparse = 0;
      if(local_id == 0)
        {
         shift_sparse = RCtoFlat(row_id, col_id, sparse_rows, sparse_cols, 0);
         Temp[0] = sparse_index[shift_sparse];
        }
      BarrierLoc
      uint full_row = (uint)Temp[0];
      if(full_row < (uint)full_rows)
         for(int i = local_id; i < full_cols; i += total_local)
           {
            int shift_out = RCtoFlat(col_id, i, sparse_cols, full_cols, row_id);
            int shift_full = RCtoFlat(full_row, i, full_rows, full_cols, 0);
            grad += IsNaNOrInf(result_gr[shift_out] * full[shift_full], 0.0f);
           }
      grad = LocalSum(grad, 1, Temp);
      if(local_id == 0)
         sparse_gr[shift_sparse] = grad;
     }
//--- Calce full gradient
   if(row_id < full_rows && col_id < full_cols)
     {
      float grad = 0;
      for(int r = 0; r < sparse_rows; r ++)
        {
         float s = 0;
         for(int c = local_id; c < sparse_cols; c += total_local)
           {
            int shift_sparse = RCtoFlat(r, c, sparse_rows, sparse_cols, 0);
            if((uint)sparse_index[shift_sparse] == (uint)row_id)
              {
               s = sparse_data[shift_sparse];
               int shift_out = RCtoFlat(c, col_id, sparse_cols, full_cols, r);
               grad += IsNaNOrInf(s * result_gr[shift_out], 0.0f);
               break;
              }
           }
        }
      grad = LocalSum(grad, 1, Temp);
      if(local_id == 0)
        {
         int shift_full = RCtoFlat(row_id, col_id, full_rows, full_cols, 0);
         full_gr[shift_full] = grad;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the mh flash attention kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void MHFlashAttention(__global const float *query,
                               __global const float *key_value,
                               __global float *logsumexp,
                               __global float *output,
                               const int dimension,
                               const int total_kv,
                               const int mask_future
                              )
  {
//--- init
   const int q_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_loc = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   __local float4 temp4[LOCAL_ARRAY_SIZE];
//---
   const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
   float prev_max = MIN_VALUE;
   float sumexp = 0;
   float out = 0;
   for(int id = 0; id < total_kv; id += total_loc)
     {
      int k_id = id + local_id;
      const int shift_k = RCtoFlat(h_id, 0, 2 * total_heads, dimension, k_id);
      const int shift_v = RCtoFlat(h_id + total_heads, 0, 2 * total_heads, dimension, k_id);
      //--- Score
      float score = 0;
      if(k_id < total_kv && (mask_future == 0 || q_id <= k_id))
        {
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q = IsNaNOrInf4((float4)(
                                      (d < dimension ? query[shift_q + d] : 0.0f),
                                      ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                   ), 0.0f);
            float4 k = IsNaNOrInf4((float4)(
                                      (d < dimension ? key_value[shift_k + d] : 0.0f),
                                      ((d + 1) < dimension ? key_value[shift_k + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? key_value[shift_k + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? key_value[shift_k + d + 3] : 0.0f)
                                   ), 0.0f);
            score += IsNaNOrInf(dot(q, k), 0.0f);
           }
         score /= sqrt((float)dimension);
        }
      else
         score = MIN_VALUE;
      //--- norm score
      float max = fmax(prev_max, LocalMax(score, 1, temp));
      if(score > MIN_VALUE)
         score = exp(score - max);
      else
         score = 0.0f;
      if(sumexp == 0.0f)
         sumexp = LocalSum(score, 1, temp);
      else
         sumexp = IsNaNOrInf(exp(prev_max - max) * sumexp + LocalSum(score, 1, temp), 0.0f);
      for(int d = 0; d < dimension; d += 4)
        {
         float4 val = (float4)0.0f;
         if(score > 0.0f && k_id < total_kv)
           {
            float4 v = (float4)(
                          (d < dimension ? key_value[shift_v + d] : 0.0f),
                          ((d + 1) < dimension ? key_value[shift_v + d + 1] : 0.0f),
                          ((d + 2) < dimension ? key_value[shift_v + d + 2] : 0.0f),
                          ((d + 3) < dimension ? key_value[shift_v + d + 3] : 0.0f)
                       );
            val = IsNaNOrInf4(v * score, 0.0f);
           }
         val = LocalSum4(val, 1, temp4);
         int idx = local_id - d;
         if(idx >= 0 && idx < 4)
           {
            if(out != 0.0f)
               out = IsNaNOrInf(exp(prev_max - max) * out + vget4(val, idx), 0.0f);
            else
               out = vget4(val, idx);
           }
        }
      prev_max = max;
     }
   if(local_id < dimension)
     {
      if(sumexp > 0.0f)
         output[shift_q + local_id] = IsNaNOrInf(out / sumexp, 0.0f);
      else
         output[shift_q + local_id] = 0.0f;
     }
   if(local_id == 0)
     {
      int shift_logse = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
      if(sumexp > 0.0f)
         logsumexp[shift_logse] = IsNaNOrInf(prev_max + log(sumexp), 0.0f);
      else
         logsumexp[shift_logse] = 0.0f;
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh flash attention grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void MHFlashAttentionGrad(__global const float *query,
                                   __global float *query_gr,
                                   __global const float *key_value,
                                   __global float *key_value_gr,
                                   __global const float *logsumexp,
                                   __global const float *output,
                                   __global const float *output_gr,
                                   const int dimension,
                                   const int total_q,
                                   const int total_kv,
                                   const int mask_future
                                  )
  {
   const int id = get_global_id(0);
   const int d_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_heads = get_global_size(2);
   __local float temp[LOCAL_ARRAY_SIZE];
//--- Query gradient: dQ[q, d]
   if(id < total_q)
     {
      const int q_id = id;
      const int shift_q = RCtoFlat(h_id, d_id, total_heads, dimension, q_id);
      const int shift_lse = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
      const float lse = IsNaNOrInf(logsumexp[shift_lse], 0.0f);
      const float q_d = IsNaNOrInf(query[shift_q], 0.0f);
      const float g_d = IsNaNOrInf(output_gr[shift_q], 0.0f);
      const float o_d = IsNaNOrInf(output[shift_q], 0.0f);
      const float D = LocalSum(IsNaNOrInf(g_d * o_d, 0.0f), 1, temp);
      float grad_q = 0.0f;
      for(int k_id = 0; k_id < total_kv; k_id++)
        {
         if(mask_future != 0 && q_id > k_id)
            continue;
         const int shift_k = RCtoFlat(h_id, d_id, 2 * total_heads, dimension, k_id);
         const int shift_v = RCtoFlat(h_id + total_heads, d_id, 2 * total_heads, dimension, k_id);
         const float k_d = IsNaNOrInf(key_value[shift_k], 0.0f);
         const float v_d = IsNaNOrInf(key_value[shift_v], 0.0f);
         const float s = LocalSum(IsNaNOrInf(q_d * k_d, 0.0f), 1, temp) / sqrt((float)dimension);
         const float p = IsNaNOrInf(exp(clamp(s - lse, -120.0f, 0.0f)), 0.0f);
         if(p == 0.0f)
            continue;
         const float dp = LocalSum(IsNaNOrInf(g_d * v_d, 0.0f), 1, temp);
         const float ds = IsNaNOrInf(p * (dp - D), 0.0f);
         grad_q += IsNaNOrInf(k_d * ds, 0.0f);
        }
      query_gr[shift_q] = IsNaNOrInf(grad_q, 0.0f);
     }
//--- Key & Value gradients: dK[k, d], dV[k, d]
   if(id < total_kv)
     {
      const int k_id = id;
      const int shift_k = RCtoFlat(h_id, d_id, 2 * total_heads, dimension, k_id);
      const int shift_v = RCtoFlat(h_id + total_heads, d_id, 2 * total_heads, dimension, k_id);
      const float k_d = IsNaNOrInf(key_value[shift_k], 0.0f);
      const float v_d = IsNaNOrInf(key_value[shift_v], 0.0f);
      float grad_k = 0.0f;
      float grad_v = 0.0f;
      for(int q_id = 0; q_id < total_q; q_id++)
        {
         if(mask_future != 0 && q_id > k_id)
            continue;
         const int shift_q = RCtoFlat(h_id, d_id, total_heads, dimension, q_id);
         const int shift_lse = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
         const float lse = IsNaNOrInf(logsumexp[shift_lse], 0.0f);
         const float q_d = IsNaNOrInf(query[shift_q], 0.0f);
         const float g_d = IsNaNOrInf(output_gr[shift_q], 0.0f);
         const float o_d = IsNaNOrInf(output[shift_q], 0.0f);
         const float D = LocalSum(IsNaNOrInf(g_d * o_d, 0.0f), 1, temp);
         const float s = LocalSum(IsNaNOrInf(q_d * k_d, 0.0f), 1, temp) / sqrt((float)dimension);
         const float p = IsNaNOrInf(exp(clamp(s - lse, -120.0f, 0.0f)), 0.0f);
         if(p == 0.0f)
            continue;
         const float dp = LocalSum(IsNaNOrInf(g_d * v_d, 0.0f), 1, temp);
         const float ds = IsNaNOrInf(p * (dp - D), 0.0f);
         grad_k += IsNaNOrInf(q_d * ds, 0.0f);
         grad_v += IsNaNOrInf(p * g_d, 0.0f);
        }
      key_value_gr[shift_k] = IsNaNOrInf(grad_k, 0.0f);
      key_value_gr[shift_v] = IsNaNOrInf(grad_v, 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Implements the mh flash stca kernel on the GPU.                  |
//+------------------------------------------------------------------+
__kernel void MHFlashSTCA(__global const float *query,
                          __global const float *X,
                          __global float *logsumexp,
                          __global float *output,
                          const int dimension,
                          const int total_X,
                          const int mask_future
                         )
  {
//--- init
   const int q_id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_q = get_global_size(0);
   const int total_loc = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   __local float4 temp4[LOCAL_ARRAY_SIZE];
//---
   const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
   float prev_max = MIN_VALUE;
   float sumexp = 0;
   float out = 0;
   for(int id = 0; id < total_X; id += total_loc)
     {
      int x_id = id + local_id;
      const int shift_x = RCtoFlat(x_id, 0, total_X, dimension, 0);
      //--- Score
      float score = 0;
      if(x_id < total_X && (mask_future == 0 || q_id <= x_id))
        {
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q = IsNaNOrInf4((float4)(
                                      (d < dimension ? query[shift_q + d] : 0.0f),
                                      ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                   ), 0.0f);
            float4 k = IsNaNOrInf4((float4)(
                                      (d < dimension ? X[shift_x + d] : 0.0f),
                                      ((d + 1) < dimension ? X[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? X[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? X[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            score += IsNaNOrInf(dot(q, k), 0.0f);
           }
         score /= sqrt((float)dimension);
        }
      else
         score = MIN_VALUE;
      //--- norm score
      float max = fmax(prev_max, LocalMax(score, 1, temp));
      if(score > MIN_VALUE)
         score = IsNaNOrInf(exp(score - max), 0.0f);
      else
         score = 0.0f;
      if(sumexp == 0.0f)
         sumexp = LocalSum(score, 1, temp);
      else
         sumexp = IsNaNOrInf(exp(prev_max - max) * sumexp, 0.0f) + LocalSum(score, 1, temp);
      for(int d = 0; d < dimension; d += 4)
        {
         float4 val = (float4)0.0f;
         if(score > 0.0f && x_id < total_X)
           {
            float4 v = IsNaNOrInf4((float4)(
                                      (d < dimension ? X[shift_x + d] : 0.0f),
                                      ((d + 1) < dimension ? X[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? X[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? X[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            val = IsNaNOrInf4(v * score, 0.0f);
           }
         val = LocalSum4(val, 1, temp4);
         float add = 0.0f;
         int idx = local_id - d;
         if(idx >= 0 && idx < 4)
           {
            if(out != 0.0f)
               out = IsNaNOrInf(exp(prev_max - max) * out + vget4(val, idx), 0.0f);
            else
               out = vget4(val, idx);
           }
        }
      prev_max = max;
     }
   if(local_id < dimension)
     {
      if(sumexp > 0.0f)
         output[shift_q + local_id] = IsNaNOrInf(out / sumexp, 0.0f);
      else
         output[shift_q + local_id] = 0.0f;
     }
   if(local_id == 0)
     {
      int shift_logse = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
      if(sumexp > 0.0f)
         logsumexp[shift_logse] = IsNaNOrInf(prev_max + log(sumexp), 0.0f);
      else
         logsumexp[shift_logse] = 0.0f;
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh flash stca grad gradient contributions on the    |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void MHFlashSTCAGrad(__global const float *query,
                              __global float *query_gr,
                              __global const float *X,
                              __global float *X_gr,
                              __global const float *logsumexp,
                              __global const float *output,
                              __global const float *output_gr,
                              const int dimension,
                              const int total_q,
                              const int total_X,
                              const int mask_future
                             )
  {
   const int id = get_global_id(0);
   const int local_id = get_local_id(1);
   const int h_id = get_global_id(2);
   const int total_loc = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   __local float4 temp4[LOCAL_ARRAY_SIZE];
//--- Query gradient: dQ[q, d]
   if(id < total_q)
     {
      float grad_q = 0.0f;
      const int q_id = id;
      const int shift_q = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
      const int shift_lse = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
      const float lse = IsNaNOrInf(logsumexp[shift_lse], 0.0f);
      float D = 0;
      for(int d = 0; d < dimension; d += 4)
        {
         float4 g_d = IsNaNOrInf4((float4)(
                                     (d < dimension ? output_gr[shift_q + d] : 0.0f),
                                     ((d + 1) < dimension ? output_gr[shift_q + d + 1] : 0.0f),
                                     ((d + 2) < dimension ? output_gr[shift_q + d + 2] : 0.0f),
                                     ((d + 3) < dimension ? output_gr[shift_q + d + 3] : 0.0f)
                                  ), 0.0f);
         float4 o_d = IsNaNOrInf4((float4)(
                                     (d < dimension ? output[shift_q + d] : 0.0f),
                                     ((d + 1) < dimension ? output[shift_q + d + 1] : 0.0f),
                                     ((d + 2) < dimension ? output[shift_q + d + 2] : 0.0f),
                                     ((d + 3) < dimension ? output[shift_q + d + 3] : 0.0f)
                                  ), 0.0f);
         D += IsNaNOrInf(dot(g_d, o_d), 0.0f);
        }
      for(int l_id = 0; l_id < total_X; l_id += total_loc)
        {
         int x_id = l_id + local_id;
         float ds = 0;
         if(x_id < total_X && (mask_future == 0 || q_id <= x_id))
           {
            const int shift_x = RCtoFlat(x_id, 0, total_X, dimension, 0);
            float score = 0;
            float dp = 0;
            for(int d = 0; d < dimension; d += 4)
              {
               float4 q_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? query[shift_q + d] : 0.0f),
                                           ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                        ), 0.0f);
               float4 x_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? X[shift_x + d] : 0.0f),
                                           ((d + 1) < dimension ? X[shift_x + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? X[shift_x + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? X[shift_x + d + 3] : 0.0f)
                                        ), 0.0f);
               score += IsNaNOrInf(dot(q_d, x_d), 0.0f);
               float4 g_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? output_gr[shift_q + d] : 0.0f),
                                           ((d + 1) < dimension ? output_gr[shift_q + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? output_gr[shift_q + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? output_gr[shift_q + d + 3] : 0.0f)
                                        ), 0.0f);
               dp += IsNaNOrInf(dot(g_d, x_d), 0.0f);
              }
            score /= sqrt((float)dimension);
            const float p = IsNaNOrInf(exp(clamp(score - lse, -120.0f, 0.0f)), 0.0f);
            ds = IsNaNOrInf(p * (dp - D), 0.0f);
           }
         for(int d = 0; d < dimension; d += 4)
           {
            float4 x_d = (float4)0;
            if(x_id < total_X && (mask_future == 0 || q_id <= x_id))
              {
               const int shift_x = RCtoFlat(x_id, 0, total_X, dimension, 0);
               x_d = IsNaNOrInf4((float4)(
                                    (d < dimension ? X[shift_x + d] : 0.0f),
                                    ((d + 1) < dimension ? X[shift_x + d + 1] : 0.0f),
                                    ((d + 2) < dimension ? X[shift_x + d + 2] : 0.0f),
                                    ((d + 3) < dimension ? X[shift_x + d + 3] : 0.0f)
                                 ), 0.0f);
              }
            float4 q_dg = LocalSum4(x_d * ds, 1, temp4);
            int idx = local_id - d;
            if(idx >= 0 && idx < 4)
               grad_q += vget4(q_dg, idx);
           }
        }
      if(local_id < dimension)
         query_gr[shift_q + local_id] = IsNaNOrInf(grad_q, 0.0f);
     }
//--- X gradients: dX[k, d]
   if(id < total_X && h_id == 0)
     {
      float grad_X = 0.0f;
      const int x_id = id;
      const int shift_x = RCtoFlat(x_id, 0, total_X, dimension, 0);
      for(int l_id = 0; l_id < total_q * total_heads; l_id += total_loc)
        {
         int loc = l_id + local_id;
         int h = loc / total_q;
         int q_id = loc % total_q;
         float ds = 0;
         float p = 0;
         if(h < total_heads && q_id < total_q &&
            (mask_future == 0 || q_id <= x_id))
           {
            const int shift_lse = RCtoFlat(q_id, h, total_q, total_heads, 0);
            const float lse = IsNaNOrInf(logsumexp[shift_lse], 0.0f);
            const int shift_q = RCtoFlat(h, 0, total_heads, dimension, q_id);
            float score = 0;
            float D = 0;
            float dp = 0;
            for(int d = 0; d < dimension; d += 4)
              {
               float4 q_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? query[shift_q + d] : 0.0f),
                                           ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                        ), 0.0f);
               float4 x_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? X[shift_x + d] : 0.0f),
                                           ((d + 1) < dimension ? X[shift_x + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? X[shift_x + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? X[shift_x + d + 3] : 0.0f)
                                        ), 0.0f);
               score += IsNaNOrInf(dot(q_d, x_d), 0.0f);
               float4 g_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? output_gr[shift_q + d] : 0.0f),
                                           ((d + 1) < dimension ? output_gr[shift_q + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? output_gr[shift_q + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? output_gr[shift_q + d + 3] : 0.0f)
                                        ), 0.0f);
               dp += IsNaNOrInf(dot(g_d, x_d), 0.0f);
               float4 o_d = IsNaNOrInf4((float4)(
                                           (d < dimension ? output[shift_q + d] : 0.0f),
                                           ((d + 1) < dimension ? output[shift_q + d + 1] : 0.0f),
                                           ((d + 2) < dimension ? output[shift_q + d + 2] : 0.0f),
                                           ((d + 3) < dimension ? output[shift_q + d + 3] : 0.0f)
                                        ), 0.0f);
               D += IsNaNOrInf(dot(g_d, o_d), 0.0f);
              }
            const float p = IsNaNOrInf(exp(clamp(score - lse, -120.0f, 0.0f)), 0.0f);
            if(p != 0.0f)
               ds = IsNaNOrInf(p * (dp - D), 0.0f);
           }
         //---
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q_d = (float4)0;
            float4 g_d = (float4)0;
            if(h < total_heads && q_id < total_q &&
               (mask_future == 0 || q_id <= x_id))
              {
               const int shift_q = RCtoFlat(h, 0, total_heads, dimension, q_id);
               q_d = IsNaNOrInf4((float4)(
                                    (d < dimension ? query[shift_q + d] : 0.0f),
                                    ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                    ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                    ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                 ), 0.0f);
               g_d = IsNaNOrInf4((float4)(
                                    (d < dimension ? output_gr[shift_q + d] : 0.0f),
                                    ((d + 1) < dimension ? output_gr[shift_q + d + 1] : 0.0f),
                                    ((d + 2) < dimension ? output_gr[shift_q + d + 2] : 0.0f),
                                    ((d + 3) < dimension ? output_gr[shift_q + d + 3] : 0.0f)
                                 ), 0.0f);
              }
            float4 x_dg = LocalSum4(q_d * ds + g_d * p, 1, temp4);
            int idx = local_id - d;
            if(idx >= 0 && idx < 4)
               grad_X += vget4(x_dg, idx);
           }
        }
      if(local_id < dimension)
         X_gr[shift_x + local_id] = IsNaNOrInf(grad_X, 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Performs the concat vec matrix data-layout operation on the GPU. |
//+------------------------------------------------------------------+
__kernel void ConcatVecMatrix(__global const float *vector_in,
                              __global const float *matrix_in,
                              __global float *matrix_out,
                              const int dimension_v,
                              const int colums_m,
                              const int multvarsecond
                             )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int v = get_global_id(2);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
   const int variables = get_global_size(2);
//---
   if(c < dimension_v)
     {
      int flat_v = RCtoFlat(0, c, 1, dimension_v, v);
      int flat_out = RCtoFlat(r, c, rows, dimension_v + colums_m, v);
      matrix_out[flat_out] = IsNaNOrInf(vector_in[flat_v], 0.0f);
     }
   if(c < colums_m)
     {
      int flat_m = RCtoFlat(r, c, rows, colums_m, multvarsecond * v);
      int flat_out = RCtoFlat(r, c + dimension_v, rows, dimension_v + colums_m, v);
      matrix_out[flat_out] = IsNaNOrInf(matrix_in[flat_m], 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Computes the concat vec matrix grad gradient contributions on the|
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void ConcatVecMatrixGrad(__global float *vector_in_gr,
                                  __global float *matrix_in_gr,
                                  __global const float *matrix_out_gr,
                                  const int dimension_v,
                                  const int colums_m,
                                  const int rows,
                                  const int variables,
                                  const int multvarsecond
                                 )
  {
   const int loc_id = get_local_id(0);
   const int c = get_global_id(1);
   const int glob_id = get_global_id(2);
   const int total_loc = get_local_size(0);
   const int cols = get_global_size(1);
   const int total_glob = get_global_size(2);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   float vec_grad = 0;
   for(int i = 0; (i < rows && glob_id < variables); i += total_loc)
     {
      int r = i + loc_id;
      if(r >= rows)
         continue;
      if(c < dimension_v)
        {
         int flat_out = RCtoFlat(r, c, rows, dimension_v + colums_m, glob_id);
         vec_grad += IsNaNOrInf(matrix_out_gr[flat_out], 0.0f);
        }
      if(c < colums_m)
        {
         int flat_m = RCtoFlat(r, c, rows, colums_m, glob_id);
         int flat_out = RCtoFlat(r, c + dimension_v, rows, dimension_v + colums_m, glob_id);
         if((bool)multvarsecond)
            matrix_in_gr[flat_m] = IsNaNOrInf(matrix_out_gr[flat_out], 0.0f);
        }
     }
   if(c < dimension_v)
     {
      vec_grad = LocalSum(vec_grad, 0, temp);
      if(loc_id == 0)
        {
         int flat_v = RCtoFlat(0, c, 1, dimension_v, glob_id);
         vector_in_gr[flat_v] = vec_grad;
        }
     }
   if(!((bool)multvarsecond) && c < colums_m && glob_id < rows)
     {
      float mat_grad = 0;
      for(int i = 0; i < variables; i += total_loc)
        {
         int v = i + loc_id;
         if(v >= variables)
            continue;
         int flat_out = RCtoFlat(glob_id, c + dimension_v, rows, dimension_v + colums_m, v);
         mat_grad += IsNaNOrInf(matrix_out_gr[flat_out], 0.0f);
        }
      mat_grad = LocalSum(mat_grad, 0, temp);
      if(loc_id == 0)
        {
         int flat_m = RCtoFlat(glob_id, c, rows, colums_m, 0);
         matrix_in_gr[flat_m] = mat_grad;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the inf net bgu kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void INFNetBGU(__global const float2*  scal_and_shift,
                        __global const float*   sequence,
                        __global const float*   context,
                        __global const float*   scenarios,
                        __global float*      sequence_out,
                        __global float*      context_out,
                        __global float*      scenarios_out,
                        const int   sequence_units,
                        const int   sequence_vars,
                        const int   context_units,
                        const int   scenarios_units
                       )
  {
   const int id_n = get_global_id(0);
   const int d = get_global_id(1);
   const int v = get_global_id(2);
   const int total_n = get_global_size(0);
   const int dimension = get_global_size(1);
   const int variables = get_global_size(2);
   __global const float* inputs = sequence;
   __global float* outputs = sequence_out;
   int units = 0;
   int var = 0;
   float2 ss = 0;
   if(v < sequence_vars)
     {
      units = sequence_units;
      var = v;
      ss = scal_and_shift[RCtoFlat(v, d, sequence_vars + scenarios_units + 1, dimension, 0)];
     }
   else
      if(v == sequence_vars)
        {
         inputs = context;
         outputs = context_out;
         units = context_units;
         if(id_n < units)
            ss = scal_and_shift[RCtoFlat(v, d, sequence_vars + scenarios_units + 1, dimension, 0)];
        }
      else
        {
         inputs = scenarios;
         outputs = scenarios_out;
         units = scenarios_units;
         if(id_n < units)
            ss = scal_and_shift[RCtoFlat(sequence_vars + 1 + id_n, d, sequence_vars + scenarios_units + 1, dimension, 0)];
        }
//---
   ss = IsNaNOrInf2(ss, (float2)0);
   float sig_a = fActivation(ss.s0, ActFunc_SIGMOID);
   for(int i = id_n; i < units; i += total_n)
     {
      int shift = RCtoFlat(i, d, units, dimension, var);
      float val = IsNaNOrInf(inputs[shift], 0);
      val = IsNaNOrInf(val * (1 + sig_a) + ss.s1, 0);
      outputs[shift] = val;
     }
  }
//+------------------------------------------------------------------+
//| Computes the inf net bgu grad gradient contributions on the GPU. |
//+------------------------------------------------------------------+
__kernel void INFNetBGUGrad(__global const float2* scal_and_shift,
                            __global float2*       scal_and_shift_grad,
                            __global const float*  sequence,
                            __global const float*  context,
                            __global const float*  scenarios,
                            __global float*        sequence_grad,
                            __global float*        context_grad,
                            __global float*        scenarios_grad,
                            __global const float*  sequence_out_grad,
                            __global const float*  context_out_grad,
                            __global const float*  scenarios_out_grad,
                            const int sequence_units,
                            const int sequence_vars,
                            const int context_units,
                            const int scenarios_units
                           )
  {
   const int id_n      = get_local_id(0);
   const int d         = get_global_id(1);
   const int v         = get_global_id(2);
   const int total_n   = get_local_size(0);
   const int dimension = get_global_size(1);
//---
   __local float4 Temp[LOCAL_ARRAY_SIZE];
//---
   __global const float* inputs   = sequence;
   __global float*       grad_in  = sequence_grad;
   __global const float* grad_out = sequence_out_grad;
   int    units          = 0;
   int    var            = 0;
   int    ss_idx         = 0;
   float2 ss             = (float2)0;
   bool   scenarios_mode = false;
   const int ss_rows = sequence_vars + scenarios_units + 1;
   if(v < sequence_vars)
     {
      units  = sequence_units;
      var    = v;
      ss_idx = RCtoFlat(v, d, ss_rows, dimension, 0);
      ss     = IsNaNOrInf2(scal_and_shift[ss_idx], (float2)0);
     }
   else
      if(v == sequence_vars)
        {
         inputs   = context;
         grad_in  = context_grad;
         grad_out = context_out_grad;
         units    = context_units;
         if(id_n < units)
           {
            ss_idx   = RCtoFlat(v, d, ss_rows, dimension, 0);
            ss       = IsNaNOrInf2(scal_and_shift[ss_idx], (float2)0);
           }
        }
      else
        {
         inputs         = scenarios;
         grad_in        = scenarios_grad;
         grad_out       = scenarios_out_grad;
         units          = scenarios_units;
         scenarios_mode = true;
         if(id_n < units)
           {
            ss_idx = RCtoFlat(sequence_vars + 1 + id_n, d, ss_rows, dimension, 0);
            ss     = IsNaNOrInf2(scal_and_shift[ss_idx], (float2)0);
           }
        }
   const float sig_a = fActivation(ss.s0, ActFunc_SIGMOID);
   float d_alpha_acc = 0.0f;
   float d_beta_acc  = 0.0f;
   for(int i = id_n; i < units; i += total_n)
     {
      int   shift = RCtoFlat(i, d, units, dimension, var);
      float x     = IsNaNOrInf(inputs[shift],  0.0f);
      float dy    = IsNaNOrInf(grad_out[shift], 0.0f);
      grad_in[shift] = IsNaNOrInf(dy * (1 + sig_a), 0.0f);
      d_alpha_acc += dy * x;
      d_beta_acc  += dy;
     }
   if(!scenarios_mode)
     {
      float4 sums = LocalSum4((float4)(d_alpha_acc, d_beta_acc, 0.0f, 0.0f), 0, Temp);
      if(get_local_id(0) == 0)
        {
         sums.s0 = Deactivation(sums.s0, sig_a, ActFunc_SIGMOID);
         scal_and_shift_grad[ss_idx] = sums.lo;
        }
     }
   else
      if(id_n < units)
         scal_and_shift_grad[ss_idx] = (float2)(Deactivation(d_alpha_acc, sig_a, ActFunc_SIGMOID),
                                                IsNaNOrInf(d_beta_acc,  0.0f));
  }
//+------------------------------------------------------------------+
//| Implements the mh cross attvs sim kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void MHCrossAttvsSim(__global const float *query,
                              __global const float *key,
                              __global const float *value,
                              __global const float *prototype,
                              __global const float *levels,
                              __global float       *logsumexp,
                              __global float       *output_at,
                              __global float       *similarity,
                              const int             dimension,
                              const int             total_X)
  {
   const int q_id        = get_global_id(0);
   const int local_id    = get_local_id(1);
   const int h_id        = get_global_id(2);
   const int total_q     = get_global_size(0);
   const int total_loc   = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float  temp[LOCAL_ARRAY_SIZE];
   __local float4 temp4[LOCAL_ARRAY_SIZE];
//---
   const int shift_q     = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
   const int logse_shift = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
   const float threshold = levels[logse_shift];
   /*
      Loop 1: compute logsumexp via online softmax (no masking)
   */
   float prev_max = MIN_VALUE;
   float sumexp   = 0.0f;
   for(int id = 0; id < total_X; id += total_loc)
     {
      int x_id = id + local_id;
      const int shift_x = RCtoFlat(h_id, 0, total_heads, dimension, x_id);
      float score = 0.0f;
      if(x_id < total_X)
        {
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q = IsNaNOrInf4((float4)(
                                      query[shift_q + d],
                                      ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                   ), 0.0f);
            float4 k = IsNaNOrInf4((float4)(
                                      key[shift_x + d],
                                      ((d + 1) < dimension ? key[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? key[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? key[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            score += IsNaNOrInf(dot(q, k), 0.0f);
           }
         score /= sqrt((float)dimension);
        }
      else
         score = MIN_VALUE;
      float max = fmax(prev_max, LocalMax(score, 1, temp));
      score   = (score > MIN_VALUE) ? IsNaNOrInf(exp(score - max), 0.0f) : 0.0f;
      if(sumexp == 0.0f)
         sumexp = LocalSum(score, 1, temp);
      else
         sumexp = IsNaNOrInf(exp(prev_max - max) * sumexp, 0.0f) + LocalSum(score, 1, temp);
      prev_max = max;
     }
//--- Write logsumexp
   const float lse = (sumexp > 0.0f) ? IsNaNOrInf(prev_max + log(sumexp), 0.0f) : 0.0f;
   if(local_id == 0)
      logsumexp[logse_shift] = lse;
   /*
   Loop 2: accumulate value and similarity
   score = exp(score - lse); skip if score < threshold
   score are already normalized — no division by sumexp needed
   */
   float out_at  = 0.0f;
   float out_sim = 0.0f;
   for(int id = 0; id < total_X; id += total_loc)
     {
      int x_id = id + local_id;
      const int shift_x = RCtoFlat(h_id, 0, total_heads, dimension, x_id);
      float score = 0.0f;
      if(x_id < total_X && sumexp > 0.0f)
        {
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q = IsNaNOrInf4((float4)(
                                      query[shift_q + d],
                                      ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                   ), 0.0f);
            float4 k = IsNaNOrInf4((float4)(
                                      key[shift_x + d],
                                      ((d + 1) < dimension ? key[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? key[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? key[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            score += IsNaNOrInf(dot(q, k), 0.0f);
           }
         score /= sqrt((float)dimension);
         score = IsNaNOrInf(exp(score - lse), 0.0f);
         if(score < threshold)
            score = 0.0f;
        }
      for(int d = 0; d < dimension; d += 4)
        {
         float4 val = (float4)0.0f;
         float4 sim = (float4)0.0f;
         if(score > 0.0f)
           {
            float4 v = IsNaNOrInf4((float4)(
                                      value[shift_x + d],
                                      ((d + 1) < dimension ? value[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? value[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? value[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            float4 p = IsNaNOrInf4((float4)(
                                      prototype[shift_x + d],
                                      ((d + 1) < dimension ? prototype[shift_x + d + 1] : 0.0f),
                                      ((d + 2) < dimension ? prototype[shift_x + d + 2] : 0.0f),
                                      ((d + 3) < dimension ? prototype[shift_x + d + 3] : 0.0f)
                                   ), 0.0f);
            val = IsNaNOrInf4(v * score, 0.0f);
            sim = IsNaNOrInf4(p * score, 0.0f);
           }
         val = LocalSum4(val, 1, temp4);
         sim = LocalSum4(sim, 1, temp4);
         int idx = local_id - d;
         if(idx >= 0 && idx < 4)
           {
            out_at  += vget4(val, idx);
            out_sim += vget4(sim, idx);
           }
        }
     }
//--- Write results (weights already normalized via lse)
   if(local_id < dimension)
     {
      output_at[shift_q + local_id]  = IsNaNOrInf(out_at,  0.0f);
      similarity[shift_q + local_id] = IsNaNOrInf(out_sim, 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Computes the mh cross attvs sim grad gradient contributions on   |
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void MHCrossAttvsSimGrad(__global const float *query,
                                  __global const float *key,
                                  __global const float *value,
                                  __global const float *prototype,
                                  __global const float *levels,
                                  __global const float *logsumexp,
                                  __global const float *output_at,
                                  __global const float *similarity,
                                  __global const float *grad_output_at,
                                  __global const float *grad_similarity,
                                  __global float       *grad_query,
                                  __global float       *grad_key,
                                  __global float       *grad_value,
                                  __global float       *grad_prototype,
                                  __global float       *grad_levels,
                                  const int             dimension,
                                  const int             total_X)
  {
   const int id          = get_global_id(0);
   const int local_id    = get_local_id(1);
   const int h_id        = get_global_id(2);
   const int total_q     = get_global_size(0);
   const int total_loc   = get_local_size(1);
   const int total_heads = get_global_size(2);
//---
   __local float  temp[LOCAL_ARRAY_SIZE];
   __local float4 temp4[LOCAL_ARRAY_SIZE];
   const float inv_sqrt_d = 1.0f / sqrt((float)dimension);
   /*
   Block 1: grad_query + grad_levels
   global_id(0) = q_id, owns shift_q — no race condition
   */
     {
      const int q_id        = id;
      const int shift_q     = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
      const int logse_shift = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
      const float threshold = levels[logse_shift];
      const float lse       = logsumexp[logse_shift];
      //--- D = dot(grad_output_at, output_at) + dot(grad_similarity, similarity)
      //--- used in softmax gradient: grad_score = score * (dot(go,v) + dot(gs,p) - D)
      float D = 0.0f;
      for(int d = local_id; d < dimension; d += total_loc)
        {
         D += IsNaNOrInf(grad_output_at[shift_q + d]  * output_at[shift_q + d],  0.0f);
         D += IsNaNOrInf(grad_similarity[shift_q + d] * similarity[shift_q + d], 0.0f);
        }
      D = LocalSum(D, 1, temp);
      //--- Iterate over x_id
      float out_gq   = 0.0f;
      float grad_lev = 0.0f;
      for(int l_id = 0; l_id < total_X; l_id += total_loc)
        {
         int x_id = l_id + local_id;
         const int shift_x = RCtoFlat(h_id, 0, total_heads, dimension, x_id);
         float score     = 0.0f;
         float grad_score = 0.0f;
         if(x_id < total_X)
           {
            float score    = 0.0f;
            float dot_go_v = 0.0f;
            float dot_gs_p = 0.0f;
            for(int d = 0; d < dimension; d += 4)
              {
               float4 q_d  = IsNaNOrInf4((float4)(
                                            query[shift_q + d],
                                            ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 k_d  = IsNaNOrInf4((float4)(
                                            key[shift_x + d],
                                            ((d + 1) < dimension ? key[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? key[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? key[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 v_d  = IsNaNOrInf4((float4)(
                                            value[shift_x + d],
                                            ((d + 1) < dimension ? value[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? value[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? value[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 p_d  = IsNaNOrInf4((float4)(
                                            prototype[shift_x + d],
                                            ((d + 1) < dimension ? prototype[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? prototype[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? prototype[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 go_d = IsNaNOrInf4((float4)(
                                            grad_output_at[shift_q + d],
                                            ((d + 1) < dimension ? grad_output_at[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? grad_output_at[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? grad_output_at[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 gs_d = IsNaNOrInf4((float4)(
                                            grad_similarity[shift_q + d],
                                            ((d + 1) < dimension ? grad_similarity[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? grad_similarity[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? grad_similarity[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               score    += IsNaNOrInf(dot(q_d,  k_d), 0.0f);
               dot_go_v += IsNaNOrInf(dot(go_d, v_d), 0.0f);
               dot_gs_p += IsNaNOrInf(dot(gs_d, p_d), 0.0f);
              }
            score /= sqrt((float)dimension);
            score = IsNaNOrInf(exp(score - lse), 0.0f);
            grad_score = IsNaNOrInf(score * (dot_go_v + dot_gs_p - D), 0.0f);
           }
         //--- Surrogate gradient for threshold: -sign(level) * sum(grad_score)
         grad_lev -= grad_score;
         //--- grad_query += grad_score * inv_sqrt_d * key
         for(int d = 0; d < dimension; d += 4)
           {
            float4 k_d = (float4)0.0f;
            if(fabs(grad_score) > 0 && x_id < total_X)
               k_d = IsNaNOrInf4((float4)(
                                    key[shift_x + d],
                                    ((d + 1) < dimension ? key[shift_x + d + 1] : 0.0f),
                                    ((d + 2) < dimension ? key[shift_x + d + 2] : 0.0f),
                                    ((d + 3) < dimension ? key[shift_x + d + 3] : 0.0f)
                                 ), 0.0f);
            float4 gq = LocalSum4(k_d * (grad_score * inv_sqrt_d), 1, temp4);
            int idx = local_id - d;
            if(idx >= 0 && idx < 4)
               out_gq += vget4(gq, idx);
           }
        }
      //--- Write grad_query
      if(local_id < dimension)
         grad_query[shift_q + local_id] = IsNaNOrInf(out_gq, 0.0f);
      //--- Write grad_levels
      grad_lev = LocalSum(grad_lev, 1, temp);
      if(local_id == 0)
        {
         grad_levels[logse_shift] = (fabs(grad_lev) > 0.0f)
                                    ? IsNaNOrInf(grad_lev, 0.0f)
                                    : 0.0f;
        }
     }
   /*
   Block 2: grad_key + grad_value + grad_prototype
   global_id(0) = x_id, global_id(2) = h_id, owns shift_x
   iterates over q_id in local dim — no race condition
   */
   for(int id2 = 0; id2 < total_X; id2 += total_q)
     {
      const int x_id    = id2 + id;
      const int shift_x = RCtoFlat(h_id, 0, total_heads, dimension, x_id);
      float grad_k = 0.0f;
      float grad_v = 0.0f;
      float grad_p = 0.0f;
      for(int l_id = 0; l_id < total_q; l_id += total_loc)
        {
         int q_id = l_id + local_id;
         float score      = 0.0f;
         float grad_score = 0.0f;
         if(q_id < total_q && x_id < total_X)
           {
            const int shift_q     = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
            const int logse_shift = RCtoFlat(q_id, h_id, total_q, total_heads, 0);
            const float threshold = levels[logse_shift];
            const float lse       = logsumexp[logse_shift];
            float D        = 0.0f;
            float dot_go_v = 0.0f;
            float dot_gs_p = 0.0f;
            for(int d = 0; d < dimension; d += 4)
              {
               float4 q_d  = IsNaNOrInf4((float4)(
                                            query[shift_q + d],
                                            ((d + 1) < dimension ? query[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? query[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? query[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 k_d  = IsNaNOrInf4((float4)(
                                            key[shift_x + d],
                                            ((d + 1) < dimension ? key[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? key[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? key[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 v_d  = IsNaNOrInf4((float4)(
                                            value[shift_x + d],
                                            ((d + 1) < dimension ? value[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? value[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? value[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 p_d  = IsNaNOrInf4((float4)(
                                            prototype[shift_x + d],
                                            ((d + 1) < dimension ? prototype[shift_x + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? prototype[shift_x + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? prototype[shift_x + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 go_d = IsNaNOrInf4((float4)(
                                            grad_output_at[shift_q + d],
                                            ((d + 1) < dimension ? grad_output_at[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? grad_output_at[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? grad_output_at[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 gs_d = IsNaNOrInf4((float4)(
                                            grad_similarity[shift_q + d],
                                            ((d + 1) < dimension ? grad_similarity[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? grad_similarity[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? grad_similarity[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 o_d  = IsNaNOrInf4((float4)(
                                            output_at[shift_q + d],
                                            ((d + 1) < dimension ? output_at[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? output_at[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? output_at[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               float4 s_d  = IsNaNOrInf4((float4)(
                                            similarity[shift_q + d],
                                            ((d + 1) < dimension ? similarity[shift_q + d + 1] : 0.0f),
                                            ((d + 2) < dimension ? similarity[shift_q + d + 2] : 0.0f),
                                            ((d + 3) < dimension ? similarity[shift_q + d + 3] : 0.0f)
                                         ), 0.0f);
               score    += IsNaNOrInf(dot(q_d,  k_d),  0.0f);
               dot_go_v += IsNaNOrInf(dot(go_d, v_d),  0.0f);
               dot_gs_p += IsNaNOrInf(dot(gs_d, p_d),  0.0f);
               D        += IsNaNOrInf(dot(go_d, o_d),  0.0f);
               D        += IsNaNOrInf(dot(gs_d, s_d),  0.0f);
              }
            score /= sqrt((float)dimension);
            score = IsNaNOrInf(exp(score - lse), 0.0f);
            if(score > 0.0f)
               grad_score = IsNaNOrInf(score * (dot_go_v + dot_gs_p - D), 0.0f);
           }
         //--- grad_key   += grad_score * inv_sqrt_d * query
         //--- grad_value += score * grad_output_at
         //--- grad_proto += score * grad_similarity
         for(int d = 0; d < dimension; d += 4)
           {
            float4 q_d  = (float4)0.0f;
            float4 go_d = (float4)0.0f;
            float4 gs_d = (float4)0.0f;
            if(q_id < total_q && x_id < total_X)
              {
               const int sq = RCtoFlat(h_id, 0, total_heads, dimension, q_id);
               q_d  = IsNaNOrInf4((float4)(
                                     query[sq + d],
                                     ((d + 1) < dimension ? query[sq + d + 1] : 0.0f),
                                     ((d + 2) < dimension ? query[sq + d + 2] : 0.0f),
                                     ((d + 3) < dimension ? query[sq + d + 3] : 0.0f)
                                  ), 0.0f);
               go_d = IsNaNOrInf4((float4)(
                                     grad_output_at[sq + d],
                                     ((d + 1) < dimension ? grad_output_at[sq + d + 1] : 0.0f),
                                     ((d + 2) < dimension ? grad_output_at[sq + d + 2] : 0.0f),
                                     ((d + 3) < dimension ? grad_output_at[sq + d + 3] : 0.0f)
                                  ), 0.0f);
               gs_d = IsNaNOrInf4((float4)(
                                     grad_similarity[sq + d],
                                     ((d + 1) < dimension ? grad_similarity[sq + d + 1] : 0.0f),
                                     ((d + 2) < dimension ? grad_similarity[sq + d + 2] : 0.0f),
                                     ((d + 3) < dimension ? grad_similarity[sq + d + 3] : 0.0f)
                                  ), 0.0f);
              }
            float4 gk = LocalSum4(q_d  * (grad_score * inv_sqrt_d), 1, temp4);
            float4 gv = LocalSum4(go_d * score,                     1, temp4);
            float4 gp = LocalSum4(gs_d * score,                     1, temp4);
            int idx = local_id - d;
            if(idx >= 0 && idx < 4)
              {
               grad_k += vget4(gk, idx);
               grad_v += vget4(gv, idx);
               grad_p += vget4(gp, idx);
              }
           }
        }
      //--- Write grad_key/value/prototype — single work-group owns shift_x, no conflicts
      if(local_id < dimension)
        {
         grad_key[shift_x + local_id]       = IsNaNOrInf(grad_k, 0.0f);
         grad_value[shift_x + local_id]     = IsNaNOrInf(grad_v, 0.0f);
         grad_prototype[shift_x + local_id] = IsNaNOrInf(grad_p, 0.0f);
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the market state density kernel on the GPU.           |
//+------------------------------------------------------------------+
__kernel void MarketStateDensity(__global const float* sequence,
                                 __global float* distribution,
                                 const uint sequence_size,
                                 const uint quantiles,
                                 const float quant_step
                                )
  {
//--- features
   const uint feature = get_global_id(0);
   const uint dimension = get_global_size(0);
//--- local worker index
   const uint lid = get_local_id(1);
//--- local size over sequence axis
   const uint local_n = get_local_size(1);
//--- total quantiles
   const int bins = 2 * quantiles + 1;
//--- variables
   const uint var = get_global_id(2);
   const uint variables = get_global_size(2);
//---
   __local float4 Temp[LOCAL_ARRAY_SIZE];
//--- current state reference
   if(lid == 0)
      Temp[0].s0 = IsNaNOrInf(sequence[RCtoFlat(0, feature, sequence_size, dimension, var)], 0.0f);
   BarrierLoc;
   const float current = Temp[0].s0;
//--- iterate through sequence
   uint count = 0;
   for(uint i = 0; i < sequence_size; i += local_n)
     {
      int q = bins + 2;
      if(i + lid < sequence_size)
        {
         const float value = IsNaNOrInf(sequence[RCtoFlat(i + lid, feature, sequence_size, dimension, var)], 0.0f);
         //--- deviation from current state
         const float diff = current - value;
         //--- quantization
         q = (int)floor(diff / quant_step);
         // center shift
         q += (int)quantiles;
         // clamp
         q = clamp(q, 0, bins - 1);
        }
      // accumulate
      for(int b = 0; b < bins; b += 4)
        {
         float4 b4 = (float4)0.0f;
         switch(q - b)
           {
            case 0:
               b4.s0 = 1;
               break;
            case 1:
               b4.s1 = 1;
               break;
            case 2:
               b4.s2 = 1;
               break;
            case 3:
               b4.s3 = 1;
               break;
           }
         BarrierLoc;
         count += (uint)vget4(LocalSum4(b4, 1, Temp), lid - b);
        }
     }
//--- normalization and save
   if(lid < bins)
      distribution[RCtoFlat(lid, feature, bins, dimension, var)] = ((float)count / sequence_size);
  }
//+------------------------------------------------------------------+
//| Computes the nll grad gradient contributions on the GPU.         |
//+------------------------------------------------------------------+
__kernel void NLLGrad(__global const float* log_b,
                      __global float* grad_in,
                      __global const float* grad_out,
                      __global float* grad_log_b
                     )
  {
   const uint i = get_global_id(0);
//--- log-scale resistance
   const float s = log_b[i];
//--- forecat gradient
   const float grad = grad_out[i];
//--- gradient
   const float g = IsNaNOrInf(grad * exp(-s), 0.0f);
//--- to inputs
   grad_in[i] = g;
//--- local gradient for s
//--- d(g*e^-s)/ds = -g*e^-s
   grad_log_b[i] = -g;
  }
//+------------------------------------------------------------------+
//| Implements the cog driver forecast shift state kernel on the GPU.|
//+------------------------------------------------------------------+
__kernel void CogDriverForecastShiftState(__global const float* plan,
      __global const float* confidence,
      __global float* shifted_plan,
      __global float* shifted_confidence,
      __global float* delta_mask,
      const int dimension,
      const int horizon
                                         )
  {
   const int id = get_global_id(0);
   const int total = dimension * horizon;
   if(id >= total || dimension <= 0 || horizon <= 0)
      return;
   const int h = id / dimension;
   const int d = id - h * dimension;
   if(h < horizon - 1)
      shifted_plan[id] = IsNaNOrInf(plan[(h + 1) * dimension + d], 0.0f);
   else
      shifted_plan[id] = 0.0f;
   if(d == 0)
     {
      if(h < horizon - 1)
        {
         shifted_confidence[h] = IsNaNOrInf(confidence[h + 1], 0.0f);
         delta_mask[h] = 1.0f;
        }
      else
        {
         shifted_confidence[h] = 0.0f;
         delta_mask[h] = 0.0f;
        }
     }
  }
//+------------------------------------------------------------------+
//| Implements the cog driver forecast build kv kernel on the GPU.   |
//+------------------------------------------------------------------+
__kernel void CogDriverForecastBuildKV(__global const float* context,
                                       __global float* context_kv,
                                       const int dimension,
                                       const int horizon,
                                       const int variables,
                                       const int heads,
                                       const int head_dim
                                      )
  {
   const int variable = get_global_id(0);
   const int h = get_global_id(1);
   const int flat_dim = get_global_id(2);
   if(variable >= variables || h >= horizon || flat_dim >= dimension ||
      heads <= 0 || head_dim <= 0)
      return;
   const int head = flat_dim / head_dim;
   if(head >= heads)
      return;
   const int local_dim = flat_dim - head * head_dim;
   const int token = variable * horizon + h;
   const int src_base = variable * 2 * horizon * dimension + h * 2 * dimension;
   const float key = IsNaNOrInf(context[src_base + flat_dim], 0.0f);
   const float value = IsNaNOrInf(context[src_base + dimension + flat_dim], 0.0f);
   const int k_shift = RCtoFlat(head, local_dim, 2 * heads, head_dim, token);
   const int v_shift = RCtoFlat(head + heads, local_dim, 2 * heads, head_dim, token);
   context_kv[k_shift] = key;
   context_kv[v_shift] = value;
  }
//+------------------------------------------------------------------+
//| Computes the cog driver forecast build kv grad gradient          |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CogDriverForecastBuildKVGrad(__global const float* context_kv_gr,
      __global float* context_gr,
      const int dimension,
      const int horizon,
      const int variables,
      const int heads,
      const int head_dim
                                          )
  {
   const int variable = get_global_id(0);
   const int h = get_global_id(1);
   const int flat_dim = get_global_id(2);
   if(variable >= variables || h >= horizon || flat_dim >= dimension ||
      heads <= 0 || head_dim <= 0)
      return;
   const int head = flat_dim / head_dim;
   if(head >= heads)
      return;
   const int local_dim = flat_dim - head * head_dim;
   const int token = variable * horizon + h;
   const int dst_base = variable * 2 * horizon * dimension + h * 2 * dimension;
   const int k_shift = RCtoFlat(head, local_dim, 2 * heads, head_dim, token);
   const int v_shift = RCtoFlat(head + heads, local_dim, 2 * heads, head_dim, token);
   context_gr[dst_base + flat_dim] += IsNaNOrInf(context_kv_gr[k_shift], 0.0f);
   context_gr[dst_base + dimension + flat_dim] += IsNaNOrInf(context_kv_gr[v_shift], 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the cog driver forecast confidence kernel on the GPU. |
//+------------------------------------------------------------------+
__kernel void CogDriverForecastConfidence(__global const float* expected_error,
      __global float* confidence,
      const int total
                                         )
  {
   const int id = get_global_id(0);
   if(id >= total)
      return;
   const float error = fmax(IsNaNOrInf(expected_error[id], 0.0f), 0.0f);
   confidence[id] = 1.0f / (1.0f + error);
  }
//+------------------------------------------------------------------+
//| Implements the cog driver forecast delta norm kernel on the GPU. |
//+------------------------------------------------------------------+
__kernel void CogDriverForecastDeltaNorm(__global const float* plan,
      __global const float* shifted_plan,
      __global float* plan_delta,
      __global float* plan_delta_norm,
      const int dimension,
      const int horizon
                                        )
  {
   const int h = get_global_id(0);
   if(h >= horizon || dimension <= 0)
      return;
   float sum = 0.0f;
   for(int d = 0; d < dimension; d++)
     {
      const int idx = h * dimension + d;
      const float delta = IsNaNOrInf(plan[idx] - shifted_plan[idx], 0.0f);
      plan_delta[idx] = delta;
      sum += delta * delta;
     }
   plan_delta_norm[h] = sqrt(sum / (float)dimension);
  }
//+------------------------------------------------------------------+
//| Generic RAG retrieval pipeline; every extent is runtime supplied |
//+------------------------------------------------------------------+
__kernel void RAGScore(__global const float *embedding,
                       __global const float *centroids,
                       __global const float *used,
                       __global float *scores,
                       const int scenario_count,
                       const int embedding_dimension)
  {
   const int scenario = get_global_id(0);
   if(scenario >= scenario_count || embedding_dimension <= 0)
      return;
   if(used[scenario] <= 0.5f)
     {
      scores[scenario] = RAG_INVALID_SCORE;
      return;
     }
   float dot = 0.0f, left = 0.0f, right = 0.0f;
   for(int d = 0; d < embedding_dimension; d++)
     {
      float a = IsNaNOrInf(embedding[d], 0.0f);
      float b = IsNaNOrInf(centroids[scenario * embedding_dimension + d], 0.0f);
      dot += a * b;
      left += a * a;
      right += b * b;
     }
   const float score = (left > 1.0e-12f && right > 1.0e-12f ?
                        dot / (sqrt(left) * sqrt(right)) : RAG_INVALID_SCORE);
   scores[scenario] = (isfinite(score) ? score : RAG_INVALID_SCORE);
  }
//+------------------------------------------------------------------+
//| Implements the rag before helper on the GPU.                     |
//+------------------------------------------------------------------+
inline int RAGBefore(const float2 left, const float2 right)
  {
   return (left.y > right.y || (left.y == right.y && left.x < right.x));
  }
//+------------------------------------------------------------------+
//| Implements the rag sort descending helper on the GPU.            |
//+------------------------------------------------------------------+
inline void RAGSortDescending(__local float2 candidates[], const int lane, const int block_size)
  {
   for(int width = 2; width <= block_size; width <<= 1)
      for(int stride = width >> 1; stride > 0; stride >>= 1)
        {
         const int partner = lane ^ stride;
         if(partner > lane)
           {
            const int descending = ((lane & width) == 0);
            const int before = RAGBefore(candidates[lane], candidates[partner]);
            if((descending && !before) || (!descending && before))
              {
               const float2 swap = candidates[lane];
               candidates[lane] = candidates[partner];
               candidates[partner] = swap;
              }
           }
         BarrierLoc;
        }
  }
//+------------------------------------------------------------------+
//| Implements the rag partial kernel on the GPU.                    |
//+------------------------------------------------------------------+
__kernel void RAGPartial(__global const float *scores,
                         __global float2 *partial,
                         const int scenario_count,
                         const int block_size,
                         const int top_k)
  {
   const int lane = get_local_id(0);
   const int block = get_group_id(0);
   const int scenario = block * block_size + lane;
   __local float2 candidates[LOCAL_ARRAY_SIZE];
   if(block_size <= 0 || block_size > LOCAL_ARRAY_SIZE || (block_size & (block_size - 1)) != 0 ||
      top_k <= 0 || top_k > block_size)
      return;
   const float score = (scenario < scenario_count ? scores[scenario] : RAG_INVALID_SCORE);
   candidates[lane] = (isfinite(score) && score > RAG_INVALID_SCORE ?
                       (float2)((float)scenario, score) :
                       (float2)(-1.0f, RAG_INVALID_SCORE));
   BarrierLoc;
   RAGSortDescending(candidates, lane, block_size);
   if(lane < top_k)
      partial[block * top_k + lane] = candidates[lane];
  }
//+------------------------------------------------------------------+
//| Implements the rag merge kernel on the GPU.                      |
//+------------------------------------------------------------------+
__kernel void RAGMerge(__global const float2 *source,
                       __global float2 *destination,
                       const int candidate_count,
                       const int block_size,
                       const int top_k)
  {
   const int lane = get_local_id(0);
   const int block = get_group_id(0);
   const int candidate_id = block * block_size + lane;
   __local float2 candidates[LOCAL_ARRAY_SIZE];
   if(block_size <= 0 || block_size > LOCAL_ARRAY_SIZE || (block_size & (block_size - 1)) != 0 ||
      top_k <= 0 || top_k > block_size)
      return;
   const float2 candidate = (candidate_id < candidate_count ? source[candidate_id] :
                             (float2)(-1.0f, RAG_INVALID_SCORE));
   candidates[lane] = ((int)candidate.x >= 0 && isfinite(candidate.y) &&
                       candidate.y > RAG_INVALID_SCORE ? candidate :
                       (float2)(-1.0f, RAG_INVALID_SCORE));
   BarrierLoc;
   RAGSortDescending(candidates, lane, block_size);
   if(lane < top_k)
      destination[block * top_k + lane] = candidates[lane];
  }
//+------------------------------------------------------------------+
//| Implements the rag gather kernel on the GPU.                     |
//+------------------------------------------------------------------+
__kernel void RAGGather(__global const float2 *top,
                        __global const float *actions,
                        __global float *tokens,
                        const int scenario_count,
                        const int action_count,
                        const int action_dimension,
                        const int token_width,
                        const int top_k,
                        __global const float *action_used,
                        __global const float *action_mean_reward,
                        __global const float *action_hit_count,
                        __global float *absolute_relevance)
  {
   const int token = get_global_id(0);
   const int total = top_k * action_count;
   if(token >= total || action_count <= 0 || action_dimension <= 0 || token_width != action_dimension + 2)
      return;
   if(token == 0)
     {
      float best_score = MIN_VALUE;
      int has_action = 0;
      for(int candidate_rank = 0; candidate_rank < top_k; candidate_rank++)
        {
         const int candidate_scenario = (int)top[candidate_rank].x;
         if(candidate_scenario < 0 || candidate_scenario >= scenario_count || top[candidate_rank].y <= MIN_VALUE)
            continue;
         for(int candidate_action = 0; candidate_action < action_count; candidate_action++)
           {
            const int candidate_slot = candidate_scenario * action_count + candidate_action;
            if(action_used[candidate_slot] > 0.5f && action_hit_count[candidate_slot] > 0.0f)
              {
               best_score = fmax(best_score, IsNaNOrInf(top[candidate_rank].y, MIN_VALUE));
               has_action = 1;
               break;
              }
           }
        }
      absolute_relevance[0] = (has_action ? IsNaNOrInf(best_score, 0.0f) : 0.0f);
     }
   const int rank = token / action_count, action = token - rank * action_count;
   const int scenario = (int)top[rank].x;
   const int width = token_width;
   const int out = token * width;
   if(scenario < 0 || scenario >= scenario_count || top[rank].y <= MIN_VALUE)
     {
      for(int d = 0; d < width; d++)
         tokens[out + d] = 0.0f;
      return;
     }
   const int slot = scenario * action_count + action;
   const int valid = (action_used[slot] > 0.5f && action_hit_count[slot] > 0.0f);
   if(!valid)
     {
      for(int d = 0; d < width; d++)
         tokens[out + d] = 0.0f;
      return;
     }
   const float raw_relevance = exp(clamp(IsNaNOrInf(top[rank].y, MIN_VALUE), -16.0f, 16.0f)) *
                               log(1.0f + fmax(IsNaNOrInf(action_hit_count[slot], 0.0f), 0.0f));
   float relevance_sum = 0.0f;
   for(int candidate_rank = 0; candidate_rank < top_k; candidate_rank++)
     {
      const int candidate_scenario = (int)top[candidate_rank].x;
      if(candidate_scenario < 0 || candidate_scenario >= scenario_count || top[candidate_rank].y <= MIN_VALUE)
         continue;
      for(int candidate_action = 0; candidate_action < action_count; candidate_action++)
        {
         const int candidate_slot = candidate_scenario * action_count + candidate_action;
         if(action_used[candidate_slot] <= 0.5f || action_hit_count[candidate_slot] <= 0.0f)
            continue;
         relevance_sum += exp(clamp(IsNaNOrInf(top[candidate_rank].y, MIN_VALUE), -16.0f, 16.0f)) *
                          log(1.0f + fmax(IsNaNOrInf(action_hit_count[candidate_slot], 0.0f), 0.0f));
        }
     }
   const float weighted_relevance = (relevance_sum > 1.0e-20f ? raw_relevance / relevance_sum : 0.0f);
   if(weighted_relevance <= 0.0f)
     {
      for(int d = 0; d < width; d++)
         tokens[out + d] = 0.0f;
      return;
     }
   for(int d = 0; d < action_dimension; d++)
      tokens[out + d] = IsNaNOrInf(actions[slot * action_dimension + d], 0.0f);
   tokens[out + action_dimension] = weighted_relevance;
   tokens[out + action_dimension + 1] = IsNaNOrInf(action_mean_reward[slot], 0.0f);
  }
//+------------------------------------------------------------------+
//| Actor-owned differentiable rejection gate. Memory remains raw.   |
//+------------------------------------------------------------------+
__kernel void RAGTokenGate(__global const float *tokens,
                           __global float *gated_tokens,
                           __global const float *absolute_relevance,
                           __global const float *threshold_weights,
                           __global float *branch_activity,
                           const int token_count,
                           const int token_width,
                           const float scale)
  {
   if(get_global_id(0) > 0 || token_count <= 0 || token_width < 2)
      return;
   int has_tokens = 0;
   for(int token = 0; token < token_count; token++)
     {
      const float weighted_relevance =
         IsNaNOrInf(tokens[token * token_width + token_width - 2], 0.0f);
      if(weighted_relevance > 0.0f)
        {
         has_tokens = 1;
         break;
        }
     }
   if(!has_tokens)
     {
      for(int i = 0; i < token_count * token_width; i++)
         gated_tokens[i] = 0.0f;
      branch_activity[0] = 0.0f;
      return;
     }
   const float threshold = IsNaNOrInf(threshold_weights[1], 0.0f);
   const float score = IsNaNOrInf(absolute_relevance[0], 0.0f);
   const float z = clamp(scale * (score - threshold), -40.0f, 40.0f);
   const float sigmoid = 1.0f / (1.0f + exp(-z));
   for(int i = 0; i < token_count * token_width; i++)
      gated_tokens[i] = IsNaNOrInf(tokens[i] * sigmoid, 0.0f);
   branch_activity[0] = sigmoid;
  }
//+------------------------------------------------------------------+
//| Accumulate only the threshold derivative; memory has no gradient |
//+------------------------------------------------------------------+
__kernel void RAGTokenGateGrad(__global const float *tokens,
                               __global const float *gated_gradient,
                               __global const float *absolute_relevance,
                               __global const float *threshold_weights,
                               __global float *threshold_gradient,
                               __global const float *branch_activity_gradient,
                               const int token_count,
                               const int token_width,
                               const float scale)
  {
   if(get_global_id(0) > 0 || token_count <= 0 || token_width < 2)
      return;
   const float activity_gradient = IsNaNOrInf(branch_activity_gradient[0], 0.0f);
   int has_tokens = 0;
   for(int token = 0; token < token_count; token++)
      if(IsNaNOrInf(tokens[token * token_width + token_width - 2], 0.0f) > 0.0f)
        {
         has_tokens = 1;
         break;
        }
   if(!has_tokens)
     {
      threshold_gradient[0] = 0.0f;
      return;
     }
   const float threshold = IsNaNOrInf(threshold_weights[1], 0.0f);
   const float score = IsNaNOrInf(absolute_relevance[0], 0.0f);
   const float z = clamp(scale * (score - threshold), -40.0f, 40.0f);
   const float g = 1.0f / (1.0f + exp(-z));
   float token_gradient = 0.0f;
   for(int token = 0; token < token_count; token++)
     {
      const int offset = token * token_width;
      for(int d = 0; d < token_width; d++)
         token_gradient += IsNaNOrInf(gated_gradient[offset + d], 0.0f) *
                           IsNaNOrInf(tokens[offset + d], 0.0f);
     }
   const float gradient = (token_gradient + activity_gradient) * (-scale * g * (1.0f - g));
   threshold_gradient[0] = clamp(IsNaNOrInf(gradient, 0.0f), -MAX_GRAD, MAX_GRAD);
  }
//+------------------------------------------------------------------+
//| Generic bounded-memory nearest-pair reduction.  One work-group   |
//| independently reduces one row of candidates; output is float4    |
//| {squared_distance,left,right,valid}.                             |
//+------------------------------------------------------------------+
inline void RAGNearestPairIndex(const int pair, const int count, int *left, int *right)
  {
   if(pair < 0 || count < 2)
     {
      *left = -1;
      *right = -1;
      return;
     }
   const long target = (long)pair;
   int low = 0;
   int high = count - 1;
// Find the last triangular row prefix not greater than target.  Exact
// 64-bit intermediates keep the mapping stable at count==65536.
   while(low + 1 < high)
     {
      const int middle = low + (high - low) / 2;
      const long before = ((long)middle * (2L * (long)count - (long)middle - 1L)) / 2L;
      if(before <= target)
         low = middle;
      else
         high = middle;
     }
   const long row_start = ((long)low * (2L * (long)count - (long)low - 1L)) / 2L;
   const long column = (long)low + 1L + (target - row_start);
   *left = (column < (long)count ? low : -1);
   *right = (column < (long)count ? (int)column : -1);
  }
//+------------------------------------------------------------------+
//| Implements the rag nearest before helper on the GPU.             |
//+------------------------------------------------------------------+
inline int RAGNearestBefore(const float4 left, const float4 right)
  {
// Validity is part of the ordering.  A valid distance saturated to
// MAXFLOAT must still beat the invalid MAXFLOAT sentinel.
   if(left.w != right.w)
      return left.w > right.w;
   if(left.w <= 0.0f)
      return 0;
   if(left.x != right.x)
      return left.x < right.x;
   if(left.y != right.y)
      return left.y < right.y;
   return left.z < right.z;
  }
//+------------------------------------------------------------------+
//| Implements the rag nearest pair kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void RAGNearestPair(__global const float *centroids,
                             __global const float *used,
                             __global float4 *result,
                             const int candidate_count,
                             const int dimension,
                             const int batch_count)
  {
   const int lane = get_local_id(0);
   const int batch = get_group_id(0);
   const int lanes = get_local_size(0);
   __local float4 best[LOCAL_ARRAY_SIZE];
   float4 current = (float4)(MAXFLOAT, -1.0f, -1.0f, 0.0f);
   if(candidate_count > 1 && dimension > 0 && batch >= 0 && batch < batch_count &&
      lanes > 0 && lanes <= LOCAL_ARRAY_SIZE)
     {
      // candidate_count<=65536: divide the even factor before multiplying so
      // the documented maximum (2,147,450,880 pairs) stays in signed int.
      const int pairs = ((candidate_count & 1) == 0 ?
                         (candidate_count / 2) * (candidate_count - 1) :
                         candidate_count * ((candidate_count - 1) / 2));
      const int base = batch * candidate_count;
      for(int pair = lane; pair < pairs; pair += lanes)
        {
         int left = -1, right = -1;
         RAGNearestPairIndex(pair, candidate_count, &left, &right);
         if(left < 0 || right < 0 || used[base + left] <= 0.5f || used[base + right] <= 0.5f)
            continue;
         float distance = 0.0f;
         const int lbase = (base + left) * dimension;
         const int rbase = (base + right) * dimension;
         for(int d = 0; d < dimension; d++)
           {
            const float delta = IsNaNOrInf(centroids[lbase + d] - centroids[rbase + d], MAXFLOAT);
            // Match the CPU fallback's explicit float rounding and prevent
            // contraction from changing the nearest-pair ordering.
            volatile float squared = delta * delta;
            distance = distance + squared;
           }
         const float4 candidate = (float4)(IsNaNOrInf(distance, MAXFLOAT), (float)left, (float)right, 1.0f);
         if(RAGNearestBefore(candidate, current))
            current = candidate;
        }
     }
   best[lane] = current;
   BarrierLoc;
   for(int stride = lanes >> 1; stride > 0; stride >>= 1)
     {
      if(lane < stride && RAGNearestBefore(best[lane + stride], best[lane]))
         best[lane] = best[lane + stride];
      BarrierLoc;
     }
   if(lane == 0 && batch >= 0 && batch < batch_count)
      result[batch] = best[0];
  }
//+------------------------------------------------------------------+
//| Computes the cog driver forecast gradient rms gradient           |
//| contributions on the GPU.                                        |
//+------------------------------------------------------------------+
__kernel void CogDriverForecastGradientRMS(__global const float* gradient,
      __global float* error_target,
      const int dimension,
      const int horizon
                                          )
  {
   const int h = get_global_id(0);
   if(h >= horizon || dimension <= 0)
      return;
   float sum = 0.0f;
   for(int d = 0; d < dimension; d++)
     {
      const int idx = h * dimension + d;
      const float g = IsNaNOrInf(gradient[idx], 0.0f);
      sum += g * g;
     }
   error_target[h] = sqrt(sum / (float)dimension);
  }
//+------------------------------------------------------------------+
//| Implements the sparse softmax hausdorff kernel on the GPU.       |
//+------------------------------------------------------------------+
__kernel void SparseSoftmaxHausdorff(__global const float *plan,
                                     __global const float *decis,
                                     __global float *output,
                                     __global float *top_index,
                                     const int dimension,
                                     const int top_k
                                    )
  {
   const int d = get_local_id(0);
   const int plan_id = get_global_id(1);
   const int decis_id = get_global_id(2);
   const int total_decis = get_global_size(2);
   const int pair = plan_id * total_decis + decis_id;
//---
   __local float values[LOCAL_ARRAY_SIZE];
   __local float top_values[LOCAL_ARRAY_SIZE];
   __local int top_indices[LOCAL_ARRAY_SIZE];
//---
   float value = 0.0f;
   if(d < dimension)
     {
      const float diff = plan[plan_id * dimension + d] - decis[decis_id * dimension + d];
      value = IsNaNOrInf(fabs(diff), 0.0f);
     }
   values[d] = value;
   BarrierLoc;
//---
   if(d == 0)
     {
      const int actual_top = min(top_k, dimension);
      for(int k = 0; k < actual_top; k++)
        {
         float best = MIN_VALUE;
         int best_id = -1;
         for(int i = 0; i < dimension; i++)
           {
            int used = 0;
            for(int p = 0; p < k; p++)
              {
               if(top_indices[p] == i)
                 {
                  used = 1;
                  break;
                 }
              }
            if(used)
               continue;
            const float candidate = values[i];
            if(candidate > best || (candidate == best && (best_id < 0 || i < best_id)))
              {
               best = candidate;
               best_id = i;
              }
           }
         top_values[k] = IsNaNOrInf(best, 0.0f);
         top_indices[k] = best_id;
        }
      //--- sparse softmax on selected dimensions
      float max_value = top_values[0];
      for(int k = 1; k < actual_top; k++)
         max_value = fmax(max_value, top_values[k]);
      float sum_exp = 0.0f;
      float result = 0.0f;
      for(int k = 0; k < actual_top; k++)
        {
         const float weight = IsNaNOrInf(exp(top_values[k] - max_value), 0.0f);
         sum_exp += weight;
         result += weight * top_values[k];
        }
      result = (sum_exp > 0.0f ? result / sum_exp : 0.0f);
      output[pair] = IsNaNOrInf(result, 0.0f);
      for(int k = 0; k < actual_top; k++)
         top_index[pair * top_k + k] = (float)top_indices[k];
     }
  }
//+------------------------------------------------------------------+
//| Computes the sparse softmax hausdorff grad gradient contributions|
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void SparseSoftmaxHausdorffGrad(__global const float *plan,
      __global float *grad_plan,
      __global const float *decis,
      __global float *grad_decis,
      __global const float *grad_out,
      __global const float *top_index,
      const int dimension,
      const int count_plan,
      const int count_decis,
      const int top_k
                                        )
  {
   const int d = get_global_id(0);
   const int id = get_global_id(1);
   const int side = get_global_id(2);
//---
   if(d >= dimension)
      return;
//---
   float result = 0.0f;
   if(side == 0)
     {
      if(id >= count_plan)
         return;
      for(int decis_id = 0; decis_id < count_decis; decis_id++)
        {
         const int pair = id * count_decis + decis_id;
         const int shift = pair * top_k;
         float max_value = MIN_VALUE;
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            const float diff = plan[id * dimension + top_d] - decis[decis_id * dimension + top_d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            max_value = fmax(max_value, value);
           }
         float sum_exp = 0.0f;
         float out_value = 0.0f;
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            const float diff = plan[id * dimension + top_d] - decis[decis_id * dimension + top_d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            const float exp_value = IsNaNOrInf(exp(value - max_value), 0.0f);
            sum_exp += exp_value;
            out_value += exp_value * value;
           }
         if(sum_exp <= 0.0f)
            continue;
         out_value /= sum_exp;
         const float grad = IsNaNOrInf(grad_out[pair], 0.0f);
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            if(top_d != d)
               continue;
            const float diff = plan[id * dimension + d] - decis[decis_id * dimension + d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            const float weight = IsNaNOrInf(exp(value - max_value) / sum_exp, 0.0f);
            const float sgn = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
            result += grad * weight * (1.0f + value - out_value) * sgn;
           }
        }
      grad_plan[id * dimension + d] = IsNaNOrInf(result, 0.0f);
     }
   else
     {
      if(id >= count_decis)
         return;
      for(int plan_id = 0; plan_id < count_plan; plan_id++)
        {
         const int pair = plan_id * count_decis + id;
         const int shift = pair * top_k;
         float max_value = MIN_VALUE;
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            const float diff = plan[plan_id * dimension + top_d] - decis[id * dimension + top_d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            max_value = fmax(max_value, value);
           }
         float sum_exp = 0.0f;
         float out_value = 0.0f;
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            const float diff = plan[plan_id * dimension + top_d] - decis[id * dimension + top_d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            const float exp_value = IsNaNOrInf(exp(value - max_value), 0.0f);
            sum_exp += exp_value;
            out_value += exp_value * value;
           }
         if(sum_exp <= 0.0f)
            continue;
         out_value /= sum_exp;
         const float grad = IsNaNOrInf(grad_out[pair], 0.0f);
         for(int k = 0; k < top_k; k++)
           {
            const int top_d = (int)(top_index[shift + k] + 0.001f);
            if(top_d != d)
               continue;
            const float diff = plan[plan_id * dimension + d] - decis[id * dimension + d];
            const float value = IsNaNOrInf(fabs(diff), 0.0f);
            const float weight = IsNaNOrInf(exp(value - max_value) / sum_exp, 0.0f);
            const float sgn = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
            result -= grad * weight * (1.0f + value - out_value) * sgn;
           }
        }
      grad_decis[id * dimension + d] = IsNaNOrInf(result, 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Implements the mom adttm mean scores kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void MomADTTMMeanScores(__global const float *distance,
                                 __global float *output,
                                 const int decisions
                                )
  {
   const int plan_id = get_global_id(0);
   if(decisions <= 0)
     {
      output[plan_id] = 0.0f;
      return;
     }
//---
   float sum = 0.0f;
   for(int j = 0; j < decisions; j++)
      sum += IsNaNOrInf(distance[plan_id * decisions + j], 0.0f);
   output[plan_id] = IsNaNOrInf(-sum / (float)decisions, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the mom adttm mean scores grad gradient contributions on|
//| the GPU.                                                         |
//+------------------------------------------------------------------+
__kernel void MomADTTMMeanScoresGrad(__global const float *scores_gr,
                                     __global float *distance_gr,
                                     const int decisions
                                    )
  {
   const int plan_id = get_global_id(0);
   const int decis_id = get_global_id(1);
   if(decisions <= 0)
      return;
//---
   distance_gr[plan_id * decisions + decis_id] =
      IsNaNOrInf(-scores_gr[plan_id] / (float)decisions, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the cog driver affine stack kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void CogDriverAffineStack(__global float* stack,
                                   __global float* snapshot,
                                   __global const float* affine,
                                   const int stack_size,
                                   const int quant_dimension
                                  )
  {
   const int feature = get_global_id(0);
   const int t = get_global_id(1);
   const int shift = feature * stack_size + t;
   const int modality = feature / quant_dimension;
   const float scale = IsNaNOrInf(affine[modality * 2], 0.0f);
   const float bias = IsNaNOrInf(affine[modality * 2 + 1], 0.0f);
   const float old = IsNaNOrInf(stack[shift], 0.0f);
//---
   snapshot[shift] = old;
   stack[shift] = IsNaNOrInf(old * scale + bias, 0.0f);
  }
//+------------------------------------------------------------------+
//| Computes the cog driver affine stack grad gradient contributions |
//| on the GPU.                                                      |
//+------------------------------------------------------------------+
__kernel void CogDriverAffineStackGrad(__global const float* snapshot,
                                       __global const float* stack_gr,
                                       __global float* affine_gr,
                                       const int stack_size,
                                       const int quant_dimension
                                      )
  {
   const int modality = get_global_id(0);
   const int param = get_global_id(1);
   const int start = modality * quant_dimension;
   float grad = 0.0f;
//---
   for(int f = 0; f < quant_dimension; f++)
     {
      const int feature = start + f;
      for(int t = 0; t < stack_size - 1; t++)
        {
         const int old_shift = feature * stack_size + t;
         const int grad_shift = feature * stack_size + t + 1;
         const float g = IsNaNOrInf(stack_gr[grad_shift], 0.0f);
         if(param == 0)
            grad += g * IsNaNOrInf(snapshot[old_shift], 0.0f);
         else
            grad += g;
        }
     }
   affine_gr[modality * 2 + param] = IsNaNOrInf(grad, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the cog driver rank tcm insert pending kernel on the  |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void CogDriverRankTCMInsertPending(__global float* memory,
      __global float* age,
      __global float* score,
      __global float* valid,
      __global const float* pending_query,
      const int memory_len,
      const int query_dim,
      const int variables
                                           )
  {
   const int variable = get_global_id(0);
   const int dim = get_global_id(1);
   const int lid = get_local_id(1);
   const int lsize = get_local_size(1);
   if(variable >= variables || dim >= query_dim || memory_len <= 0 || query_dim <= 0)
      return;
   __local int invalid_slot[LOCAL_ARRAY_SIZE];
   __local int min_slot[LOCAL_ARRAY_SIZE];
   __local int min_age[LOCAL_ARRAY_SIZE];
   __local float min_score[LOCAL_ARRAY_SIZE];
   __local int insert_slot[1];
   int local_invalid = memory_len;
   int local_min_slot = -1;
   int local_min_age = 0;
   float local_min_score = MAX_VALUE;
//--- each work-item scans a part of one variable memory row
   int step = min((int)LOCAL_ARRAY_SIZE, (int)lsize);
   if(lid < step)
     {
      for(int slot = lid; slot < memory_len; slot += step)
        {
         const int state_shift = RCtoFlat(slot, variable, memory_len, variables, 0);
         if(valid[state_shift] < 0.5f)
           {
            if(slot < local_invalid)
               local_invalid = slot;
            continue;
           }
         const float sc = IsNaNOrInf(score[state_shift], MIN_VALUE);
         const int t_age = (int)IsNaNOrInf(age[state_shift], 0.0f);
         age[state_shift] = (float)(t_age + 1);
         if(local_min_slot < 0 || sc < local_min_score ||
            (sc == local_min_score && t_age > local_min_age))
           {
            local_min_slot = slot;
            local_min_score = sc;
            local_min_age = t_age;
           }
        }
      invalid_slot[lid] = local_invalid;
      min_slot[lid] = local_min_slot;
      min_score[lid] = local_min_score;
      min_age[lid] = local_min_age;
     }
   BarrierLoc
//--- reduce to one replacement slot per variable
   if(lid == 0)
     {
      int best_invalid = memory_len;
      int best_min_slot = -1;
      int best_min_age = 0;
      float best_min_score = MAX_VALUE;
      for(int i = 0; i < lsize; i++)
        {
         if(invalid_slot[i] < best_invalid)
            best_invalid = invalid_slot[i];
         if(min_slot[i] >= 0 &&
            (best_min_slot < 0 || min_score[i] < best_min_score ||
             (min_score[i] == best_min_score && min_age[i] < best_min_age)))
           {
            best_min_slot = min_slot[i];
            best_min_score = min_score[i];
            best_min_age = min_age[i];
           }
        }
      insert_slot[0] = (best_invalid < memory_len ? best_invalid : best_min_slot);
     }
   BarrierLoc
   const int target_slot = insert_slot[0];
   if(target_slot < 0)
      return;
//--- write the delayed query to the replacement slot
   memory[RCtoFlat(variable, dim, variables, query_dim, target_slot)] =
      IsNaNOrInf(pending_query[RCtoFlat(variable, dim, variables, query_dim, 0)], 0.0f);
   if(lid == 0)
     {
      const int state_shift = RCtoFlat(target_slot, variable, memory_len, variables, 0);
      age[state_shift] = 1.0f;
      score[state_shift] = 0.0f;
      valid[state_shift] = 1.0f;
     }
  }
//+------------------------------------------------------------------+
//| Implements the cog driver rank tcm build kv kernel on the GPU.   |
//+------------------------------------------------------------------+
__kernel void CogDriverRankTCMBuildKV(__global const float* memory,
                                      __global const float* age,
                                      __global const float* valid,
                                      __global float* memory_kv,
                                      const int memory_count,
                                      const int variables,
                                      const int heads,
                                      const int head_dim
                                     )
  {
   const int slot = get_global_id(0);
   const int variable = get_global_id(1);
   const int flat_dim = get_global_id(2);
   if(slot >= memory_count || variable >= variables || heads <= 0 || head_dim <= 0)
      return;
   const int query_dim = heads * head_dim;
   if(flat_dim >= query_dim)
      return;
//---
   const int head = flat_dim / head_dim;
   const int local_dim = flat_dim - head * head_dim;
   const int state_shift = RCtoFlat(slot, variable, memory_count, variables, 0);
   float value = 0.0f;
   if(valid[state_shift] >= 0.5f)
     {
      const float a = IsNaNOrInf(age[state_shift], 0.0f);
      const float period = pow(10000.0f, (float)(2 * (local_dim / 2)) / (float)max(head_dim, 1));
      const float phase = a / period;
      const float pe = ((local_dim & 1) == 0 ? sin(phase) : cos(phase));
      value = IsNaNOrInf(memory[RCtoFlat(variable, flat_dim, variables, query_dim, slot)] + pe, 0.0f);
     }
   const int total_heads = variables * heads;
   const int effective_head = variable * heads + head;
   const int k_shift = RCtoFlat(effective_head, local_dim, 2 * total_heads, head_dim, slot);
   const int v_shift = RCtoFlat(effective_head + total_heads, local_dim, 2 * total_heads, head_dim, slot);
   memory_kv[k_shift] = value;
   memory_kv[v_shift] = value;
  }
//+------------------------------------------------------------------+
//| Implements the cog driver rank tcm mean flash score kernel on the|
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void CogDriverRankTCMMeanFlashScore(__global const float* query,
      __global const float* memory_kv,
      __global const float* valid,
      __global float* out_score,
      __global const float* logsumexp,
      const int memory_count,
      const int variables,
      const int heads,
      const int head_dim
                                            )
  {
   const int slot = get_global_id(0);
   const int variable = get_global_id(1);
   if(slot >= memory_count || variable >= variables || heads <= 0 || head_dim <= 0)
      return;
   const int state_shift = RCtoFlat(slot, variable, memory_count, variables, 0);
   if(valid[state_shift] < 0.5f)
     {
      out_score[state_shift] = MIN_VALUE;
      return;
     }
//---
   float score = 0.0f;
   const int total_heads = variables * heads;
   for(int head = 0; head < heads; head++)
     {
      const int effective_head = variable * heads + head;
      const int shift_q = RCtoFlat(effective_head, 0, total_heads, head_dim, 0);
      const int shift_k = RCtoFlat(effective_head, 0, 2 * total_heads, head_dim, slot);
      const int shift_lse = RCtoFlat(0, effective_head, 1, total_heads, 0);
      const float head_lse = IsNaNOrInf(logsumexp[shift_lse], 0.0f);
      float dotv = 0.0f;
      for(int d = 0; d < head_dim; d++)
         dotv += IsNaNOrInf(query[shift_q + d], 0.0f) *
                 IsNaNOrInf(memory_kv[shift_k + d], 0.0f);
      const float raw_score = IsNaNOrInf(dotv / sqrt((float)head_dim), MIN_VALUE);
      score += IsNaNOrInf(exp(clamp(raw_score - head_lse, -120.0f, 0.0f)), 0.0f);
     }
   out_score[state_shift] = IsNaNOrInf(score / (float)heads, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the cog driver rank tcm self kv kernel on the GPU.    |
//+------------------------------------------------------------------+
__kernel void CogDriverRankTCMSelfKV(__global const float* source,
                                     __global float* self_kv,
                                     const int query_dim,
                                     const int variables,
                                     const int heads,
                                     const int head_dim
                                    )
  {
   const int key_variable = get_global_id(0);
   const int flat_dim = get_global_id(1);
   if(key_variable >= variables || flat_dim >= query_dim || heads <= 0 || head_dim <= 0)
      return;
//---
   const int head = flat_dim / head_dim;
   const int local_dim = flat_dim - head * head_dim;
   const int total_heads = variables * heads;
   const float value = IsNaNOrInf(source[RCtoFlat(key_variable, flat_dim, variables, query_dim, 0)], 0.0f);
   for(int query_variable = 0; query_variable < variables; query_variable++)
     {
      const int effective_head = query_variable * heads + head;
      const int k_shift = RCtoFlat(effective_head, local_dim, 2 * total_heads, head_dim, key_variable);
      const int v_shift = RCtoFlat(effective_head + total_heads, local_dim, 2 * total_heads, head_dim, key_variable);
      self_kv[k_shift] = value;
      self_kv[v_shift] = value;
     }
  }
//+------------------------------------------------------------------+
//| Scenario-major forecast: Z[k][h][d], U[k][h], Pi[k]              |
//+------------------------------------------------------------------+
__kernel void ScenarioForecastForward(__global const float *input,
                                      __global const float *prototypes,
                                      __global float *generator_input,
                                      __global float *router_input,
                                      __global float *confidence_input,
                                      __global const float *generator_output,
                                      __global const float *router_output,
                                      __global const float *confidence_output,
                                      __global float *z,
                                      __global float *u,
                                      __global float *pi,
                                      const int scenarios,
                                      const int horizon,
                                      const int dimension,
                                      const int phase,
                                      __global float *log_pi,
                                      __global float *log_u,
                                      const int variables,
                                      __global float *forward_validity,
                                      __global const float *generator_weights,
                                      __global const float *router_weights,
                                      __global const float *confidence_weights,
                                      const int generator_weights_count,
                                      const int router_weights_count,
                                      const int confidence_weights_count)
  {
   const int d = get_global_id(0);
   const int v = get_global_id(1);
   const int k = get_global_id(2);
   const int ld = get_local_id(0);
   const int lv = get_local_id(1);
   __local float Temp[SCENARIO_LOCAL_CAP];
   if(scenarios <= 0 || variables <= 0 || horizon <= 0 || dimension <= 0 ||
      dimension * variables > SCENARIO_LOCAL_CAP || d >= dimension || v >= variables || k >= scenarios)
      return;
   const float raw_state = input[v * dimension + d];
   const float state = IsNaNOrInf(raw_state, 0.0f);
   float valid = (isnan(raw_state) || isinf(raw_state) ? 0.0f : 1.0f);
   if(phase == 0)
     {
      const int global_lane = (k * variables + v) * dimension + d;
      const int total_lanes = scenarios * variables * dimension;
      for(int index = global_lane; index < generator_weights_count; index += total_lanes)
         if(isnan(generator_weights[index]) || isinf(generator_weights[index]))
            valid = 0.0f;
      for(int index = global_lane; index < router_weights_count; index += total_lanes)
         if(isnan(router_weights[index]) || isinf(router_weights[index]))
            valid = 0.0f;
      for(int index = global_lane; index < confidence_weights_count; index += total_lanes)
         if(isnan(confidence_weights[index]) || isinf(confidence_weights[index]))
            valid = 0.0f;
      float prototype_mean = 0.0f;
      for(int h = 0; h < horizon; h++)
        {
         const int prototype_shift = ((k * variables + v) * horizon + h) * dimension + d;
         const int condition_shift = ((k * variables + v) * horizon + h) * 2 * dimension;
         const float raw_prototype = prototypes[prototype_shift];
         const float prototype = IsNaNOrInf(raw_prototype, 0.0f);
         if(isnan(raw_prototype) || isinf(raw_prototype))
            valid = 0.0f;
         generator_input[condition_shift + d] = state;
         generator_input[condition_shift + dimension + d] = prototype;
         confidence_input[condition_shift + d] = state;
         confidence_input[condition_shift + dimension + d] = prototype;
         prototype_mean = IsNaNOrInf(prototype_mean + prototype, 0.0f);
        }
      const int router_shift = k * 2 * variables * dimension;
      router_input[router_shift + v * dimension + d] = state;
      router_input[router_shift + variables * dimension + v * dimension + d] =
         IsNaNOrInf(prototype_mean / (float)horizon, 0.0f);
      const float variable_valid = LocalMin(valid, 0, Temp + lv * dimension);
      BarrierLoc
      const float joint_valid = LocalMin(ld == 0 ? variable_valid : 1.0f, 1, Temp + ld * variables);
      if(ld == 0 && lv == 0)
         forward_validity[k] = (joint_valid > 0.5f ? 1.0f : 0.0f);
      return;
     }
   for(int h = 0; h < horizon; h++)
     {
      const int prototype_shift = ((k * variables + v) * horizon + h) * dimension + d;
      const float raw_prototype = prototypes[prototype_shift];
      const float prototype = IsNaNOrInf(raw_prototype, 0.0f);
      const float raw_residual = generator_output[prototype_shift];
      const float residual = IsNaNOrInf(raw_residual, 0.0f);
      if(isnan(raw_state) || isinf(raw_state) || isnan(raw_prototype) || isinf(raw_prototype) ||
         isnan(raw_residual) || isinf(raw_residual))
         valid = 0.0f;
      z[prototype_shift] = IsNaNOrInf(state + prototype + residual, 0.0f);
      if(d == 0)
        {
         const int confidence_shift = (k * variables + v) * horizon + h;
         const float raw_confidence = confidence_output[confidence_shift];
         if(isnan(raw_confidence) || isinf(raw_confidence))
            valid = 0.0f;
         const float confidence = clamp(IsNaNOrInf(raw_confidence, 0.0f) + 0.001f, 0.001f, 10.0f);
         u[confidence_shift] = IsNaNOrInf(confidence, 0.001f);
         log_u[confidence_shift] = IsNaNOrInf(log(confidence + 1.0e-6f), log(0.001001f));
        }
     }
   if(d == 0 && v == 0)
     {
      const float raw_router = router_output[k];
      const float raw_probability = pi[k];
      if(isnan(raw_router) || isinf(raw_router) || isnan(raw_probability) || isinf(raw_probability))
         valid = 0.0f;
      const float probability = clamp(IsNaNOrInf(raw_probability, 0.0f), 0.0f, 1.0f);
      log_pi[k] = IsNaNOrInf(log(probability + 1.0e-6f), log(1.0e-6f));
     }
   const float variable_valid = LocalMin(valid, 0, Temp + lv * dimension);
   BarrierLoc
   const float joint_valid = LocalMin(ld == 0 ? variable_valid : 1.0f, 1, Temp + ld * variables);
   if(ld == 0 && lv == 0)
      forward_validity[k] = (forward_validity[k] > 0.5f && joint_valid > 0.5f ? 1.0f : 0.0f);
  }
//+------------------------------------------------------------------+
//| Stable Laplace mixture posterior with TopK renormalization       |
//+------------------------------------------------------------------+
__kernel void ScenarioResponsibilities(__global float *target,
                                       __global const float *z,
                                       __global const float *u,
                                       __global const float *pi,
                                       __global float *responsibilities,
                                       __global float *generator_gradient,
                                       __global float *router_gradient,
                                       __global float *confidence_gradient,
                                       const int scenarios,
                                       const int top_k,
                                       const int horizon,
                                       const int dimension,
                                       __global float *scores,
                                       __global const float *sparse,
                                       __global const float *indexes,
                                       const int phase,
                                       const int variables,
                                       __global const float *prototypes,
                                       __global float *batch_control,
                                       const float huber_delta,
                                       __global const float *target_source,
                                       __global const float *forward_validity,
                                       __global float *diagnostic_partials,
                                       __global float *epoch_diagnostics,
                                       __global const float *observation_by_scenario,
                                       __global float *diagnostic_scratch,
                                       __global const float *usage,
                                       __global const float *inactive,
                                       __global float *recovery_events,
                                       __global float *age_zero_events)
  {
   if(variables <= 0 || scenarios <= 0 || top_k <= 0 || top_k > scenarios ||
      horizon <= 0 || dimension <= 0 || dimension * variables > SCENARIO_LOCAL_CAP)
      return;
   const int d = get_global_id(0);
   const int v = get_global_id(1);
   const int k = get_global_id(2);
   const int ld = get_local_id(0);
   const int lv = get_local_id(1);
   __local float Temp[SCENARIO_LOCAL_CAP];
   __local float TempSecond[SCENARIO_LOCAL_CAP];
   if(phase == 0)
     {
      if(d >= dimension || v >= variables)
         return;
      float valid = 1.0f;
      for(int h = 0; h < horizon; h++)
        {
         const int shift = (v * horizon + h) * dimension + d;
         const float raw = target_source[shift];
         if(isnan(raw) || isinf(raw))
            valid = 0.0f;
         target[shift] = IsNaNOrInf(raw, 0.0f);
        }
      const float variable_valid = LocalMin(valid, 0, Temp + lv * dimension);
      BarrierLoc
      const float joint_valid = LocalMin(ld == 0 ? variable_valid : 1.0f, 1, Temp + ld * variables);
      if(ld == 0 && lv == 0)
        {
         float forecast_valid = 1.0f;
         for(int scenario = 0; scenario < scenarios; scenario++)
            forecast_valid = min(forecast_valid, forward_validity[scenario]);
         batch_control[0] = (joint_valid > 0.5f && forecast_valid > 0.5f ? 1.0f : 0.0f);
        }
      return;
     }
   if(phase == 1)
     {
      if(d >= dimension || v >= variables || k >= scenarios)
         return;
      float laplace = 0.0f;
      float trajectory = 0.0f;
      float confidence_cost = 0.0f;
      float latent_abs = 0.0f;
      for(int h = 0; h < horizon; h++)
        {
         const int shift = ((k * variables + v) * horizon + h) * dimension + d;
         const int target_shift = (v * horizon + h) * dimension + d;
         const float scale = clamp(IsNaNOrInf(u[(k * variables + v) * horizon + h], 1.0f), 1.0e-3f, 10.0f);
         const float error = IsNaNOrInf(target[target_shift] - IsNaNOrInf(z[shift], 0.0f), 0.0f);
         laplace = IsNaNOrInf(laplace + fabs(error) / scale + log(2.0f * scale), 0.0f);
         const float absolute_error = fabs(error);
         trajectory = IsNaNOrInf(trajectory + (absolute_error <= huber_delta ?
                                               0.5f * absolute_error * absolute_error :
                                               huber_delta * (absolute_error - 0.5f * huber_delta)), 0.0f);
         confidence_cost = IsNaNOrInf(confidence_cost + log(scale) + absolute_error / scale, 0.0f);
         latent_abs = IsNaNOrInf(latent_abs + absolute_error, 0.0f);
        }
      const float variable_laplace = LocalSum(laplace, 0, Temp + lv * dimension);
      BarrierLoc
      const float joint_laplace = LocalSum(ld == 0 ? variable_laplace : 0.0f, 1, Temp + ld * variables);
      const float variable_trajectory = LocalSum(trajectory, 0, Temp + lv * dimension);
      const float variable_confidence = LocalSum(confidence_cost, 0, TempSecond + lv * dimension);
      BarrierLoc
      const float joint_trajectory = LocalSum(ld == 0 ? variable_trajectory : 0.0f, 1, Temp + ld * variables);
      const float joint_confidence = LocalSum(ld == 0 ? variable_confidence : 0.0f, 1, TempSecond + ld * variables);
      const float variable_latent = LocalSum(latent_abs, 0, Temp + lv * dimension);
      BarrierLoc
      const float joint_latent = LocalSum(ld == 0 ? variable_latent : 0.0f, 1, Temp + ld * variables);
      if(ld == 0 && lv == 0)
        {
         const bool valid = (batch_control[0] > 0.5f);
         //--- The likelihood is accumulated across the whole joint trajectory. Scale
         //--- its complete log-posterior by the root coordinate count so the
         //--- sparse TopK posterior remains representable for all selected scenarios.
         const float score_scale = sqrt(max((float)variables * (float)horizon * (float)dimension, 1.0f));
         scores[k] = (valid ? IsNaNOrInf((log(max(IsNaNOrInf(pi[k], 0.0f), 1.0e-12f)) - joint_laplace) /
                                         score_scale, MIN_VALUE) : MIN_VALUE);
         diagnostic_partials[k] = (valid ? IsNaNOrInf(joint_trajectory, 0.0f) : 0.0f);
         diagnostic_partials[scenarios + k] = (valid ? IsNaNOrInf(joint_confidence, 0.0f) : 0.0f);
         diagnostic_partials[2 * scenarios + k] = (valid ? IsNaNOrInf(joint_latent, 0.0f) : 0.0f);
        }
      return;
     }
   if(phase == 2)
     {
      if(d >= dimension || v >= variables || k >= scenarios)
         return;
      float responsibility = 0.0f;
      if(batch_control[0] > 0.5f)
         for(int selected = 0; selected < top_k; selected++)
            if((int)indexes[selected] == k)
              {
               responsibility = IsNaNOrInf(sparse[selected], 0.0f);
               break;
              }
      if(d == 0 && v == 0)
        {
         responsibilities[k] = responsibility;
         //--- Update kernels add their gradient.  Use the library convention
         //--- target-output, i.e. posterior responsibility minus router Pi.
         router_gradient[k] = (batch_control[0] > 0.5f ?
                               responsibility - IsNaNOrInf(pi[k], 0.0f) : 0.0f);
        }
      for(int h = 0; h < horizon; h++)
        {
         const int confidence_shift = (k * variables + v) * horizon + h;
         const int shift = confidence_shift * dimension + d;
         const int target_shift = (v * horizon + h) * dimension + d;
         const float scale = clamp(IsNaNOrInf(u[confidence_shift], 1.0f), 1.0e-3f, 10.0f);
         const float error = IsNaNOrInf(target[target_shift] - z[shift], 0.0f);
         generator_gradient[shift] = (responsibility > 0.0f ?
                                      (error > 0.0f ? 1.0f : (error < 0.0f ? -1.0f : 0.0f)) / scale : 0.0f);
         const float absolute_sum = LocalSum(responsibility > 0.0f ? fabs(error) : 0.0f,
                                             0, Temp + lv * dimension);
         if(ld == 0)
           {
            //--- U is SoftPlus output plus epsilon.  Convert -dL/dU to the
            //--- same target-output convention used by convolution updates.
            //--- A tiny floor lets a confidence collapsed at epsilon recover.
            const float softplus_output = max(scale - 0.001f, 0.0f);
            const float softplus_gradient = max(1.0f - exp(-softplus_output), 1.0e-3f);
            const float confidence_signal = absolute_sum / (scale * scale) -
                                            (float)dimension / scale;
            confidence_gradient[confidence_shift] = (responsibility > 0.0f && scale < 10.0f ?
                                                    IsNaNOrInf(confidence_signal * softplus_gradient, 0.0f) : 0.0f);
           }
         BarrierLoc
        }
      return;
     }
   if(phase == 3)
     {
      const int scenario = get_local_id(0);
      if(scenario >= scenarios)
         return;
      const float score = scores[scenario];
      const float responsibility = clamp(IsNaNOrInf(responsibilities[scenario], 0.0f), 0.0f, 1.0f);
      const float maximum = LocalMax(score, 0, Temp);
      const float sum_exp = LocalSum(exp(score - maximum), 0, TempSecond);
      const float router_value = -responsibility * log(max(IsNaNOrInf(pi[scenario], 0.0f), 1.0e-12f));
      const float router_sum = LocalSum(router_value, 0, Temp);
      const float trajectory_sum = LocalSum(responsibility * diagnostic_partials[scenario], 0, TempSecond);
      const float confidence_sum = LocalSum(responsibility * diagnostic_partials[scenarios + scenario], 0, Temp);
      const float latent_sum = LocalSum(responsibility * diagnostic_partials[2 * scenarios + scenario], 0, TempSecond);
      const float observation_sum = LocalSum(IsNaNOrInf(observation_by_scenario[scenario], 0.0f), 0, Temp);
      if(scenario == 0)
        {
         if(batch_control[0] > 0.5f)
           {
            epoch_diagnostics[0] += IsNaNOrInf(-(maximum + log(max(sum_exp, 1.0e-30f))), 0.0f);
            epoch_diagnostics[1] += IsNaNOrInf(router_sum, 0.0f);
            epoch_diagnostics[2] += IsNaNOrInf(trajectory_sum, 0.0f);
            epoch_diagnostics[3] += IsNaNOrInf(confidence_sum, 0.0f);
            epoch_diagnostics[4] += IsNaNOrInf(latent_sum, 0.0f);
            epoch_diagnostics[5] += IsNaNOrInf(observation_sum, 0.0f);
            epoch_diagnostics[6] += 1.0f;
           }
         else
            epoch_diagnostics[7] += 1.0f;
        }
      return;
     }
   if(phase == 4)
     {
      if(get_global_id(0) == 0)
         for(int i = 0; i < 12; i++)
            epoch_diagnostics[i] = 0.0f;
      if(get_global_id(0) < scenarios)
        {
         recovery_events[get_global_id(0)] = 0.0f;
         age_zero_events[get_global_id(0)] = 0.0f;
        }
      return;
     }
   if(phase == 5)
     {
      const int pair = get_global_id(2);
      const int pairs = scenarios * (scenarios - 1) / 2;
      if(d >= dimension || v >= variables || pair >= pairs)
         return;
      int left = 0;
      int right = 1;
      int remainder = pair;
      for(left = 0; left < scenarios - 1; left++)
        {
         const int row = scenarios - left - 1;
         if(remainder < row)
           { right = left + 1 + remainder; break; }
         remainder -= row;
        }
      float square_sum = 0.0f;
      for(int h = 0; h < horizon; h++)
        {
         const int coordinate = (v * horizon + h) * dimension + d;
         const int stride = variables * horizon * dimension;
         const float difference = IsNaNOrInf(prototypes[left * stride + coordinate] -
                                             prototypes[right * stride + coordinate], 0.0f);
         square_sum += difference * difference;
        }
      const float variable_sum = LocalSum(square_sum, 0, Temp + lv * dimension);
      BarrierLoc
      const float joint_sum = LocalSum(ld == 0 ? variable_sum : 0.0f, 1, Temp + ld * variables);
      if(ld == 0 && lv == 0)
         diagnostic_scratch[pair] = sqrt(max(joint_sum, 0.0f) /
                                         max((float)(variables * horizon * dimension), 1.0f));
      return;
     }
   if(phase == 6)
     {
      const int lane = get_local_id(0);
      const int pairs = scenarios * (scenarios - 1) / 2;
      const float usage_value = (lane < scenarios ? max(IsNaNOrInf(usage[lane], 0.0f), 0.0f) : 0.0f);
      const float usage_sum = LocalSum(usage_value, 0, Temp);
      const float probability = (usage_sum > 0.0f ? usage_value / usage_sum : 0.0f);
      const float entropy_value = (probability > 0.0f ? -probability * log(probability) : 0.0f);
      const float entropy_sum = LocalSum(entropy_value, 0, TempSecond);
      if(lane == 0)
         epoch_diagnostics[8] = IsNaNOrInf(entropy_sum, 0.0f);
      BarrierLoc
      const float inactive_sum = LocalSum(lane < scenarios && IsNaNOrInf(inactive[lane], 0.0f) > 0.0f ?
                                          1.0f : 0.0f, 0, Temp);
      if(lane == 0)
         epoch_diagnostics[10] = IsNaNOrInf(inactive_sum, 0.0f);
      BarrierLoc
      const float pair_sum = LocalSum(lane < pairs ? IsNaNOrInf(diagnostic_scratch[lane], 0.0f) : 0.0f,
                                      0, Temp);
      if(lane == 0)
         epoch_diagnostics[9] = (pairs > 0 ? IsNaNOrInf(pair_sum / (float)pairs, 0.0f) : 0.0f);
      BarrierLoc
      const float recovered_sum = LocalSum(lane < scenarios ? IsNaNOrInf(recovery_events[lane], 0.0f) : 0.0f,
                                           0, Temp);
      if(lane == 0)
         epoch_diagnostics[11] = IsNaNOrInf(recovered_sum, 0.0f);
      return;
     }
  }
//+------------------------------------------------------------------+
//| Codebook EMA; prototypes are updated statistically, not by grad  |
//+------------------------------------------------------------------+
__kernel void ScenarioCodebookEMA(__global const float *target,
                                  __global float *prototypes,
                                  __global float *ema_sums,
                                  __global float *ema_counts,
                                  __global float *usage,
                                  __global float *inactive,
                                  __global const float *batch_control,
                                  const int scenarios,
                                  const int dimension,
                                  const float decay,
                                  __global float *inactivity_age,
                                  const int seed,
                                  const int recovery_age,
                                  __global float *recovery,
                                  const int phase,
                                  const int horizon,
                                  const int variables,
                                  __global const float *responsibilities,
                                  __global float *recovery_events,
                                  __global float *age_zero_events)
  {
   if(phase == 0)
     {
      const int k = get_global_id(0);
      if(k >= scenarios)
         return;
      recovery[k] = 0.0f;
      if(batch_control[0] <= 0.5f)
         return;
      //--- The same normalized TopK responsibility that trains Z_k updates
      //--- this scenario's Delta-Z EMA.  Non-TopK scenarios receive zero.
      const float hit = clamp(IsNaNOrInf(responsibilities[k], 0.0f), 0.0f, 1.0f);
      const float count = decay * IsNaNOrInf(ema_counts[k], 0.0f) + (1.0f - decay) * hit;
      const float age = (hit > 0.0f ? 0.0f : IsNaNOrInf(inactivity_age[k], 0.0f) + 1.0f);
      const bool recover = (hit == 0.0f && age >= (float)recovery_age);
      recovery[k] = (recover ? 1.0f : 0.0f);
      if(recover)
         recovery_events[k] = IsNaNOrInf(recovery_events[k], 0.0f) + 1.0f;
      ema_counts[k] = (recover ? 1.0f : count);
      usage[k] = IsNaNOrInf(usage[k], 0.0f) + hit;
      const float final_age = (recover ? 0.0f : age);
      inactivity_age[k] = final_age;
      if(final_age == 0.0f)
         age_zero_events[k] = IsNaNOrInf(age_zero_events[k], 0.0f) + 1.0f;
      inactive[k] = (!recover && count < 1.0e-4f ? 1.0f : 0.0f);
      return;
     }
   const int d = get_global_id(0);
   const int v = get_global_id(1);
   const int k = get_global_id(2);
   if(phase != 1 || d >= dimension || v >= variables || k >= scenarios || batch_control[0] <= 0.5f)
      return;
   const float hit = clamp(IsNaNOrInf(responsibilities[k], 0.0f), 0.0f, 1.0f);
   const float count = IsNaNOrInf(ema_counts[k], 0.0f);
   const bool recover = (recovery[k] > 0.5f);
   for(int h = 0; h < horizon; h++)
     {
      const int coordinate = (v * horizon + h) * dimension + d;
      const int shift = k * variables * horizon * dimension + coordinate;
      const float sample = IsNaNOrInf(target[coordinate], 0.0f);
      if(recover)
        {
         const float prototype = IsNaNOrInf(sample + 0.01f * sin((float)seed + 17.0f * (float)k +
                                            0.61803398875f * (float)(coordinate + 1)), sample);
         prototypes[shift] = prototype;
         ema_sums[shift] = prototype;
        }
      else
        {
         const float sum = IsNaNOrInf(decay * ema_sums[shift] + (1.0f - decay) * hit * sample, 0.0f);
         ema_sums[shift] = sum;
         if(count > 1.0e-6f)
            prototypes[shift] = IsNaNOrInf(sum / count, 0.0f);
        }
     }
  }
//+------------------------------------------------------------------+
//| Per-scenario RMS normalization; inactive branches remain zero    |
//+------------------------------------------------------------------+
__kernel void ScenarioGradientNormalize(__global float *generator_gradient,
                                        __global const float *generator_input_gradient,
                                        __global const float *router_input_gradient,
                                        __global const float *confidence_input_gradient,
                                        __global float *input_gradient,
                                        __global const float *responsibilities,
                                        const int scenarios,
                                        const int horizon,
                                        const int dimension,
                                        const int phase,
                                        __global float *confidence_gradient,
                                        const int variables,
                                        __global const float *batch_control)
  {
   const int d = get_global_id(0);
   const int v = get_global_id(1);
   const int k = get_global_id(2);
   const int ld = get_local_id(0);
   const int lv = get_local_id(1);
   __local float Temp[SCENARIO_LOCAL_CAP];
   __local float TempSecond[SCENARIO_LOCAL_CAP];
   if(phase == 0)
     {
      if(d >= dimension || v >= variables || k >= scenarios)
         return;
      const float responsibility = (batch_control[0] > 0.5f ?
                                    IsNaNOrInf(responsibilities[k], 0.0f) : 0.0f);
      float generator_square_sum = 0.0f;
      float confidence_square_sum = 0.0f;
      for(int h = 0; h < horizon; h++)
        {
         const int confidence_shift = (k * variables + v) * horizon + h;
         const float generator_value = IsNaNOrInf(generator_gradient[confidence_shift * dimension + d], 0.0f);
         generator_square_sum += generator_value * generator_value;
         if(d == 0)
           {
            const float confidence_value = IsNaNOrInf(confidence_gradient[confidence_shift], 0.0f);
            confidence_square_sum += confidence_value * confidence_value;
           }
        }
      const float variable_generator = LocalSum(generator_square_sum, 0, Temp + lv * dimension);
      const float variable_confidence = LocalSum(confidence_square_sum, 0, TempSecond + lv * dimension);
      BarrierLoc
      const float joint_generator = LocalSum(ld == 0 ? variable_generator : 0.0f, 1, Temp + ld * variables);
      const float joint_confidence = LocalSum(ld == 0 ? variable_confidence : 0.0f, 1, TempSecond + ld * variables);
      if(ld == 0 && lv == 0)
        {
         Temp[0] = joint_generator;
         TempSecond[0] = joint_confidence;
        }
      BarrierLoc
      const float generator_rms = sqrt(Temp[0] / max((float)(variables * horizon * dimension), 1.0f) + 1.0e-12f);
      const float confidence_rms = sqrt(TempSecond[0] / max((float)(variables * horizon), 1.0f) + 1.0e-12f);
      const float generator_scale = (responsibility > 0.0f ? responsibility / max(generator_rms, 1.0e-6f) : 0.0f);
      const float confidence_scale = (responsibility > 0.0f ? responsibility / max(confidence_rms, 1.0e-6f) : 0.0f);
      for(int h = 0; h < horizon; h++)
        {
         const int confidence_shift = (k * variables + v) * horizon + h;
         const int shift = confidence_shift * dimension + d;
         generator_gradient[shift] = IsNaNOrInf(generator_gradient[shift] * generator_scale, 0.0f);
         if(d == 0)
            confidence_gradient[confidence_shift] = IsNaNOrInf(confidence_gradient[confidence_shift] * confidence_scale, 0.0f);
        }
      return;
     }
   if(phase == 1)
     {
      const int lane = get_local_id(0);
      const int dimension_index = get_global_id(1);
      const int variable = get_global_id(2);
      const int lanes = get_local_size(0);
      if(dimension_index >= dimension || variable >= variables)
         return;
      float partial = 0.0f;
      if(batch_control[0] > 0.5f)
         for(int q = lane; q < scenarios * horizon; q += lanes)
           {
            const int scenario = q / horizon;
            const int h = q - scenario * horizon;
            const int router_shift = scenario * 2 * variables * dimension;
            if(h == 0)
               partial += IsNaNOrInf(router_input_gradient[router_shift + variable * dimension + dimension_index], 0.0f);
            const int shift = ((scenario * variables + variable) * horizon + h) * 2 * dimension + dimension_index;
            const int direct_shift = ((scenario * variables + variable) * horizon + h) * dimension + dimension_index;
            partial += IsNaNOrInf(generator_gradient[direct_shift], 0.0f) +
                       IsNaNOrInf(generator_input_gradient[shift], 0.0f) +
                       IsNaNOrInf(confidence_input_gradient[shift], 0.0f);
           }
      const float aggregate = LocalSum(partial, 0, Temp);
      if(lane == 0)
         input_gradient[variable * dimension + dimension_index] = IsNaNOrInf(aggregate, 0.0f);
      return;
     }
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill item partial kernel on the GPU.           |
//+------------------------------------------------------------------+
__kernel void D2SkillItemPartial(__global const float *source,
                                 __global float *partials,
                                 const int dimension,
                                 const int output_offset)
  {
   const int id = get_global_id(0);
   if(id < 0 || id >= dimension)
      return;
   const float value = IsNaNOrInf(source[id], 0.0f);
   partials[output_offset + id] = IsNaNOrInf(value * value, 0.0f);
  }
//+------------------------------------------------------------------+
//| Recursive no-atomics reduction into disjoint scratch regions.    |
//+------------------------------------------------------------------+
__kernel void D2SkillItemReduce(__global float *partials,
                                __global float *reduced,
                                const int input_count,
                                const int input_offset,
                                const int output_offset,
                                const int final_output)
  {
   const int index = get_global_id(0);
   const int left = 2 * index;
   if(left >= input_count)
      return;
   const float sum = IsNaNOrInf(partials[input_offset + left], 0.0f) +
                     (left + 1 < input_count ? IsNaNOrInf(partials[input_offset + left + 1], 0.0f) : 0.0f);
   if(final_output != 0)
      reduced[0] = IsNaNOrInf(sum, 0.0f);
   else
      partials[output_offset + index] = IsNaNOrInf(sum, 0.0f);
  }
//+------------------------------------------------------------------+
//| Parallel Item state update and correction materialization.       |
//+------------------------------------------------------------------+
#define D2_ITEM_APPLY_UPDATE    0
#define D2_ITEM_APPLY_CORRECTION 1

//+------------------------------------------------------------------+
//| Implements the D2Skill item apply kernel on the GPU.             |
//+------------------------------------------------------------------+
__kernel void D2SkillItemApply(__global const float *gradient,
                               __global float *correction,
                               __global float *direction,
                               __global float *scale,
                               __global float *observations,
                               __global float *mass,
                               __global const float *reduced_norm,
                               const int dimension,
                               const int representation,
                               const float beta_correction,
                               const float beta_direction,
                               const float beta_magnitude,
                               const int phase)
  {
   const int id = get_global_id(0);
   const float norm = sqrt(fmax(IsNaNOrInf(reduced_norm[0], 0.0f), 0.0f));
   if(phase == D2_ITEM_APPLY_UPDATE)
     {
      if(id < dimension)
        {
         const float g = IsNaNOrInf(gradient[id], 0.0f);
         if(representation == 0)
           {
            const float old_value = IsNaNOrInf(correction[id], 0.0f);
            correction[id] = (1.0f - beta_correction) * old_value + beta_correction * g;
           }
         else
           {
            const float inv_norm = (norm > 1.0e-12f ? 1.0f / norm : 0.0f);
            const float old_direction = IsNaNOrInf(direction[id], 0.0f);
            direction[id] = (1.0f - beta_direction) * old_direction +
                            beta_direction * g * inv_norm;
           }
        }
      if(id == 0)
        {
         if(representation != 0)
            scale[0] = clamp((1.0f - beta_magnitude) * IsNaNOrInf(scale[0], 0.0f) +
                             beta_magnitude * norm, 0.0f, MAX_VALUE);
         observations[0] = IsNaNOrInf(observations[0], 0.0f) + 1.0f;
         mass[0] = IsNaNOrInf(mass[0], 0.0f) + norm;
        }
      return;
     }
   if(id >= dimension || representation == 0)
      return;
   const float inv_direction = (norm > 1.0e-12f ? 1.0f / norm : 0.0f);
   correction[id] = IsNaNOrInf(scale[0], 0.0f) *
                    IsNaNOrInf(direction[id], 0.0f) * inv_direction;
  }
//+------------------------------------------------------------------+
//| Write one score contribution per slot/component lane.            |
//+------------------------------------------------------------------+
__kernel void D2SkillScoreSlots(__global const float *query,
                                __global const float *keys,
                                __global float *partials,
                                const int slots,
                                const int dimension)
  {
   const int index = get_global_id(0);
   const int total = slots * dimension;
   if(index < 0 || index >= total || dimension <= 0)
      return;
   const int component = index % dimension;
   const float q = IsNaNOrInf(query[component], 0.0f);
   const float k = IsNaNOrInf(keys[index], 0.0f);
   const float dot = q * k;
   const float query2 = q * q;
   const float key2 = k * k;
   const int base = index * 4;
   partials[base] = IsNaNOrInf(dot, 0.0f);
   partials[base + 1] = IsNaNOrInf(query2, 0.0f);
   partials[base + 2] = IsNaNOrInf(key2, 0.0f);
   partials[base + 3] = (!isfinite(dot) || !isfinite(query2) || !isfinite(key2) ? 1.0f : 0.0f);
  }
//+------------------------------------------------------------------+
//| Finalize one D2Skill score per slot after staged reduction.      |
//+------------------------------------------------------------------+
__kernel void D2SkillScoreFinalize(__global const float *metrics,
                                   __global const float *utility,
                                   __global const float *used,
                                   __global const float *state,
                                   __global float *scores,
                                   const int slots,
                                   const float similarity_threshold,
                                   const float utility_weight,
                                   const float min_utility,
                                   const int utility_policy)
  {
   const int slot = get_global_id(0);
   if(slot < 0 || slot >= slots)
      return;
   const int base = slot * 4;
   const float dot = IsNaNOrInf(metrics[base], 0.0f);
   const float query2 = IsNaNOrInf(metrics[base + 1], 0.0f);
   const float key2 = IsNaNOrInf(metrics[base + 2], 0.0f);
   const float invalid_reduction = IsNaNOrInf(metrics[base + 3], 1.0f);
   float score = RAG_INVALID_SCORE;
   const int slot_state = (int)IsNaNOrInf(state[slot], (float)D2SKILL_INACTIVE);
   if(invalid_reduction <= 0.0f && IsNaNOrInf(used[slot], 0.0f) > 0.5f &&
      slot_state != D2SKILL_INACTIVE && slot_state != D2SKILL_CANDIDATE &&
      query2 > 1.0e-12f && key2 > 1.0e-12f)
     {
      const float cosine = dot / (sqrt(query2) * sqrt(key2));
      const float slot_utility = clamp(IsNaNOrInf(utility[slot], 0.0f), -1.0f, 1.0f);
      if(isfinite(cosine) && cosine >= similarity_threshold &&
         (utility_policy == 0 || slot_utility >= min_utility))
        {
         const float candidate_score = cosine + utility_weight * slot_utility;
         if(isfinite(candidate_score))
            score = candidate_score;
        }
     }
   scores[slot] = score;
  }
//+------------------------------------------------------------------+
//| Gather the selected signed correction in parallel by component.  |
//+------------------------------------------------------------------+
__kernel void D2SkillGatherCorrection(__global const float2 *top,
                                      __global const float *corrections,
                                      __global const float *directions,
                                      __global const float *scales,
                                      __global float *selected_correction,
                                      __global float *selected_slot,
                                      __global float *selected_score,
                                      __global float *uses,
                                      __global float *diagnostics,
                                      const int slots,
                                      const int dimension,
                                      const int representation,
                                      const float alpha,
                                      __global float *retrieved,
                                      const int mark_retrieved)
  {
   const int component = get_global_id(0);
   const float2 top_slot = top[0];
   const int winner = (int)top_slot.x;
   const float score = IsNaNOrInf(top_slot.y, RAG_INVALID_SCORE);
   const int valid = (winner >= 0 && winner < slots && score > RAG_INVALID_SCORE);
   if(component < dimension)
     {
      float correction = 0.0f;
      if(valid)
        {
         const int shift = winner * dimension + component;
         correction = (representation == 1 ?
                       IsNaNOrInf(directions[shift], 0.0f) *
                       IsNaNOrInf(scales[winner], 0.0f) :
                       IsNaNOrInf(corrections[shift], 0.0f));
         correction = IsNaNOrInf(correction * IsNaNOrInf(alpha, 0.0f), 0.0f);
        }
      selected_correction[component] = correction;
     }
   if(component == 0)
     {
      selected_slot[0] = (valid ? (float)winner : -1.0f);
      selected_score[0] = (valid ? score : RAG_INVALID_SCORE);
      if(valid && mark_retrieved != 0)
        {
         uses[winner] = IsNaNOrInf(uses[winner], 0.0f) + 1.0f;
         retrieved[winner] = 1.0f;
         diagnostics[0] = IsNaNOrInf(diagnostics[0], 0.0f) + 1.0f;
        }
      else
         if(!valid && mark_retrieved != 0)
           {
            diagnostics[1] = IsNaNOrInf(diagnostics[1], 0.0f) + 1.0f;
           }
     }
  }
//+------------------------------------------------------------------+
//| D2Skill metric and decision constants used by the rank kernels.  |
//+------------------------------------------------------------------+
#define D2SKILL_METRIC_COUNT       10
#define D2SKILL_DIAGNOSTIC_PARTIALS 22
#define D2SKILL_DECISION_NONE       0
#define D2SKILL_DECISION_MATCH      1
#define D2SKILL_DECISION_SEED       2
#define D2SKILL_DECISION_RESET      3
#define D2SKILL_DECISION_CANDIDATE  4
#define D2SKILL_DECISION_FREE       5
#define D2SKILL_DECISION_EVICT      6

//+------------------------------------------------------------------+
//| Implements the D2Skill age lifecycle kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void D2SkillAgeLifecycle(__global const float *used,
                                  __global float *state,
                                  __global float *age,
                                  __global float *protection,
                                  __global const float *utility,
                                  __global float *free_scores,
                                  __global float *eviction_scores,
                                  const int slots,
                                  const int inactivity_age,
                                  __global float *retrieved,
                                  __global float *candidate_scores)
  {
   const int slot = get_global_id(0);
   if(slot >= slots)
      return;
   const float is_used = IsNaNOrInf(used[slot], 0.0f);
   int slot_state = (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE);
   float slot_age = max(IsNaNOrInf(age[slot], 0.0f), 0.0f);
   float slot_protection = max(IsNaNOrInf(protection[slot], 0.0f), 0.0f);
   if(is_used > 0.5f && slot_state != D2SKILL_CANDIDATE)
     {
      const bool was_retrieved = (IsNaNOrInf(retrieved[slot], 0.0f) > 0.5f);
      slot_age = (was_retrieved ? 0.0f : min(slot_age + 1.0f, (float)MAX_VALUE));
      retrieved[slot] = 0.0f;
      slot_protection = max(slot_protection - 1.0f, 0.0f);
      if(slot_state == D2SKILL_PROTECTED && slot_protection <= 0.0f)
         slot_state = D2SKILL_ACTIVE;
      if(slot_state == D2SKILL_ACTIVE && slot_age >= (float)max(1, inactivity_age))
         slot_state = D2SKILL_INACTIVE;
      state[slot] = (float)slot_state;
      age[slot] = slot_age;
      protection[slot] = slot_protection;
     }
   free_scores[slot] = (is_used <= 0.5f && slot_state == D2SKILL_FREE ?
                        -(float)slot : RAG_INVALID_SCORE);
   eviction_scores[slot] = (is_used > 0.5f && slot_state == D2SKILL_INACTIVE &&
                            slot_protection <= 0.0f ?
                            -clamp(IsNaNOrInf(utility[slot], 0.0f), -1.0f, 1.0f) +
                            1.0e-4f * slot_age : RAG_INVALID_SCORE);
   candidate_scores[slot] = (is_used > 0.5f && slot_state == D2SKILL_CANDIDATE ?
                             -(float)slot : RAG_INVALID_SCORE);
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill metric partial kernel on the GPU.         |
//+------------------------------------------------------------------+
__kernel void D2SkillMetricPartial(__global const float *query,
                                   __global const float *gradient,
                                   __global const float *selected_slot,
                                   __global const float *keys,
                                   __global const float *directions,
                                   __global const float *used,
                                   __global const float2 *candidate_top,
                                   __global const float *candidate_key,
                                   __global const float *candidate_direction,
                                   __global float *partials,
                                   const int slots,
                                   const int dimension)
  {
   const int component = get_global_id(0);
   if(component < 0 || component >= dimension)
      return;
   const int selected = (int)round(selected_slot[0]);
   const int selected_valid = (selected >= 0 && selected < slots &&
                               IsNaNOrInf(used[selected], 0.0f) > 0.5f);
   const int candidate_index = (int)round(candidate_top[0].x);
   const int candidate = (candidate_index >= 0 && candidate_index < slots &&
                          candidate_top[0].y > RAG_INVALID_SCORE ? candidate_index : -1);
   const float q = IsNaNOrInf(query[component], 0.0f);
   const float g = IsNaNOrInf(gradient[component], 0.0f);
   float values[D2SKILL_METRIC_COUNT];
   for(int metric = 0; metric < D2SKILL_METRIC_COUNT; metric++)
      values[metric] = 0.0f;
   values[0] = IsNaNOrInf(g * g, 0.0f);
   values[1] = IsNaNOrInf(q * q, 0.0f);
   if(selected_valid)
     {
      const int shift = selected * dimension + component;
      const float key = IsNaNOrInf(keys[shift], 0.0f);
      const float direction = IsNaNOrInf(directions[shift], 0.0f);
      values[2] = IsNaNOrInf(q * key, 0.0f);
      values[3] = IsNaNOrInf(key * key, 0.0f);
      values[4] = IsNaNOrInf(g * direction, 0.0f);
      values[5] = IsNaNOrInf(direction * direction, 0.0f);
     }
   if(candidate >= 0)
     {
      const int shift = candidate * dimension + component;
      const float key = IsNaNOrInf(candidate_key[shift], 0.0f);
      const float direction = IsNaNOrInf(candidate_direction[shift], 0.0f);
      values[6] = IsNaNOrInf(q * key, 0.0f);
      values[7] = IsNaNOrInf(key * key, 0.0f);
      values[8] = IsNaNOrInf(g * direction, 0.0f);
      values[9] = IsNaNOrInf(direction * direction, 0.0f);
     }
   const int base = component * D2SKILL_METRIC_COUNT;
   for(int metric = 0; metric < D2SKILL_METRIC_COUNT; metric++)
      partials[base + metric] = values[metric];
  }
//+------------------------------------------------------------------+
//| Generic radix-8 field reduction with disjoint global outputs.    |
//+------------------------------------------------------------------+
inline float D2SkillMetricReducePair(const float left, const float right,
                                     const int field, const int max_field_a,
                                     const int max_field_b)
  {
   if(field == max_field_a || field == max_field_b)
      return(max(left, right));
   return(IsNaNOrInf(left + right, 0.0f));
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill metric reduce kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void D2SkillMetricReduce(__global float *partials,
                                  __global float *metrics,
                                  const int batch_count,
                                  const int input_count,
                                  const int fields,
                                  const int input_offset,
                                  const int output_offset,
                                  const int max_field_a,
                                  const int max_field_b,
                                  const int final_output)
  {
   const int index = get_global_id(0);
   const int reduction_radix = 8;
   const int output_count = (input_count + reduction_radix - 1) / reduction_radix;
   const int total = batch_count * output_count * fields;
   if(index < 0 || index >= total || input_count <= 0 || fields <= 0)
      return;
   const int field = index % fields;
   const int item_index = index / fields;
   const int output_index = item_index % output_count;
   const int batch = item_index / output_count;
   const int first = reduction_radix * output_index;
   const int input_base = input_offset + (batch * input_count + first) * fields + field;
   float value = IsNaNOrInf(partials[input_base], 0.0f);
   if(first + 1 < input_count)
     {
      const float right = IsNaNOrInf(partials[input_base + fields], 0.0f);
      value = D2SkillMetricReducePair(value, right, field, max_field_a, max_field_b);
     }
   if(first + 2 < input_count)
     {
      float right = IsNaNOrInf(partials[input_base + 2 * fields], 0.0f);
      if(first + 3 < input_count)
        {
         const float pair = IsNaNOrInf(partials[input_base + 3 * fields], 0.0f);
         right = D2SkillMetricReducePair(right, pair, field, max_field_a, max_field_b);
        }
      value = D2SkillMetricReducePair(value, right, field, max_field_a, max_field_b);
     }
   if(first + 4 < input_count)
     {
      float right = IsNaNOrInf(partials[input_base + 4 * fields], 0.0f);
      if(first + 5 < input_count)
        {
         const float pair = IsNaNOrInf(partials[input_base + 5 * fields], 0.0f);
         right = D2SkillMetricReducePair(right, pair, field, max_field_a, max_field_b);
        }
      if(first + 6 < input_count)
        {
         float pair = IsNaNOrInf(partials[input_base + 6 * fields], 0.0f);
         if(first + 7 < input_count)
           {
            const float pair_right = IsNaNOrInf(partials[input_base + 7 * fields], 0.0f);
            pair = D2SkillMetricReducePair(pair, pair_right, field, max_field_a, max_field_b);
           }
         right = D2SkillMetricReducePair(right, pair, field, max_field_a, max_field_b);
        }
      value = D2SkillMetricReducePair(value, right, field, max_field_a, max_field_b);
     }
   if(final_output != 0)
      metrics[batch * fields + field] = value;
   else
      partials[output_offset + (batch * output_count + output_index) * fields + field] = value;
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill candidate leader kernel on the GPU.       |
//+------------------------------------------------------------------+
__kernel void D2SkillCandidateLeader(__global const float *selected_slot,
                                     __global float *used,
                                     __global float *scales,
                                     __global float *observations,
                                     __global float *mass,
                                     __global float *age,
                                     __global float *candidate_streak,
                                     __global float *candidate_valid,
                                     __global float *candidate_scale,
                                     __global const float *metrics,
                                     __global const float2 *free_top,
                                     __global const float2 *eviction_top,
                                     __global float *decision,
                                     __global float *diagnostics,
                                     const int slots,
                                     const int representation,
                                     const float beta_magnitude,
                                     const float max_correction,
                                     const int min_confirmations,
                                     const float similarity_threshold,
                                     const float direction_threshold,
                                     __global const float2 *candidate_top,
                                     __global float *candidate_protection)
  {
   if(get_global_id(0) != 0)
      return;
   decision[0] = -1.0f;
   decision[1] = D2SKILL_DECISION_NONE;
   const float gradient_norm = sqrt(fmax(IsNaNOrInf(metrics[0], 0.0f), 0.0f));
   decision[2] = gradient_norm;
   decision[3] = 0.0f;
//--- Collection must preserve a non-zero pseudo-residual even before the
//--- untrained Actor has amplified it.  Only an exact zero has no direction.
   if(slots <= 0 || gradient_norm <= 0.0f)
      return;
   const float query_norm = sqrt(fmax(IsNaNOrInf(metrics[1], 0.0f), 0.0f));
   const int selected = (int)round(selected_slot[0]);
   const int selected_valid = (selected >= 0 && selected < slots &&
                               IsNaNOrInf(used[selected], 0.0f) > 0.5f);
   const float selected_key_norm = sqrt(fmax(IsNaNOrInf(metrics[3], 0.0f), 0.0f));
   const float selected_direction_norm = sqrt(fmax(IsNaNOrInf(metrics[5], 0.0f), 0.0f));
   const float key_cosine = (query_norm > 1.0e-8f && selected_key_norm > 1.0e-8f ?
                             metrics[2] / (query_norm * selected_key_norm) : -1.0f);
   const float direction_cosine = (selected_direction_norm > 1.0e-8f ?
                                   metrics[4] / (gradient_norm * selected_direction_norm) : 0.0f);
   if(selected_valid)
     {
      diagnostics[2] = IsNaNOrInf(diagnostics[2], 0.0f) + 1.0f;
      if(IsNaNOrInf(metrics[5], 0.0f) > 1.0e-8f)
        {
         diagnostics[19] = IsNaNOrInf(diagnostics[19], 0.0f) + 1.0f;
         diagnostics[20] = IsNaNOrInf(diagnostics[20], 0.0f) + direction_cosine;
        }
     }
   if(selected_valid && isfinite(key_cosine) && isfinite(direction_cosine) &&
      key_cosine >= similarity_threshold && direction_cosine >= direction_threshold)
     {
      if(representation == 1)
         scales[selected] = clamp((1.0f - beta_magnitude) * IsNaNOrInf(scales[selected], 0.0f) +
                                  beta_magnitude * gradient_norm, 0.0f, max_correction);
      observations[selected] = IsNaNOrInf(observations[selected], 0.0f) + 1.0f;
      mass[selected] = 0.95f * IsNaNOrInf(mass[selected], 0.0f) + 0.05f;
      age[selected] = 0.0f;
      decision[0] = (float)selected;
      decision[1] = D2SKILL_DECISION_MATCH;
      return;
     }
   const int candidate_index = (int)round(candidate_top[0].x);
   int candidate = (candidate_index >= 0 && candidate_index < slots &&
                    candidate_top[0].y > RAG_INVALID_SCORE ? candidate_index : -1);
   if(candidate < 0)
     {
      const int free_slot = (int)round(free_top[0].x);
      const int evict_slot = (int)round(eviction_top[0].x);
      const bool free_available = (free_slot >= 0 && free_slot < slots &&
                                   free_top[0].y > RAG_INVALID_SCORE);
      const bool evict_available = (evict_slot >= 0 && evict_slot < slots &&
                                    eviction_top[0].y > RAG_INVALID_SCORE);
      candidate = (free_available ? free_slot : (evict_available ? evict_slot : -1));
      if(candidate < 0)
        {
         diagnostics[5] = IsNaNOrInf(diagnostics[5], 0.0f) + 1.0f;
         return;
        }
      candidate_valid[candidate] = D2SKILL_CANDIDATE;
      used[candidate] = 1.0f;
      candidate_streak[candidate] = 1.0f;
      candidate_scale[candidate] = gradient_norm;
      mass[candidate] = gradient_norm;
      age[candidate] = 0.0f;
      candidate_protection[candidate] = (free_available ? 0.0f : -1.0f);
      decision[0] = (float)candidate;
      decision[1] = D2SKILL_DECISION_SEED;
      diagnostics[3] = IsNaNOrInf(diagnostics[3], 0.0f) + 1.0f;
      return;
     }
   const float candidate_key_norm = sqrt(fmax(IsNaNOrInf(metrics[7], 0.0f), 0.0f));
   const float candidate_direction_norm = sqrt(fmax(IsNaNOrInf(metrics[9], 0.0f), 0.0f));
   const float candidate_key_cosine = (query_norm > 1.0e-8f && candidate_key_norm > 1.0e-8f ?
                                       metrics[6] / (query_norm * candidate_key_norm) : -1.0f);
   const float candidate_direction_cosine =
      (candidate_direction_norm > 1.0e-8f ?
       metrics[8] / (gradient_norm * candidate_direction_norm) : 0.0f);
   if(candidate_direction_norm > 1.0e-8f && isfinite(candidate_direction_cosine))
     {
      diagnostics[19] = IsNaNOrInf(diagnostics[19], 0.0f) + 1.0f;
      diagnostics[20] = IsNaNOrInf(diagnostics[20], 0.0f) + candidate_direction_cosine;
     }
   const bool direction_rejected = (IsNaNOrInf(metrics[9], 0.0f) > 1.0e-8f &&
                                    (!isfinite(candidate_direction_cosine) ||
                                     candidate_direction_cosine < direction_threshold));
   if(!isfinite(candidate_key_cosine) || candidate_key_cosine < similarity_threshold ||
      direction_rejected)
     {
      candidate_streak[candidate] = 1.0f;
      candidate_scale[candidate] = gradient_norm;
      age[candidate] = 0.0f;
      decision[0] = (float)candidate;
      decision[1] = D2SKILL_DECISION_RESET;
      return;
     }
   candidate_streak[candidate] = IsNaNOrInf(candidate_streak[candidate], 0.0f) + 1.0f;
   candidate_scale[candidate] = clamp((1.0f - beta_magnitude) * IsNaNOrInf(candidate_scale[candidate], 0.0f) +
                                      beta_magnitude * gradient_norm, 0.0f, max_correction);
   decision[0] = (float)candidate;
   if(candidate_streak[candidate] < (float)max(1, min_confirmations))
     {
      decision[1] = D2SKILL_DECISION_CANDIDATE;
      return;
     }
   decision[1] = (candidate_protection[candidate] < 0.0f ?
                  D2SKILL_DECISION_EVICT : D2SKILL_DECISION_FREE);
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill vector update kernel on the GPU.          |
//+------------------------------------------------------------------+
__kernel void D2SkillVectorUpdate(__global const float *query,
                                  __global const float *gradient,
                                  __global float *keys,
                                  __global float *corrections,
                                  __global float *directions,
                                  __global const float *scales,
                                  __global float *candidate_key,
                                  __global float *candidate_direction,
                                  __global const float *candidate_scale,
                                  __global const float *decision,
                                  __global const float *reduced_norm,
                                  const int dimension,
                                  const int representation,
                                  const float beta_correction,
                                  const float beta_direction,
                                  const float beta_key,
                                  const float max_correction,
                                  const int phase)
  {
   const int component = get_global_id(0);
   if(component >= dimension || dimension <= 0)
      return;
   const int action = (int)round(decision[1]);
   const int target = (int)round(decision[0]);
   const float gradient_norm = decision[2];
   const float query_value = IsNaNOrInf(query[component], 0.0f);
   const float gradient_value = IsNaNOrInf(gradient[component], 0.0f);
   const float unit = (gradient_norm > 0.0f ? gradient_value / gradient_norm : 0.0f);
   if(phase == 1)
     {
      if((action != D2SKILL_DECISION_MATCH && action != D2SKILL_DECISION_FREE &&
          action != D2SKILL_DECISION_EVICT) || target < 0)
         return;
      const int shift = target * dimension + component;
      const float vector_norm = sqrt(fmax(IsNaNOrInf(reduced_norm[0], 0.0f), 0.0f));
      if(representation == 0)
        {
         const float factor = (vector_norm > max_correction && vector_norm > 1.0e-12f ?
                               max_correction / vector_norm : 1.0f);
         corrections[shift] = IsNaNOrInf(corrections[shift] * factor, 0.0f);
        }
      else
        {
         const float factor = (vector_norm > 1.0e-12f ? 1.0f / vector_norm : 0.0f);
         directions[shift] = IsNaNOrInf(directions[shift] * factor, 0.0f);
         corrections[shift] = IsNaNOrInf(directions[shift] *
                                         fmin(scales[target], max_correction), 0.0f);
        }
      return;
     }
   if((action == D2SKILL_DECISION_MATCH || action == D2SKILL_DECISION_FREE ||
       action == D2SKILL_DECISION_EVICT) && target >= 0)
     {
      const int shift = target * dimension + component;
      keys[shift] = IsNaNOrInf((1.0f - beta_key) * keys[shift] + beta_key * query_value, 0.0f);
      if(representation == 0)
         corrections[shift] = IsNaNOrInf((1.0f - beta_correction) *
                                         IsNaNOrInf(corrections[shift], 0.0f) +
                                         beta_correction * gradient_value, 0.0f);
      else
         directions[shift] = IsNaNOrInf((1.0f - beta_direction) *
                                        IsNaNOrInf(directions[shift], 0.0f) +
                                        beta_direction * unit, 0.0f);
      return;
     }
   if((action == D2SKILL_DECISION_SEED || action == D2SKILL_DECISION_RESET) && target >= 0)
     {
      const int shift = target * dimension + component;
      keys[shift] = query_value;
      directions[shift] = unit;
      return;
     }
   if((action == D2SKILL_DECISION_CANDIDATE || action == D2SKILL_DECISION_FREE ||
       action == D2SKILL_DECISION_EVICT) && target >= 0)
     {
      const int shift = target * dimension + component;
      keys[shift] =
         IsNaNOrInf((1.0f - beta_key) * keys[shift] +
                    beta_key * query_value, 0.0f);
      directions[shift] =
         IsNaNOrInf((1.0f - beta_direction) * directions[shift] +
                    beta_direction * unit, 0.0f);
      return;
     }
  }
//+------------------------------------------------------------------+
//| Applies the online direction EMA to one retrieved stable slot.   |
//| It intentionally does not touch keys, correction, scale or       |
//| lifecycle.                                                       |
//+------------------------------------------------------------------+
__kernel void D2SkillDirectionEMA(__global const float *gradient,
                                  __global float *directions,
                                  __global const float *selected_slot,
                                  __global const float *used,
                                  __global const float *state,
                                  __global float *reduced_norm,
                                  const int slots,
                                  const int dimension,
                                  const float beta_direction,
                                  __global float *partials,
                                  const int phase)
  {
   const int component = get_global_id(0);
   if(component < 0 || component >= dimension || slots <= 0)
      return;
   const int slot = (int)round(IsNaNOrInf(selected_slot[0], -1.0f));
   if(slot < 0 || slot >= slots || IsNaNOrInf(used[slot], 0.0f) <= 0.5f)
      return;
   const int slot_state = (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE);
   if(slot_state == D2SKILL_FREE || slot_state == D2SKILL_CANDIDATE ||
      slot_state == D2SKILL_INACTIVE)
      return;
   const int index = slot * dimension + component;
   float gradient_squared = IsNaNOrInf(reduced_norm[0], 0.0f);
   if(phase == 0 && component == 0)
      reduced_norm[1] = gradient_squared;
   if(phase == 1)
      gradient_squared = IsNaNOrInf(reduced_norm[1], 0.0f);
   const float gradient_norm = sqrt(fmax(gradient_squared, 0.0f));
   const float unit = (gradient_norm > 1.0e-12f ?
                       IsNaNOrInf(gradient[component], 0.0f) / gradient_norm : 0.0f);
   const float candidate = IsNaNOrInf((1.0f - beta_direction) *
                                      IsNaNOrInf(directions[index], 0.0f) +
                                      beta_direction * unit, 0.0f);
   if(phase == 0)
     {
      partials[component] = IsNaNOrInf(candidate * candidate, 0.0f);
      return;
     }
   if(phase == 1)
     {
      const float candidate_norm = sqrt(fmax(IsNaNOrInf(reduced_norm[0], 0.0f), 0.0f));
      if(candidate_norm > 1.0e-12f)
         directions[index] = IsNaNOrInf(candidate / candidate_norm, 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Reduces and normalizes only the retrieved stable direction row.  |
//| Phase 0 emits per-component squares; phase 1 consumes its staged |
//| reduction.  No durable buffer except directions is written.      |
//+------------------------------------------------------------------+
__kernel void D2SkillNormalizeDirection(__global float *directions,
                                        __global const float *selected_slot,
                                        __global const float *used,
                                        __global const float *state,
                                        __global float *partials,
                                        __global const float *reduced_norm,
                                        const int slots,
                                        const int dimension,
                                        const int phase)
  {
   const int component = get_global_id(0);
   if(component < 0 || component >= dimension || slots <= 0)
      return;
   const int slot = (int)round(IsNaNOrInf(selected_slot[0], -1.0f));
   if(slot < 0 || slot >= slots || IsNaNOrInf(used[slot], 0.0f) <= 0.5f)
      return;
   const int slot_state = (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE);
   if(slot_state == D2SKILL_FREE || slot_state == D2SKILL_CANDIDATE ||
      slot_state == D2SKILL_INACTIVE)
      return;
   const int index = slot * dimension + component;
   if(phase == 0)
     {
      const float value = IsNaNOrInf(directions[index], 0.0f);
      partials[component] = IsNaNOrInf(value * value, 0.0f);
      return;
     }
   if(phase == 1)
     {
      const float norm = sqrt(fmax(IsNaNOrInf(reduced_norm[0], 0.0f), 0.0f));
      if(norm > 1.0e-12f)
         directions[index] = IsNaNOrInf(directions[index] / norm, 0.0f);
      else
         directions[index] = (component == 0 ? 1.0f : 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill bank vector partial kernel on the GPU.    |
//+------------------------------------------------------------------+
__kernel void D2SkillBankVectorPartial(__global const float *corrections,
                                       __global const float *directions,
                                       __global const float *decision,
                                       __global float *partials,
                                       const int dimension,
                                       const int representation,
                                       const int output_offset)
  {
   const int component = get_global_id(0);
   if(component < 0 || component >= dimension)
      return;
   const int target = (int)round(decision[0]);
   const int action = (int)round(decision[1]);
   float value = 0.0f;
   if((action == D2SKILL_DECISION_MATCH || action == D2SKILL_DECISION_FREE ||
       action == D2SKILL_DECISION_EVICT) && target >= 0)
     {
      const int shift = target * dimension + component;
      value = (representation == 0 ? IsNaNOrInf(corrections[shift], 0.0f) :
               IsNaNOrInf(directions[shift], 0.0f));
     }
   partials[output_offset + component] = IsNaNOrInf(value * value, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill promote slot kernel on the GPU.           |
//+------------------------------------------------------------------+
__kernel void D2SkillPromoteSlot(__global float *scales,
                                 __global float *utility,
                                 __global float *observations,
                                 __global float *uses,
                                 __global float *mass,
                                 __global float *used,
                                 __global float *state,
                                 __global float *age,
                                 __global float *protection,
                                 __global float *candidate_streak,
                                 __global float *candidate_valid,
                                 __global const float *candidate_scale,
                                 __global const float *decision,
                                 __global float *diagnostics,
                                 const int slots,
                                 const int protection_age)
  {
   if(get_global_id(0) != 0)
      return;
   const int action = (int)round(decision[1]);
   const int target = (int)round(decision[0]);
   if((action != D2SKILL_DECISION_FREE && action != D2SKILL_DECISION_EVICT) ||
      target < 0 || target >= slots ||
      (int)IsNaNOrInf(candidate_valid[target], (float)D2SKILL_FREE) != D2SKILL_CANDIDATE)
      return;
   scales[target] = IsNaNOrInf(candidate_scale[target], 0.0f);
   utility[target] = 0.0f;
   observations[target] = IsNaNOrInf(candidate_streak[target], 1.0f);
   uses[target] = 0.0f;
   mass[target] = IsNaNOrInf(candidate_streak[target], 1.0f);
   used[target] = 1.0f;
   state[target] = D2SKILL_PROTECTED;
   age[target] = 0.0f;
   protection[target] = (float)protection_age;
   if(action == D2SKILL_DECISION_EVICT)
      diagnostics[7] = IsNaNOrInf(diagnostics[7], 0.0f) + 1.0f;
   else
      diagnostics[4] = IsNaNOrInf(diagnostics[4], 0.0f) + 1.0f;
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill diagnostics norm partial kernel on the GPU|
//+------------------------------------------------------------------+
__kernel void D2SkillDiagnosticsNormPartial(__global const float *corrections,
      __global float *partials,
      const int slots,
      const int dimension)
  {
   const int index = get_global_id(0);
   const int total = slots * dimension;
   if(index < 0 || index >= total)
      return;
   const float value = IsNaNOrInf(corrections[index], 0.0f);
   partials[index] = IsNaNOrInf(value * value, 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill diagnostics partial kernel on the GPU.    |
//+------------------------------------------------------------------+
__kernel void D2SkillDiagnosticsPartial(__global const float *norms,
                                        __global const float *utility,
                                        __global const float *uses,
                                        __global const float *used,
                                        __global const float *state,
                                        __global const float *age,
                                        __global float *partials,
                                        const int slots)
  {
   const int slot = get_global_id(0);
   if(slot < 0 || slot >= slots)
      return;
   const int base = slot * D2SKILL_DIAGNOSTIC_PARTIALS;
   for(int item = 0; item < D2SKILL_DIAGNOSTIC_PARTIALS; item++)
      partials[base + item] = 0.0f;
   const float slot_used = IsNaNOrInf(used[slot], 0.0f);
   const int slot_state = (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE);
   if(slot_used <= 0.5f && slot_state == D2SKILL_FREE)
      partials[base] = 1.0f;
   if(slot_state == D2SKILL_PROTECTED)
      partials[base + 1] = 1.0f;
   if(slot_state == D2SKILL_ACTIVE)
      partials[base + 2] = 1.0f;
   if(slot_state == D2SKILL_INACTIVE)
      partials[base + 3] = 1.0f;
   if(slot_state == D2SKILL_CANDIDATE)
      partials[base + 21] = 1.0f;
   if(slot_used <= 0.5f)
      return;
   const float slot_age = max(IsNaNOrInf(age[slot], 0.0f), 0.0f);
   const float slot_usage = max(IsNaNOrInf(uses[slot], 0.0f), 0.0f);
   partials[base + 4] = 1.0f;
   partials[base + 5] = IsNaNOrInf(utility[slot], 0.0f);
   partials[base + 6] = sqrt(fmax(IsNaNOrInf(norms[slot], 0.0f), 0.0f));
   partials[base + 7] = slot_age;
   partials[base + 8] = slot_usage;
   partials[base + 9] = slot_age;
   partials[base + 10] = slot_usage;
   int utility_bin = clamp((int)floor((clamp(IsNaNOrInf(utility[slot], 0.0f),
                                       -1.0f, 1.0f) + 1.0f) * 2.5f), 0, 4);
   const int usage_bin = (slot_usage < 1.0f ? 0 : (slot_usage < 2.0f ? 1 :
                          (slot_usage < 4.0f ? 2 : (slot_usage < 8.0f ? 3 : 4))));
   partials[base + 11 + utility_bin] = 1.0f;
   partials[base + 16 + usage_bin] = 1.0f;
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill diagnostics reduce kernel on the GPU.     |
//+------------------------------------------------------------------+
__kernel void D2SkillDiagnosticsReduce(__global const float *totals,
                                       __global float *diagnostics,
                                       __global float *utility_distribution,
                                       __global float *usage_distribution,
                                       const int slots)
  {
   const int output = get_global_id(0);
   const float used_count = IsNaNOrInf(totals[4], 0.0f);
   switch(output)
     {
      case 0:
         diagnostics[8] = (float)slots;
         break;
      case 1:
         diagnostics[9] = IsNaNOrInf(totals[0], 0.0f);
         break;
      case 2:
         diagnostics[10] = (IsNaNOrInf(totals[21], 0.0f) > 0.0f ? 1.0f : 0.0f);
         break;
      case 3:
         diagnostics[11] = IsNaNOrInf(totals[1], 0.0f);
         break;
      case 4:
         diagnostics[12] = IsNaNOrInf(totals[2], 0.0f);
         break;
      case 5:
         diagnostics[13] = IsNaNOrInf(totals[3], 0.0f);
         break;
      case 6:
         diagnostics[14] = (used_count > 0.0f ? totals[5] / used_count : 0.0f);
         break;
      case 7:
         diagnostics[15] = (used_count > 0.0f ? totals[6] / used_count : 0.0f);
         break;
      case 8:
         diagnostics[16] = (diagnostics[19] > 0.0f ? diagnostics[20] / diagnostics[19] : 0.0f);
         break;
      case 9:
         diagnostics[17] = (used_count > 0.0f ? totals[7] / used_count : 0.0f);
         break;
      case 10:
         diagnostics[18] = (used_count > 0.0f ? totals[8] / used_count : 0.0f);
         break;
      case 11:
         diagnostics[21] = IsNaNOrInf(totals[9], 0.0f);
         break;
      case 12:
         diagnostics[22] = IsNaNOrInf(totals[10], 0.0f);
         break;
      default:
         if(output >= 13 && output < 18)
            utility_distribution[output - 13] = IsNaNOrInf(totals[output - 2], 0.0f);
         else
            if(output >= 18 && output < 23)
               usage_distribution[output - 18] = IsNaNOrInf(totals[output - 2], 0.0f);
         break;
     }
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill influence total kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void D2SkillInfluenceTotal(__global const float *influence,
                                    __global const float *used,
                                    __global const float *state,
                                    __global float *partials,
                                    const int slots)
  {
   const int slot = get_global_id(0);
   if(slot < 0 || slot >= slots)
      return;
   const bool eligible = (IsNaNOrInf(used[slot], 0.0f) > 0.5f &&
                          (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE) != D2SKILL_CANDIDATE);
   partials[slot] = (eligible ? fmax(IsNaNOrInf(influence[slot], 0.0f), 0.0f) : 0.0f);
  }
//+------------------------------------------------------------------+
//| Gradient contribution for the selected forward correction.       |
//+------------------------------------------------------------------+
__kernel void D2SkillInfluencePartial(__global const float *gradient,
                                      __global const float *correction,
                                      __global float *partials,
                                      const int dimension)
  {
   const int component = get_global_id(0);
   if(component < 0 || component >= dimension)
      return;
   partials[component] = IsNaNOrInf(IsNaNOrInf(gradient[component], 0.0f) *
                                    IsNaNOrInf(correction[component], 0.0f), 0.0f);
  }
//+------------------------------------------------------------------+
//| Single writer after staged reduction: no atomics are required.   |
//+------------------------------------------------------------------+
__kernel void D2SkillInfluenceAccumulate(__global const float *selected_slot,
      __global const float *reduced_influence,
      __global float *influence,
      const int slots)
  {
   if(get_global_id(0) != 0 || slots <= 0)
      return;
   const int slot = (int)round(IsNaNOrInf(selected_slot[0], -1.0f));
   if(slot < 0 || slot >= slots)
      return;
   influence[slot] = IsNaNOrInf(influence[slot], 0.0f) +
                     IsNaNOrInf(reduced_influence[0], 0.0f);
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill reset influence kernel on the GPU.        |
//+------------------------------------------------------------------+
__kernel void D2SkillResetInfluence(__global float *influence,
                                    const int slots)
  {
   const int slot = get_global_id(0);
   if(slot < slots)
      influence[slot] = 0.0f;
  }
//+------------------------------------------------------------------+
//| Implements the D2Skill utility kernel on the GPU.                |
//+------------------------------------------------------------------+
__kernel void D2SkillUtility(__global float *influence,
                             __global const float *influence_total,
                             __global float *utility,
                             __global const float *used,
                             __global const float *state,
                             const float delta_j,
                             const float beta,
                             const int slots,
                             __global float *diagnostics,
                             __global float *utility_applied)
  {
   const int slot = get_global_id(0);
   if(slot < 0 || slot >= slots || slots <= 0)
      return;
   utility_applied[slot] = 0.0f;
   const float mass = fmax(IsNaNOrInf(influence[slot], 0.0f), 0.0f);
   const float total = fmax(IsNaNOrInf(influence_total[0], 0.0f), 0.0f);
   const bool eligible = (IsNaNOrInf(used[slot], 0.0f) > 0.5f &&
                          (int)IsNaNOrInf(state[slot], (float)D2SKILL_FREE) != D2SKILL_CANDIDATE);
   if(eligible && total > 0.0f && mass > 0.0f && fabs(delta_j) > 0.0f)
     {
      const float utility_before = utility[slot];
      const float utility_current = IsNaNOrInf(utility_before, 0.0f);
      const float utility_target = clamp(IsNaNOrInf(delta_j, 0.0f), -1.0f, 1.0f) * mass / total;
      const float utility_proposed = (1.0f - beta) * utility_current + beta * utility_target;
      utility[slot] = utility_proposed;
      if(utility_before != utility_proposed)
         utility_applied[slot] = 1.0f;
     }
   influence[slot] = 0.0f;
   if(slot == 0 && total > 0.0f && fabs(delta_j) > 0.0f)
      diagnostics[6] = IsNaNOrInf(diagnostics[6], 0.0f) + 1.0f;
  }
//+------------------------------------------------------------------+
//| OMPB regularizers: elementwise reductions only                   |
//+------------------------------------------------------------------+
__kernel void OMPBDisagreement(__global const float *reference,
                               __global const float *current,
                               __global float *output,
                               const int source_states,
                               const int target_states,
                               const int samples,
                               const float inv_tau2,
                               __global const float *gate,
                               __global float *alpha_gradient,
                               const int gate_stride,
                               const float multiplier,
                               __global float *numeric_flags,
                               const int flags_enabled)
  {
   const int lane = (int)get_local_id(0);
   const int variable = (int)get_global_id(1);
   __local float source_scratch[LOCAL_ARRAY_SIZE];
   __local float target_scratch[LOCAL_ARRAY_SIZE];
   __local float source_alpha_scratch[LOCAL_ARRAY_SIZE];
   __local float target_alpha_scratch[LOCAL_ARRAY_SIZE];
   __local float invalid_scratch[LOCAL_ARRAY_SIZE];
   float source_sum = 0.0f;
   float target_sum = 0.0f;
   float source_alpha = 0.0f;
   float target_alpha = 0.0f;
   int invalid = 0;
   if(source_states > 0 && target_states > 0 && samples > 1 && gate_stride > 0)
     {
      const float raw_gate = gate[variable * gate_stride];
      if(flags_enabled)
         invalid |= isnan(raw_gate) || isinf(raw_gate) || isnan(inv_tau2) || isinf(inv_tau2) ||
                    isnan(multiplier) || isinf(multiplier);
      const float g = IsNaNOrInf(raw_gate, 0.0f);
      const float gate2 = g * g;
      const float gate_alpha = 2.0f * gate2 * (1.0f - g);
      if(flags_enabled)
         invalid |= isnan(gate2) || isinf(gate2) || isnan(gate_alpha) || isinf(gate_alpha);
      const int source_total = source_states * samples * samples;
      const int target_total = target_states * samples * samples;
      //--- Traverse source states and unordered posterior-sample pairs.
      for(int index = lane; index < source_total; index += get_local_size(0))
        {
         const int sample_index = index % (samples * samples);
         const int left = sample_index / samples;
         const int right = sample_index % samples;
         if(left < right)
           {
            const float raw = reference[(variable * source_states * samples * samples) + index];
            const float raw_scaled = raw * gate2 * inv_tau2;
            if(flags_enabled)
               invalid |= isnan(raw) || isinf(raw) || isnan(raw_scaled) || isinf(raw_scaled);
            const float scaled = IsNaNOrInf(raw_scaled, 1.0f);
            source_sum += fmin(scaled, 1.0f);
            if(flags_enabled)
               invalid |= isnan(source_sum) || isinf(source_sum);
            if(scaled < 1.0f)
              {
               const float raw_alpha = gate_alpha * raw * inv_tau2;
               if(flags_enabled)
                  invalid |= isnan(raw_alpha) || isinf(raw_alpha);
               source_alpha += raw_alpha;
               if(flags_enabled)
                  invalid |= isnan(source_alpha) || isinf(source_alpha);
              }
           }
        }
      //--- Traverse target states and unordered posterior-sample pairs.
      for(int index = lane; index < target_total; index += get_local_size(0))
        {
         const int sample_index = index % (samples * samples);
         const int left = sample_index / samples;
         const int right = sample_index % samples;
         if(left < right)
           {
            const float raw = current[(variable * target_states * samples * samples) + index];
            const float raw_scaled = raw * gate2 * inv_tau2;
            if(flags_enabled)
               invalid |= isnan(raw) || isinf(raw) || isnan(raw_scaled) || isinf(raw_scaled);
            const float scaled = IsNaNOrInf(raw_scaled, 1.0f);
            target_sum += fmin(scaled, 1.0f);
            if(flags_enabled)
               invalid |= isnan(target_sum) || isinf(target_sum);
            if(scaled < 1.0f)
              {
               const float raw_alpha = gate_alpha * raw * inv_tau2;
               if(flags_enabled)
                  invalid |= isnan(raw_alpha) || isinf(raw_alpha);
               target_alpha += raw_alpha;
               if(flags_enabled)
                  invalid |= isnan(target_alpha) || isinf(target_alpha);
              }
           }
        }
     }
   //--- Reduce source and target statistics through independent local scratch.
   source_sum = LocalSum(source_sum, 0, source_scratch);
   target_sum = LocalSum(target_sum, 0, target_scratch);
   source_alpha = LocalSum(source_alpha, 0, source_alpha_scratch);
   target_alpha = LocalSum(target_alpha, 0, target_alpha_scratch);
   if(flags_enabled)
     {
      invalid |= isnan(source_scratch[0]) || isinf(source_scratch[0]) ||
                 isnan(target_scratch[0]) || isinf(target_scratch[0]) ||
                 isnan(source_alpha_scratch[0]) || isinf(source_alpha_scratch[0]) ||
                 isnan(target_alpha_scratch[0]) || isinf(target_alpha_scratch[0]);
      invalid = (LocalSum((float)invalid, 0, invalid_scratch) > 0.0f ? 1 : 0);
     }
   //--- Finalize the per-variable disagreement and alpha update direction.
   if(lane == 0)
     {
      float delta = 0.0f;
      float alpha_direction = 0.0f;
      if(source_states > 0 && target_states > 0 && samples > 1 && gate_stride > 0)
        {
         const int pairs = samples * (samples - 1) / 2;
         const float source_mean = source_sum / (float)(source_states * pairs);
         const float target_mean = target_sum / (float)(target_states * pairs);
         const float raw_delta = source_mean - target_mean;
         if(flags_enabled)
            invalid |= isnan(source_mean) || isinf(source_mean) || isnan(target_mean) ||
                       isinf(target_mean) || isnan(raw_delta) || isinf(raw_delta);
         delta = IsNaNOrInf(raw_delta, 0.0f);
         const float sign_delta = (delta > 0.0f ? 1.0f : (delta < 0.0f ? -1.0f : 0.0f));
         const float raw_alpha_direction = multiplier * sign_delta *
                                           (source_alpha / (float)(source_states * pairs) -
                                            target_alpha / (float)(target_states * pairs));
         if(flags_enabled)
            invalid |= isnan(raw_alpha_direction) || isinf(raw_alpha_direction);
         alpha_direction = raw_alpha_direction;
        }
      output[variable] = delta;
      const float final_alpha = IsNaNOrInf(alpha_direction, 0.0f);
      if(flags_enabled)
         invalid |= isnan(final_alpha) || isinf(final_alpha);
      alpha_gradient[variable] = final_alpha;
      if(flags_enabled)
         numeric_flags[variable] = (invalid ? 1.0f : 0.0f);
     }
  }
//+------------------------------------------------------------------+
//| Computes the OMPB disagreement grad gradient contributions on    |
//| GPU.                                                             |
//+------------------------------------------------------------------+
__kernel void OMPBDisagreementGrad(__global const float *corrections,
                                   __global const float *distance,
                                   __global const float *disagreement,
                                   __global float *gradient,
                                   const int states,
                                   const int variables,
                                   const int samples,
                                   const int dimension,
                                   const int source_sign,
                                   const float multiplier,
                                   const float inv_tau2,
                                   __global const float *gate,
                                   const int gate_stride,
                                   __global float *numeric_flags,
                                   const int flags_enabled)
   {
    const int group = (int)get_global_id(0);
    const int sample = (int)get_global_id(1);
    const int coordinate = (int)get_global_id(2);
    //--- All buffers are indexed as [group, sample, coordinate] with fixed extents.
    if(states <= 0 || variables <= 0 || samples <= 0 || dimension <= 0 ||
       group < 0 || sample < 0 || coordinate < 0 ||
       group >= states * variables || sample >= samples || coordinate >= dimension)
      return;
    const int variable = group / states;
    const int distance_base = group * samples * samples;
    const int correction_base = (group * samples + sample) * dimension + coordinate;
    int invalid = 0;
    const float raw_delta = disagreement[variable];
    const float raw_correction = corrections[correction_base];
    if(flags_enabled)
       invalid |= isnan(raw_delta) || isinf(raw_delta) || isnan(raw_correction) ||
                  isinf(raw_correction) || isnan(multiplier) || isinf(multiplier) ||
                  isnan(inv_tau2) || isinf(inv_tau2);
    const float delta = IsNaNOrInf(raw_delta, 0.0f);
    if(gate_stride <= 0)
      {
       if(flags_enabled)
          numeric_flags[correction_base] = 1.0f;
       return;
      }
    const float raw_gate = gate[variable * gate_stride];
    if(flags_enabled)
       invalid |= isnan(raw_gate) || isinf(raw_gate);
    const float g = IsNaNOrInf(raw_gate, 0.0f);
    const float gate2 = g * g;
    if(flags_enabled)
       invalid |= isnan(gate2) || isinf(gate2);
    if(samples == 1)
      {
       if(flags_enabled)
          numeric_flags[correction_base] = (invalid ? 1.0f : 0.0f);
       return;
      }
    const int pairs = samples * (samples - 1) / 2;
    //--- delta = D_S - D_T; its sign drives the pair direction. The buffer is
    //--- appended (+=) and must be pre-zeroed by the wrapper. In line with the
    //--- library contract (optimizers add gradient buffers), the multiplier
    //--- carries the negative derivative of the mismatch penalty.
    const float sign_delta = (delta > 0.0f ? 1.0f : (delta < 0.0f ? -1.0f : 0.0f));
    float value = 0.0f;
    const float correction = IsNaNOrInf(raw_correction, 0.0f);
    //--- Saturated pair-distance terms do not contribute to gradient.
    for(int other = 0; other < samples; other++)
      {
       if(other == sample)
          continue;
       const int didx = distance_base + sample * samples + other;
       const float raw_distance = distance[didx];
       const float raw_pair_distance = raw_distance * gate2 * inv_tau2;
       if(flags_enabled)
          invalid |= isnan(raw_distance) || isinf(raw_distance) || isnan(raw_pair_distance) ||
                     isinf(raw_pair_distance);
       const float pair_distance = IsNaNOrInf(raw_pair_distance, 1.0f);
       if(pair_distance >= 1.0f)
          continue;
       const int other_index = (group * samples + other) * dimension + coordinate;
       const float raw_other = corrections[other_index];
       const float raw_difference = raw_correction - raw_other;
       const float raw_pair_value = 2.0f * gate2 * raw_difference;
       if(flags_enabled)
          invalid |= isnan(raw_other) || isinf(raw_other) || isnan(raw_difference) ||
                     isinf(raw_difference) || isnan(raw_pair_value) || isinf(raw_pair_value);
       value += 2.0f * gate2 * (correction - IsNaNOrInf(raw_other, 0.0f));
       if(flags_enabled)
          invalid |= isnan(value) || isinf(value);
      }
    const float direction_factor = multiplier * (float)source_sign * sign_delta * inv_tau2 /
                                   (float)(states * pairs);
    const float raw_direction = value * direction_factor;
    if(flags_enabled)
       invalid |= isnan(direction_factor) || isinf(direction_factor) ||
                  isnan(raw_direction) || isinf(raw_direction);
    const float raw_gradient = gradient[correction_base];
    const float final_gradient = raw_gradient + clamp(IsNaNOrInf(raw_direction, 0.0f), -MAX_GRAD, MAX_GRAD);
    if(flags_enabled)
       invalid |= isnan(raw_gradient) || isinf(raw_gradient) || isnan(final_gradient) || isinf(final_gradient);
    gradient[correction_base] = final_gradient;
    if(flags_enabled)
       numeric_flags[correction_base] = (invalid ? 1.0f : 0.0f);
   }
//+------------------------------------------------------------------+
//| Implements the OMPB gaussian kl kernel on the GPU.               |
//+------------------------------------------------------------------+
__kernel void OMPBGaussianKL(__global const float *mean,
                             __global const float *logvar,
                             __global float *output, const int total)
  {
   const int id = (int)get_global_id(0);
   if(id < total)
     {
      const float mu = IsNaNOrInf(mean[id], 0.0f);
      const float lv = IsNaNOrInf(logvar[id], 0.0f);
      //--- expm1(lv) = exp(lv) - 1 avoids cancellation for small lv;
      //--- the exact KL is non-negative, so guard float round-off.
      output[id] = fmax(0.5f * (expm1(lv) - lv + mu * mu), 0.0f);
     }
  }
//+------------------------------------------------------------------+
