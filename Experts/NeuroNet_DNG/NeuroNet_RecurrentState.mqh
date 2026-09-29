//+------------------------------------------------------------------+
//|                                      NeuroNet_RecurrentState.mqh |
//|                                              Copyright 2026, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+-------------------------------------------------------------------------------+
//| Logical method implementations generated from  NeuroNet.mqh: RecurrentState.  |
//+-------------------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronLSTM layer defaults.         |
//+------------------------------------------------------------------+
CNeuronLSTM::CNeuronLSTM(void)
  {
   ForgetGate     =  new CLayer();
   InputGate      =  new CLayer();
   OutputGate     =  new CLayer();
   NewContent     =  new CLayer();
   Memory         =  new CArrayFloat();
   PrevMemory     =  new CArrayFloat();
   Input          =  new CArrayFloat();
   InputGradient  =  new CArrayFloat();
  }
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronLSTM GPU resources.              |
//+------------------------------------------------------------------+
CNeuronLSTM::~CNeuronLSTM(void)
  {
   DeleteObj(ForgetGate);
   DeleteObj(InputGate);
   DeleteObj(OutputGate);
   DeleteObj(NewContent);
   DeleteObj(Memory);
   DeleteObj(PrevMemory);
   DeleteObj(Input);
   DeleteObj(InputGradient);
  }
//+-------------------------------------------------------------------+
//| Initializes the CNeuronLSTM layer and allocates its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronLSTM::Init(uint numOutputs, uint myIndex, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type)
  {
   if(units_count <= 0)
      ReturnFalse;
//--- Init Layers
   if(!CNeuronProof::Init(numOutputs, myIndex, window, step, units_count, optimization_type))
      ReturnFalse;
   if(!InitLayer(ForgetGate, units_count, window + units_count, optimization_type))
      ReturnFalse;
   if(!InitLayer(InputGate, units_count, window + units_count, optimization_type))
      ReturnFalse;
   if(!InitLayer(OutputGate, units_count, window + units_count, optimization_type))
      ReturnFalse;
   if(!InitLayer(NewContent, units_count, window + units_count, optimization_type))
      ReturnFalse;
   if(!Memory.Reserve(units_count))
      ReturnFalse;
   if(!PrevMemory.Reserve(units_count))
      ReturnFalse;
   CNeuron *temp;
   for(int i = 0; i < units_count; i++)
     {
      if(!Memory.Add(0))
         ReturnFalse;
      if(!PrevMemory.Add(0))
         ReturnFalse;
      temp = OutputLayer.At(i);
      temp.setOutputVal(0);
     }
//---
   return true;
  }
//+---------------------------------------------------------------------+
//| Initializes one LSTM gate sublayer and allocates its GPU  buffers.  |
//+---------------------------------------------------------------------+
bool CNeuronLSTM::InitLayer(CLayer *layer, int numUnits, int numOutputs, ENUM_OPTIMIZATION optimization_type)
  {
   if(CheckPointer(layer) == POINTER_INVALID)
     {
      layer = new CLayer(numOutputs);
      if(CheckPointer(layer) == POINTER_INVALID)
         ReturnFalse;
     }
   else
      layer.Clear();
   if(!layer.Reserve(numUnits))
      ReturnFalse;
//---
   CNeuron *temp;
   for(int i = 0; i < numUnits; i++)
     {
      temp = new CNeuron();
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      if(!temp.Init(numOutputs + 1, i, optimization_type))
         ReturnFalse;
      if(!layer.Add(temp))
         ReturnFalse;
     }
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Runs the CNeuronLSTM feed-forward pass and writes the layer  output.  |
//+-----------------------------------------------------------------------+
bool CNeuronLSTM::feedForward(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || prevLayer.Total() <= 0)
      ReturnFalse;
   CNeuronBase *temp;
   CConnection *temp_con;
   if(CheckPointer(Input) == POINTER_INVALID)
     {
      Input = new CArrayFloat();
      if(CheckPointer(Input) == POINTER_INVALID)
         ReturnFalse;
     }
   else
      Input.Clear();
//--- Concatenate input sequence
   int total = prevLayer.Total();
   if(!Input.Reserve(total + OutputLayer.Total()))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      temp = prevLayer.At(i);
      if(CheckPointer(temp) == POINTER_INVALID || !Input.Add(temp.getOutputVal()))
         ReturnFalse;
     }
   total = OutputLayer.Total();
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      if(CheckPointer(temp) == POINTER_INVALID || !Input.Add(temp.getOutputVal()))
         ReturnFalse;
     }
   int total_data = Input.Total();
//--- Calculated forget gate
   CArrayFloat *forget_gate = CalculateGate(ForgetGate, Input);
   if(CheckPointer(forget_gate) == POINTER_INVALID)
      ReturnFalse;
//--- Calculated input gate
   CArrayFloat *input_gate = CalculateGate(InputGate, Input);
   if(CheckPointer(input_gate) == POINTER_INVALID)
      ReturnFalse;
//--- Calculated output gate
   CArrayFloat *output_gate = CalculateGate(OutputGate, Input);
   if(CheckPointer(output_gate) == POINTER_INVALID)
      ReturnFalse;
//--- Calculated new content
   CArrayFloat *new_content = new CArrayFloat();
   if(CheckPointer(new_content) == POINTER_INVALID)
      ReturnFalse;
   total = NewContent.Total();
   for(int i = 0; i < total; i++)
     {
      temp = NewContent.At(i);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      float val = 0;
      for(int c = 0; c < total_data; c++)
        {
         temp_con = temp.Connections.At(c);
         if(CheckPointer(temp_con) == POINTER_INVALID)
            ReturnFalse;
         val += temp_con.weight * Input.At(c);
        }
      val = TanhFunction(val);
      temp.setOutputVal(val);
      if(!new_content.Add(val))
         ReturnFalse;
     }
//--- Calculated output sequences
   for(int i = 0; i < total; i++)
     {
      if(PrevMemory.Total() <= i)
         PrevMemory.Add(Memory.At(i));
      else
         PrevMemory.Update(i, Memory.At(i));
      float value = Memory.At(i) * forget_gate.At(i) + new_content.At(i) * input_gate.At(i);
      if(!Memory.Update(i, value))
         ReturnFalse;
      temp = OutputLayer.At(i);
      value = TanhFunction(value) * output_gate.At(i);
      temp.setOutputVal(value);
     }
//---
   DeleteObj(forget_gate);
   DeleteObj(input_gate);
   DeleteObj(new_content);
   DeleteObj(output_gate);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronLSTM calculate gate operation.                |
//+------------------------------------------------------------------+
CArrayFloat *CNeuronLSTM::CalculateGate(CLayer *gate, CArrayFloat *sequence)
  {
   CNeuronBase *temp;
   CConnection *temp_con;
   CArrayFloat *result = new CArrayFloat();
   if(CheckPointer(gate) == POINTER_INVALID)
      return NULL;
   int total = gate.Total();
   int total_data = sequence.Total();
   for(int i = 0; i < total; i++)
     {
      temp = gate.At(i);
      if(CheckPointer(temp) == POINTER_INVALID)
        {
         DeleteObj(result);
         return NULL;
        }
      float val = 0;
      for(int c = 0; c < total_data; c++)
        {
         temp_con = temp.Connections.At(c);
         if(CheckPointer(temp_con) == POINTER_INVALID)
           {
            DeleteObj(result);
            return NULL;
           }
         val += temp_con.weight * (sequence.At(c) == DBL_MAX ? 1 : sequence.At(c));
        }
      val = SigmoidFunction(val);
      temp.setOutputVal(val);
      if(!result.Add(val))
        {
         DeleteObj(result);
         return NULL;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronLSTM calc hidden gradients operation.         |
//+------------------------------------------------------------------+
bool CNeuronLSTM::calcHiddenGradients(CLayer *&nextLayer)
  {
   if(CheckPointer(InputGradient) == POINTER_INVALID)
     {
      InputGradient = new CArrayFloat();
      if(CheckPointer(InputGradient) == POINTER_INVALID)
         ReturnFalse;
     }
   else
      InputGradient.Clear();
//---
   int total = OutputLayer.Total();
   CNeuron *temp;
   CArrayFloat *MemoryGradient = new CArrayFloat();
   CNeuron *gate;
   CConnection *con;
//---
   if(nextLayer != OutputLayer)
      for(int i = 0; i < total; i++)
        {
         temp = OutputLayer.At(i);
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         temp.setGradient(temp.sumDOW(nextLayer));
        }
//--- Calculated memory  and output gate gradients
   if(CheckPointer(MemoryGradient) == POINTER_INVALID)
      ReturnFalse;
   if(!MemoryGradient.Reserve(total))
      ReturnFalse;
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      gate = OutputGate.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      float value = temp.getGradient() * gate.getOutputVal();
      value = TanhFunctionDerivative(Memory.At(i)) * value;
      if(i >= MemoryGradient.Total())
        {
         if(!MemoryGradient.Add(value))
            ReturnFalse;
        }
      else
        {
         value = MemoryGradient.At(i) + value;
         if(!MemoryGradient.Update(i, value))
            ReturnFalse;
        }
      gate.setGradient(gate.getOutputVal() != 0 && temp.getGradient() != 0 ? temp.getGradient()*temp.getOutputVal()*SigmoidFunctionDerivative(gate.getOutputVal()) / gate.getOutputVal() : 0);
      //--- Calcculated gates and new content gradients
      gate = ForgetGate.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      gate.setGradient(gate.getOutputVal() != 0 && value != 0 ? value * SigmoidFunctionDerivative(gate.getOutputVal()) : 0);
      gate = InputGate.At(i);
      temp = NewContent.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      gate.setGradient(gate.getOutputVal() != 0 && value != 0 ? value * temp.getOutputVal()*SigmoidFunctionDerivative(gate.getOutputVal()) : 0);
      temp.setGradient(temp.getOutputVal() != 0 && value != 0 ? value * gate.getOutputVal()*TanhFunctionDerivative(temp.getOutputVal()) : 0);
     }
//--- Calculated input gradients
   int total_inp = temp.getConnections().Total();
   for(int n = 0; n < total_inp; n++)
     {
      float value = 0;
      for(int i = 0; i < total; i++)
        {
         temp = ForgetGate.At(i);
         con = temp.getConnections().At(n);
         value += temp.getGradient() * con.weight;
         //---
         temp = InputGate.At(i);
         con = temp.getConnections().At(n);
         value += temp.getGradient() * con.weight;
         //---
         temp = OutputGate.At(i);
         con = temp.getConnections().At(n);
         value += temp.getGradient() * con.weight;
         //---
         temp = NewContent.At(i);
         con = temp.getConnections().At(n);
         value += temp.getGradient() * con.weight;
        }
      if(InputGradient.Total() >= n)
        {
         if(!InputGradient.Add(value))
            ReturnFalse;
        }
      else
         if(!InputGradient.Update(n, value))
            ReturnFalse;
     }
//--- Calculated gradients for prev. state
   int shift = total_inp - total;
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      temp.setGradient(InputGradient.At(shift + i));
     }
//--- Calculated memory  and output gate gradients
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      gate = OutputGate.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      float value = temp.getGradient() * gate.getPrevVal();
      value = MemoryGradient.At(i) + TanhFunctionDerivative(PrevMemory.At(i)) * value;
      if(!MemoryGradient.Update(i, value))
         ReturnFalse;
      gate.setGradient(gate.getGradient() + (gate.getPrevVal() != 0 && temp.getGradient() != 0 ? temp.getGradient()*temp.getPrevVal()*SigmoidFunctionDerivative(gate.getPrevVal()) / gate.getPrevVal() : 0));
      //--- Calcculated gates and new content gradients
      gate = ForgetGate.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      gate.setGradient(gate.getGradient() + (gate.getPrevVal() != 0 && value != 0 ? value * SigmoidFunctionDerivative(gate.getPrevVal()) : 0));
      gate = InputGate.At(i);
      temp = NewContent.At(i);
      if(CheckPointer(gate) == POINTER_INVALID)
         ReturnFalse;
      gate.setGradient(gate.getGradient() + (gate.getPrevVal() != 0 && value != 0 ? value * temp.getPrevVal()*SigmoidFunctionDerivative(gate.getPrevVal()) : 0));
      temp.setGradient(temp.getGradient() + (temp.getPrevVal() != 0 && value != 0 ? value * gate.getPrevVal()*TanhFunctionDerivative(temp.getPrevVal()) : 0));
     }
//---
   DeleteObj(MemoryGradient);
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Updates the CNeuronLSTM trainable weights from the accumulated  gradients.  |
//+-----------------------------------------------------------------------------+
bool CNeuronLSTM::updateInputWeights(CLayer *&prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || CheckPointer(Input) == POINTER_INVALID)
      ReturnFalse;
//---
   if(!updateInputWeights(ForgetGate, Input) || !updateInputWeights(InputGate, Input) || !updateInputWeights(OutputGate, Input)
      || !updateInputWeights(NewContent, Input))
     {
      ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------------------+
//| Updates the CNeuronLSTM trainable weights from the accumulated   gradients.  |
//+------------------------------------------------------------------------------+
bool CNeuronLSTM::updateInputWeights(CLayer *gate, CArrayFloat *input_data)
  {
   if(CheckPointer(gate) == POINTER_INVALID || CheckPointer(input_data) == POINTER_INVALID)
      ReturnFalse;
   CNeuronBase *neuron;
   CConnection *con;
   int total_n = gate.Total();
   int total_data = input_data.Total();
   float lt = (float)(lr * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t)));
   for(int n = 0; n < total_n; n++)
     {
      neuron = gate.At(n);
      if(CheckPointer(neuron) == POINTER_INVALID)
         ReturnFalse;
      for(int i = 0; i < total_data; i++)
        {
         con = neuron.getConnections().At(i);
         if(CheckPointer(con) == POINTER_INVALID)
            ReturnFalse;
         float data = input_data.At(i);
         float g = neuron.getGradient();
         if(optimization == SGD)
            con.weight += con.deltaWeight = (g != 0 && data != 0 ? lr * g * (data != DBL_MAX ? data : 1) : 0) + alpha * con.deltaWeight;
         else
           {
            con.mt = b1 * con.mt + (1 - b1) * g;
            con.vt = (float)(b2 * con.vt + (1 - b2) * pow(g, 2) + 0.00000001);
            con.weight += con.deltaWeight = (float)(lt * con.mt / sqrt(con.vt));
            t++;
           }
        }
     }
//---
   return true;
  }
//+----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronLSTM  layer.  |
//+----------------------------------------------------------------------+
bool CNeuronLSTM::calcInputGradients(CNeuronBase *prevNeuron, uint index)
  {
   if(CheckPointer(prevNeuron) == POINTER_INVALID || CheckPointer(InputGradient) == POINTER_INVALID || InputGradient.Total() <= (int)index)
      ReturnFalse;
//---
   prevNeuron.setGradient(InputGradient.At(index));
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronLSTM   layer.  |
//+-----------------------------------------------------------------------+
bool CNeuronLSTM::calcInputGradients(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   int total = prevLayer.Total();
   if(total <= 0)
      ReturnFalse;
   CNeuronBase *neuron;
   bool result = true;
   for(int i = 0; (i < total && result); i++)
     {
      neuron = prevLayer.At(i);
      if(CheckPointer(neuron) == POINTER_INVALID)
        {
         result = false;
         break;
        }
      result = calcInputGradients(neuron, i);
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronLSTM state to the file handle.                  |
//+------------------------------------------------------------------+
bool CNeuronLSTM::Save(const int file_handle)
  {
   if(!CNeuronProof::Save(file_handle))
      ReturnFalse;
   if(!ForgetGate.Save(file_handle))
      ReturnFalse;
   if(!InputGate.Save(file_handle))
      ReturnFalse;
   if(!OutputGate.Save(file_handle))
      ReturnFalse;
   if(!NewContent.Save(file_handle))
      ReturnFalse;
   if(!Memory.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronLSTM state from the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronLSTM::Load(const int file_handle)
  {
   if(!CNeuronProof::Load(file_handle))
      ReturnFalse;
   if(!ForgetGate.Load(file_handle))
      ReturnFalse;
   if(!InputGate.Load(file_handle))
      ReturnFalse;
   if(!OutputGate.Load(file_handle))
      ReturnFalse;
   if(!NewContent.Load(file_handle))
      ReturnFalse;
   if(!Memory.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Constructor: initializes the CNeuronLSTMOCL layer defaults.      |
//+------------------------------------------------------------------+
CNeuronLSTMOCL::CNeuronLSTMOCL(void)   :  m_iMemory(-1),
   m_iConcatenated(-1),
   m_iConcatenatedGradient(-1),
   m_iHiddenState(-1),
   m_iWeightsGradient(-1),
   m_iInputs(-1),
   m_iVariables(1)
  {}
//+------------------------------------------------------------------+
//| Destructor: releases the CNeuronLSTMOCL GPU resources.           |
//+------------------------------------------------------------------+
CNeuronLSTMOCL::~CNeuronLSTMOCL(void)
  {
   if(!OpenCL)
      return;
   OpenCL.BufferFree(m_iConcatenated);
   OpenCL.BufferFree(m_iConcatenatedGradient);
   OpenCL.BufferFree(m_iHiddenState);
   OpenCL.BufferFree(m_iMemory);
   OpenCL.BufferFree(m_iWeightsGradient);
   m_cFirstMomentumLSTM.BufferFree();
   m_cSecondMomentumLSTM.BufferFree();
   m_cWeightsLSTM.BufferFree();
  }
//+-----------------------------------------------------------------------+
//| Initializes the CNeuronLSTMOCL layer and allocates its GPU  buffers.  |
//+-----------------------------------------------------------------------+
bool CNeuronLSTMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return Init(numOutputs, myIndex, open_cl, numNeurons, 1, optimization_type, batch);
  }
//+------------------------------------------------------------------------+
//| Initializes the CNeuronLSTMOCL layer and allocates its GPU   buffers.  |
//+------------------------------------------------------------------------+
bool CNeuronLSTMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons * variables, optimization_type, batch))
      ReturnFalse;
//---
   m_iVariables = int(variables);
   m_iMemory = OpenCL.AddBuffer(sizeof(float) * (numNeurons * m_iVariables) * 2, CL_MEM_READ_WRITE);
   if(m_iMemory < 0)
      ReturnFalse;
   m_iHiddenState = OpenCL.AddBuffer(sizeof(float) * (numNeurons * m_iVariables), CL_MEM_READ_WRITE);
   if(m_iHiddenState < 0)
      ReturnFalse;
   m_iConcatenated = OpenCL.AddBuffer(sizeof(float) * (numNeurons * m_iVariables) * 4, CL_MEM_READ_WRITE);
   if(m_iConcatenated < 0)
      ReturnFalse;
   m_iConcatenatedGradient = OpenCL.AddBuffer(sizeof(float) * (numNeurons * m_iVariables) * 4, CL_MEM_READ_WRITE);
   if(m_iConcatenatedGradient < 0)
      ReturnFalse;
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Runs the CNeuronLSTMOCL feed-forward pass and writes the layer  output.  |
//+--------------------------------------------------------------------------+
bool CNeuronLSTMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() <= 0 ||
      NeuronOCL.getOutputIndex() < 0 || !OpenCL)
      ReturnFalse;
//---
   if(m_iInputs != NeuronOCL.Neurons() / m_iVariables)
     {
      if(!SetInputs(NeuronOCL.Neurons() / m_iVariables))
         ReturnFalse;
     }
//---
   if(m_iMemory < 0 || m_iConcatenated < 0)
      ReturnFalse;
//---
   setBuffer(def_k_LSTM_FeedForward, def_k_lstmff_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_LSTM_FeedForward, def_k_lstmff_concatenated, m_iConcatenated)
   setArgument(def_k_LSTM_FeedForward, def_k_lstmff_inputs_size, m_iInputs)
   setBuffer(def_k_LSTM_FeedForward, def_k_lstmff_memory, m_iMemory)
   setBuffer(def_k_LSTM_FeedForward, def_k_lstmff_outputs, getOutputIndex())
   setBuffer(def_k_LSTM_FeedForward, def_k_lstmff_weights, m_cWeightsLSTM.GetIndex())
   uint global_work_offset[] = {0, 0, 0};
   uint global_work_size[] = {Neurons() / m_iVariables, 4, m_iVariables};
   uint local_work_size[] = {1, 4, 1};
   kernelExecuteLoc(def_k_LSTM_FeedForward, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Sets the inputs of the CNeuronLSTMOCL.                           |
//+------------------------------------------------------------------+
bool CNeuronLSTMOCL::SetInputs(int count)
  {
   m_iInputs = count;
   count = (int)((m_iInputs + Neurons() / m_iVariables + 1) * (Neurons() / m_iVariables) * 4);
   if(!m_cWeightsLSTM.Reserve(count))
      ReturnFalse;
   float k = (float)(1 / sqrt(Neurons() + 1));
   for(int i = 0; i < count; i++)
     {
      if(!m_cWeightsLSTM.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!m_cWeightsLSTM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!m_cFirstMomentumLSTM.BufferInit(count, 0))
      ReturnFalse;
   if(!m_cFirstMomentumLSTM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!m_cSecondMomentumLSTM.BufferInit(count, 0))
      ReturnFalse;
   if(!m_cSecondMomentumLSTM.BufferCreate(OpenCL))
      ReturnFalse;
   if(m_iWeightsGradient >= 0)
      OpenCL.BufferFree(m_iWeightsGradient);
   m_iWeightsGradient = OpenCL.AddBuffer(sizeof(float) * count, CL_MEM_READ_WRITE);
   if(m_iWeightsGradient < 0)
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronLSTMOCL  layer.  |
//+-------------------------------------------------------------------------+
bool CNeuronLSTMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || NeuronOCL.Neurons() <= 0 || NeuronOCL.getGradientIndex() < 0 ||
      NeuronOCL.getOutputIndex() < 0 || !OpenCL)
      ReturnFalse;
//---
   if(m_cWeightsLSTM.GetIndex() < 0 || m_cFirstMomentumLSTM.GetIndex() < 0 ||
      m_cSecondMomentumLSTM.GetIndex() < 0)
      ReturnFalse;
   if(m_iInputs < 0 || m_iConcatenated < 0 || m_iMemory < 0 ||
      m_iConcatenatedGradient < 0 || m_iHiddenState < 0 || m_iInputs != NeuronOCL.Neurons() / m_iVariables)
      ReturnFalse;
//---
   setBuffer(def_k_LSTM_ConcatenatedGradient, def_k_lstmcg_concatenated, m_iConcatenated)
   setBuffer(def_k_LSTM_ConcatenatedGradient, def_k_lstmcg_concatenated_gradient, m_iConcatenatedGradient)
   setBuffer(def_k_LSTM_ConcatenatedGradient, def_k_lstmcg_gradient, getGradientIndex())
   setBuffer(def_k_LSTM_ConcatenatedGradient, def_k_lstmcg_memory, m_iMemory)
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {Neurons() / m_iVariables, m_iVariables};
   kernelExecute(def_k_LSTM_ConcatenatedGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   vector<float> temp = vector<float>::Zeros(Neurons() * 4);
   if(!OpenCL.BufferToVector(m_iConcatenatedGradient, temp, temp.Size()))
      ReturnFalse;
#endif
//---
   uint local_work_size[] = {1, m_iVariables};
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_concatenated_gradient, m_iConcatenatedGradient)
   setArgument(def_k_LSTM_HiddenGradient, def_k_lstmhg_hidden_size, Neurons() / m_iVariables)
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_hidden_state, m_iHiddenState)
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_inputs_gradient, NeuronOCL.getGradientIndex())
   setArgument(def_k_LSTM_HiddenGradient, def_k_lstmhg_inputs_size, m_iInputs)
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_output, getOutputIndex())
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_weeights, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_LSTM_HiddenGradient, def_k_lstmhg_weights_gradient, m_iWeightsGradient)
   kernelExecuteLoc(def_k_LSTM_HiddenGradient, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Updates the CNeuronLSTMOCL trainable weights from the  accumulated  gradients.  |
//+---------------------------------------------------------------------------------+
bool CNeuronLSTMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || m_cWeightsLSTM.GetIndex() < 0 || m_iWeightsGradient < 0 ||
      m_cFirstMomentumLSTM.GetIndex() < 0 || m_cSecondMomentumLSTM.GetIndex() < 0)
      ReturnFalse;
//---
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_weights_gradient, m_iWeightsGradient)
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_m, m_cFirstMomentumLSTM.GetIndex())
   setBuffer(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_matrix_v, m_cSecondMomentumLSTM.GetIndex())
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_l, lr)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b1, b1)
   setArgument(def_k_LSTM_UpdateWeightsAdam, def_k_lstmuw_b2, b2)
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {(m_iInputs + Neurons()) / m_iVariables + 1, Neurons() / m_iVariables};
   kernelExecute(def_k_LSTM_UpdateWeightsAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!m_cWeightsLSTM.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Append LSTM parameters and optimizer buffers.                    |
//+------------------------------------------------------------------+
bool CNeuronLSTMOCL::AppendParameterFingerprint(ulong &fingerprint)
  {
   if(!CNeuronBaseOCL::AppendParameterFingerprint(fingerprint) ||
      !AppendParameterBufferFingerprint(GetPointer(m_cWeightsLSTM), fingerprint, 31) ||
      !AppendParameterBufferFingerprint(GetPointer(m_cFirstMomentumLSTM), fingerprint, 32) ||
      !AppendParameterBufferFingerprint(GetPointer(m_cSecondMomentumLSTM), fingerprint, 33))
      ReturnFalse;
   return(true);
  }
//+--------------------------------------------------------------------------------+
//| Averages the CNeuronLSTMOCL weights toward the source model with  factor tau.  |
//+--------------------------------------------------------------------------------+
bool CNeuronLSTMOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronLSTMOCL *Source = source;
   if(!OpenCL || m_cWeightsLSTM.GetIndex() < 0 ||
      m_cFirstMomentumLSTM.GetIndex() < 0 || m_cSecondMomentumLSTM.GetIndex() < 0)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {m_cWeightsLSTM.Total()};
   ResetLastError();
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_target, m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_source, Source.m_cWeightsLSTM.GetIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_m, m_cFirstMomentumLSTM.GetIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_v, m_cSecondMomentumLSTM.GetIndex())
   setArgument(def_k_SoftUpdateAdam, def_k_sua_tau, (float)tau)
   setArgument(def_k_SoftUpdateAdam, def_k_sua_b1, (float)b1)
   setArgument(def_k_SoftUpdateAdam, def_k_sua_b2, (float)b2)
   kernelExecute(def_k_SoftUpdateAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!m_cWeightsLSTM.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronLSTMOCL state to the file handle.               |
//+------------------------------------------------------------------+
bool CNeuronLSTMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, m_iInputs, INT_VALUE) < sizeof(m_iInputs))
      ReturnFalse;
   if(FileWriteInteger(file_handle, m_iVariables, INT_VALUE) < sizeof(m_iInputs))
      ReturnFalse;
   if((m_cWeightsLSTM.GetIndex() >= 0 && !m_cWeightsLSTM.BufferRead()) || !m_cWeightsLSTM.Save(file_handle))
      ReturnFalse;
   if((m_cFirstMomentumLSTM.GetIndex() >= 0 && !m_cFirstMomentumLSTM.BufferRead()) || !m_cFirstMomentumLSTM.Save(file_handle))
      ReturnFalse;
   if((m_cSecondMomentumLSTM.GetIndex() >= 0 && !m_cSecondMomentumLSTM.BufferRead()) || !m_cSecondMomentumLSTM.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronLSTMOCL state from the file handle.          |
//+------------------------------------------------------------------+
bool CNeuronLSTMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   m_iInputs = FileReadInteger(file_handle);
   m_iVariables = FileReadInteger(file_handle);
//---
   m_cWeightsLSTM.BufferFree();
   if(!m_cWeightsLSTM.Load(file_handle) || !m_cWeightsLSTM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   m_cFirstMomentumLSTM.BufferFree();
   if(!m_cFirstMomentumLSTM.Load(file_handle) || !m_cFirstMomentumLSTM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   m_cSecondMomentumLSTM.BufferFree();
   if(!m_cSecondMomentumLSTM.Load(file_handle) || !m_cSecondMomentumLSTM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(m_iMemory >= 0)
      OpenCL.BufferFree(m_iMemory);
   m_iMemory = OpenCL.AddBuffer(sizeof(float) * 2 * Neurons(), CL_MEM_READ_WRITE);
   if(m_iMemory < 0)
      ReturnFalse;
//---
   if(m_iConcatenated >= 0)
      OpenCL.BufferFree(m_iConcatenated);
   m_iConcatenated = OpenCL.AddBuffer(sizeof(float) * 4 * Neurons(), CL_MEM_READ_WRITE);
   if(m_iConcatenated < 0)
      ReturnFalse;
//---
   if(m_iConcatenatedGradient >= 0)
      OpenCL.BufferFree(m_iConcatenatedGradient);
   m_iConcatenatedGradient = OpenCL.AddBuffer(sizeof(float) * 4 * Neurons(), CL_MEM_READ_WRITE);
   if(m_iConcatenatedGradient < 0)
      ReturnFalse;
//---
   if(m_iHiddenState >= 0)
      OpenCL.BufferFree(m_iHiddenState);
   m_iHiddenState = OpenCL.AddBuffer(sizeof(float) * Neurons(), CL_MEM_READ_WRITE);
   if(m_iHiddenState < 0)
      ReturnFalse;
//---
   if(m_iWeightsGradient >= 0)
      OpenCL.BufferFree(m_iWeightsGradient);
   m_iWeightsGradient = OpenCL.AddBuffer(sizeof(float) * m_cWeightsLSTM.Total(), CL_MEM_READ_WRITE);
   if(m_iWeightsGradient < 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronLSTMOCL execution state and its GPU buffers.   |
//+------------------------------------------------------------------+
bool CNeuronLSTMOCL::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
//---
   float emp[];
   ArrayResize(emp, Neurons() * 2);
   ArrayInitialize(emp, 0);
   if(!OpenCL.BufferWrite(m_iHiddenState, emp, 0, 0, Neurons()))
      ReturnFalse;
   if(!OpenCL.BufferWrite(m_iMemory, emp, 0, 0, Neurons() * 2))
      ReturnFalse;
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Rebinds the CNeuronLSTMOCL GPU buffers to the given OpenCL  context.  |
//+-----------------------------------------------------------------------+
void CNeuronLSTMOCL::SetOpenCL(COpenCLMy * obj)
  {
   if(!!OpenCL)
     {
      OpenCL.BufferFree(m_iMemory);
      OpenCL.BufferFree(m_iHiddenState);
      OpenCL.BufferFree(m_iConcatenated);
      OpenCL.BufferFree(m_iConcatenatedGradient);
      OpenCL.BufferFree(m_iWeightsGradient);
     }
   CNeuronBaseOCL::SetOpenCL(obj);
   m_cWeightsLSTM.BufferCreate(OpenCL);
   m_cFirstMomentumLSTM.BufferCreate(OpenCL);
   m_cSecondMomentumLSTM.BufferCreate(OpenCL);
   m_iWeightsGradient = OpenCL.AddBuffer(sizeof(float) * m_cWeightsLSTM.Total(), CL_MEM_READ_WRITE);
   int numNeurons = Neurons();
   m_iMemory = OpenCL.AddBuffer(sizeof(float) * numNeurons * 2, CL_MEM_READ_WRITE);
   m_iHiddenState = OpenCL.AddBuffer(sizeof(float) * numNeurons, CL_MEM_READ_WRITE);
   m_iConcatenated = OpenCL.AddBuffer(sizeof(float) * numNeurons * 4, CL_MEM_READ_WRITE);
   m_iConcatenatedGradient = OpenCL.AddBuffer(sizeof(float) * numNeurons * 4, CL_MEM_READ_WRITE);
  }
//+----------------------------------------------------------------------+
//| Initializes the CNeuronSSMOCL layer and allocates its GPU  buffers.  |
//+----------------------------------------------------------------------+
bool CNeuronSSMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
//---
   if(!cHiddenStates.Init(0, 0, OpenCL, window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cHiddenStates.SetActivationFunction(None);
   iWindowHidden = window_key;
//---
   if(!cA.Init(0, 1, OpenCL, iWindowHidden, iWindowHidden, iWindowHidden, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cA.SetActivationFunction(SIGMOID);
//---
   if(!cB.Init(0, 2, OpenCL, window, window, iWindowHidden, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cB.SetActivationFunction(SIGMOID);
//---
   if(!cAB.Init(0, 3, OpenCL, 2 * iWindowHidden * units_count, optimization, iBatch))
      ReturnFalse;
   cAB.SetActivationFunction(None);
//---
   if(!cC.Init(0, 4, OpenCL, 2 * iWindowHidden, 2 * iWindowHidden, window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cC.SetActivationFunction(None);
//---
   SetActivationFunction(None);
   if(!SetOutput(cC.getOutput()) || !SetGradient(cC.getGradient()))
      ReturnFalse;
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Runs the CNeuronSSMOCL feed-forward pass and writes the layer  output.  |
//+-------------------------------------------------------------------------+
bool CNeuronSSMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cA.FeedForward(cHiddenStates.AsObject()))
      ReturnFalse;
   if(!cB.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!Concat(cA.getOutput(), cB.getOutput(), cAB.getOutput(), iWindowHidden, iWindowHidden, cA.Neurons() / iWindowHidden))
      ReturnFalse;
   if(!cC.FeedForward(cAB.AsObject()))
      ReturnFalse;
//---
   if(!bTrain)
      if(!SumAndNormalize(cA.getOutput(), cB.getOutput(), cHiddenStates.getOutput(), iWindowHidden, true))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronSSMOCL  layer.  |
//+------------------------------------------------------------------------+
bool CNeuronSSMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cAB.CalcHiddenGradients(cC.AsObject()))
      ReturnFalse;
//---
   if(!DeConcat(cA.getGradient(), cB.getGradient(), cAB.getGradient(), iWindowHidden, iWindowHidden, cA.Neurons() / iWindowHidden))
      ReturnFalse;
   if(cA.Activation() != None && !DeActivation(cA.getOutput(), cA.getGradient(), cA.getGradient(), cA.Activation()))
      ReturnFalse;
   if(cB.Activation() != None && !DeActivation(cB.getOutput(), cB.getGradient(), cB.getGradient(), cB.Activation()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cB.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Updates the CNeuronSSMOCL trainable weights from the accumulated  gradients.  |
//+-------------------------------------------------------------------------------+
bool CNeuronSSMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cA.UpdateInputWeights(cHiddenStates.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cA.getOutput(), cB.getOutput(), cHiddenStates.getOutput(), iWindowHidden, true))
      ReturnFalse;
//---
   if(!cB.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cC.UpdateInputWeights(cAB.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------+
//| Rebinds the CNeuronSSMOCL GPU buffers to the given OpenCL  context.  |
//+----------------------------------------------------------------------+
void CNeuronSSMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cHiddenStates.SetOpenCL(OpenCL);
   cA.SetOpenCL(OpenCL);
   cB.SetOpenCL(OpenCL);
   cAB.SetOpenCL(OpenCL);
   cC.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronSSMOCL state to the file handle.                |
//+------------------------------------------------------------------+
bool CNeuronSSMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, int(iWindowHidden)) < INT_VALUE)
      ReturnFalse;
   if(!cA.Save(file_handle))
      ReturnFalse;
   if(!cB.Save(file_handle))
      ReturnFalse;
   if(!cC.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronSSMOCL state from the file handle.           |
//+------------------------------------------------------------------+
bool CNeuronSSMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindowHidden = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cB.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cC.AsObject()))
      ReturnFalse;
//---
   if(!cHiddenStates.Init(0, 0, OpenCL, cA.Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cAB.Init(0, 3, OpenCL, 2 * cA.Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   if(!SetOutput(cC.getOutput()) || !SetGradient(cC.getGradient()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronSSMOCL execution state and its GPU buffers.    |
//+------------------------------------------------------------------+
bool CNeuronSSMOCL::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   CBufferFloat *output = cHiddenStates.getOutput();
   if(!output ||
      !output.Fill(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Initializes the CNeuronMambaOCL layer and allocates its GPU  buffers.  |
//+------------------------------------------------------------------------+
bool CNeuronMambaOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
   if(!cXProject.Init(0, 0, OpenCL, window, window, window_key + 2, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cXProject.SetActivationFunction(None);
   if(!cZProject.Init(0, 1, OpenCL, window, window, window_key, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cZProject.SetActivationFunction(SIGMOID);
   if(!cInsideConv.Init(0, 2, OpenCL, 3, 1, 1, window_key, units_count, optimization, iBatch))
      ReturnFalse;
   cInsideConv.SetActivationFunction(SIGMOID);
   if(!cSSM.Init(0, 3, OpenCL, window_key, window_key, units_count, optimization, iBatch))
      ReturnFalse;
   if(!cZSSM.Init(0, 4, OpenCL, 2 * window_key * units_count, optimization, iBatch))
      ReturnFalse;
   cZSSM.SetActivationFunction(None);
   if(!cOutProject.Init(0, 5, OpenCL, 2 * window_key, 2 * window_key, window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cOutProject.SetActivationFunction(None);
//---
   if(!Temp.BufferInit(window * units_count, 0))
      ReturnFalse;
   if(!Temp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!SetOutput(cOutProject.getOutput()))
      ReturnFalse;
   if(!SetGradient(cOutProject.getGradient()))
      ReturnFalse;
   SetActivationFunction(None);
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Runs the CNeuronMambaOCL feed-forward pass and writes the layer  output.  |
//+---------------------------------------------------------------------------+
bool CNeuronMambaOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cXProject.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cZProject.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(!cInsideConv.FeedForward(cXProject.AsObject()))
      ReturnFalse;
   if(!cSSM.FeedForward(cInsideConv.AsObject()))
      ReturnFalse;
   if(!Concat(cSSM.getOutput(), cZProject.getOutput(), cZSSM.getOutput(), 1, 1, cSSM.Neurons()))
      ReturnFalse;
   if(!cOutProject.FeedForward(cZSSM.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronMambaOCL  layer.  |
//+--------------------------------------------------------------------------+
bool CNeuronMambaOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!cZSSM.CalcHiddenGradients(cOutProject.AsObject()))
      ReturnFalse;
   if(!DeConcat(cSSM.getGradient(), cZProject.getGradient(), cZSSM.getGradient(), 1, 1, cSSM.Neurons()))
      ReturnFalse;
   if(cZProject.Activation() != None &&
      !DeActivation(cZProject.getOutput(), cZProject.getGradient(), cZProject.getGradient(), cZProject.Activation()))
      ReturnFalse;
   if(!cInsideConv.CalcHiddenGradients(cSSM.AsObject()))
      ReturnFalse;
   if(!cXProject.CalcHiddenGradients(cInsideConv.AsObject()))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cZProject.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), NeuronOCL.getGradient(), GetPointer(Temp), 1, false, 0, 0, 0, 0.5f))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cXProject.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getGradient(), GetPointer(Temp), NeuronOCL.getGradient(), 1, false, 0, 0, 0, 1.0f))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Updates the CNeuronMambaOCL trainable weights from the  accumulated gradients.  |
//+---------------------------------------------------------------------------------+
bool CNeuronMambaOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cXProject.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cZProject.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cInsideConv.UpdateInputWeights(cXProject.AsObject()))
      ReturnFalse;
   if(!cSSM.UpdateInputWeights(cInsideConv.AsObject()))
      ReturnFalse;
   if(!cOutProject.UpdateInputWeights(cZSSM.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Rebinds the CNeuronMambaOCL GPU buffers to the given OpenCL  context.  |
//+------------------------------------------------------------------------+
void CNeuronMambaOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cXProject.SetOpenCL(OpenCL);
   cZProject.SetOpenCL(OpenCL);
   cInsideConv.SetOpenCL(OpenCL);
   cSSM.SetOpenCL(OpenCL);
   cZSSM.SetOpenCL(OpenCL);
   cOutProject.SetOpenCL(OpenCL);
   Temp.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMambaOCL state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuronMambaOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cXProject.Save(file_handle))
      ReturnFalse;
   if(!cZProject.Save(file_handle))
      ReturnFalse;
   if(!cInsideConv.Save(file_handle))
      ReturnFalse;
   if(!cSSM.Save(file_handle))
      ReturnFalse;
   if(!cOutProject.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMambaOCL state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronMambaOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cXProject.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cZProject.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInsideConv.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cSSM.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cOutProject.AsObject()))
      ReturnFalse;
//---
   if(!cZSSM.Init(0, 3, OpenCL, 2 * cZProject.Neurons(), optimization, iBatch))
      ReturnFalse;
   cZSSM.SetActivationFunction(None);
//---
   if(!SetOutput(cOutProject.getOutput()))
      ReturnFalse;
   if(!SetGradient(cOutProject.getGradient()))
      ReturnFalse;
//---
   if(!Temp.BufferInit(Neurons(), 0))
      ReturnFalse;
   if(!Temp.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuronMambaOCL execution state and its GPU buffers.  |
//+------------------------------------------------------------------+
bool CNeuronMambaOCL::Clear(void)
  {
   return cSSM.Clear();
  }
//+-----------------------------------------------------------------------------+
//| Initializes the CNeuronMambaBlockOCL layer and allocates its GPU  buffers.  |
//+-----------------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = window;
//---
   if(!cMamba.Init(0, 0, OpenCL, window, window_key, units_count, optimization, iBatch))
      ReturnFalse;
   if(!cMambaResidual.Init(0, 1, OpenCL, window * units_count, optimization, iBatch))
      ReturnFalse;
   cMambaResidual.SetActivationFunction(None);
   if(!cFF[0].Init(0, 2, OpenCL, window, window, 4 * window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cFF[0].SetActivationFunction(LReLU);
   if(!cFF[1].Init(0, 2, OpenCL, 4 * window, 4 * window, window, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cFF[1].SetActivationFunction(None);
//---
   SetActivationFunction(None);
   SetGradient(cFF[1].getGradient(), true);
//---
   return true;
  }
//+--------------------------------------------------------------------------------+
//| Runs the CNeuronMambaBlockOCL feed-forward pass and writes the  layer output.  |
//+--------------------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cMamba.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!SumAndNormalize(cMamba.getOutput(), NeuronOCL.getOutput(), cMambaResidual.getOutput(), iWindow, true))
      ReturnFalse;
   if(!cFF[0].FeedForward(cMambaResidual.AsObject()))
      ReturnFalse;
   if(!cFF[1].FeedForward(cFF[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cMambaResidual.getOutput(), cFF[1].getOutput(), getOutput(), iWindow, true))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CNeuronMambaBlockOCL layer.  |
//+-------------------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!cFF[0].CalcHiddenGradients(cFF[1].AsObject()))
      ReturnFalse;
   if(!cMambaResidual.CalcHiddenGradients(cFF[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(cMambaResidual.getGradient(), getGradient(), cMamba.getGradient(), iWindow, false))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cMamba.AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
     {
      if(!DeActivation(NeuronOCL.getOutput(), cMambaResidual.getGradient(), cMamba.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
      if(!SumAndNormalize(cMambaResidual.getGradient(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), iWindow, false))
         ReturnFalse;
     }
   else
      if(!SumAndNormalize(cMamba.getGradient(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), iWindow, false))
         ReturnFalse;
//---
   return true;
  }
//+--------------------------------------------------------------------------------------+
//| Updates the CNeuronMambaBlockOCL trainable weights from the  accumulated gradients.  |
//+--------------------------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cMamba.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cFF[0].UpdateInputWeights(cMambaResidual.AsObject()))
      ReturnFalse;
   if(!cFF[1].UpdateInputWeights(cFF[0].AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronMambaBlockOCL state to the file handle.         |
//+------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iWindow)) < INT_VALUE)
      ReturnFalse;
//---
   if(!cMamba.Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!cFF[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronMambaBlockOCL state from the file handle.    |
//+------------------------------------------------------------------+
bool CNeuronMambaBlockOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iWindow = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cMamba.AsObject()))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, cFF[i].AsObject()))
         ReturnFalse;
   if(!cMambaResidual.Init(0, 1, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
//---
   SetGradient(cFF[1].getGradient(), true);
//---
   return true;
  }
//+-----------------------------------------------------------------------------+
//| Rebinds the CNeuronMambaBlockOCL GPU buffers to the given OpenCL  context.  |
//+-----------------------------------------------------------------------------+
void CNeuronMambaBlockOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cMamba.SetOpenCL(OpenCL);
   cMambaResidual.SetOpenCL(OpenCL);
   cFF[0].SetOpenCL(OpenCL);
   cFF[1].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------------+
//| Initializes the CNeuron2DSSMOCL layer and allocates its GPU  buffers.  |
//+------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                           uint window_in, uint window_out, uint units_in, uint units_out,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units_out, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   iWindowOut = window_out;
   iUnitsOut = units_out;
//---
   int index = 0;
   CNeuronConvOCL *conv = NULL;
   CNeuronTransposeOCL *transp = NULL;
//--- Projection Time
   cProjectionX_Time.Clear();
   cProjectionX_Time.SetOpenCL(OpenCL);
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, window_in, optimization, iBatch) ||
      !cProjectionX_Time.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, iUnitsOut, window_in, 1, optimization, iBatch) ||
      !cProjectionX_Time.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, window_in, iUnitsOut, optimization, iBatch) ||
      !cProjectionX_Time.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_in, window_in, iWindowOut, iUnitsOut, 1, optimization, iBatch) ||
      !cProjectionX_Time.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//--- Projection Variables
   cProjectionX_Variable.Clear();
   cProjectionX_Variable.SetOpenCL(OpenCL);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window_in, window_in, iUnitsOut, units_in, 1, optimization, iBatch) ||
      !cProjectionX_Variable.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
   index++;
   transp = new CNeuronTransposeOCL();
   if(!transp ||
      !transp.Init(0, index, OpenCL, units_in, iUnitsOut, optimization, iBatch) ||
      !cProjectionX_Variable.Add(transp))
     {
      DeleteObj(transp);
      ReturnFalse;
     }
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, units_in, units_in, iWindowOut, iUnitsOut, 1, optimization, iBatch) ||
      !cProjectionX_Variable.Add(conv))
     {
      DeleteObj(conv);
      ReturnFalse;
     }
//--- HiddenState
   index++;
   if(!cHiddenStates.Init(0, index, OpenCL, 2 * iUnitsOut * iWindowOut, optimization, iBatch))
      ReturnFalse;
//--- A*H
   index++;
   if(!cA.Init(0, index, OpenCL, iWindowOut, iWindowOut, 2 * iWindowOut, iUnitsOut, 2, optimization, iBatch))
      ReturnFalse;
   if(!SumAndNormalize(cA.GetWeightsConv(), cA.GetWeightsConv(), cA.GetWeightsConv(), iWindowOut, false, 0, 0, 0, 0.05f))
      ReturnFalse;
   cA.SetActivationFunction(MinusSoftPlus);
//--- B
   index++;
   if(!cB_Time.Init(0, index, OpenCL, iWindowOut, iWindowOut, 1, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cB_Time.SetActivationFunction(TANH);
   index++;
   if(!cB_Variable.Init(0, index, OpenCL, iWindowOut, iWindowOut, 1, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cB_Variable.SetActivationFunction(TANH);
//--- C
   index++;
   if(!cC_Time.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cC_Time.SetActivationFunction(TANH);
   index++;
   if(!cC_Variable.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cC_Variable.SetActivationFunction(TANH);
//--- Delta
   index++;
   if(!cDelta_Time.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cDelta_Time.SetActivationFunction(SoftPlus);
   index++;
   if(!cDelta_Variable.Init(0, index, OpenCL, iWindowOut, iWindowOut, iUnitsOut, iUnitsOut, 1, optimization, iBatch))
      ReturnFalse;
   cDelta_Variable.SetActivationFunction(SoftPlus);
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Runs the CNeuron2DSSMOCL feed-forward pass and writes the layer  output.  |
//+---------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   CNeuronBaseOCL *inp = NeuronOCL;
   CNeuronBaseOCL *x_time = NULL;
   CNeuronBaseOCL *x_var = NULL;
//--- Projection Time
   int total = cProjectionX_Time.Total();
   for(int i = 0; i < total; i++)
     {
      x_time = cProjectionX_Time.At(i);
      if(!x_time ||
         !x_time.FeedForward(inp))
         ReturnFalse;
      inp = x_time;
     }
//--- Projection Variable
   inp = NeuronOCL;
   total = cProjectionX_Variable.Total();
   for(int i = 0; i < total; i++)
     {
      x_var = cProjectionX_Variable.At(i);
      if(!x_var ||
         !x_var.FeedForward(inp))
         ReturnFalse;
      inp = x_var;
     }
//---
   if(!cA.FeedForward(cHiddenStates.AsObject()))
      ReturnFalse;
   if(!cB_Time.FeedForward(x_time) ||
      !cB_Variable.FeedForward(x_var))
      ReturnFalse;
   if(!cC_Time.FeedForward(x_time) ||
      !cC_Variable.FeedForward(x_var))
      ReturnFalse;
   if(!cDelta_Time.FeedForward(x_time) ||
      !cDelta_Variable.FeedForward(x_var))
      ReturnFalse;
//---
   if(!cHiddenStates.SwapOutputs())
      ReturnFalse;
//---
   return feedForwardSSM2D();
  }
//+----------------------------------------------------------------------------+
//| Runs the CNeuron2DSSMOCL feed-forward pass and writes the layer   output.  |
//+----------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::feedForwardSSM2D(void)
  {
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!OpenCL || !x_time || !x_var)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnitsOut, iWindowOut};
   uint local_work_size[2] = {global_work_size[0], 1};
   int kernel = def_k_SSM2D_FeedForward;
//---
   ResetLastError();
   setBuffer(kernel, def_k_ssm2d_ah, cA.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_b_time, cB_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_b_var, cB_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_c_time, cC_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_c_var, cC_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_delta_time, cDelta_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_delta_var, cDelta_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_hidden, cHiddenStates.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_px_time, x_time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_px_var, x_var.getOutputIndex())
   setBuffer(kernel, def_k_ssm2d_y, getOutputIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cHiddenStates.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+--------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuron2DSSMOCL  layer.  |
//+--------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::calcInputGradientsSSM2D(void)
  {
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!OpenCL || !x_time || !x_var)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iUnitsOut, iWindowOut};
   uint local_work_size[2] = {1, global_work_size[1]};
   int kernel = def_k_SSM2D_CalcHiddenGradient;
//---
   ResetLastError();
   setBuffer(kernel, def_k_ssm2dhg_ah, cA.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_ah, cA.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_b_time, cB_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_b_time, cB_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_b_var, cB_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_b_var, cB_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_c_time, cC_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_c_time, cC_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_c_var, cC_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_c_var, cC_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_delta_time, cDelta_Time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_delta_time, cDelta_Time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_delta_var, cDelta_Variable.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_delta_var, cDelta_Variable.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_hidden, cHiddenStates.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_px_time, x_time.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_px_time, x_time.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_px_var, x_var.getOutputIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_px_var, x_var.getGradientIndex())
   setBuffer(kernel, def_k_ssm2dhg_grad_y, getGradientIndex())
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!x_var.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+---------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuron2DSSMOCL   layer.  |
//+---------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!calcInputGradientsSSM2D())
      ReturnFalse;
//--- Deactivation
   CNeuronBaseOCL *x_time = cProjectionX_Time[-1];
   CNeuronBaseOCL *x_var = cProjectionX_Variable[-1];
   if(!x_time || !x_var)
      ReturnFalse;
   if(x_time.Activation() != None)
      if(!DeActivation(x_time.getOutput(), x_time.getGradient(), x_time.getGradient(), x_time.Activation()))
         ReturnFalse;
   if(x_var.Activation() != None)
      if(!DeActivation(x_var.getOutput(), x_var.getGradient(), x_var.getGradient(), x_var.Activation()))
         ReturnFalse;
   if(cB_Time.Activation() != None)
      if(!DeActivation(cB_Time.getOutput(), cB_Time.getGradient(), cB_Time.getGradient(), cB_Time.Activation()))
         ReturnFalse;
   if(cB_Variable.Activation() != None)
      if(!DeActivation(cB_Variable.getOutput(), cB_Variable.getGradient(), cB_Variable.getGradient(), cB_Variable.Activation()))
         ReturnFalse;
   if(cC_Time.Activation() != None)
      if(!DeActivation(cC_Time.getOutput(), cC_Time.getGradient(), cC_Time.getGradient(), cC_Time.Activation()))
         ReturnFalse;
   if(cC_Variable.Activation() != None)
      if(!DeActivation(cC_Variable.getOutput(), cC_Variable.getGradient(), cC_Variable.getGradient(), cC_Variable.Activation()))
         ReturnFalse;
   if(cDelta_Time.Activation() != None)
      if(!DeActivation(cDelta_Time.getOutput(), cDelta_Time.getGradient(), cDelta_Time.getGradient(), cDelta_Time.Activation()))
         ReturnFalse;
   if(cDelta_Variable.Activation() != None)
      if(!DeActivation(cDelta_Variable.getOutput(), cDelta_Variable.getGradient(), cDelta_Variable.getGradient(), cDelta_Variable.Activation()))
         ReturnFalse;
   if(cA.Activation() != None)
      if(!DeActivation(cA.getOutput(), cA.getGradient(), cA.getGradient(), cA.Activation()))
         ReturnFalse;
//--- Gradient to projections X
   CBufferFloat *grad_x_time = x_time.getGradient();
   CBufferFloat *grad_x_var = x_var.getGradient();
   if(!x_time.SetGradient(x_time.getPrevOutput(), false) ||
      !x_var.SetGradient(x_var.getPrevOutput(), false))
      ReturnFalse;
//--- B -> X
   if(!x_time.CalcHiddenGradients(cB_Time.AsObject()) ||
      !SumAndNormalize(grad_x_time, x_time.getGradient(), grad_x_time, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_var.CalcHiddenGradients(cB_Variable.AsObject()) ||
      !SumAndNormalize(grad_x_var, x_var.getGradient(), grad_x_var, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
//--- C -> X
   if(!x_time.CalcHiddenGradients(cC_Time.AsObject()) ||
      !SumAndNormalize(grad_x_time, x_time.getGradient(), grad_x_time, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_var.CalcHiddenGradients(cC_Variable.AsObject()) ||
      !SumAndNormalize(grad_x_var, x_var.getGradient(), grad_x_var, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
//--- Delta -> X
   if(!x_time.CalcHiddenGradients(cDelta_Time.AsObject()) ||
      !SumAndNormalize(grad_x_time, x_time.getGradient(), grad_x_time, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_var.CalcHiddenGradients(cDelta_Variable.AsObject()) ||
      !SumAndNormalize(grad_x_var, x_var.getGradient(), grad_x_var, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!x_time.SetGradient(grad_x_time, false) ||
      !x_var.SetGradient(grad_x_var, false))
      ReturnFalse;
//--- Projection Variable
   int total = cProjectionX_Variable.Total() - 2;
   for(int i = total; i >= 0; i--)
     {
      x_var = cProjectionX_Variable[i];
      if(!x_var ||
         !x_var.CalcHiddenGradients(cProjectionX_Variable[i + 1]))
         ReturnFalse;
     }
//--- Projection Time
   total = cProjectionX_Time.Total() - 2;
   for(int i = total; i >= 0; i--)
     {
      x_time = cProjectionX_Time[i];
      if(!x_time ||
         !x_time.CalcHiddenGradients(cProjectionX_Time[i + 1]))
         ReturnFalse;
     }
//--- Projections -> inputs
   if(!NeuronOCL.CalcHiddenGradients(x_var.AsObject()))
      ReturnFalse;
   grad_x_time = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(x_time.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(x_time.AsObject()) ||
      !SumAndNormalize(grad_x_time, NeuronOCL.getGradient(), grad_x_time, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(grad_x_time, false))
      ReturnFalse;
//---
   return true;
  }
//+---------------------------------------------------------------------------------+
//| Updates the CNeuron2DSSMOCL trainable weights from the  accumulated gradients.  |
//+---------------------------------------------------------------------------------+
bool CNeuron2DSSMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cB_Time.UpdateInputWeights(cProjectionX_Time[-1]))
      ReturnFalse;
   if(!cB_Variable.UpdateInputWeights(cProjectionX_Variable[-1]))
      ReturnFalse;
//--- C -> X
   if(!cC_Time.UpdateInputWeights(cProjectionX_Time[-1]))
      ReturnFalse;
   if(!cC_Variable.UpdateInputWeights(cProjectionX_Variable[-1]))
      ReturnFalse;
//--- Delta -> X
   if(!cDelta_Time.UpdateInputWeights(cProjectionX_Time[-1]))
      ReturnFalse;
   if(!cDelta_Variable.UpdateInputWeights(cProjectionX_Variable[-1]))
      ReturnFalse;
//--- Projection Variable
   CNeuronBaseOCL *x_time = NULL;
   CNeuronBaseOCL *x_var = NULL;
   int total = cProjectionX_Variable.Total() - 1;
   for(int i = total; i > 0; i--)
     {
      x_var = cProjectionX_Variable[i];
      if(!x_var ||
         !x_var.UpdateInputWeights(cProjectionX_Variable[i - 1]))
         ReturnFalse;
     }
//--- Projection Time
   total = cProjectionX_Time.Total() - 1;
   for(int i = total; i > 0; i--)
     {
      x_time = cProjectionX_Time[i];
      if(!x_time ||
         !x_time.UpdateInputWeights(cProjectionX_Time[i - 1]))
         ReturnFalse;
     }
//--- Projections -> inputs
   x_var = cProjectionX_Variable[0];
   x_time = cProjectionX_Time[0];
   if(!x_var.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!x_time.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//--- A*H
   if(!cHiddenStates.SwapOutputs() ||
      !cA.UpdateInputWeights(cHiddenStates.AsObject()) ||
      !cHiddenStates.SwapOutputs())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------+
//| Rebinds the CNeuron2DSSMOCL GPU buffers to the given OpenCL  context.  |
//+------------------------------------------------------------------------+
void CNeuron2DSSMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cHiddenStates.SetOpenCL(OpenCL);
   cProjectionX_Time.SetOpenCL(OpenCL);
   cProjectionX_Variable.SetOpenCL(OpenCL);
   cA.SetOpenCL(OpenCL);
   cB_Time.SetOpenCL(OpenCL);
   cB_Variable.SetOpenCL(OpenCL);
   cC_Time.SetOpenCL(OpenCL);
   cC_Variable.SetOpenCL(OpenCL);
   cDelta_Time.SetOpenCL(OpenCL);
   cDelta_Variable.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//| Saves the CNeuron2DSSMOCL state to the file handle.              |
//+------------------------------------------------------------------+
bool CNeuron2DSSMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cHiddenStates.Save(file_handle))
      ReturnFalse;
   if(!cProjectionX_Time.Save(file_handle))
      ReturnFalse;
   if(!cProjectionX_Variable.Save(file_handle))
      ReturnFalse;
   if(!cA.Save(file_handle))
      ReturnFalse;
   if(!cB_Time.Save(file_handle))
      ReturnFalse;
   if(!cB_Variable.Save(file_handle))
      ReturnFalse;
   if(!cC_Time.Save(file_handle))
      ReturnFalse;
   if(!cC_Variable.Save(file_handle))
      ReturnFalse;
   if(!cDelta_Time.Save(file_handle))
      ReturnFalse;
   if(!cDelta_Variable.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuron2DSSMOCL state from the file handle.         |
//+------------------------------------------------------------------+
bool CNeuron2DSSMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cHiddenStates.AsObject()))
      ReturnFalse;
   if(!cProjectionX_Time.Load(file_handle))
      ReturnFalse;
   if(!cProjectionX_Variable.Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cA.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cB_Time.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cB_Variable.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cC_Time.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cC_Variable.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDelta_Time.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cDelta_Variable.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Clears the CNeuron2DSSMOCL execution state and its GPU buffers.  |
//+------------------------------------------------------------------+
bool CNeuron2DSSMOCL::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cHiddenStates.Clear())
      ReturnFalse;
   if(!cA.Clear())
      ReturnFalse;
   if(!cB_Time.Clear())
      ReturnFalse;
   if(!cB_Variable.Clear())
      ReturnFalse;
   if(!cC_Time.Clear())
      ReturnFalse;
   if(!cC_Variable.Clear())
      ReturnFalse;
   if(!cDelta_Time.Clear())
      ReturnFalse;
   if(!cDelta_Variable.Clear())
      ReturnFalse;
//---
   for(int i = 0; i < cProjectionX_Time.Total(); i++)
     {
      if(!cProjectionX_Time[i] ||
         !((CNeuronBaseOCL*)cProjectionX_Time[i]).Clear())
         ReturnFalse;
     }
   for(int i = 0; i < cProjectionX_Variable.Total(); i++)
     {
      if(!cProjectionX_Variable[i] ||
         !((CNeuronBaseOCL*)cProjectionX_Variable[i]).Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Initializes the CNeuronCGLSTMOCL layer and allocates its GPU  buffers.  |
//+-------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                            uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return CNeuronCGLSTMOCL::Init(numOutputs, myIndex, open_cl, numNeurons, numNeurons, 1, optimization_type, batch);
  }
//+--------------------------------------------------------------------------+
//| Initializes the CNeuronCGLSTMOCL layer and allocates its GPU   buffers.  |
//+--------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                            uint count, uint window, uint variables,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, count * variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   if(!cConcatenateInputs.Init(0, 0, OpenCL, (count + window)*variables, optimization, iBatch))
      ReturnFalse;
   cConcatenateInputs.SetActivationFunction(None);
//---
   if(!cProjection.Init(0, 1, OpenCL, count + window, count + window, count * 4, 1, variables, optimization, iBatch))
      ReturnFalse;
   cProjection.SetActivationFunction(None);
//---
   if(!Clear())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Handles the CNeuronCGLSTMOCL cslstm_feed forward operation.      |
//+------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::CSLSTM_feedForward(void)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {cProjection.GetFilters() / 4, cProjection.GetVariables()};
   uint kernel = def_k_CSLSTM_FeedForward;
   setBuffer(kernel, def_k_cslstmff_concatenated, cProjection.getOutputIndex())
   setBuffer(kernel, def_k_cslstmff_memory, cProjection.getPrevOutIndex())
   setBuffer(kernel, def_k_cslstmff_output, getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+-----------------------------------------------------------------------+
//| Handles the CNeuronCGLSTMOCL cslstm_calc hidden gradient  operation.  |
//+-----------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::CSLSTM_CalcHiddenGradient(void)
  {
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {cProjection.GetFilters(), cProjection.GetVariables()};
   uint kernel = def_k_CSLSTM_CalcHiddenGradient;
   setBuffer(kernel, def_k_cslstmhg_concatenated, cProjection.getOutputIndex())
   setBuffer(kernel, def_k_cslstmhg_concatenated_grad, cProjection.getGradientIndex())
   setBuffer(kernel, def_k_cslstmhg_memory, cProjection.getPrevOutIndex())
   setBuffer(kernel, def_k_cslstmhg_output_grad, getGradientIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cProjection.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Runs the CNeuronCGLSTMOCL feed-forward pass and writes the layer  output.  |
//+----------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   int hidden = (int)cProjection.GetFilters() / 4;
   int inputs = (int)cProjection.GetWindow() - hidden;
   int variables = (int)cProjection.GetVariables();
//---
   if(!Concat(NeuronOCL.getOutput(), getOutput(), cConcatenateInputs.getOutput(), inputs, hidden, variables))
      ReturnFalse;
   if(!cProjection.FeedForward(cConcatenateInputs.AsObject()))
      ReturnFalse;
//---
   return CSLSTM_feedForward();
  }
//+---------------------------------------------------------------------------+
//| Back-propagates the input gradients through the CNeuronCGLSTMOCL  layer.  |
//+---------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   int hidden = (int)cProjection.GetFilters() / 4;
   int inputs = (int)cProjection.GetWindow() - hidden;
   int variables = (int)cProjection.GetVariables();
//---
   if(!CSLSTM_CalcHiddenGradient())
      ReturnFalse;
   if(!cConcatenateInputs.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
   if(!DeConcat(NeuronOCL.getGradient(), getPrevOutput(), cConcatenateInputs.getGradient(), inputs, hidden, variables))
      ReturnFalse;
   if(NeuronOCL.Activation() != None)
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), NeuronOCL.getGradient(), NeuronOCL.Activation()))
         ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------------+
//| Updates the CNeuronCGLSTMOCL trainable weights from the  accumulated gradients.  |
//+----------------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return cProjection.UpdateInputWeights(cConcatenateInputs.AsObject());
  }
//+------------------------------------------------------------------+
//| Saves the CNeuronCGLSTMOCL state to the file handle.             |
//+------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CNeuronCGLSTMOCL state from the file handle.        |
//+------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!cConcatenateInputs.Init(0, 0, OpenCL, cProjection.GetWindow()*cProjection.GetVariables(), optimization, iBatch))
      ReturnFalse;
   cConcatenateInputs.SetActivationFunction(None);
//---
   return true;
  }
//+-------------------------------------------------------------------------+
//| Rebinds the CNeuronCGLSTMOCL GPU buffers to the given OpenCL  context.  |
//+-------------------------------------------------------------------------+
void CNeuronCGLSTMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cProjection.SetOpenCL(OpenCL);
   cConcatenateInputs.SetOpenCL(OpenCL);
  }
//+----------------------------------------------------------------------------------+
//| Averages the CNeuronCGLSTMOCL weights toward the source model  with factor tau.  |
//+----------------------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   if(!cProjection.WeightsUpdate(((CNeuronCGLSTMOCL*)source).cProjection.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------+
//| Clears the CNeuronCGLSTMOCL execution state and its GPU buffers.  |
//+-------------------------------------------------------------------+
bool CNeuronCGLSTMOCL::Clear(void)
  {
   if(!CNeuronBaseOCL::Clear())
      ReturnFalse;
   if(!cProjection.Clear())
      ReturnFalse;
   if(!cProjection.getPrevOutput().Fill(0))
      ReturnFalse;
   if(!cConcatenateInputs.Clear())
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Initializes the CMamba4CastEmbeding layer and allocates its GPU  buffers.  |
//+----------------------------------------------------------------------------+
bool CMamba4CastEmbeding::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                               uint window, uint window_out, uint units_count, uint & periods[],
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(periods.Size() <= 0)
      ReturnFalse;
   int freqs = (int(window_out / 2 + 2 * periods.Size()) - 1) / int(2 * periods.Size());
   if(freqs <= 0)
      ReturnFalse;
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * units_count, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cProjection.Init(0, index, OpenCL, window, window, window_out - 2 * freqs * periods.Size(), units_count, 1, optimization, iBatch))
      ReturnFalse;
   cProjection.SetActivationFunction(TANH);
   index++;
   if(!cNorm.Init(0, index, OpenCL, cProjection.Neurons(), iBatch, optimization))
      ReturnFalse;
   cNorm.SetActivationFunction(None);
   index++;
   if(!cProjectionWithTE.Init(0, index, OpenCL, window, units_count, periods, freqs, optimization, iBatch))
      ReturnFalse;
   SetActivationFunction(None);
//---
   return true;
  }
//+-------------------------------------------------------------------------------+
//| Runs the CMamba4CastEmbeding feed-forward pass and writes the  layer output.  |
//+-------------------------------------------------------------------------------+
bool CMamba4CastEmbeding::feedForward(CNeuronBaseOCL* NeuronOCL, CBufferFloat* SecondInput)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNorm.FeedForward(cProjection.AsObject()))
      ReturnFalse;
   if(!cProjectionWithTE.FeedForward(NeuronOCL, SecondInput))
      ReturnFalse;
   if(!Concat(cNorm.getOutput(), cProjectionWithTE.getOutput(), Output, cProjection.GetFilters(),
              cProjectionWithTE.GetWindowOut(), cProjection.GetUnits()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------------------+
//| Back-propagates the input gradients through the  CMamba4CastEmbeding layer.  |
//+------------------------------------------------------------------------------+
bool CMamba4CastEmbeding::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!DeConcat(cNorm.getGradient(), cProjectionWithTE.getGradient(), Gradient, cProjection.GetFilters(),
                cProjectionWithTE.GetWindowOut(), cProjection.GetUnits()))
      ReturnFalse;
   if(!cProjection.CalcHiddenGradients(cNorm.AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cProjection.AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(cProjectionWithTE.AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, cProjection.GetWindow(), false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false)
     )
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------------+
//| Updates the CMamba4CastEmbeding trainable weights from the  accumulated gradients.  |
//+-------------------------------------------------------------------------------------+
bool CMamba4CastEmbeding::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cProjection.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cNorm.UpdateInputWeights(cProjection.AsObject()))
      ReturnFalse;
   if(!cProjectionWithTE.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+-------------------------------------------------------------------------------------+
//| Averages the CMamba4CastEmbeding weights toward the source model  with factor tau.  |
//+-------------------------------------------------------------------------------------+
bool CMamba4CastEmbeding::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CMamba4CastEmbeding* Source = source;
   if(!cProjection.WeightsUpdate(Source.cProjection.AsObject(), tau))
      ReturnFalse;
   if(!cNorm.WeightsUpdate(Source.cNorm.AsObject(), tau))
      ReturnFalse;
   if(!cProjectionWithTE.WeightsUpdate(Source.cProjectionWithTE.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Saves the CMamba4CastEmbeding state to the file handle.          |
//+------------------------------------------------------------------+
bool CMamba4CastEmbeding::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(!cNorm.Save(file_handle))
      ReturnFalse;
   if(!cProjectionWithTE.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Restores the CMamba4CastEmbeding state from the file handle.     |
//+------------------------------------------------------------------+
bool CMamba4CastEmbeding::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNorm.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjectionWithTE.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+----------------------------------------------------------------------------+
//| Rebinds the CMamba4CastEmbeding GPU buffers to the given OpenCL  context.  |
//+----------------------------------------------------------------------------+
void CMamba4CastEmbeding::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cProjection.SetOpenCL(OpenCL);
   cNorm.SetOpenCL(OpenCL);
   cProjectionWithTE.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+