//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#ifndef NEURONET_DECLARATIONS_MQH
#ifndef NEURONET_DEFINITIONS_MQH
#define NEURONET_BUILDING_FACADE
#include "NeuroNet_Definitions.mqh"
#undef NEURONET_BUILDING_FACADE
#endif // NEURONET_DEFINITIONS_MQH
#define NEURONET_DECLARATIONS_MQH
#ifndef NEURONET_DEFINITIONS_MQH
#include "NeuroNet_Definitions.mqh"
#endif // NEURONET_DEFINITIONS_MQH
//+------------------------------------------------------------------+
//| NeuroNet class declarations in original dependency order
//+------------------------------------------------------------------+
class CConnection : public CObject
  {
public:
   float             weight;        ///< Current weight
   float             deltaWeight;   ///< last delta of weight used in SGD optimization
   float             mt;            ///< First moment in Adam optimization
   float             vt;            ///< Second moment in Adam optimization

   /** Constructor @param[in] w initial weight */ CConnection(float w) { weight = w; deltaWeight = 0; mt = 0; vt = 0; }
   /** Destructor */ ~CConnection() {};
   //--- methods for working with files
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual int       Type(void)   const   {  return defConnect;   }  ///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CArrayCon
///\brief Array of connections to anothe neuron
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para51">the link.</A>
//+------------------------------------------------------------------+
class CArrayCon  :    public CArrayObj
  {
public:
   /** Constructor */
                     CArrayCon(void) {};
   /** Destructor */ ~CArrayCon(void) {};
   //---
   virtual bool      CreateElement(int const index); ///< Method for cearing new element by index @param[in] index Index of new element @return logical result of operation
   virtual void      IncreaseTotal()   {  m_data_total++; } ///< Increase number of elements in array
   virtual int       Type(void)  const { return defArrayConnects; }  ///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CLayer
/// Class of neurons collection in one layer of Neural Net.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para53">the link.</A>
//+------------------------------------------------------------------+
class CLayer;
class CNeuronBaseOCL;
//+------------------------------------------------------------------+
///\class CNeuronBase
/// The base class of neuron.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para52">the link.</A>
//+------------------------------------------------------------------+
class CNeuronBase    :  public CObject
  {
protected:
   float             outputVal;  ///< Output value
   float             prevVal;    ///< Previous output value
   uint              m_myIndex;  ///< Index of neuron in layer
   float             gradient;   ///< Current gradient of neuron
   CArrayCon         *Connections;  ///< Array of connections with neurons in next layer
   ENUM_ACTIVATION   activation;    ///< Activation type (#ENUM_ACTIVATION)
   ENUM_OPTIMIZATION optimization;  ///< Optimization method (#ENUM_OPTIMIZATION)
   int               t;             ///< Count of iterations
   //---
   virtual bool      feedForward(CLayer *prevLayer)               {  ReturnFalse;    }   ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      calcHiddenGradients(CLayer *&nextLayer)     {  ReturnFalse;     }   ///< Method to transfer gradient to previous layer. @param nextLayer Pointer to next layer.
   virtual bool      updateInputWeights(CLayer *&prevLayer)       {  ReturnFalse;    }   ///< Method for updating weights.@param prevLayer Pointer to previos layer.
   virtual float     activationFunction(float x);                                        ///< Method to calculate activation function.@param x Input data. @return Result of activation function.
   virtual float     SigmoidFunction(float x)                    {  return (float)MathPow(1 + exp(-x), -1); } ///< Calculating Sigmoid \f$\frac{1}{1+e^x}\f$.@param x Input data.@return Result of calculation
   virtual float     TanhFunction(float x)                       {  return (float)tanh(x);   }              ///< Calculating \f$tanh(x)\f$.@param x Input data.@return Result of calculation
   virtual CLayer    *getOutputLayer(void)                        {  return NULL;      }  ///< Method for getting a pointer to the resulting neural layer. Not used in fully connected neural networks.@return Pointer to layer.
public:
   /** Constructor */
                     CNeuronBase(void);
   /** Destructor */~CNeuronBase(void);
   virtual bool      Init(uint numOutputs, uint myIndex, ENUM_OPTIMIZATION optimization_type);  ///< Method of initialization class.@param numOutputs Number of connections to next layer.@param myIndex Index of neuron in layer.@param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) {  activation = value; }      ///< Set the type of activation function (#ENUM_ACTIVATION)
   //---
   //static float            lr;
   static float            alpha;        ///< Multiplier to momentum in SGD optimization
   //---
   virtual void      setOutputVal(float val)                     {  prevVal = outputVal;   outputVal = val;    } ///< Set the output value
   virtual float     getOutputVal()                               {  return outputVal; }                      ///< Return result of feed forward operations.@return Output value
   virtual float     getPrevVal()                                 {  return prevVal;   }                      ///< Return result of feed forward operations at previous iteration.@return Previous output value
   virtual void      setGradient(float val)                      {  gradient = val;     }                    ///< Set gradient value to neuron.
   virtual float     getGradient()                                {  return gradient;  }                      ///< Return gradient of neuron.@return Gradient
   virtual CArrayCon *getConnections()                            {  return Connections;}                     ///< Method to get access to array of connections.@return Pointer to connections array
   virtual float     activationFunctionDerivative(float x);   ///< Calculate derivative of activation function.@param[in] x Input data@return Derivative
   virtual float     SigmoidFunctionDerivative(float x)          {  return x * (1 - x);   } ///< Calculate derivative of Sigmoid function.@param x Input data@return Derivative
   virtual float     TanhFunctionDerivative(float x)             {  return (1 + x) * (1 - x);  } ///< Calculate derivative of \f$tanh(x)\f$.@param x Input data@return Derivative
   //---
   virtual bool      feedForward(CObject *&SourceObject);            ///< Dispatch method for defining the subroutine for Feed Forward process.@param SourceObject Pointer to previos layer.
   virtual bool      calcHiddenGradients(CObject *&TargetObject);    ///< Dispatch method for defining the subroutine for transfer gradient to previous layer.@param TargetObject Pointer to next layer.
   virtual bool      updateInputWeights(CObject *&SourceObject);     ///< Dispatch method for defining the subroutine for updating weights.@param SourceObject Pointer to previos layer.
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle)///< Load method @param[in] file_handle handle of file @return logical result of operation
     {
      activation = (ENUM_ACTIVATION)FileReadInteger(file_handle, INT_VALUE);
      optimization = (ENUM_OPTIMIZATION)FileReadInteger(file_handle, INT_VALUE);
      t = (ENUM_OPTIMIZATION)FileReadInteger(file_handle, INT_VALUE);
      return(Connections.Load(file_handle));
     }
   //---
   virtual int       Type(void)        const                      {  return defNeuronBase;                  }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CNeuron
/// Class of neuron for full connected layers.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para52">the link.</A>
//+------------------------------------------------------------------+
class CNeuron  :  public CNeuronBase
  {
private:
   virtual bool      feedForward(CLayer *prevLayer);                          ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      calcHiddenGradients(CLayer *&nextLayer);                 ///< Method to transfer gradient to previous layer. @param nextLayer Pointer to next layer.
   virtual bool      updateInputWeights(CLayer *&prevLayer);                  ///< Method for updating weights.@param prevLayer Pointer to previos layer.

public:
   /** Constructor */
                     CNeuron(void)  {};
   /** Destructor */~CNeuron(void) { Connections.Shutdown(); }
   //---
   virtual bool      calcOutputGradients(float targetVals);                  ///< Method of output gradients calculation.@param targetVals Traget value
   virtual float     sumDOW(CLayer *&nextLayer) ;                             ///< A method for collecting gradients from the next layer.@param[in] nextLayer Pointer to next layer@return Total gradient to neuron.
   virtual int       Type(void)   const   {  return defNeuron;   } ///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class COpenCLMy
/// Class for working with OpenCL
//+------------------------------------------------------------------+
class COpenCLMy   :  public COpenCL
  {
protected:
   uaLongChar              uLocalDimension;

public:
   /** Constructor */
                     COpenCLMy(void)   {};
   /** Destructor */~COpenCLMy(void)   { for(int i = 0; i < m_kernels_total; i++) KernelFree(i); }
   bool              Initialize(const string program, const bool show_log = true);
   template<typename T>
   int               AddBufferFromArray(T &data[], const uint data_array_offset, const uint data_array_count, const uint flags);
   ///< Method for creating OpenCL buffer from array.@param data[] Array of data.@param data_array_offset Offset of data in array.@param data_array_count Number of data items in array.@param flags Buffer's properties (CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_ALLOC_HOST_PTR)
   int               AddBuffer(uint size_in_bytes, const uint flags);
   //---
   long              GetMaxLocalSize(const int dimension) const { return uLocalDimension.data[dimension]; }
   ulong             GetBufferSize(const int buffer_index) const;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CLayer: public CArrayObj
  {
private:
   uint              iOutputs;                                 ///< Number of output connections from 1 neuron to neurons in next layer.
   int               iFileHandle;                              ///< File handle for download result of previous study.
   COpenCLMy         *OpenCL;                                   ///< Class for working with OpenCL

public:
   /** Constructor */
                     CLayer(uint outputs = 0, int handle = INVALID_HANDLE, COpenCLMy *OpenCL = NULL);
   ///< @param[in] outputs Number of output connections from 1 neuron to neurons in next layer @param[in] handle File handle for download result of previous study @param[in] OpenCL Pointer to class for working with OpenCL
   /** Destructor */~CLayer(void) {};
   //---
   virtual bool      CreateElement(int const index);           ///< Method for creating new element in layer
   virtual void      IncreaseTotal()   {  m_data_total++;   }  ///< Method for increase number of items in layer
   virtual int       Type(void)  const { return defLayer; }    ///< Identificator of class.@return Type of class
   virtual bool      Load(const int file_handle);              ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual uint      Outputs(void)  { return iOutputs;   }
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);
   virtual bool      WeightsUpdate(CLayer *source, float tau);
   virtual bool      ClearStates(void);
   CObject           *At(const int index) const
     {
      if(index >= m_data_total || (-index) > m_data_total)
         return NULL;
      if(index < 0)
         return(m_data[m_data_total + index]);
      return(m_data[index]);
     }
   virtual CNeuronBaseOCL* operator[](const int index) const { return (CNeuronBaseOCL*)At(index); }
   virtual CObject*        AsObject(void) { return GetPointer(this); }
  };
//+------------------------------------------------------------------+
///\class CArrayLayer
/// Class of layers collection in Neural Net.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para53">the link.</A>
//+------------------------------------------------------------------+
class CArrayLayer  :    public CArrayObj
  {
public:
   /** Constructor */
                     CArrayLayer(void) {};
   /** Destructor */~CArrayLayer(void) {};
   //---
   virtual bool      CreateElement(uint neurons,  uint outputs);  ///< Method for creating new element
   virtual int       Type(void)  const { return defArrayLayer; }  ///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CNeuronProof
/// Class of pooling layer
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8234#para42">the link.</A>
//+------------------------------------------------------------------+
class CNeuronProof : public CNeuronBase
  {
protected:
   CLayer            *OutputLayer;                                      ///< Layer of output data. Used for connection with next layer.
   int               iWindow;                                           ///< Input window size
   int               iStep;                                             ///< Size of step

   virtual bool      feedForward(CLayer *prevLayer);                    ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      calcHiddenGradients(CLayer *&nextLayer);           ///< Method to transfer gradient to previous layer. @param nextLayer Pointer to next layer.

public:
   /** Constructor */
                     CNeuronProof(void) {};
   /** Destructor */~CNeuronProof(void);
   virtual bool      Init(uint numOutputs, uint myIndex, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type);
   ///< Method of initialization class.@param numOutputs Number of connections to next layer.@param myIndex Index of neuron in layer.@param window Size of input window @param step Step size.@param units_countNumber of neurons.@param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual CLayer    *getOutputLayer(void)  { return OutputLayer;  }    ///< Method for getting a pointer to the resulting neural layer. Not used in fully connected neural networks.@return Pointer to layer.
   virtual bool      calcInputGradients(CLayer *prevLayer);             ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   virtual bool      calcInputGradients(CNeuronBase *prevNeuron, uint index); ///< Method to transfer gradients to neuron in previous layer @param[in] prevNeuron Pointer to neuron.@param[in] index Index of neuron in previous layer
   //--- methods for working with files
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual int       Type(void)   const   {  return defNeuronProof;   }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CNeuronConv
/// Class of convolution layer
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8234#para43">the link.</A>
//+------------------------------------------------------------------+
class CNeuronConv  :  public CNeuronProof
  {
protected:
   float             param;   //PReLU param
   virtual bool      feedForward(CLayer *prevLayer);                    ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      calcHiddenGradients(CLayer *&nextLayer);           ///< Method to transfer gradient to previous layer. @param nextLayer Pointer to next layer.
   virtual float     activationFunction(float x);                      ///< Method to calculate activation function.@param x Input data. @return Result of activation function.
   virtual bool      updateInputWeights(CLayer *&prevLayer);            ///< Method for updating weights.@param prevLayer Pointer to previos layer.
public:
   /** Constructor */
                     CNeuronConv() :   param((float)0.01) { };
   /** Destructor */~CNeuronConv(void)             { };
   //---
   virtual bool      calcInputGradients(CLayer *prevLayer);                  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   virtual bool      calcInputGradients(CNeuronBase *prevNeuron, uint index);///< Method to transfer gradients to neuron in previous layer @param[in] prevNeuron Pointer to neuron.@param[in] index Index of neuron in previous layer
   virtual float     activationFunctionDerivative(float x);                 ///< Calculate derivative of activation function.@param[in] x Input data@return Derivative
   virtual int       Type(void)   const   {  return defNeuronConv;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
  };
//+------------------------------------------------------------------+
///\class CLayerDescription
/// Class of layer decription. Used to describe the structure of a neural network from the main program.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8234#para44">the link.</A>
//+------------------------------------------------------------------+
class CLayerDescription    :  public CObject
  {
public:
   /** Constructor */
                     CLayerDescription(void);
   /** Destructor */~CLayerDescription(void) {};
   //---
   uint              type;          ///< Type of neurons in layer (\ref ObjectTypes)
   uint              count;         ///< Number of neurons
   uint              window;        ///< Size of input window
   uint              window_out;    ///< Size of output window
   uint              step;          ///< Step size
   uint              layers;        ///< Layers count
   uint              batch;         ///< Batch Size
   ENUM_ACTIVATION   activation;    ///< Type of activation function (#ENUM_ACTIVATION)
   ENUM_OPTIMIZATION optimization;  ///< Type of optimization method (#ENUM_OPTIMIZATION)
   float             probability;   ///< Probability of neurons shutdown, only Dropout used
   uint              windows[];
   uint              units[];
   uint              heads[];
   vector<float>     radius;
   uint              variables;
   //---
   virtual bool      Copy(CLayerDescription *source);
   //---
   virtual bool      operator= (CLayerDescription *source)  { return Copy(source); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CBufferFloat;
//+------------------------------------------------------------------+
///\class CNet
/// The main class of the neural network. Contains basic methods for the functioning of a neural network.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para53">the link.</A>
//+------------------------------------------------------------------+
class CNet : public CObject
  {
protected:
   bool                    backPropOCL(CBufferFloat *targetVals, CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL, bool layer_by_layer = false, ENUM_ACTIVATION SecondActivation = None);       ///< Back propagation method for GPU calculation. @param[in] targetVals Target values
   int                     backPropCount;
   float                   recentAverageError;                 ///< Average error
   CArrayLayer            *layers;                             ///< Array of layers
   COpenCLMy              *opencl;                             ///< Class for working with OpenCL
   virtual bool            OpenCLInit(void);

public:
   /** Constructor */
                     CNet(void)  { Create(NULL); }
                     CNet(CArrayObj *Description)  { Create(Description); }
   bool                    Create(CArrayObj *Description);
   /** Destructor */      ~CNet(void);
   bool                    feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true, CBufferFloat *SecondInput = NULL); ///< Feed Forward method.@param[in] prevLayer Pointer to previos layer. @param[in] window Window of input data. @param[in] tem Use Time Embedding.
   bool                    feedForward(CNet *inputNet, int inputLayer = -1, CBufferFloat *SecondInput = NULL);
   bool                    feedForward(CNet *inputNet, int inputLayer = -1, CNet *secondNet = NULL, int secondLayer = -1);
   bool                    feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true, CNet *secondNet = NULL, int secondLayer = -1);
   bool                    backProp(CArrayFloat *targetVals, CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL);               ///< Back propagation method. @param[in] targetVals Target values
   bool                    backProp(CArrayFloat *targetVals, CNet *secondNet = NULL, int secondLayer = -1);               ///< Back propagation method. @param[in] targetVals Target values
   bool                    backPropGradient(CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL, int LastLayer = -1, bool load_weights = true);     ///< Back propagation method for GPU calculation. @param[in] targetVals Target values
   bool                    backPropGradient(CNet *secondNet, int secondLayer = -1, int LastLayer = -1, bool load_weights = true);
   void                    getResults(CBufferFloat *&resultVals);                ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   void                    getResults(vector<float> &resultVals);                ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   void                    getResults(float &resultVals[]);                      ///< Method to get results of feed forward process.@param[out] resultVals Array of result values
   float                   getRecentAverageError()  { return recentAverageError; } ///< Method to check quality of study. @return Average error
   bool                    Save(string file_name, float error, float undefine, float forecast, datetime time, bool common = true);
   ///< Save method. @param[in] file_name File name to save @param[in] error Average error @param[in] undefine Undefined percent @param[in] Foecast percent @param[in] time Last study time @param[in] common Common flag
   virtual bool            Save(const int file_handle);
   bool                    Load(string file_name, float &error, float &undefine, float &forecast, datetime &time, bool common = true);
   ///< Load method. @param[in] file_name File name to save @param[out] error Average error @param[out] undefine Undefined percent @param[out] Foecast percent @param[out] time Last study time @param[in] common Common flag
   virtual bool            Load(const int file_handle);
   //---
   static float            recentAverageSmoothingFactor;             ///< Smoothing factor of average error
   virtual int             Type(void)   const   {  return defNet;   }///< Identificator of class.@return Type of class
   virtual bool            TrainMode(bool flag);                     ///< Set Training Mode Flag
   virtual bool            GetLayerOutput(uint layer, CBufferFloat *&result); ///< Retutn Output data of layer. @param[in] layer Number of layer @param[out] return Buffer with data
   virtual bool            GetLayerOutput(uint layer, vector<float> &result); ///< Retutn Output data of layer. @param[in] layer Number of layer @param[out] return Buffer with data
   //---
   virtual void            SetOpenCL(COpenCLMy *obj);
   virtual COpenCLMy*      GetOpenCL(void)   {  return opencl; }
   //---
   virtual bool            WeightsUpdate(CNet *net, float tau);
   //--- Soft Actor-Critic
   virtual bool            GetLogProbs(vector<float> &log_probs);
   virtual bool            AlphasGradient(CNet *PolicyNet);
   virtual bool            CalcLogProbs(CBufferFloat *buffer);
   virtual bool            SetResult(float &result[]);
   virtual bool            Clear(void);
  };
//+------------------------------------------------------------------+
///\class CNeuronLSTM
/// Class of recurrent LSTM unit
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8385#para5">the link.</A>
//+------------------------------------------------------------------+
class CNeuronLSTM    :  public CNeuronProof
  {
protected:
   CLayer            *ForgetGate;                                    ///< Object of forget gate
   CLayer            *InputGate;                                     ///< Object of input gate
   CLayer            *OutputGate;                                    ///< Object of output gate
   CLayer            *NewContent;                                    ///< Object of new content
   CArrayFloat       *Memory;                                        ///< Memory array
   CArrayFloat       *PrevMemory;                                    ///< Ravious iteration memory array
   CArrayFloat       *Input;                                         ///< Input data
   CArrayFloat       *InputGradient;                                 ///< Gradient on previous layer
   //---
   virtual bool      feedForward(CLayer *prevLayer);                    ///< Feed Forward method. Detailed description on <A HREF="https://www.mql5.com/ru/articles/8385#para52">the link.</A>@param prevLayer Pointer to previos layer.
   virtual bool      calcHiddenGradients(CLayer *&nextLayer);           ///< Method to transfer gradient to previous layer. @param nextLayer Pointer to next layer.
   virtual bool      updateInputWeights(CLayer *&prevLayer);            ///< Method for updating weights.@param prevLayer Pointer to previos layer.
   virtual bool      updateInputWeights(CLayer *gate, CArrayFloat *input_data); ///< Method for updating gates' weights.@param gate Pointer to gate. @param input_data Pointer to tensor with input data.
   virtual bool      InitLayer(CLayer *layer, int numUnits, int numOutputs, ENUM_OPTIMIZATION optimization_type);
   ///< Method of gate initialization @param[in] layer Pointer to gate @param[in] numUnits Number of units in gate @param[in] numOutputs Number of outputs @param[in] optimization_type Type of optimization (#ENUM_OPTIMIZATION)
   virtual CArrayFloat *CalculateGate(CLayer *gate, CArrayFloat *sequence); ///< Method of calculation gate iteration @param[in] gate Pointer to gate @param[in] sequence Input data @return Array of output data

public:
   /** Constructor */
                     CNeuronLSTM(void);
   /** Destructor */~CNeuronLSTM(void);
   virtual bool      Init(uint numOutputs, uint myIndex, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type);
   ///< Unit initialization method. Detailed description on <A HREF="https://www.mql5.com/ru/articles/8385#para51">the link.</A>
   //---
   virtual CLayer    *getOutputLayer(void)  { return OutputLayer;  }    ///< Method for getting a pointer to the resulting neural layer. Not used in fully connected neural networks.@return Pointer to layer.
   virtual bool      calcInputGradients(CLayer *prevLayer) ;
   virtual bool      calcInputGradients(CNeuronBase *prevNeuron, uint index) ;
   //--- methods for working with files
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual int       Type(void)   const   {  return defNeuronLSTM;   }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
///\class CBufferFloat
/// Class of OpenCL buffer data. Used for transfer data from CPU to GPU and back.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para44">the link.</A>
//+------------------------------------------------------------------+
class CBufferFloat     :  public CArrayFloat
  {
protected:
   COpenCLMy         *OpenCL;                ///< Object for working with OpenCL
   int               m_myIndex;              ///< Index of buffer
public:
   /** Constructor */
                     CBufferFloat(void);
   /** Destructor */~CBufferFloat(void);
   //---
   virtual bool      BufferInit(uint count, float value);  ///< Method for buffer initialization @param[in] count Number of items @param[in] value Initialization value
   virtual bool      BufferInitLike(CBufferFloat* master);  ///< Method for buffer initialization @param[in] count Number of items @param[in] value Initialization value
   virtual bool      BufferCreate(COpenCL *opencl);       ///< Method for creating new buffer @param[in] opencl Pointer to #COpenCLMy object
   virtual bool      BufferFree(void);                      ///< Method for deleting buffer from GPU
   virtual bool      BufferRead(void);                      ///< Method for reading buffer data from GPU
   virtual bool      BufferWrite(void);                     ///< Method for writing buffer data to GPU
   virtual bool      BufferSet(int index) {  if(!OpenCL.BufferFree(m_myIndex)) ReturnFalse; m_myIndex = index; return true; } ///< Change buffer index number @param index New index
   template<typename T>
   int               GetData(vector<T> &values);             ///< Read data from buffer to vector @param[out] values Vector to read data
   template<typename T>
   int               GetData(T &values[]);             ///< Read data from buffer to array @param[out] values Array to read data
   template<typename T>
   int               GetData(matrix<T> &values);             ///< Read data from buffer to vector @param[out] values Vector to read data
   virtual int       GetData(CArrayFloat *values);         ///< Read data from buffer to array @param[out] values Array to read data
   virtual int       GetIndex(void) const {  return m_myIndex; }  ///< Get buffer index @return Index
   virtual float     Maximum(void) {  return m_data[Argmax()];}
   virtual float     Minimum(void) {  return m_data[Argmin()];}
   virtual int       Argmax(void) {  return ArrayMaximum(m_data, 0, m_data_total);}
   virtual int       Argmin(void) {  return ArrayMinimum(m_data, 0, m_data_total);}
   //---
   template<typename T>
   bool              AssignArray(const T &src[]);
   virtual bool      AssignArray(const matrix<float> &src);
   virtual bool      AssignArray(const vector<float> &src);
   virtual bool      AssignArray(const CArrayFloat *obj);
   virtual bool      AssignArray(const CBufferFloat *obj)  { return AssignArray((CArrayFloat*)obj); }
   virtual bool      AddArray(const CBufferFloat *src)  { return CArrayFloat::AddArray((CArrayFloat*)src); }
   virtual bool      AddArray(const vector<float> &src);
   template<typename T>
   bool              AddArray(const T &src[]);
   //---
   virtual bool      Fill(const float value);
   virtual bool      Fill(const CBufferFloat *obj);
   template<typename T>
   bool              Fill(const matrix<T> &obj);
   template<typename T>
   bool              Fill(const vector<T> &obj);
   virtual bool      Random(const float min_value, const float  max_value);
   //---
   virtual int       Type(void)                                    const { return defBufferDouble;      }///< Identificator of class.@return Type of class
   virtual void      BufferToCSV(const string file_name);   ///< Save buffer data to CSV file @param[in] file_name File name to write data
   //--- methods for working with files
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   //---
   virtual COpenCL   *GetOpenCL(void) const { return OpenCL; }
   bool              operator=(const CBufferFloat *obj) {return AssignArray(obj);}
   float             operator[](const int index)  const
     {
      if(index < 0)
         return(At(m_data_total + index));
      return(At(index));
     }
  };
//+------------------------------------------------------------------+
///\ingroup neuron_base
///\class CNeuronBaseOCL
///\brief The base class of neuron for GPU calculation.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para45">the link.</A>
//+------------------------------------------------------------------+
class CNeuronBaseOCL    :  public CObject
  {
protected:
   bool               bTrain;             ///< Training Mode Flag
   COpenCLMy         *OpenCL;             ///< Object for working with OpenCL
   CBufferFloat      *Output;             ///< Buffer of Output tensor
   CBufferFloat      *PrevOutput;         ///< Buffer of previous iteration Output tensor
   CBufferFloat      *Weights;            ///< Buffer of weights matrix
   CBufferFloat      *DeltaWeights;       ///< Buffer of last delta weights matrix (#SGD)
   CBufferFloat      *Gradient;           ///< Buffer of gradient tensor
   CBufferFloat      *FirstMomentum;      ///< Buffer of first momentum matrix (#ADAM)
   CBufferFloat      *SecondMomentum;     ///< Buffer of second momentum matrix (#ADAM)
   vector<float>      prev_output;
   //---
   //const float      lr;
   float             alpha;               ///< Multiplier to momentum in #SGD optimization
   uint              iBatch;              ///< Batch size used in LS optimization
   uint              t;                   ///< Count of iterations
   //---
   int               m_myIndex;           ///< Index of neuron in layer
   ENUM_ACTIVATION   activation;          ///< Activation type (#ENUM_ACTIVATION)
   ENUM_OPTIMIZATION optimization;        ///< Optimization method (#ENUM_OPTIMIZATION)
   //---
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) { return feedForward(NeuronOCL); }               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)  {  return updateInputWeights(NeuronOCL); }        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);        ///< Method to transfer gradients to previous layer @param[in] NeuronOCL Pointer to previous layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) { return calcInputGradients(NeuronOCL); }
   virtual bool      calcHiddenGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) { ReturnFalse; }
   virtual bool      calcHiddenGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.
   virtual float     GenerateWeight(void);
   virtual bool      WeightsUpdateAdam(CNeuronBaseOCL *source, float tau);
   virtual bool      SumAndNormalize(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, bool normilize = true, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 0.5f);
   virtual bool      Normalize(CBufferFloat *tensor, int dimension);
   virtual bool      Different(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 1.0f);
   virtual bool      DifferentGrad(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 1.0f);
   //---
   virtual bool      IdentSum(CBufferFloat *tensor, CBufferFloat *out, int dimension, int shift_in = 0, int shift_out = 0, float multiplyer = 1.0f);
   virtual bool      IdentDifferent(CBufferFloat *tensor, CBufferFloat *out, int dimension, int shift_in = 0, int shift_out = 0, float multiplyer = 1.0f);
   virtual bool      IdentDifferentGrad(CBufferFloat *tensor, CBufferFloat *out, int dimension, int shift_in = 0, int shift_out = 0, float multiplyer = 1.0f);
   //---
   virtual bool      SumVecMatrix(CBufferFloat *vector_in, CBufferFloat *matrix_in, CBufferFloat *out, uint dimension, uint variables, uint shift_in1 = 0, uint shift_in2 = 0, uint shift_out = 0, float multiplyer = 1.0f);
   virtual bool      SumVecMatrixGrad(CBufferFloat *vector_in, CBufferFloat *matrix_in, CBufferFloat *out, uint dimension, uint variables, uint shift_in1 = 0, uint shift_in2 = 0, uint shift_out = 0, float multiplyer = 1.0f);
   virtual bool      MatMul(const CBufferFloat *matr1, const CBufferFloat *matr2, CBufferFloat *result, const int rows1, const int cols1, const int cols2, const int variables = 1, const bool mult_var_second = true);
   virtual bool      MatMulGrad(const CBufferFloat *matr1, CBufferFloat *matr1_gr, const CBufferFloat *matr2, CBufferFloat *matr2_gr, const CBufferFloat *result_gr, const int rows1, const int cols1, const int cols2, const int variables = 1, const bool mult_var_second = true);
   virtual bool      DiagMatMul(const CBufferFloat *diag, const CBufferFloat *matr, CBufferFloat *result, const int rows, const int cols, const int variables = 1, const int activation = None);
   virtual bool      DiagMatMulGrad(const CBufferFloat *diag, CBufferFloat *diag_gr, const CBufferFloat *matr, CBufferFloat *matr_gr, const CBufferFloat *result_gr, const int rows, const int cols, const int variables = 1);
   virtual bool      SparseMatMul(const CBufferFloat *sparse_indexs, const CBufferFloat *sparse_data, const CBufferFloat *full, CBufferFloat *result, const int sparse_rows, const int sparse_cols, const int full_rows, const int full_cols);
   virtual bool      SparseMatMulGrad(const CBufferFloat *sparse_indexs, const CBufferFloat *sparse_data, CBufferFloat *sparse_grad, const CBufferFloat *full, CBufferFloat *full_gr, const CBufferFloat *grad, const int sparse_rows, const int sparse_cols, const int full_rows, const int full_cols);
   virtual bool      SparseConcatenate(const CBufferFloat *sparse_indexs, const CBufferFloat *sparse_data, const CBufferFloat *full, CBufferFloat *result, const int sparse_rows, const int sparse_cols, const int full_rows, const int full_cols);
   virtual bool      SparseConcatenateGrad(const CBufferFloat *sparse_indexs, const CBufferFloat *sparse_data, CBufferFloat *sparse_grad, const CBufferFloat *full, CBufferFloat *full_gr, const CBufferFloat *grad, const int sparse_rows, const int sparse_cols, const int full_rows, const int full_cols);
   virtual bool      ScalarToVector(CBufferFloat *scalar, CBufferFloat *vector_in, CBufferFloat *vector_out, int dimension);
   virtual bool      ScalarToVectorGrad(CBufferFloat *scalar, CBufferFloat *scalar_gr, CBufferFloat *vector_in, CBufferFloat *vector_in_gr, CBufferFloat *vector_out_gr, int dimension);
   ///< \brief Method sum and normilize 2 tensors by calling 2 kernels ::SumMatrix() and ::Normalize().
   virtual bool      LoadInsideLayer(int file_handle, CNeuronBaseOCL *neuron);
   virtual bool      Activation(CBufferFloat *inputs, CBufferFloat *outputs, int activat);
   virtual bool      DeActivation(CBufferFloat *inputs, CBufferFloat *inputs_gr, CBufferFloat *output_gr, int activat);
   virtual bool      NLLGrad(const CBufferFloat *log_b, CBufferFloat *grad_in, const CBufferFloat *grad_out, CBufferFloat *grad_log_b);
   virtual bool      SparseSoftmaxHausdorff(const CBufferFloat *plan, const CBufferFloat *decis, CBufferFloat *output, CBufferFloat *top_index, const int dimension, const int top_k);
   virtual bool      SparseSoftmaxHausdorffGrad(const CBufferFloat *plan, CBufferFloat *grad_plan, const CBufferFloat *decis, CBufferFloat *grad_decis, const CBufferFloat *grad_out, const CBufferFloat *top_index, const int dimension, const int top_k);
   //---
   virtual bool      Concat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *output, int window1, int window2, int count);
   virtual bool      Concat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *input3, CBufferFloat *output, int window1, int window2, int window3, int count);
   virtual bool      Concat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *input3, CBufferFloat *input4, CBufferFloat *output, int window1, int window2, int window3, int window4, int count);
   virtual bool      Concat(int input1, int input2, int input3, int input4, int output, int window1, int window2, int window3, int window4, int count);
   virtual bool      DeConcat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *output, int window1, int window2, int count);
   virtual bool      DeConcat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *input3, CBufferFloat *output, int window1, int window2, int window3, int count);
   virtual bool      DeConcat(CBufferFloat *input1, CBufferFloat *input2, CBufferFloat *input3, CBufferFloat *input4, CBufferFloat *output, int window1, int window2, int window3, int window4, int count);
   virtual bool      ConcatVecMatrix(CBufferFloat *vector_in, CBufferFloat *matrix_in, CBufferFloat *out, uint dimension_v, uint  rows, uint cols, uint variables, bool multvarsecond);
   virtual bool      ConcatVecMatrixGrad(CBufferFloat *vector_in, CBufferFloat *matrix_in, CBufferFloat *out, uint dimension_v, uint rows, uint  cols, uint variables, bool multvarsecond);
   //---
   virtual bool      ElementMult(const CBufferFloat *input1, const CBufferFloat *input2, CBufferFloat *output);
   virtual bool      ElementMultGrad(const CBufferFloat *input1, CBufferFloat *input1_gr, const CBufferFloat *input2, CBufferFloat *input2_gr, const CBufferFloat *output, const int activation1, const int activation2);
   virtual bool      GateElementMult(const CBufferFloat *input1, const CBufferFloat *input2, const CBufferFloat *gate, CBufferFloat *output);
   virtual bool      GateElementMultGrad(const CBufferFloat *input1, CBufferFloat *input1_gr, const CBufferFloat *input2, CBufferFloat *input2_gr, const CBufferFloat *gate, CBufferFloat *gate_gr, const CBufferFloat *output, const int activation1, const int activation2, const int activation_gate);
   int               RND(int total)   { xor128; return (int)((float)(total - 1) / UINT_MAX * rnd_w);  } ///< Generates a random neuron position to turn off
   //---
   virtual bool      DiversityLoss(CNeuronBaseOCL *neuron, const int units, const int dimension, const bool add = false);
   //---
   virtual bool      SwapBuffers(CBufferFloat *&buffer1, CBufferFloat *&buffer2, bool check_size = true);
   //---
   virtual bool      DilatedCorrelation(CNeuronBaseOCL* feature, CNeuronBaseOCL* correlations, uint shifts_index, uint dimension, uint units);
   virtual bool      DilatedCorrelationGrad(CNeuronBaseOCL* feature, CNeuronBaseOCL* correlations, uint shifts_index, uint dimension, uint units);
   virtual bool      DilatedDifference(CNeuronBaseOCL* feature, CNeuronBaseOCL* difference, uint shifts_index, uint dimension, uint units);
   virtual bool      DilatedDifferenceGrad(CNeuronBaseOCL* feature, CNeuronBaseOCL* difference, uint shifts_index, uint dimension, uint units);
   virtual bool      PerturbedMatrix(CNeuronBaseOCL* inputs, CNeuronBaseOCL* perturb, CNeuronBaseOCL* outputs, float perturb_mult, uint dimension, uint units);
   virtual bool      PerturbedMatrixGrad(CNeuronBaseOCL* inputs, CNeuronBaseOCL* perturb, CNeuronBaseOCL* outputs, float perturb_mult, uint dimension, uint units);
   virtual bool      LinearUpsample(CNeuronBaseOCL* data, CNeuronBaseOCL* upsample, uint upsampl_mult, uint units, uint variables);
   virtual bool      LinearUpsampleGrad(CNeuronBaseOCL* data, CNeuronBaseOCL* upsample, uint upsampl_mult, uint units, uint variables);
   //---
   virtual int       CreateShifts(const uint max_displacement, int &count);

public:
   /** Constructor */
                     CNeuronBaseOCL(void);
   /** Destructor */~CNeuronBaseOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) {  activation = value; }      ///< Set the type of activation function (#ENUM_ACTIVATION)
   //---
   virtual int       getOutputIndex(void)    const {  return Output.GetIndex();        }  ///< Get index of output buffer @return Index
   virtual int       getPrevOutIndex(void)   const {  return PrevOutput.GetIndex();    }  ///< Get index of previous iteration output buffer @return Index
   virtual int       getGradientIndex(void)  const {  return Gradient.GetIndex();      }  ///< Get index of gradient buffer @return Index
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true);
   virtual bool      SetOutput(CBufferFloat *buffer, bool delete_prev = true);
   virtual bool      SetGradientIndex(int index)   {  return Gradient.BufferSet(index);   }  ///< Method for change index of gradient buffer.@param[in] New index of buffer
   virtual int       getWeightsIndex(void)         const
     {
      if(!Weights)
         return INVALID_HANDLE;
      return Weights.GetIndex();
     }  ///< Get index of weights matrix buffer @return Index
   virtual int       getDeltaWeightsIndex(void)    const {  return DeltaWeights.GetIndex();  }  ///< Get index of delta weights matrix buffer (SGD)@return Index
   virtual int       getFirstMomentumIndex(void)   const {  return FirstMomentum.GetIndex(); }  ///< Get index of first momentum matrix buffer (Adam)@return Index
   virtual int       getSecondMomentumIndex(void)  const {  return SecondMomentum.GetIndex();}  ///< Get index of Second momentum matrix buffer (Adam)@return Index
   virtual int       getWeightsSAMIndex(void)      const {  return -1;                     }
   //---
   virtual int       getOutputVal(float &values[]) {  return Output.GetData(values);      }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(CArrayFloat *values)   {  return Output.GetData(values);  }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(vector<float> &values)   {  return Output.GetData(values);  }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getPrevVal(float &values[])   {  return PrevOutput.GetData(values);  }  ///< Get values of previous iteration output buffer @param[out] values Array of data @return number of items
   virtual int       getGradient(float &values[])  {  return Gradient.GetData(values);    }  ///< Get values of gradient buffer @param[out] values Array of data @return number of items
   virtual int       getGradient(vector<float> &values)  {  return Gradient.GetData(values);    }  ///< Get values of gradient buffer @param[out] values Array of data @return number of items
   virtual bool      calcAlphaGradients(CNeuronBaseOCL *NeuronOCL)    { return true; }
   virtual bool      GetAlphaLogProbs(vector<float> &log_probs)       { ReturnFalse; }
   virtual bool      ReCalcLogProbs(void)                             { ReturnFalse; }
   virtual CBufferFloat   *getOutput(void)      const    {  return Output;       }                 ///< Get pointer of output buffer @return Pointer to object
   virtual CBufferFloat   *getGradient(void)    const    {  return Gradient;     }                 ///< Get pointer of gradient buffer @return Pointer to object
   virtual CBufferFloat   *getPrevOutput(void)  const    {  return PrevOutput;   }                 ///< Get pointer of gradient buffer @return Pointer to object
   virtual int       getWeights(float &values[])   {  return Weights.GetData(values);     }  ///< Get values of weights matrix buffer @param[out] values Array of data @return number of items
   virtual CBufferFloat   *getWeights(void)        {  return Weights;     }                 ///< Get pointer of gradient buffer @return Pointer to object
   virtual int       Neurons(void) const           {  return Output.Total();              }  ///< Get number of neurons in layer @return Number of neurons
   virtual int       Activation(void) const        {  return (int)activation;             }  ///< Get type of activation function @return Type (#ENUM_ACTIVATION)
   virtual int       getConnections(void)          {  return (CheckPointer(Weights) != POINTER_INVALID ? Weights.Total() / (Output.Total() + 1) : 0);   } ///< Get number of connections 1 neuron to next layer @return Number of connections
   virtual ENUM_OPTIMIZATION Optimization(void)    {  return optimization; }
   //---
   virtual bool      FeedForward(CObject *SourceObject, CBufferFloat *SecondInput = NULL);                    ///< Dispatch method for defining the subroutine for feed forward process. @param SourceObject Pointer to the previous layer.
   virtual bool      UpdateInputWeights(CObject *SourceObject, CBufferFloat *SecondInput = NULL);             ///< Dispatch method for defining the subroutine for updating weights.@param SourceObject Pointer to previos layer.
   ///\ingroup neuron_base_gr
   ///@{
   virtual bool      CalcHiddenGradients(CObject *TargetObject, CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL, ENUM_ACTIVATION SecondActivation = None);        ///< Dispatch method for defining the subroutine for transferring the gradient to the previous layer. @param TargetObject Pointer to the next layer.
   virtual bool      calcOutputGradients(CBufferFloat *Target, float &error);               ///< Method of output gradients calculation by calling kernel ::CalcOutputGradient().@param Target Traget value
   virtual bool      calcAlignmentGradient(const CNeuronBaseOCL *neuron, const bool add = false);
   ///@}
   //---
   virtual bool      Save(const int file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(const int file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronBaseOCL;                  }///< Identificator of class.@return Type of class
   virtual void      TrainMode(bool flag)                         {  bTrain = flag;                              } ///< Set Training Mode Flag
   virtual bool      TrainMode(void)                              {  return bTrain;                            }///< Get Training Mode Flag
   virtual CLayerDescription* GetLayerInfo(void);
   virtual bool      numOutputs(const uint outputs, ENUM_OPTIMIZATION optimization_type);
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual CObject*  AsObject(void) {  return GetPointer(this);   }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   //---
   virtual bool      Clear(void) { return !(!Output || !Output.Fill(0) || !Gradient || !Gradient.Fill(0)); }
   //---
   virtual bool      SwapOutputs(void) { return SwapBuffers(Output, PrevOutput, true); }
  };
//+------------------------------------------------------------------+
///\class CNeuronProofOCL
/// Class of pooling layer GPU calculation
//+------------------------------------------------------------------+
class CNeuronProofOCL : public CNeuronBaseOCL
  {
protected:
   uint               iWindow;                                             ///< Input window size
   uint               iStep;                                               ///< Size of step

   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< Feed Forward method.@param NeuronOCL Pointer to previos layer.
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);        ///< Method to transfer gradients to previous layer @param[in] NeuronOCL Pointer to previous layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   { return true;};        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.

public:
   /** Constructor */
                     CNeuronProofOCL(void) : iWindow(2), iStep(1) {};
   /** Destructor */~CNeuronProofOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of input window @param[in] step Step size.@param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //--- methods for working with files
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual int       Type(void)   const   {  return defNeuronProofOCL;   }///< Identificator of class.@return Type of class
   virtual CLayerDescription* GetLayerInfo(void);
   virtual uint      GetWindow(void) const { return iWindow;   }
  };
//+------------------------------------------------------------------+
///\class CNeuronConvOCL
/// Class of convolution layer GPU calculation
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para41">the link.</A>
//+------------------------------------------------------------------+
class CNeuronConvOCL    :  public CNeuronProofOCL
  {
protected:
   uint              iWindowOut;                                           ///< Size of out window
   uint              iVariables;                                           ///< Size of out window
   //---
   CBufferFloat      *WeightsConv;                                         ///< Matrix of weights to previous layer
   CBufferFloat      *DeltaWeightsConv;                                    ///< Matrix of delta weights to previous layer (#SGD)
   CBufferFloat      *FirstMomentumConv;                                   ///< Matrix of first momentum to previous layer (#ADAM)
   CBufferFloat      *SecondMomentumConv;                                  ///< Matrix of second momentum to previous layer (#ADAM)
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;      ///< Feed Forward method.@param NeuronOCL Pointer to previos layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;  ///< Method for updating weights.@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;  ///< Method to transfer gradients to previous layer @param[in] NeuronOCL Pointer to previous layer.

public:
   /** Constructor */
                     CNeuronConvOCL(void) :   iWindowOut(1), iVariables(1) {  activation = SoftPlus;   }
   /** Destructor */~CNeuronConvOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint window_out, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint window_out, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of input window @param[in] step Step size.@param[in] window_out Size of output window @param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual CBufferFloat* GetWeightsConv(void)   {  return WeightsConv;  }
   //---
   virtual int       Type(void) override const  {  return defNeuronConvOCL;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetFilters(void) const { return iWindowOut; }
   virtual uint      GetVariables(void) const { return iVariables; }
   virtual uint      GetUnits(void) const { return Output.Total() / (iWindowOut * iVariables); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronConvSAMOCL    :  public CNeuronConvOCL
  {
protected:
   float             fRho;
   //---
   CBufferFloat      cWeightsSAM;
   CBufferFloat      cWeightsSAMConv;
   //---
   virtual bool      calcEpsilonWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForwardSAM(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      InitBufferLike(CBufferFloat *&buffer, CBufferFloat *master);
   virtual void      ReplaceBuffer(CBufferFloat *&buffer, CBufferFloat *master);

public:
                     CNeuronConvSAMOCL(void) {  activation = GELU;   }
                    ~CNeuronConvSAMOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint window_out, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint window_out, uint units_count, uint variables, float rho, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      InitPS(CNeuronConvSAMOCL *master);
   //---
   virtual int       Type(void)   const         {  return defNeuronConvSAMOCL;   }
   virtual int       Activation(void)  const    {  return (fRho == 0 ? (int)None : (int)activation);   }
   virtual int       getWeightsSAMIndex(void) const override {  return cWeightsSAM.GetIndex();   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
  };
//+------------------------------------------------------------------+
///\class CNeuronAttentionOCL
/// Class of Self-Attention layer GPU calculation
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para42">the link.</A>
//+------------------------------------------------------------------+
class CNeuronAttentionOCL : public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    *Querys;             ///< Convolution layer for Querys
   CNeuronConvOCL    *Values;             ///< Convolution layer for Values
   CBufferFloat      *Scores;             ///< Buffer for Scores matrix
   CNeuronBaseOCL    *AttentionOut;       ///< Layer of Self-Attention Out
   CNeuronConvOCL    *FF1;                ///< Convolution layer for first layer of Feed Forward block
   CNeuronConvOCL    *FF2;                ///< Convolution layer for second layer of Feed Forward block
   //---
   uint              iWindow;             ///< Window size
   uint              iUnits;              ///< Number of units
   //---
   virtual bool      feedForward(CNeuronBaseOCL *prevLayer);                  ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *prevLayer);           ///< Method for updating weights.@param prevLayer Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.

public:
   /** Constructor */
                     CNeuronAttentionOCL(void) : iWindow(1), iUnits(0) {};
   /** Destructor */~CNeuronAttentionOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of in/out window and step.@param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       Type(void)   const   {  return defNeuronAttentionOCL;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual CLayerDescription* GetLayerInfo(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
///\ingroup neuron_base
///\class CNeuronMHAttentionOCL
///\brief Class of Multi-Head Self-Attention layer GPU calculation
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8909#para52">the link.</A>
//+------------------------------------------------------------------+
class CNeuronMHAttentionOCL   :  public CNeuronAttentionOCL
  {
protected:
   CNeuronConvOCL    *Querys2;            ///< Convolution layer for Querys Head 2
   CNeuronConvOCL    *Querys3;            ///< Convolution layer for Querys Head 3
   CNeuronConvOCL    *Querys4;            ///< Convolution layer for Querys Head 4
   CNeuronConvOCL    *Keys2;              ///< Convolution layer for Keys Head 2
   CNeuronConvOCL    *Keys3;              ///< Convolution layer for Keys Head 3
   CNeuronConvOCL    *Keys4;              ///< Convolution layer for Keys Head 4
   CNeuronConvOCL    *Values2;            ///< Convolution layer for Values Head 2
   CNeuronConvOCL    *Values3;            ///< Convolution layer for Values Head 3
   CNeuronConvOCL    *Values4;            ///< Convolution layer for Values Head 4
   CBufferFloat      *Scores2;            ///< Buffer for Scores matrix Head 2
   CBufferFloat      *Scores3;            ///< Buffer for Scores matrix Head 3
   CBufferFloat      *Scores4;            ///< Buffer for Scores matrix Head 4
   CNeuronBaseOCL    *AttentionOut2;      ///< Layer of Self-Attention Out
   CNeuronBaseOCL    *AttentionOut3;      ///< Layer of Self-Attention Out
   CNeuronBaseOCL    *AttentionOut4;      ///< Layer of Self-Attention Out
   CNeuronBaseOCL    *AttentionConcatenate;///< Layer of Concatenate Self-Attention Out
   CNeuronConvOCL    *Weights0;           ///< Convolution layer for Weights0
   //---
   virtual bool      feedForward(CNeuronBaseOCL *prevLayer);                  ///< Feed Forward method.@param prevLayer Pointer to previos layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *prevLayer);            ///< Method for updating weights.@param prevLayer Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   /// Method to transfer gradients inside Head Self-Attention
   virtual bool      calcHeadGradient(CNeuronConvOCL *query, CNeuronConvOCL *value, CBufferFloat *score, CNeuronBaseOCL *attention, CNeuronBaseOCL *prevLayer);

public:
   /** Constructor */
                     CNeuronMHAttentionOCL(void) {};
   /** Destructor */~CNeuronMHAttentionOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of in/out window and step.@param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       Type(void)   const   {  return defNeuronMHAttentionOCL;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);   ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);   ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual CLayerDescription* GetLayerInfo(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
///\class CCollection
///\brief Class of objects collection.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9025#para41">the link.</A>
//+------------------------------------------------------------------+
class CCollection    :  public CArrayObj
  {
public:
                     CCollection(void) {};
                    ~CCollection(void) {};
   //---
   CObject*          operator[](const int i) const               { return(m_data[i]); }
   //---
   virtual bool      CreateElement(const int index)
     {
      if(index < 0)
         ReturnFalse;
      //---
      if(index >= m_data_max)
        {
         if(!Reserve(index - m_data_total))
            ReturnFalse;
        }
      m_data[index] = new CBufferFloat();
      return(CheckPointer(m_data[index]) != POINTER_INVALID);
     }
   //---
   virtual bool      SetOpenCL(COpenCLMy *open_cl)
     {
      if(!open_cl)
         ReturnFalse;
      bool result = true;
      for(int i = 0; (i < m_data_total && result); i++)
        {
         CBufferFloat* temp = m_data[i];
         if(!temp || !temp.BufferCreate(open_cl))
           {
            result = false;
            break;
           }
        }
      //---
      return result;
     }
   //---
  };
//+------------------------------------------------------------------+
///\ingroup neuron_base
///\class CNeuronMLMHAttentionOCL
///\brief Class of Multilayer multi-headed attention neuron.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9025#para41">the link.</A>
//+------------------------------------------------------------------+
class CNeuronMLMHAttentionOCL       :  public CNeuronBaseOCL
  {
protected:
   uint              iLayers;                                     ///< Number of inner layers
   uint              iHeads;                                      ///< Number of heads
   uint              iWindow;                                     ///< Input window size
   uint              iUnits;                                      ///< Number of units
   uint              iWindowKey;                                  ///< Size of Key/Query window
   //---
   CCollection       *QKV_Tensors;                                ///< The collection of tensors of Queries, Keys and Values
   CCollection       *QKV_Weights;                                ///< The collection of Matrix of weights to previous layer
   CCollection       *S_Tensors;                                  ///< The collection of Scores tensors
   CCollection       *AO_Tensors;                                 ///< The collection of Attention Out tensors
   CCollection       *FF_Tensors;                                 ///< The collection of tensors of Feed Forward output
   CCollection       *FF_Weights;                                 ///< The collection of Matrix of Feed Forward weights

   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   virtual bool      ConvolutionForward(CBufferFloat *weights, CBufferFloat *inputs, CBufferFloat *outputs, uint window, uint window_out, ENUM_ACTIVATION activ, uint step = 0, uint variables = 1);
   ///< \brief Convolution Feed Forward method of calling kernel ::FeedForwardConv().
   virtual bool      AttentionScore(CBufferFloat *qkv, CBufferFloat *scores, bool mask = false);
   ///< \brief Multi-heads attention scores method of calling kernel ::MHAttentionScore().
   virtual bool      AttentionOut(CBufferFloat *qkv, CBufferFloat *scores, CBufferFloat *out);
   ///< \brief Multi-heads attention out method of calling kernel ::MHAttentionOut().
   ///\ingroup neuron_base_opt
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      ConvolutuionUpdateWeights(CBufferFloat *weights, CBufferFloat *gradient, CBufferFloat *inputs, CBufferFloat *momentum1, CBufferFloat *momentum2, uint window, uint window_out, uint step = 0, uint heads = 0, uint variables = 1);
   virtual bool      ConvolutuionUpdateWeights(CBufferFloat *weights, CBufferFloat *source, CBufferFloat *momentum1, CBufferFloat *momentum2, float tau);
   ///< Method for updating weights in convolution layer.\details Calling one of kernels ::UpdateWeightsConvMomentum() or ::UpdateWeightsConvAdam() in depends of optimization type (#ENUM_OPTIMIZATION).
   virtual bool      ConvolutionInputGradients(CBufferFloat *weights, CBufferFloat *gradient, CBufferFloat *inputs, CBufferFloat *inp_gradient, uint window, uint window_out, uint activ, uint shift_out = 0, uint step = 0, uint variables = 1);
   ///< Method of passing gradients through a convolutional layer.
   virtual bool      AttentionInsideGradients(CBufferFloat *qkv, CBufferFloat *qkv_g, CBufferFloat *scores, CBufferFloat *gradient);
   ///< Method of passing gradients through attention layer.
public:
   /** Constructor */
                     CNeuronMLMHAttentionOCL(void);
   /** Destructor */~CNeuronMLMHAttentionOCL(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of in/out window and step.@param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       Type(void)   const   {  return defNeuronMLMHAttentionOCL;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual CLayerDescription* GetLayerInfo(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
///\ingroup neuron_base
///\class CNeuronDropoutOCL
///\brief The Dropout neuron for GPU calculation.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9112#para2">the link.</A>
//+------------------------------------------------------------------+
class CNeuronDropoutOCL    :  public   CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    *PrevLayer;
   float             OutProbability;
   int               OutNumber;
   CBufferFloat      *DropOutMultiplier;
   float             dInitValue;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method.@param NeuronOCL Pointer to previos layer.
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) {return true;}        ///< Blank method for updating weights.@param NeuronOCL Pointer to previos layer.
   //---

public:
   /** Constructor */
                     CNeuronDropoutOCL(void);
   /** Destructor */
                    ~CNeuronDropoutOCL(void);
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, float out_prob, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer #param[in] out_prob Probability of neurons shutdown @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       getOutputIndex(void) const override  {  return (bTrain ? Output.GetIndex() : PrevLayer.getOutputIndex());             }  ///< Get index of output buffer @return Index
   virtual int       getGradientIndex(void) const override {  return (bTrain ? Gradient.GetIndex() : PrevLayer.getGradientIndex());          }  ///< Get index of gradient buffer @return Index
   //---
   virtual int       getOutputVal(float &values[]) override {  return (bTrain ? Output.GetData(values) : PrevLayer.getOutputVal(values)); }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(vector<float> &values) override {  return (bTrain ? Output.GetData(values) : PrevLayer.getOutputVal(values)); }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(CArrayFloat *values) override {  return (bTrain ? Output.GetData(values) : PrevLayer.getOutputVal(values)); }  ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getGradient(float &values[]) override {  return (bTrain ? Gradient.GetData(values) : PrevLayer.getGradient(values));    }  ///< Get values of gradient buffer @param[out] values Array of data @return number of items
   virtual int       getGradient(vector<float> &values) override {  return (bTrain ? Gradient.GetData(values) : PrevLayer.getGradient(values));    }  ///< Get values of gradient buffer @param[out] values Array of data @return number of items
   virtual CBufferFloat   *getOutput(void) const override {  return (bTrain ? Output : PrevLayer.getOutput());      }                 ///< Get pointer of output buffer @return Pointer to object
   virtual CBufferFloat   *getGradient(void) const override {  return (bTrain ? Gradient : PrevLayer.getGradient());  }                 ///< Get pointer of gradient buffer @return Pointer to object
   //---
   virtual bool      Save(int const file_handle) override ;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override ;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronDropoutOCL;                }///< Identificator of class.@return Type of class
   virtual CLayerDescription* GetLayerInfo(void) override;
  };
//+------------------------------------------------------------------+
///\class CNeuronBatchNormOCL
///\brief The class of Batch Normalization neuron for GPU calculation.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9207#para4">the link.</A>
//+------------------------------------------------------------------+
class CNeuronBatchNormOCL  :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    *PrevLayer;       ///< Pointer to the object of the previous layer
   int               iBatchSize;       ///< Batch size
   int               iBatchCount;      ///< Batch count
   CBufferFloat      *BatchOptions;    ///< Container of method parameters

   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method of calling kernel ::BatchFeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;        ///< Method for updating weights.\details Calling one of kernels ::UpdateBatchOptionsMomentum() or ::UpdateBatchOptionsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradientBatch(). @param NeuronOCL Pointer to next layer.

public:
   /** Constructor */
                     CNeuronBatchNormOCL(void);
   /** Destructor */~CNeuronBatchNormOCL(void);
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, uint batchSize, ENUM_OPTIMIZATION optimization_type);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       getOutputIndex(void)          {  return (iBatchSize > 1 || !PrevLayer ? Output.GetIndex() : PrevLayer.getOutputIndex());             } ///< Get index of output buffer @return Index
   virtual int       getGradientIndex(void)        {  return (iBatchSize >  1 || !PrevLayer ? Gradient.GetIndex() : PrevLayer.getGradientIndex());          } ///< Get index of gradient buffer @return Index
   //---
   virtual int       getOutputVal(float &values[])   {  return (iBatchSize > 1 || !PrevLayer ? Output.GetData(values) : PrevLayer.getOutputVal(values)); } ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(CArrayFloat *values)   {  return (iBatchSize > 1 || !PrevLayer ? Output.GetData(values) : PrevLayer.getOutputVal(values)); } ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getOutputVal(vector<float> &values)   {  return (iBatchSize > 1 || !PrevLayer ? Output.GetData(values) : PrevLayer.getOutputVal(values)); } ///< Get values of output buffer @param[out] values Array of data @return number of items
   virtual int       getGradient(float &values[])    {  return (iBatchSize > 1 || !PrevLayer ? Gradient.GetData(values) : PrevLayer.getGradient(values));    } ///< Get values of gradient buffer @param[out] values Array of data @return number of items
   virtual CBufferFloat   *getOutput(void)           {  return (iBatchSize > 1 || !PrevLayer ? Output : PrevLayer.getOutput());      }               ///< Get pointer of output buffer @return Pointer to object
   virtual CBufferFloat   *getGradient(void)         {  return (iBatchSize > 1 || !PrevLayer ? Gradient : PrevLayer.getGradient());  }               ///< Get pointer of gradient buffer @return Pointer to object
   virtual CBufferFloat   *getBatchOptions(void)         {  return (iBatchSize > 1 ? BatchOptions : NULL);  }               ///< Get pointer of Batch Options buffer @return Pointer to object
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronBatchNormOCL;    }///< Identificator of class.@return Type of class
   virtual CLayerDescription* GetLayerInfo(void);
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CBufferFloat   *getWeights(void) override { return BatchOptions; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLSTMOCL : public CNeuronBaseOCL
  {
protected:
   CBufferFloat      m_cWeightsLSTM;
   CBufferFloat      m_cFirstMomentumLSTM;      ///< Buffer of first momentum matrix (#ADAM) or last delta weights matrix (#SGD)
   CBufferFloat      m_cSecondMomentumLSTM;     ///< Buffer of second momentum matrix (#ADAM)

   int               m_iMemory;
   int               m_iConcatenated;
   int               m_iConcatenatedGradient;
   int               m_iHiddenState;
   int               m_iWeightsGradient;
   int               m_iInputs;
   int               m_iVariables;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method.@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;        ///< Method for updating weights.
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronLSTMOCL(void);
                    ~CNeuronLSTMOCL(void);
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      SetInputs(int count);
   ///@}
   //---
   virtual bool      Save(int const file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void) override       const                      {  return defNeuronLSTMOCL;                  }///< Identificator of class.@return Type of class
   virtual bool      Clear(void);
   virtual CBufferFloat *getLSTMWeights(void) { return GetPointer(m_cWeightsLSTM);}
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#ifndef class_vae
#include "..\Unsupervised\AE\VAE.mqh"
#endif
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSoftMaxOCL    :  public CNeuronBaseOCL
  {
protected:
   uint              iHeads;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronSoftMaxOCL(void) : iHeads(1) { activation = None; }
                    ~CNeuronSoftMaxOCL(void) {};
   //---
   ///\ingroup neuron_base_gr
   ///@{
   virtual bool      calcOutputGradients(CBufferFloat *Target, float &error) override;               ///< Method of output gradients calculation by calling kernel ::CalcOutputGradient().@param Target Traget value
   virtual void      SetHeads(int heads)  { iHeads = heads; }
   ///@}
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual int       Type(void) override        const                      {  return defNeuronSoftMaxOCL;                  }///< Identificator of class.@return Type of class
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   virtual uint      Heads(void) const {  return iHeads; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFQF : protected CNeuronBaseOCL
  {
protected:
   //--- Fractal Net
   CNeuronBaseOCL    cFraction;
   CNeuronSoftMaxOCL cSoftMax;
   //--- Cosine embeding
   CNeuronBaseOCL    cCosine;
   CNeuronBaseOCL    cCosineEmbeding;
   //--- Quantile Net
   CNeuronBaseOCL    cQuantile0;
   CNeuronBaseOCL    cQuantile1;
   CNeuronBaseOCL    cQuantile2;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronFQF();
                    ~CNeuronFQF();
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint actions, uint quantiles, uint numInputs, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual bool      Save(int const file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void) override        const                      {  return defNeuronFQF; }///< Identificator of class.@return Type of class
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual CObject*  AsObject(void) {  return GetPointer(this);   }
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      SetActivationFunction(ENUM_ACTIVATION function)    {  CNeuronBaseOCL::SetActivationFunction(function); }
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMLMHSparseAttention  : public CNeuronMLMHAttentionOCL
  {
protected:
   float             m_dSparse;
   //---
   virtual bool      AttentionScore(CBufferFloat *qkv, CBufferFloat *scores, bool mask = true);
   ///< \brief Multi-heads attention scores method of calling kernel ::MHAttentionScore().
   virtual bool      AttentionOut(CBufferFloat *qkv, CBufferFloat *scores, CBufferFloat *out);
   ///< \brief Multi-heads attention out method of calling kernel ::MHAttentionOut().

public:
                     CNeuronMLMHSparseAttention(void)   :  m_dSparse(0.3f) {};
                    ~CNeuronMLMHSparseAttention(void) {};
   //---
   void              Sparse(float value)  { m_dSparse = value;}
   float             Sparse(void)         { return m_dSparse; }
   virtual int       Type(void)   const   {  return defNeuronMLMHSparseAttentionOCL;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiModel : public CNeuronBaseOCL
  {
protected:
   int               iModels;
   int               iUpdateModel;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.

public:
                     CNeuronMultiModel(void) {};
                    ~CNeuronMultiModel(void) {};
   virtual bool      Init(uint numInputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, int models);
   //---
   ///\ingroup neuron_base_gr
   ///@{
   virtual bool      calcHiddenGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.
   ///@}
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronMultiModels;                  }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronConcatenate   :  public CNeuronBaseOCL
  {
protected:
   int               i_SecondInputs;
   CBufferFloat      *ConcWeights;            ///< Buffer of weights matrix
   CBufferFloat      *ConcDeltaWeights;       ///< Buffer of last delta weights matrix (#SGD)
   CBufferFloat      *ConcFirstMomentum;      ///< Buffer of first momentum matrix (#ADAM)
   CBufferFloat      *ConcSecondMomentum;     ///< Buffer of second momentum matrix (#ADAM)
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;        ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronConcatenate(void);
   /** Destructor */~CNeuronConcatenate(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, uint inputs1, uint inputs2, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void) override       const                      {  return defNeuronConcatenate; }///< Identificator of class.@return Type of class
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CBufferFloat   *getConcatWeights(void)        {  return ConcWeights;     }                 ///< Get pointer of gradient buffer @return Pointer to object
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSoftActorCritic  :  public CNeuronFQF
  {
protected:
   CNeuronConcatenate   cAlphas;
   CBufferFloat         cLogProbs;
   CBufferFloat         cRandomize;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronSoftActorCritic(void) {};
                    ~CNeuronSoftActorCritic(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint actions, uint quantiles, uint numInputs, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   virtual bool      calcAlphaGradients(CNeuronBaseOCL *NeuronOCL);
   virtual bool      GetAlphaLogProbs(vector<float> &log_probs)       { return (cLogProbs.GetData(log_probs) > 0); }
   virtual bool      CalcLogProbs(CBufferFloat *buffer);
   virtual bool      ReCalcLogProbs(void);
   //---
   virtual bool      Save(int const file_handle) override;///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle) override;///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void) override        const                      {  return defNeuronSoftActorCritic; }///< Identificator of class.@return Type of class
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual void      getCheckData(vector<float> &output, vector<float> &grad, vector<float> &quantiles, vector<float> &probability, vector<float> &alphas);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronEmbeddingOCL  :  public CNeuronBaseOCL
  {
protected:
   int               a_Windows[];
   int               i_WindowOut;
   int               i_StackSize;
   int               i_WindowsBuffer;
   int               i_STDBuffer;
   //---
   CBufferFloat      WeightsEmbedding;
   CBufferFloat      FirstMomentumEmbed;
   CBufferFloat      SecondMomentumEmbed;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   ///\ingroup neuron_base_gr
   ///@{
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.
   ///@}
   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
public:
                     CNeuronEmbeddingOCL(void);
                    ~CNeuronEmbeddingOCL(void);
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint stack_size, uint window_out, uint &windows[]);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object. #param[in] numNeurons Number of neurons in layer @param optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronEmbeddingOCL;                  }///< Identificator of class.@return Type of class
   virtual CLayerDescription* GetLayerInfo(void)   { return CNeuronBaseOCL::GetLayerInfo(); }
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual bool      Clear(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPositionEncoder  :  public CNeuronBaseOCL
  {
protected:
   CBufferFloat      PositionEncoder;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)         { return true; }        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)         { return true; }          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronPositionEncoder(void) {};
                    ~CNeuronPositionEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint count, uint window, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronPEOCL;                  }///< Identificator of class.@return Type of class
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTransposeOCL : public CNeuronBaseOCL
  {
protected:
   uint               iWindow;
   uint               iCount;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override       { return true; }        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;          ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronTransposeOCL(void) {};
                    ~CNeuronTransposeOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint count, uint window, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronTransposeOCL; }///< Identificator of class.@return Type of class
   //---
   virtual uint      GetWindow(void) const { return iWindow; }
   virtual uint      GetCount(void) const { return iCount; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMH2AttentionOCL       :  public CNeuronBaseOCL
  {
protected:
   uint              iHeads;                                      ///< Number of heads
   uint              iWindow;                                     ///< Input window size
   uint              iUnits;                                      ///< Number of units
   uint              iWindowKey;                                  ///< Size of Key/Query window
   //---
   CNeuronConvOCL    Q_Embedding;
   CNeuronConvOCL    KV_Embedding;
   CNeuronTransposeOCL Transpose;
   int               ScoreIndex;
   CNeuronBaseOCL    MHAttentionOut;
   CNeuronConvOCL    W0;
   CNeuronBaseOCL    AttentionOut;
   CNeuronConvOCL    FF[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      attentionOut(void);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionInsideGradients(void);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
   /** Constructor */
                     CNeuronMH2AttentionOCL(void);
   /** Destructor */~CNeuronMH2AttentionOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count, ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMH2AttentionOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCGConvOCL  :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cInputF;
   CNeuronBaseOCL    cInputS;
   CNeuronBaseOCL    cF;
   CNeuronBaseOCL    cS;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronCGConvOCL(void) {};
                    ~CNeuronCGConvOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint numNeurons,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   //---
   virtual int       Type(void)   const   {  return defNeuronCGConvOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMFTOCL  : public CNeuronMLMHAttentionOCL
  {
protected:
   //---
   CCollection       cTranspose;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      Transpose(CBufferFloat *in, CBufferFloat *out, int rows, int cols);
   virtual bool      MHCA(CBufferFloat *q, CBufferFloat *kv,
                          CBufferFloat *score, CBufferFloat *out);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      MHCAInsideGradients(CBufferFloat *q, CBufferFloat *qg,
                                         CBufferFloat *kv, CBufferFloat *kvg,
                                         CBufferFloat *score, CBufferFloat *aog);

public:
                     CNeuronMFTOCL(void) {};
                    ~CNeuronMFTOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count, uint features,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   //---
   virtual int       Type(void)   const   {  return defNeuronMFTOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronXCiTOCL  : public CNeuronMLMHAttentionOCL
  {
protected:
   //---
   uint              iLPIWindow;
   uint              iLPIStep;
   uint              iBatchCount;
   //---
   CCollection       cLPI;
   CCollection       cLPI_Weights;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      XCiT(CBufferFloat *qkv, CBufferFloat *score, CBufferFloat *out);
   virtual bool      BatchNorm(CBufferFloat *inputs, CBufferFloat *options, CBufferFloat *out);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      XCiTInsideGradients(CBufferFloat *qkv, CBufferFloat *qkvg,
                                         CBufferFloat *score, CBufferFloat *aog);
   virtual bool      BatchNormInsideGradient(CBufferFloat *inputs, CBufferFloat *inputs_g,
         CBufferFloat *options, CBufferFloat *out,
         CBufferFloat *out_g, ENUM_ACTIVATION activation);
   virtual bool      BatchNormUpdateWeights(CBufferFloat *options, CBufferFloat *out_g);

public:
                     CNeuronXCiTOCL(void) {};
                    ~CNeuronXCiTOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint lpi_window, uint heads,
                          uint units_count, uint layers,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronXCiTOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDOTOCL     :  public CNeuronBaseOCL
  {
protected:
   uint              iWindowSize;
   uint              iPrevWindowSize;
   uint              iDimension;
   uint              iUnits;
   uint              iHeads;
   //---
   CNeuronConvOCL    cProjInput;
   CNeuronConvOCL    cQKV;
   int               iScoreBuffer;
   CNeuronBaseOCL    cRelativePositionsBias;
   CNeuronBaseOCL    MHAttentionOut;
   CNeuronConvOCL    cProj;
   CNeuronBaseOCL    AttentionOut;
   CNeuronConvOCL    cFF1;
   CNeuronConvOCL    cFF2;
   CNeuronBaseOCL    SAttenOut;
   CNeuronXCiTOCL    cCAtten;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      DOT(void);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateRelativePositionsBias(void);
   virtual bool      DOTInsideGradients(void);

public:
                     CNeuronDOTOCL(void) {};
                    ~CNeuronDOTOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint dimension, uint heads,
                          uint units_count, uint prev_window,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   //---
   virtual int       Type(void)   const   {  return defNeuronDOTOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFAQOCL  : public CNeuronBaseOCL
  {
protected:
   //---
   CNeuronConvOCL    cF;
   CNeuronBaseOCL    cWv;
   CNeuronBatchNormOCL cNormV;
   CNeuronBaseOCL    cQd;
   CNeuronXCiTOCL    cDQd;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronFAQOCL(void) {};
                    ~CNeuronFAQOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint heads,
                          uint units_count, uint input_units,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   //---
   virtual int       Type(void)   const   {  return defNeuronFAQOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCrossAttention : public CNeuronMH2AttentionOCL
  {
protected:
   uint              iWindow_K;
   uint              iUnits_K;
   CNeuronBaseOCL    *cContext;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CNeuronBaseOCL *Context);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);
   virtual bool      attentionOut(void);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CNeuronBaseOCL *Context);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CNeuronBaseOCL *Context);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);
   virtual bool      AttentionInsideGradients(void);

public:
                     CNeuronCrossAttention(void) : cContext(NULL) {};
                    ~CNeuronCrossAttention(void) { DeleteObj(cContext); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count, uint window_k, uint units_k,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronCrossAttenOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGTE : public CNeuronBaseOCL
  {
protected:
   uint              iHeads;                                      ///< Number of heads
   uint              iWindow;                                     ///< Input window size
   uint              iUnits;                                      ///< Number of units
   uint              iWindowKey;                                  ///< Size of Key/Query window
   //---
   CNeuronConvOCL    cQKV;
   CNeuronSoftMaxOCL cSoftMax;
   int               ScoreIndex;
   CNeuronBaseOCL    cMHAttentionOut;
   CNeuronConvOCL    cW0;
   CNeuronBaseOCL    cAttentionOut;
   CNeuronCGConvOCL  cGraphConv[2];
   CNeuronConvOCL    cFF[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionOut(void);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionInsideGradients(void);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronGTE(void) {};
                    ~CNeuronGTE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronGTE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);                ///< Set Training Mode Flag
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CResidualConv  :  public CNeuronBaseOCL
  {
protected:
   int               iWindowOut;
   //---
   CNeuronConvSAMOCL    cConvs[3];
   CNeuronBatchNormOCL cNorm[3];
   CNeuronBaseOCL    cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CResidualConv(void) {};
                    ~CResidualConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defResidualConv;   }
   //--- methods for working with files
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;                ///< Set Training Mode Flag
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CCCMREncoder   :  public CNeuronBaseOCL
  {
protected:
   CResidualConv     cResidual[6];
   CNeuronConvOCL    cInput;
   CNeuronBatchNormOCL cNorm;
   CNeuronConvOCL    cOutput;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CCCMREncoder(void) {};
                    ~CCCMREncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defCCMREncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);                ///< Set Training Mode Flag
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCrossXCiTOCL  :  public CNeuronXCiTOCL
  {
protected:
   CCollection       cConcat;
   CCollection       cValue;
   CCollection       cV_Weights;
   CBufferFloat      TempBuffer;
   uint              iWindow2;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Motion);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Motion);

public:
                     CNeuronCrossXCiTOCL(void) {};
                    ~CNeuronCrossXCiTOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window1, uint window2, uint lpi_window, uint heads,
                          uint units_count, uint layers,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CNeuronBaseOCL *Motion);
   virtual int       Type(void)   const   {  return defNeuronCrossXCiTOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCCMROCL    :  public CNeuronBaseOCL
  {
protected:
   CCCMREncoder      FeatureExtractor;
   CNeuronBaseOCL    PrevFeatures;
   CNeuronBaseOCL    Motion;
   CNeuronBaseOCL    Temp;
   CCCMREncoder      LocalContext;
   CNeuronXCiTOCL    GlobalContext;
   CNeuronCrossXCiTOCL  MotionContext;
   CNeuronLSTMOCL    RecurentUnit;
   CNeuronConvOCL    UpScale;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronCCMROCL(void) {};
                    ~CNeuronCCMROCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronCCMROCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);                ///< Set Training Mode Flag
   virtual bool      Clear(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronNODEOCL     :  public CNeuronBaseOCL
  {
protected:
   uint              iDimension;
   uint              iVariables;
   uint              iLenth;

   int               iBuffersK[];
   int               iInputsK[];
   int               iMeadl[];
   CBufferFloat      cAlpha;
   CBufferFloat      cTemp;
   CCollection       cBeta;
   CBufferFloat      cSolution;
   CCollection       cWeights;
   //---
   virtual bool      CalculateKBuffer(int k);
   virtual bool      CalculateInputK(CBufferFloat* inputs, int k);
   virtual bool      CalculateOutput(CBufferFloat* inputs);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      CalculateOutputGradient(CBufferFloat* inputs);
   virtual bool      CalculateInputKGradient(CBufferFloat* inputs, int k);
   virtual bool      CalculateKBufferGradient(int k);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronNODEOCL(void) {};
                    ~CNeuronNODEOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint variables, uint lenth,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronNODEOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronConformer  :  public CNeuronBaseOCL
  {
protected:
   //---
   int               iWindow;
   int               iDimension;
   int               iHeads;
   int               iVariables;
   int               iCount;
   //---
   CNeuronConvOCL    cQKV;
   CNeuronBaseOCL    cdQKV;
   int               iScore;
   CNeuronBaseOCL    cAttentionOut;
   CNeuronConvOCL    cW0;
   CNeuronNODEOCL    cNODE[3];
   CNeuronConvOCL    cFF[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      attentionOut(void);
   //---
   virtual bool      AttentionInsideGradients(void);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronConformer(void)  :  iScore(-1) {};
                    ~CNeuronConformer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint variables, uint units_count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronConformerOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRevINDenormOCL   :  public CNeuronBatchNormOCL
  {
protected:
   int               iBatchNormLayer;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   {  return true;   }
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronRevINDenormOCL(void)   :  iBatchNormLayer(-1) {};
                    ~CNeuronRevINDenormOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, int NormLayer, CNeuronBatchNormOCL *normLayer);
   virtual int       GetNormLayer(void)   {  return iBatchNormLayer; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronRevInDenormOCL;    }///< Identificator of class.@return Type of class
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override  {  return true;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronClientOCL  :  public CNeuronBaseOCL
  {
protected:
   //--- Attention
   CNeuronMLMHAttentionOCL cTransformerEncoder;
   CNeuronConvOCL          cProjection;
   //--- Linear model
   CNeuronConvOCL          cLinearModel[];
   //---
   CNeuronBaseOCL          cInput;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronClientOCL(void) {};
                    ~CNeuronClientOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint at_layers, uint count, uint &mlp[],
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronClientOCL;   }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLearnabledPE  :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cInputs;
   CNeuronBaseOCL    cPositionEncoder;
   ///\ingroup neuron_base_ff
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);               ///< \brief Feed Forward method of calling kernel ::FeedForward().@param NeuronOCL Pointer to previos layer.
   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);        ///< Method to transfer gradient to previous layer by calling kernel ::CalcHiddenGradient(). @param NeuronOCL Pointer to next layer.

public:
                     CNeuronLearnabledPE(void) {};
                    ~CNeuronLearnabledPE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual int       Type(void)        const                      {  return defNeuronLearnabledPE;    }///< Identificator of class.@return Type of class
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual CBufferFloat* GetPE(void)   const { return cPositionEncoder.getOutput(); }
   virtual bool      AddPEGradient(CBufferFloat *grad)
     {
      if(!grad ||
         !SumAndNormalize(cPositionEncoder.getGradient(), grad, cPositionEncoder.getGradient(), 1, false, 0, 0, 0, 1))
         ReturnFalse;
      return true;
     }
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUShapeAttention  :  public CNeuronBaseOCL
  {
protected:
   CNeuronMLMHAttentionOCL    cAttention[2];
   CNeuronConvOCL             cMergeSplit[2];
   CNeuronBaseOCL             *cNeck;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   //---
public:
                     CNeuronUShapeAttention(void) { activation = None; }
                    ~CNeuronUShapeAttention(void) { DeleteObj(cNeck); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, uint inside_block, ENUM_OPTIMIZATION optimization_type, uint batch);
   ///< Method of initialization class.@param[in] numOutputs Number of connections to next layer.@param[in] myIndex Index of neuron in layer.@param[in] open_cl Pointer to #COpenCLMy object.@param[in] window Size of in/out window and step.@param[in] units_countNumber of neurons.@param[in] optimization_type Optimization type (#ENUM_OPTIMIZATION)@return Boolen result of operations.
   //---
   virtual int       Type(void)   const   {  return defNeuronUShapeAttention;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPatching   :  public CNeuronBaseOCL
  {
protected:
   uint              iWindowIn;
   uint              iStep;
   uint              iWindowOut;
   uint              iVariables;
   uint              iCount;
   //---
   CBufferFloat      cPatchWeights;
   CBufferFloat      cPatchFirstMomentum;
   CBufferFloat      cPatchSecondMomentum;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronPatching(void) {};
                    ~CNeuronPatching(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window_in, uint step, uint window_out, uint count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronPatchingOCL; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTiDEOCL :  public CNeuronBaseOCL
  {
protected:
   uint              iHistory;
   uint              iForecast;
   uint              iVariables;
   uint              iFeatures;
   //---
   CResidualConv     acEncoderDecoder[];
   CNeuronConvOCL    cGlobalResidual;
   CResidualConv     acFeatureProjection[2];
   CResidualConv     cTemporalDecoder;
   //---
   CNeuronBaseOCL    cHistoryInput;
   CNeuronBaseOCL    cFeatureInput;
   CNeuronBaseOCL    cEncoderInput;
   CNeuronBaseOCL    cTemporalDecoderInput;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None);

public:
                     CNeuronTiDEOCL(void) {};
                    ~CNeuronTiDEOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint history, uint forecast, uint variables, uint features, uint &encoder_decoder[], ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronTiDEOCL; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLegendreWavelets :  public CNeuronConvOCL
  {
protected:
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   { return true; }        ///< Method for updating weights.@param NeuronOCL Pointer to previos layer.

public:
                     CNeuronLegendreWavelets(void) {};
                    ~CNeuronLegendreWavelets(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint units_count, uint filters, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronLegendreWavelets;   }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFEDW : public CNeuronBaseOCL
  {
protected:
   //---
   uint                    iWindow;
   uint                    iCount;
   uint                    iFilters;
   //---
   CNeuronLegendreWavelets cWavlets;
   CNeuronBatchNormOCL     cNorm;
   CNeuronSoftMaxOCL       cSoftMax;
   CNeuronConvOCL          cFF[2];
   CNeuronBaseOCL          cReconstruct;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      Reconsruct(CBufferFloat* inputs, CBufferFloat *outputs);

public:
                     CNeuronFEDW(void) {};
                    ~CNeuronFEDW(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint count, uint filters,  ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronFEDW; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFITSOCL :  public CNeuronBaseOCL
  {
protected:
   //---
   uint              iWindow;
   uint              iWindowOut;
   uint              iCount;
   uint              iFFTin;
   uint              iIFFTin;
   //---
   CNeuronBaseOCL    cInputsRe;
   CNeuronBaseOCL    cInputsIm;
   CNeuronBaseOCL    cFFTRe;
   CNeuronBaseOCL    cFFTIm;
   CNeuronDropoutOCL cDropRe;
   CNeuronDropoutOCL cDropIm;
   CNeuronConvOCL    cInsideRe1;
   CNeuronConvOCL    cInsideIm1;
   CNeuronConvOCL    cInsideRe2;
   CNeuronConvOCL    cInsideIm2;
   CNeuronBaseOCL    cComplexRe;
   CNeuronBaseOCL    cComplexIm;
   CNeuronBaseOCL    cIFFTRe;
   CNeuronBaseOCL    cIFFTIm;
   CBufferFloat      cClear;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   virtual bool      ComplexLayerOut(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im);
   virtual bool      ComplexLayerGradient(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronFITSOCL(void) {};
                    ~CNeuronFITSOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_out, uint count, float dropout, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronFITSOCL; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFreDFOCL   :  public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iCount;
   uint              iFFTin;
   bool              bTranspose;
   float             fAlpha;
   //---
   CBufferFloat      cForecastFreRe;
   CBufferFloat      cForecastFreIm;
   CBufferFloat      cTargetFreRe;
   CBufferFloat      cTargetFreIm;
   CBufferFloat      cLossFreRe;
   CBufferFloat      cLossFreIm;
   CBufferFloat      cGradientFreRe;
   CBufferFloat      cGradientFreIm;
   CBufferFloat      cTranspose;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   virtual bool      Transpose(CBufferFloat *inputs, CBufferFloat *outputs, uint rows, uint cols);
   virtual bool      FreqMSA(CBufferFloat *target, CBufferFloat *forecast, CBufferFloat *gradient);
   virtual bool      CumulativeGradient(CBufferFloat *gradient1, CBufferFloat *gradient2, CBufferFloat *cummulative, float alpha);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   {  return true;   }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronFreDFOCL(void)   {};
                    ~CNeuronFreDFOCL(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint count, float alpha, bool need_transpose = true, ENUM_OPTIMIZATION optimization_type = ADAM, uint batch = 1);
   virtual bool      calcOutputGradients(CArrayFloat *Target, float &error);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronFreDFOCL; }
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronComplexMLMHAttention :  public CNeuronMLMHAttentionOCL
  {
protected:
   virtual bool      ConvolutionForward(CBufferFloat *weights, CBufferFloat *inputs, CBufferFloat *outputs, uint window, uint window_out, ENUM_ACTIVATION activ, uint step = 0);
   virtual bool      AttentionScore(CBufferFloat *qkv, CBufferFloat *scores, bool mask = false);
   virtual bool      AttentionOut(CBufferFloat *qkv, CBufferFloat *scores, CBufferFloat *out);
   virtual bool      ConvolutuionUpdateWeights(CBufferFloat *weights, CBufferFloat *gradient, CBufferFloat *inputs, CBufferFloat *momentum1, CBufferFloat *momentum2, uint window, uint window_out, uint step = 0);
   virtual bool      ConvolutionInputGradients(CBufferFloat *weights, CBufferFloat *gradient, CBufferFloat *inputs, CBufferFloat *inp_gradient, uint window, uint window_out, uint activ, uint shift_out = 0, uint step = 0);
   virtual bool      AttentionInsideGradients(CBufferFloat *qkv, CBufferFloat *qkv_g, CBufferFloat *scores, CBufferFloat *gradient);
   virtual bool      SumAndNormalize(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, bool normilize = true, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 0.5f);

public:
                     CNeuronComplexMLMHAttention(void) {};
                    ~CNeuronComplexMLMHAttention(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronComplexMLMHAttentionOCL;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronATFNetOCL  :  public CNeuronBaseOCL
  {
protected:
   uint              iHistory;
   uint              iForecast;
   uint              iVariables;
   uint              iFFT;
   //--- T-Block
   CNeuronBatchNormOCL  cNorm;
   CNeuronTransposeOCL  cTranspose;
   CNeuronPositionEncoder  cPositionEncoder;
   CNeuronPatching      cPatching;
   CLayer               caAttention;
   CLayer               caProjection;
   CNeuronRevINDenormOCL cRevIN;
   //--- F-Block
   CBufferFloat*     cInputs;
   CBufferFloat      cInputFreqRe;
   CBufferFloat      cInputFreqIm;
   CNeuronBaseOCL    cInputFreqComplex;
   CBufferFloat      cMainFreqWeights;
   CNeuronBaseOCL    cNormFreqComplex;
   CBufferFloat      cMeans;
   CBufferFloat      cVariances;
   CNeuronComplexMLMHAttention cFreqAtteention;
   CNeuronBaseOCL    cUnNormFreqComplex;
   CBufferFloat      cOutputFreqRe;
   CBufferFloat      cOutputFreqIm;
   CBufferFloat      cOutputTimeSeriasRe;
   CBufferFloat      cOutputTimeSeriasIm;
   CBufferFloat      cOutputTimeSeriasReGrad;
   CBufferFloat      cReconstructInput;
   CBufferFloat      cForecast;
   CBufferFloat      cReconstructInputGrad;
   CBufferFloat      cForecastGrad;
   CBufferFloat      cZero;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);
   virtual bool      ComplexNormalize(void);
   virtual bool      ComplexUnNormalize(void);
   virtual bool      ComplexNormalizeGradient(void);
   virtual bool      ComplexUnNormalizeGradient(void);
   virtual bool      MainFreqWeights(void);
   virtual bool      WeightedSum(void);
   virtual bool      WeightedSumGradient(void);
   virtual bool      calcReconstructGradient(void);

public:
                     CNeuronATFNetOCL(void) {};
                    ~CNeuronATFNetOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint history, uint forecast, uint variables, uint heads, uint layers, uint &patch[], ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronATFNetOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual CBufferFloat   *getWeights(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronS3   :  public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iSegments;
   //---
   CNeuronBaseOCL    cOne;
   CNeuronConvOCL    cShufle;
   CNeuronSoftMaxOCL cProbability;
   CNeuronConvOCL    cWeights;
   CBufferFloat      cPositions;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForwardS3(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradientsS3(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronS3(void)   {};
                    ~CNeuronS3(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronS3;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMLMHAttentionMLKV :  public CNeuronMLMHAttentionOCL
  {
protected:
   uint              iLayersToOneKV;                             ///< Number of inner layers
   uint              iHeadsKV;                                   ///< Number of heads
   CCollection       KV_Tensors;                                 ///< The collection of tensors of Keys and Values
   CCollection       KV_Weights;                                 ///< The collection of Matrix of weights to previous layer
   CBufferFloat      Temp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out);
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g, CBufferFloat *kv, CBufferFloat *kv_g, CBufferFloat *scores, CBufferFloat *gradient);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.

public:
                     CNeuronMLMHAttentionMLKV(void) {};
                    ~CNeuronMLMHAttentionMLKV(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMLMHAttentionMLKV;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMLCrossAttentionMLKV :  public CNeuronMLMHAttentionMLKV
  {
protected:
   uint              iWindowKV;
   uint              iUnitsKV;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out) override;
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g, CBufferFloat *kv,
         CBufferFloat *kv_g, CBufferFloat *scores, CBufferFloat *gradient) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }

public:
                     CNeuronMLCrossAttentionMLKV(void) {};
                    ~CNeuronMLCrossAttentionMLKV(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint window_kv, uint heads_kv, uint units_count, uint units_count_kv,
                          uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMLCrossAttentionMLKV;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual uint      GetSecondBufferSize(void)  {  return (iWindowKV * iUnitsKV); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCSCMOCL :  public CNeuronBaseOCL
  {
protected:
   uint              i_Count;
   uint              i_Variables;
   bool              b_NeedTranspose;
   //---
   CArrayInt         ia_Windows;
   CArrayObj         caTranspose;
   CArrayObj         caConvolutions;
   CArrayObj         caMLP;
   CArrayObj         caTemp;
   CArrayObj         caConvOutputs;
   CArrayObj         caConvGradients;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronCSCMOCL(void) {};
                    ~CNeuronCSCMOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint &windows[], uint variables, uint inputs_count, bool need_transpose, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronCSCMOCL;   }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSPyrAttentionOCL :  public CNeuronBaseOCL
  {
protected:
   uint              iWindowIn;
   uint              iWindowKey;
   uint              iHeads;
   uint              iHeadsKV;
   uint              iCount;
   uint              iPAMLayers;
   //---
   CArrayObj         caS3;
   CArrayObj         caQuery;
   CArrayObj         caKV;
   CArrayInt         caScore;
   CArrayObj         caAttentionOut;
   CArrayObj         caW0;
   CNeuronConvOCL    cFF1;
   CNeuronConvOCL    cFF2;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, int scores, CBufferFloat *out, int window);
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g, CBufferFloat *kv, CBufferFloat *kv_g, int scores, CBufferFloat *gradient);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);  ///< Method to transfer gradients to previous layer @param[in] prevLayer Pointer to previous layer.
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.
   virtual void      ArraySetOpenCL(CArrayObj *array, COpenCLMy *obj);

public:
                     CNeuronSPyrAttentionOCL(void) {};
                    ~CNeuronSPyrAttentionOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window_in, uint window_key, uint heads, uint heads_kv, uint units_count, uint pam_layers, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSPyrAttentionMLKV;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPLROCL  :  public CNeuronBaseOCL
  {
protected:
   bool              bTranspose;
   int               icIsTTP;
   int               iVariables;
   int               iCount;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  { return true; }

public:
                     CNeuronPLROCL(void)  : bTranspose(false) {};
                    ~CNeuronPLROCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window_in, uint units_count, bool transpose, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronPLROCL;   }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTPMEncoder :  public CNeuronLSTMOCL
  {
protected:
   bool                 bTSinRow;
   //---
   CNeuronCSCMOCL       cFeatureExtraction;
   CNeuronBaseOCL       cMemAndHidden;
   CNeuronConcatenate   cConcatenated;
   CNeuronSoftMaxOCL    cSoftMax;
   CNeuronBaseOCL       cAttentionOut;
   CNeuronTransposeOCL  cTranspose;
   CBufferFloat         cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTPMEncoder(void) {};
                    ~CNeuronTPMEncoder(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint variables, uint lenth, uint hidden_size, bool ts_in_row, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void) override       const                      {  return defNeuronTPMEncoder; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTPM :  public CNeuronLSTMOCL
  {
protected:
   CNeuronTPMEncoder    cEncoder;
   CNeuronPLROCL        cFeatureExtraction;
   CNeuronBaseOCL       cMemAndHidden;
   CNeuronConcatenate   cConcatenated;
   CNeuronSoftMaxOCL    cSoftMax;
   CNeuronBaseOCL       cAttentionOut;
   CNeuronConcatenate   cAttAndFeature;
   CBufferFloat         cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTPM(void) {};
                    ~CNeuronTPM(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint variables, uint lenth, uint hidden_size, bool ts_in_row, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void) override       const                      {  return defNeuronTPM; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTNNEncoder  :  public CNeuronMLMHAttentionMLKV
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTNNEncoder(void)  {};
                    ~CNeuronSTNNEncoder(void)  {};
   //---
   virtual int       Type(void)   override const   {  return defNeuronSTNNEncoder;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTNNDecoder   :  public CNeuronMLCrossAttentionMLKV
  {
protected:
   CNeuronSTNNEncoder      cEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronSTNNDecoder(void) {};
                    ~CNeuronSTNNDecoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint window_kv, uint heads_kv, uint units_count, uint units_count_kv, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSTNNDecoder;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSparseTSF  :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL       cConvolution;
   CNeuronTransposeOCL  acTranspose[4];
   CNeuronConvOCL       cForecast;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSparseTSF(void) {};
                    ~CNeuronSparseTSF(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint sequence, uint variables, uint period, uint forecast, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSparseTSF;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTEMPOOCL   :  public CNeuronBaseOCL
  {
protected:
   //--- constants
   uint              iVariables;
   uint              iSequence;
   uint              iForecast;
   uint              iFFT;
   //--- Trend
   CNeuronPLROCL     cPLR;
   CNeuronBaseOCL    cTrend;
   //--- Seasons
   CNeuronBaseOCL    cInputSeasons;
   CNeuronTransposeOCL cTranspose[2];
   CBufferFloat      cInputFreqRe;
   CBufferFloat      cInputFreqIm;
   CNeuronBaseOCL    cInputFreqComplex;
   CNeuronBaseOCL    cNormFreqComplex;
   CBufferFloat      cMeans;
   CBufferFloat      cVariances;
   CNeuronComplexMLMHAttention cFreqAtteention;
   CNeuronBaseOCL    cUnNormFreqComplex;
   CBufferFloat      cOutputFreqRe;
   CBufferFloat      cOutputFreqIm;
   CNeuronBaseOCL    cOutputTimeSeriasRe;
   CBufferFloat      cOutputTimeSeriasIm;
   CBufferFloat      cZero;
   //--- Noise
   CNeuronBaseOCL    cResidual;
   //--- Forecast
   CNeuronBaseOCL    cConcatInput;
   CNeuronBatchNormOCL cNormalize;
   CNeuronPatching   cPatching;
   CNeuronBatchNormOCL cNormalizePLR;
   CNeuronPatching   cPatchingPLR;
   CNeuronPositionEncoder acPE[2];
   CNeuronMLCrossAttentionMLKV cAttention;
   CNeuronTransposeOCL  cTransposeAtt;
   CNeuronConvOCL    acForecast[2];
   CNeuronTransposeOCL  cTransposeFrc;
   CNeuronRevINDenormOCL cRevIn;
   CNeuronConvOCL    cSum;
   //--- Complex functions
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   virtual bool      ComplexNormalize(void);
   virtual bool      ComplexUnNormalize(void);
   virtual bool      ComplexNormalizeGradient(void);
   virtual bool      ComplexUnNormalizeGradient(void);
   //---
   bool              CutTrendAndOther(CBufferFloat *inputs);
   bool              CutTrendAndOtherGradient(CBufferFloat *inputs_gr);
   bool              CutOneFromAnother(void);
   bool              CutOneFromAnotherGradient(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronTEMPOOCL(void)   {};
                    ~CNeuronTEMPOOCL(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint sequence, uint variables, uint forecast, uint heads, uint layers, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronTEMPOOCL;   }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual CBufferFloat   *getWeights(void)  override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMVMHAttentionMLKV :  public CNeuronMLMHAttentionOCL
  {
protected:
   uint              iLayersToOneKV;                              ///< Number of inner layers
   uint              iHeadsKV;                                    ///< Number of heads
   uint              iVariables;                                  ///< Number of variables
   CCollection       KV_Tensors;                                  ///< The collection of tensors of Keys and Values
   CCollection       K_Tensors;                                   ///< The collection of tensors of Keys
   CCollection       K_Weights;                                   ///< The collection of Matrix of weights to previous layer
   CCollection       V_Tensors;                                   ///< The collection of tensors of Values
   CCollection       V_Weights;                                   ///< The collection of Matrix of weights to previous layer
   CBufferFloat      Temp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out);
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g, CBufferFloat *kv, CBufferFloat *kv_g, CBufferFloat *scores, CBufferFloat *gradient);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMVMHAttentionMLKV(void) {};
                    ~CNeuronMVMHAttentionMLKV(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMVMHAttentionMLKV;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMVCrossAttentionMLKV :  public CNeuronMVMHAttentionMLKV
  {
protected:
   uint              iWindowKV;
   uint              iUnitsKV;
   uint              iVariablesKV;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, CBufferFloat *scores, CBufferFloat *out);
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g, CBufferFloat *kv, CBufferFloat *kv_g, CBufferFloat *scores, CBufferFloat *gradient);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);        ///< Method for updating weights.\details Calling one of kernels ::UpdateWeightsMomentum() or ::UpdateWeightsAdam() in depends of optimization type (#ENUM_OPTIMIZATION).@param NeuronOCL Pointer to previos layer.

public:
                     CNeuronMVCrossAttentionMLKV(void) {};
                    ~CNeuronMVCrossAttentionMLKV(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint window_kv,
                          uint heads_kv, uint units_count, uint units_count_kv,
                          uint layers, uint layers_to_one_kv, uint variables_q, uint variables_kv,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMVCrossMHAttentionMLKV;   }
   //---
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   virtual CBufferFloat* getWeights(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronInjectTST  :  public CNeuronBaseOCL
  {
protected:
   CNeuronPatching            cPatching;
   CNeuronLearnabledPE        cCIPosition;
   CNeuronLearnabledPE        cCMPosition;
   CNeuronMVMHAttentionMLKV   cChanelIndependentAttention;
   CNeuronMLMHAttentionMLKV   cChanelMixAttention;
   CNeuronMVCrossAttentionMLKV   cGlobalInjectionAttention;
   CBufferFloat               cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronInjectTST(void) {};
                    ~CNeuronInjectTST(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronInjectTST; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual CBufferFloat   *getWeights(void)  override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSSMOCL  :  public CNeuronBaseOCL
  {
protected:
   uint              iWindowHidden;
   CNeuronBaseOCL    cHiddenStates;
   CNeuronConvOCL    cA;
   CNeuronConvOCL    cB;
   CNeuronBaseOCL    cAB;
   CNeuronConvOCL    cC;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronSSMOCL(void)  {};
                    ~CNeuronSSMOCL(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSSMOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMambaOCL   :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    cXProject;
   CNeuronConvOCL    cZProject;
   CNeuronConvOCL    cInsideConv;
   CNeuronSSMOCL     cSSM;
   CNeuronBaseOCL    cZSSM;
   CNeuronConvOCL    cOutProject;
   CBufferFloat      Temp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronMambaOCL(void) {};
                    ~CNeuronMambaOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMambaOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMambaBlockOCL :  public   CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   CNeuronMambaOCL   cMamba;
   CNeuronBaseOCL    cMambaResidual;
   CNeuronConvOCL    cFF[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronMambaBlockOCL(void) {};
                    ~CNeuronMambaBlockOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronMambaBlockOCL; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTrajLLMOCL :  public CNeuronBaseOCL
  {
protected:
   //--- State Encoder
   CNeuronLSTMOCL                cStateRNN;
   CNeuronConvOCL                cStateMLP[2];
   //--- Variables Encoder
   CNeuronTransposeOCL           cTranspose;
   CNeuronLSTMOCL                cVariablesRNN;
   CNeuronConvOCL                cVariablesMLP[2];
   //--- Context Encoder
   CNeuronLearnabledPE           cStatePE;
   CNeuronLearnabledPE           cVariablesPE;
   CNeuronMLMHAttentionMLKV      cStateToState;
   CNeuronMLCrossAttentionMLKV   cVariableToState;
   CNeuronMLCrossAttentionMLKV   cStateToVariable;
   CNeuronBaseOCL                cContext;
   CNeuronConvOCL                cContextMLP[2];
   //---
   CNeuronMLMHAttentionMLKV      cHighLevelInteraction;
   CNeuronMambaBlockOCL          caMamba[3];
   CNeuronMLCrossAttentionMLKV   cLaneAware;
   CNeuronConvOCL                caForecastMLP[2];
   CNeuronTransposeOCL           cTransposeOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronTrajLLMOCL(void) {};
                    ~CNeuronTrajLLMOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronTrajLLMOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUniTraj    :  public CNeuronBaseOCL
  {
protected:
   uint              iVariables;
   float             fDropout;
   //---
   CBufferFloat      cHistoryMask;
   CBufferFloat      cFutureMask;
   CNeuronBaseOCL    cData;
   CNeuronLearnabledPE cPE;
   CNeuronMVMHAttentionMLKV   cEncoder;
   CNeuronBaseOCL    cDForw;
   CNeuronBaseOCL    cDBakw;
   CNeuronConvOCL    cProjDForw;
   CNeuronConvOCL    cProjDBakw;
   CNeuronBaseOCL    cDataDForw;
   CNeuronBaseOCL    cDataDBakw;
   CNeuronBaseOCL    cConcatDataDForwBakw;
   CNeuronMambaBlockOCL cSSM[4];
   CNeuronConvOCL    cStat;
   CNeuronTransposeOCL cTranspStat;
   CVAE              cVAE;
   CNeuronTransposeOCL cTranspVAE;
   CNeuronConvOCL    cDecoder[2];
   CNeuronTransposeOCL cTranspResult;
   //---
   virtual bool      Prepare(const CBufferFloat* history, const CBufferFloat* future);
   virtual bool      PrepareGrad(CBufferFloat* history_gr, CBufferFloat* future_gr);
   virtual bool      BTS(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { return feedForward(NeuronOCL, NULL); }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { return calcInputGradients(NeuronOCL, NULL, NULL, None); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return updateInputWeights(NeuronOCL, NULL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   //---

public:
                     CNeuronUniTraj(void) {};
                    ~CNeuronUniTraj(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, float dropout, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronUniTrajOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHiVTAAEncoder : public CNeuronMVMHAttentionMLKV
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHiVTAAEncoder(void) {};
                    ~CNeuronHiVTAAEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHiVTAAEncoder;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTransposeRCDOCL  :  public CNeuronTransposeOCL
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTransposeRCDOCL(void) {};
                    ~CNeuronTransposeRCDOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint count, uint window, uint dimension, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)        const                      {  return defNeuronTransposeRCDOCL; }///< Identificator of class.@return Type of class
   virtual uint      GetDimension(void) const { return Neurons() / (iCount * iWindow); }

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHiVTOCL    :  public CNeuronBaseOCL
  {
protected:
   uint              iHistory;
   uint              iVariables;
   uint              iForecast;
   uint              iNumTraj;
   //---
   CNeuronBaseOCL    cDataTAD;
   CNeuronConvOCL    cEmbeddingTAD;
   CNeuronTransposeRCDOCL cTransposeATD;
   CNeuronHiVTAAEncoder cAAEncoder;
   CNeuronTransposeRCDOCL cTransposeTAD;
   CNeuronLearnabledPE  cPosEmbeddingTAD;
   CNeuronMVMHAttentionMLKV cTemporalEncoder;
   CNeuronLearnabledPE  cPosLineEmbeddingTAD;
   CNeuronPatching   cLineEmbeddibg;
   CNeuronMVCrossAttentionMLKV cALEncoder;
   CNeuronMLMHAttentionMLKV cGlobalEncoder;
   CNeuronTransposeOCL cTransposeADT;
   CNeuronConvOCL    cDecoder[3]; // Agent* Traj * Forecast
   CNeuronConvOCL    cProbProj;
   CNeuronSoftMaxOCL cProbability; // Agent* Traj
   CNeuronBaseOCL    cForecast;
   CNeuronTransposeOCL cTransposeTA;
   //---
   virtual bool      Prepare(const CNeuronBaseOCL *history);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHiVTOCL(void) {};
                    ~CNeuronHiVTOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint forecast, uint num_traj, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHiVTOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPointNetOCL   :  public CNeuronBaseOCL
  {
protected:
   CNeuronPointNetOCL   *cTNet1;
   CNeuronBaseOCL       *cTurned1;
   CNeuronConvOCL       cPreNet[2];
   CNeuronBatchNormOCL  cPreNetNorm[2];
   CNeuronPointNetOCL   *cTNet2;
   CNeuronBaseOCL       *cTurned2;
   CNeuronConvOCL       cFeatureNet[3];
   CNeuronBatchNormOCL  cFeatureNetNorm[3];
   CNeuronTransposeOCL  cTranspose;
   CNeuronProofOCL      cMaxPool;
   CNeuronBaseOCL       cFinalMLP[2];
   //---
   virtual bool         OrthoganalLoss(CNeuronBaseOCL *NeuronOCL, bool add = false);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPointNetOCL(void) {};
                    ~CNeuronPointNetOCL(void);
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint output, bool use_tnets,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronPointNetOCL; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPointNet2Local   :  public CNeuronConvOCL
  {
protected:
   float                fRadius;
   uint                 iUnits;
   //---
   CBufferFloat         cDistance;
   CNeuronConvOCL       cFeatureNet[3];
   CNeuronBatchNormOCL  cFeatureNetNorm[3];
   CNeuronBaseOCL       cLocalMaxPool;
   CNeuronConvOCL       cFinalMLP;
   //---
   virtual bool         CalcDistance(CNeuronBaseOCL *NeuronOCL);
   virtual bool         LocalMaxPool(void);
   virtual bool         LocalMaxPoolGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
public:
                     CNeuronPointNet2Local(void) {};
                    ~CNeuronPointNet2Local(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint window_out, float radius,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronPointNet2LocalOCL; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPointNet2OCL   :  public CNeuronPointNetOCL
  {
protected:
   CNeuronPointNetOCL   *cTNetG;
   CNeuronBaseOCL       *cTurnedG;
   //---
   CNeuronPointNet2Local caLocalPointNet[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
public:
                     CNeuronPointNet2OCL(void) {};
                    ~CNeuronPointNet2OCL(void) ;
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint output, bool use_tnets,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronPointNet2OCL; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPointFormer   :  public CNeuronPointNet2OCL
  {
protected:
   CNeuronMLMHSparseAttention    caLocalAttention[2];
   CNeuronMLCrossAttentionMLKV   caLocalGlobalAttention[2];
   CNeuronMLMHAttentionMLKV      caGlobalAttention[2];
   CNeuronLearnabledPE           caLocalPE[2];
   CNeuronLearnabledPE           caGlobalPE[2];
   CNeuronBaseOCL                cConcatenate;
   CNeuronConvOCL                cScale;
   //---
   CBufferFloat                  *cbTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPointFormer(void) {};
                    ~CNeuronPointFormer(void) { DeleteObj(cbTemp); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint output, bool use_tnets,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronPointFormer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSceneSpecific :  public CNeuronMLCrossAttentionMLKV
  {
protected:
   CNeuronBaseOCL    cOne;
   CNeuronBaseOCL    cSceneSpecificKnowledge;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override { return feedForward(NeuronOCL); }
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override { return calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override { return updateInputWeights(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSceneSpecific(void) {};
                    ~CNeuronSceneSpecific(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint units_count_kv, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronSceneSpecific;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMLMHSceneConditionAttention  :  public CNeuronMLMHAttentionMLKV
  {
protected:
   CLayer            cSceneAgnostic;
   CLayer            cSceneSpecific;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMLMHSceneConditionAttention(void) {};
                    ~CNeuronMLMHSceneConditionAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override  const   {  return defNeuronMLMHSceneConditionAttention;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHyperDet   :  public CNeuronPointNet2OCL
  {
protected:
   CNeuronMLMHSparseAttention    caLocalAttention[2];
   CNeuronMLCrossAttentionMLKV   caLocalGlobalAttention[2];
   CNeuronMLMHSceneConditionAttention  caGlobalAttention[2];
   CNeuronLearnabledPE           caLocalPE[2];
   CNeuronLearnabledPE           caGlobalPE[2];
   CNeuronBaseOCL                cConcatenate;
   CNeuronConvOCL                cScale;
   //---
   CBufferFloat                  *cbTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHyperDet(void) {};
                    ~CNeuronHyperDet(void) { DeleteObj(cbTemp); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint output, bool use_tnets,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronHyperDet; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSEFormer   :    public CNeuronPointNet2OCL
  {
protected:
   uint              iUnits;
   uint              iPoints;
   //---
   CLayer            cQuery;
   CLayer            cKey;
   CLayer            cValue;
   CLayer            cKeyValue;
   CArrayInt         cScores;
   CLayer            cMHAttentionOut;
   CLayer            cAttentionOut;
   CLayer            cResidual;
   CLayer            cFeedForward;
   CLayer            cCenterPoints;
   CLayer            cFinalAttention;
   CNeuronMLCrossAttentionMLKV SEOut;
   CBufferFloat      cbTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      AttentionOut(CBufferFloat *q, CBufferFloat *kv, int scores, CBufferFloat *out);
   virtual bool      AttentionInsideGradients(CBufferFloat *q, CBufferFloat *q_g,
         CBufferFloat *kv, CBufferFloat *kv_g,
         int scores, CBufferFloat *gradient);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   //---

public:
                     CNeuronSEFormer(void) {};
                    ~CNeuronSEFormer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint output, bool use_tnets,
                          uint center_points, uint center_window,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronSEFormer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSPFormer   : public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   uint              iHeads;
   uint              iSPWindow;
   uint              iSPUnits;
   uint              iSPHeads;
   uint              iWindowKey;
   uint              iLayers;
   uint              iLayersSP;
   //---
   CLayer            cSuperPoints;
   CLayer            cQuery;
   CLayer            cSPKeyValue;
   CLayer            cMask;
   CArrayInt         cScores;
   CLayer            cMHCrossAttentionOut;
   CLayer            cCrossAttentionOut;
   CLayer            cResidual;
   CLayer            cQKeyValue;
   CLayer            cMHSelfAttentionOut;
   CLayer            cSelfAttentionOut;
   CLayer            cFeedForward;
   CBufferFloat      cTempSP;
   CBufferFloat      cTempQ;
   CBufferFloat      cTempSelfKV;
   CBufferFloat      cTempCrossKV;
   //---
   virtual bool      CreateBuffers(void);
   virtual bool      AttentionOut(CNeuronBaseOCL *q, CNeuronBaseOCL *kv, const int scores,
                                  CNeuronBaseOCL *out, CNeuronBaseOCL *mask,
                                  const int units,
                                  const int heads,
                                  const int units_kv,
                                  const int heads_kv,
                                  const int dimension,
                                  const float mask_level = 0.5f);
   virtual bool      AttentionInsideGradients(CNeuronBaseOCL *q, CNeuronBaseOCL *kv, const int scores,
         CNeuronBaseOCL *out, CNeuronBaseOCL *mask,
         const int units,
         const int heads,
         const int units_kv,
         const int heads_kv,
         const int dimension,
         const float mask_level = 0.5f);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSPFormer(void) {};
                    ~CNeuronSPFormer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint window_sp, uint units_sp, uint heads_sp,
                          uint layers, uint layers_to_sp,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronSPFormer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMAFT   : public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   uint              iHeads;
   uint              iSPWindow;
   uint              iSPUnits;
   uint              iSPHeads;
   uint              iWindowKey;
   uint              iLayers;
   uint              iLayersSP;
   //---
   CLayer            cSuperPoints;
   CLayer            cQuery;
   CLayer            cQPosition;
   CLayer            cQKey;
   CLayer            cQValue;
   CLayer            cMHSelfAttentionOut;
   CLayer            cSelfAttentionOut;
   CLayer            cSPKey;
   CLayer            cSPValue;
   CArrayInt         cScores;
   CArrayInt         cPositionBias;
   CLayer            cMHCrossAttentionOut;
   CLayer            cCrossAttentionOut;
   CLayer            cResidual;
   CLayer            cFeedForward;
   CBufferFloat      cTempSP;
   CBufferFloat      cTempQ;
   CBufferFloat      cTempCrossK;
   CBufferFloat      cTempCrossV;
   //---
   virtual bool      CreateBuffers(void);
   virtual bool      CalcPositionBias(CBufferFloat *pos_q, CBufferFloat *pos_k, const int pos_bias,
                                      const int units,
                                      const int units_kv,
                                      const int dimension);
   virtual bool      AttentionOut(CNeuronBaseOCL *q, CNeuronBaseOCL *k, CNeuronBaseOCL *v,
                                  const int scores, CNeuronBaseOCL *out, const int pos_bias,
                                  const int units,
                                  const int heads,
                                  const int units_kv,
                                  const int heads_kv,
                                  const int dimension,
                                  const bool use_pos_bias);
   virtual bool      AttentionInsideGradients(CNeuronBaseOCL *q, CNeuronBaseOCL *k, CNeuronBaseOCL *v,
         const int scores, CNeuronBaseOCL *out,
         const int units, const int heads,
         const int units_kv, const int heads_kv,
         const int dimension);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMAFT(void) {};
                    ~CNeuronMAFT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint window_sp, uint units_sp, uint heads_sp,
                          uint layers, uint layers_to_sp,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMAFT; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGRES :  public CNeuronMAFT
  {
protected:
   CLayer            cReference;
   CLayer            cRefKey;
   CLayer            cRefValue;
   CLayer            cMHRefAttentionOut;
   CLayer            cRefAttentionOut;
   //---
   virtual bool      CreateBuffers(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronGRES(void) {};
                    ~CNeuronGRES(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint window_sp, uint units_sp, uint heads_sp,
                          uint ref_size, uint layers, uint layers_to_sp,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronGRES; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGEGWA   :  public CNeuronUShapeAttention
  {
protected:
   CNeuronBaseOCL    cResidual;
   CNeuronMLCrossAttentionMLKV   cCrossAttention;
   CBufferFloat      cTemp;
   bool              bAddNeckGradient;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronGEGWA(void)   :  bAddNeckGradient(false) {};
                    ~CNeuronGEGWA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint units_count,
                          uint window_kv, uint heads_kv, uint units_count_kv,
                          uint layers, uint inside_block, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronGEGWA;   }///< Identificator of class.@return Type of class
   //--- methods for working with files
   virtual bool      Save(int const file_handle);  ///< Save method @param[in] file_handle handle of file @return logical result of operation
   virtual bool      Load(int const file_handle);  ///< Load method @param[in] file_handle handle of file @return logical result of operation
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual CNeuronBaseOCL* GetInsideLayer(const int layer) const;
   virtual void      AddNeckGradient(const bool flag) {  bAddNeckGradient = flag; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLPC  :  public CNeuronMLCrossAttentionMLKV
  {
protected:
   CNeuronBaseOCL    cOne;
   CNeuronBaseOCL    cPrimitives;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override { return feedForward(NeuronOCL); }
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override { return calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override { return updateInputWeights(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronLPC(void) {};
                    ~CNeuronLPC(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint heads_kv, uint units_count, uint units_count_kv, uint layers, uint layers_to_one_kv, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronLPC;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronOCM  :  public CNeuronBaseOCL
  {
protected:
   uint              iPrimWindow;
   uint              iPrimUnits;
   uint              iPrimHeads;
   uint              iContWindow;
   uint              iContUnits;
   uint              iContHeads;
   uint              iWindowKey;
   //---
   CLayer            cQuery;
   CLayer            cKey;
   CLayer            cValue;
   CLayer            cMHAttentionOut;
   CLayer            cAttentionOut;
   CArrayInt         cScores;
   CLayer            cResidual;
   CLayer            cFeedForward;
   //---
   virtual bool      CreateBuffers(void);
   virtual bool      AttentionOut(CNeuronBaseOCL *q, CNeuronBaseOCL *k, CNeuronBaseOCL *v,
                                  const int scores, CNeuronBaseOCL *out,
                                  const int units,
                                  const int heads,
                                  const int units_kv,
                                  const int heads_kv,
                                  const int dimension);
   virtual bool      AttentionInsideGradients(CNeuronBaseOCL *q, CNeuronBaseOCL *k, CNeuronBaseOCL *v,
         const int scores, CNeuronBaseOCL *out,
         const int units, const int heads,
         const int units_kv, const int heads_kv,
         const int dimension);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }

public:
                     CNeuronOCM(void) {};
                    ~CNeuronOCM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint prim_window, uint window_key, uint prim_units, uint prim_heads,
                          uint cont_window, uint cont_units, uint cont_heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronOCM; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *Primitives, CNeuronBaseOCL *Context);
   virtual bool      calcInputGradients(CNeuronBaseOCL *Primitives, CNeuronBaseOCL *Context);
   virtual bool      updateInputWeights(CNeuronBaseOCL *Primitives, CNeuronBaseOCL *Context);
   //---
   virtual uint      GetPrimitiveWindow(void)   const { return iPrimWindow; }
   virtual uint      GetContextWindow(void)     const { return iContWindow; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRefMask    :  public CNeuronBaseOCL
  {
protected:
   CNeuronGEGWA      cGEGWA;
   CLayer            cContentEncoder;
   CLayer            cBackGround;
   CNeuronLPC        cLPC;
   CLayer            cDecoder;
   CNeuronOCM        cOCM;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronRefMask(void) {};
                    ~CNeuronRefMask(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint content_size, uint content_units,
                          uint primitive_units, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRefMask; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRelativeSelfAttention   :  public CNeuronBaseOCL
  {
protected:
   uint                    iWindow;
   uint                    iWindowKey;
   uint                    iHeads;
   uint                    iUnits;
   int                     iScore;
   //---
   CNeuronConvOCL       cQuery;
   CNeuronConvOCL       cKey;
   CNeuronConvOCL       cValue;
   CNeuronTransposeOCL     cTranspose;
   CNeuronBaseOCL          cDistance;
   CLayer                  cBKey;
   CLayer                  cBValue;
   CLayer                  cGlobalContentBias;
   CLayer                  cGlobalPositionalBias;
   CLayer                  cMHAttentionPooling;
   CLayer                  cScale;
   CBufferFloat            cTemp;
   //---
   virtual bool      AttentionOut(void);
   virtual bool      AttentionGradient(void);

   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronRelativeSelfAttention(void) : iScore(-1) {};
                    ~CNeuronRelativeSelfAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRelativeSelfAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetWindow(void) const { return iWindow; }
   virtual uint      GetUnits(void) const { return iUnits; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRMAT :  public CNeuronBaseOCL
  {
protected:
   CLayer            cLayers;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronRMAT(void) {};
                    ~CNeuronRMAT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRMAT; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override;
   //---
   virtual uint      GetWindow(void) const
     {
      CNeuronRelativeSelfAttention* neuron = cLayers[0];
      return (!neuron ? 0 : neuron.GetWindow());
     }
   virtual uint      GetUnits(void) const
     {
      CNeuronRelativeSelfAttention* neuron = cLayers[0];
      return (!neuron ? 0 : neuron.GetUnits());
     }
   virtual void      TrainMode(bool flag) override
     {
      bTrain = flag;
      CNeuronBaseOCL *neuron = NULL;
      for(int i = 0; i < cLayers.Total(); i++)
        {
         neuron = cLayers[i];
         if(!!neuron)
            neuron.TrainMode(bTrain);
        }
     }
   //---
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true)
     {
      CNeuronBaseOCL *last = cLayers[-1];
      if(!last ||
         !last.SetGradient(buffer, delete_prev) ||
         !CNeuronBaseOCL::SetGradient(last.getGradient(), delete_prev))
         ReturnFalse;
      return true;
     }
   virtual bool      SetOutput(CBufferFloat *buffer, bool delete_prev = true)
     {
      CNeuronBaseOCL *last = cLayers[-1];
      if(!last ||
         !last.SetOutput(buffer, delete_prev) ||
         !CNeuronBaseOCL::SetOutput(last.getOutput(), delete_prev))
         ReturnFalse;
      return true;
     }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHAttentionPooling  :  public   CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iHeads;
   uint              iUnits;
   CLayer            cNeurons;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHAttentionPooling(void) {};
                    ~CNeuronMHAttentionPooling(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMHAttentionPooling; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMotifs    :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    cMotifs;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMotifs(void) {};
                    ~CNeuronMotifs(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint window, uint step, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMotifs; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleAttention :  public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   //---
   CNeuronBaseOCL    cWideInputs;
   CNeuronRelativeSelfAttention  cAttentions[4];
   CNeuronBaseOCL    cConcatAttentions;
   CNeuronMHAttentionPooling cPooling;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultiScaleAttention(void) {};
                    ~CNeuronMultiScaleAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMultiScaleAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMolformer  :  public CNeuronRMAT
  {
public:
                     CNeuronMolformer(void) {};
                    ~CNeuronMolformer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads, uint layers,
                          uint motif_window, uint motif_step,
                          ENUM_OPTIMIZATION optimization_type, uint batch); //Molformer
   //---
   virtual int       Type(void) override   const   {  return defNeuronMolformer; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRelativeCrossAttention   :  public CNeuronRelativeSelfAttention
  {
protected:
   uint                    iUnitsKV;
   //---
   CLayer                  cKVProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronRelativeCrossAttention(void) {};
                    ~CNeuronRelativeCrossAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          uint window_kv, uint units_kv,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRelativeCrossAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetUnitsKV(void) const { return iUnitsKV; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMotifEncoder  :  public CNeuronRMAT
  {
public:
                     CNeuronMotifEncoder(void) {};
                    ~CNeuronMotifEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronMotifEncoder; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPropertyAwareAttention    :  public CNeuronRMAT
  {
protected:
   CBufferFloat      cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPropertyAwareAttention(void) {};
                    ~CNeuronPropertyAwareAttention(void) {};

   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint properties,
                          uint units_count, uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronPropertyAwareAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAMCT    :  public CNeuronBaseOCL
  {
protected:
   CNeuronRMAT                   cAtomEncoder;
   CNeuronMotifEncoder           cMotifEncoder;
   CLayer                        cMotifProjection;
   CNeuronPropertyAwareAttention cPropertyDecoder;
   CLayer                        cPropertyProjection;
   CNeuronBaseOCL                cConcatenate;
   CNeuronMHAttentionPooling     cPooling;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAMCT(void) {};
                    ~CNeuronAMCT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint properties,
                          uint units_count, uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronAMCT; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronNAFS :  public CNeuronBaseOCL
  {
protected:
   uint                 iDimension;
   uint                 iSmoothing;
   uint                 iUnits;
   //---
   CNeuronBaseOCL       cFeatureSmoothing;
   CNeuronTransposeOCL  cTranspose;
   CNeuronBaseOCL       cDistance;
   CNeuronSoftMaxOCL    cAdaptation;
   //---
   virtual bool      FeatureSmoothing(const CNeuronBaseOCL *neuron, const CNeuronBaseOCL *smoothing);
   virtual bool      FeatureSmoothingGradient(const CNeuronBaseOCL *neuron, const CNeuronBaseOCL *smoothing);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }

public:
                     CNeuronNAFS(void) {};
                    ~CNeuronNAFS(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronNAFS; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronBatchNormWithNoise  :  public CNeuronBatchNormOCL
  {
protected:
   CBufferFloat      cNoise;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronBatchNormWithNoise(void) {};
                    ~CNeuronBatchNormWithNoise(void) {};
   //---
   virtual int       Type(void)        const                      {  return defNeuronBatchNormWithNoise;    }///< Identificator of class.@return Type of class
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDiffusion  : public CNeuronUShapeAttention
  {
protected:
   CNeuronBatchNormWithNoise  cAddNoise;
   CNeuronBaseOCL             cResidual;
   CNeuronRevINDenormOCL      cRevIn;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronDiffusion(void) {};
                    ~CNeuronDiffusion(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint layers, uint inside_block, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronDiffusion;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag)                         {  bTrain = flag; cAddNoise.TrainMode(bTrain); } ///< Set Training Mode Flag
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHyperProjection   :  public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   { return true; }

public:
                     CNeuronHyperProjection(void) : iWindow(-1), iUnits(-1) {};
                    ~CNeuronHyperProjection(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHyperProjection;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHyperboloids  :  public CNeuronBaseOCL
  {
protected:
   uint              iWindows;
   uint              iUnits;
   uint              iCentroids;
   //---
   CLayer            cHyperCentroids;
   CLayer            cHyperCurvatures;
   //---
   int               iProducts;
   int               iDistances;
   int               iNormes;
   //---
   virtual bool      LogMap(CNeuronBaseOCL *featers, CNeuronBaseOCL *centroids, CNeuronBaseOCL *curvatures, CNeuronBaseOCL *outputs);
   virtual bool      LogMapGrad(CNeuronBaseOCL *featers, CNeuronBaseOCL *centroids, CNeuronBaseOCL *curvatures, CNeuronBaseOCL *outputs);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronHyperboloids(void) : iWindows(0), iUnits(0), iCentroids(0), iProducts(-1), iDistances(-1), iNormes(-1) {};
                    ~CNeuronHyperboloids(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint units_count, uint centroids, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHyperboloids;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHypDiff :  public CNeuronRMAT
  {
public:
                     CNeuronHypDiff(void) {};
                    ~CNeuronHypDiff(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint centroids,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHypDiff; }
   //---
   virtual uint      GetWindow(void) const override
     {
      CNeuronRMAT* neuron = cLayers[1];
      return (!neuron ? 0 : neuron.GetWindow() - 1);
     }
   virtual uint      GetUnits(void) const override
     {
      CNeuronRMAT* neuron = cLayers[1];
      return (!neuron ? 0 : neuron.GetUnits());
     }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronBaseSAMOCL  :  public CNeuronBaseOCL
  {
protected:
   float             fRho;
   CBufferFloat      cWeightsSAM;
   //---
   virtual bool      calcEpsilonWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForwardSAM(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcHiddenGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronBaseSAMOCL(void) {};
                    ~CNeuronBaseSAMOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, float rho, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle)   override;
   virtual bool      Load(int const file_handle)   override;
   //---
   virtual int       Type(void)        const override          {  return defNeuronBaseSAMOCL;      }
   virtual int       Activation(void)  const override          {  return (fRho == 0 ? (int)None : (int)activation);   }
   virtual int       getWeightsSAMIndex(void) const override   {  return cWeightsSAM.GetIndex();   }
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPSBlock :  public CNeuronConvSAMOCL
  {
protected:
   CNeuronConvSAMOCL acConvolution[2];
   CNeuronBaseOCL    cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPSBlock(void) {};
                    ~CNeuronPSBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_out, uint units_count, uint variables, float rho, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      InitPS(CNeuronPSBlock *master);
   //---
   virtual int       Type(void)   const override   {  return defNeuronPSBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPSformer   :  public CNeuronBaseSAMOCL
  {
protected:
   CNeuronTransposeOCL           acTranspose[2];
   CNeuronPSBlock                acPSBlocks[3];
   CNeuronRelativeSelfAttention  acAttention[2];
   CNeuronBaseOCL                cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPSformer(void)   {};
                    ~CNeuronPSformer(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint segments, float rho,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const         {  return defNeuronPSformer;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMarketObserver   :  public CNeuronRMAT
  {
public:
                     CNeuronMarketObserver(void)   {};
                    ~CNeuronMarketObserver(void)   {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint forecast,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMarketObserver; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRLAgent :  public   CNeuronRMAT
  {
public:
                     CNeuronRLAgent(void) {};
                    ~CNeuronRLAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint segments, float rho,
                          uint layers, uint n_actions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   override {  return defNeuronRLAgent;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronControlAgent  :  public CNeuronRMAT
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronControlAgent(void) {};
                    ~CNeuronControlAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint window_kv, uint units_kv, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronControlAgent; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMASA    :  public CNeuronBaseSAMOCL
  {
protected:
   CNeuronMarketObserver   cMarketObserver;
   CNeuronRevINDenormOCL   cRevIN;
   CNeuronRLAgent          cRLAgent;
   CNeuronControlAgent     cControlAgent;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMASA(void) {};
                    ~CNeuronMASA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers_mo, uint forecast,
                          uint segments_rl, float rho,
                          uint layers_rl, uint n_actions,
                          uint heads_contr,
                          uint layers_contr,
                          int NormLayer, CNeuronBatchNormOCL *normLayer,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMASA; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override;
   //---
   virtual int       GetNormLayer(void)   {  return cRevIN.GetNormLayer(); }
   virtual bool      SetNormLayer(int NormLayer, CNeuronBatchNormOCL *normLayer);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPLRMultiAgentsOCL  :  public CNeuronPLROCL
  {
protected:
   int               iAgents;
   CBufferFloat      cMinDistance;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronPLRMultiAgentsOCL(void)  : iAgents(1) {};
                    ~CNeuronPLRMultiAgentsOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint units_count, bool transpose,
                          vector<float> &min_distance,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronPLRMultiAgentsOCL;   }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCrossSectionalAnalysis :  public CNeuronMVMHAttentionMLKV
  {
protected:
   CNeuronConvOCL          cEmbeding;
   CNeuronTransposeRCDOCL  cTransposeRCD;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCrossSectionalAnalysis(void) {};
                    ~CNeuronCrossSectionalAnalysis(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint heads_kv,
                          uint units_count, uint layers, uint layers_to_one_kv,
                          uint variables, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void)   const override   {  return defNeuronCrossSectionalAnalysis;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTransposeVRCOCL  :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL     cTranspose;
   CNeuronTransposeRCDOCL  cTransposeRCD;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;

public:
                     CNeuronTransposeVRCOCL(void) {};
                    ~CNeuronTransposeVRCOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint variables, uint count, uint window, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)  const override    {  return defNeuronTransposeVRCOCL; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetWindow(void) override const { return cTransposeRCD.GetWindow(); }
   virtual uint      GetCount(void) override const { return cTransposeRCD.GetCount(); }
   virtual uint      GetVariables(void) const { return iWindow; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTemporalAnalysis :  public CNeuronCrossSectionalAnalysis
  {
protected:
   CNeuronTransposeVRCOCL  cTranspose;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTemporalAnalysis(void) {};
                    ~CNeuronTemporalAnalysis(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint heads_kv,
                          uint units_count, uint layers, uint layers_to_one_kv,
                          uint variables, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void)   const override   {  return defNeuronTemporalAnalysis;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPortfolioGenerator  :  public CNeuronBaseOCL
  {
protected:
   uint              iAssets;
   uint              iTimePoints;
   uint              iAgents;
   uint              iDimension;
   //---
   CNeuronBaseOCL          cAssetTime[2];
   CNeuronTransposeVRCOCL  cTransposeVRC;
   CNeuronSoftMaxOCL       cSoftMax;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPortfolioGenerator(void) {};
                    ~CNeuronPortfolioGenerator(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint assets, uint time_points, uint dimension,
                          uint agents,  uint projection,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronPortfolioGenerator;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMASAAT  :  public CNeuronPortfolioGenerator
  {
protected:
   CNeuronTransposeOCL              cTranspose;
   CNeuronPLRMultiAgentsOCL         cPLR;
   CNeuronBaseOCL                   cConcat;
   CNeuronCrossSectionalAnalysis    cCrossSectionalAnalysis;
   CNeuronTemporalAnalysis          cTemporalAnalysis;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)  override
     { return        feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMASAAT(void)  {};
                    ~CNeuronMASAAT(void)  {};
   //---
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint units_cout,
                          uint layers, vector<float> &min_distance, uint projection,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronMASAAT;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHConvOCL  :  public CNeuronConvOCL
  {
protected:
   uint              iHeads;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHConvOCL(void)  :  iHeads(1)   {};
                    ~CNeuronMHConvOCL(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint window_out, uint units_count, uint variables, uint heads, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronMHConvOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHFeedForward   :  public CNeuronBaseOCL
  {
protected:
   CNeuronMHConvOCL  acConvolutions[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHFeedForward(void) {};
                    ~CNeuronMHFeedForward(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out,
                          uint units_count, uint variables, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronMHFeedForward;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true)
     {
      CNeuronBaseOCL *last = GetPointer(acConvolutions[acConvolutions.Size() - 1]);
      if(!last ||
         !last.SetGradient(buffer, delete_prev) ||
         !CNeuronBaseOCL::SetGradient(last.getGradient(), delete_prev))
         ReturnFalse;
      return true;
     }
   virtual bool      SetOutput(CBufferFloat *buffer, bool delete_prev = true)
     {
      CNeuronBaseOCL *last = GetPointer(acConvolutions[acConvolutions.Size() - 1]);
      if(!last ||
         !last.SetOutput(buffer, delete_prev) ||
         !CNeuronBaseOCL::SetOutput(last.getOutput(), delete_prev))
         ReturnFalse;
      return true;
     }
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDMHAttention  :  public CNeuronRMAT
  {
public:
                     CNeuronDMHAttention(void) {};
                    ~CNeuronDMHAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual int       Type(void) override   const   {  return defNeuronDMHAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCrossDMHAttention   :  public CNeuronRMAT
  {
protected:
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronCrossDMHAttention(void)  {};
                    ~CNeuronCrossDMHAttention(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint window_cross, uint units_cross,
                          uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronCrossDMHAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLegendreWaveletsHL :  public CNeuronConvOCL
  {
protected:
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   { return true; }        ///< Method for updating weights.@param NeuronOCL Pointer to previos layer.

public:
                     CNeuronLegendreWaveletsHL(void) {};
                    ~CNeuronLegendreWaveletsHL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint step, uint units_count, uint filters, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void)   const   {  return defNeuronLegendreWaveletsHL;   }///< Identificator of class.@return Type of class
   //---
   virtual uint      GetFilters(void)  const {return (iWindowOut / 2); }
   virtual uint      GetVariables(void)  const {return (iVariables); }
   virtual uint      GetUnits(void)  const {return (Neurons() / (iVariables * iWindowOut)); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDecouplingFlow   :  public   CNeuronLegendreWaveletsHL
  {
protected:
   CNeuronTransposeOCL  cTranspose;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronDecouplingFlow(void) {};
                    ~CNeuronDecouplingFlow(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint units_count,
                          uint filters, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void)   const   {  return defNeuronDecouplingFlow;   }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDilatedCasualConv   :  public CNeuronRMAT
  {
public:
                     CNeuronDilatedCasualConv(void) {};
                    ~CNeuronDilatedCasualConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint dimension,
                          uint units_count, uint variables, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronDilatedCasualConv; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultitaskStockformer   :  public CNeuronBaseOCL
  {
protected:
   CNeuronDecouplingFlow      cDecouplingFlow;
   CNeuronBaseOCL             cLowFreqSignal;
   CNeuronBaseOCL             cHighFreqSignal;
   CNeuronRMAT                cTemporalAttention;
   CNeuronDilatedCasualConv   cDilatedCasualConvolution;
   CNeuronLearnabledPE        cLowFreqPE;
   CNeuronLearnabledPE        cHighFreqPE;
   CNeuronNAFS                cLowFreqGraphAttention;
   CNeuronNAFS                cHighFreqGraphAttention;
   CNeuronDMHAttention        cLowFreqFusionDecoder;
   CNeuronCrossDMHAttention   cLowHighFreqFusionDecoder;
   CNeuronBaseOCL             cLowHigh;
   CNeuronConvOCL             cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultitaskStockformer(void) {};
                    ~CNeuronMultitaskStockformer(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint neurons_out, uint filters,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMultitaskStockformer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMemory  :  public CNeuronRelativeCrossAttention
  {
protected:
   CNeuronLSTMOCL    cLSTM;
   CNeuronMambaOCL   cMamba;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override  { return feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        updateInputWeights(NeuronOCL); }

public:
                     CNeuronMemory(void) {};
                    ~CNeuronMemory(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronMemory; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFinMem   :  public CNeuronRelativeCrossAttention
  {
protected:
   CNeuronTransposeOCL  cTransposeState;
   CNeuronMemory        cMemory[2];
   CNeuronRelativeCrossAttention cCrossMemory;
   CNeuronRelativeCrossAttention cMemoryToAccount;
   CNeuronRelativeCrossAttention cActionToAccount;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override   { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override   { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override   { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronFinMem(void) {};
                    ~CNeuronFinMem(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint account_descr, uint nactions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronFinMem; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLowLevelReflection :   public CNeuronMemory
  {
protected:
   CNeuronLSTMOCL    cChangeLSTM;
   CNeuronMambaOCL   cChangeMamba;
   CNeuronRelativeCrossAttention cCrossAttention[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronLowLevelReflection(void) {};
                    ~CNeuronLowLevelReflection(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronLowLevelReflection; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHighLevelReflection :  public CNeuronMemory
  {
protected:
   CNeuronBaseOCL    cAccount;
   CNeuronLSTMOCL    cHistoryAccount;
   CNeuronRelativeCrossAttention cActionReason;
   CNeuronLSTMOCL    cHistoryActions;
   CNeuronRelativeCrossAttention cActionResult;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronHighLevelReflection(void) {};
                    ~CNeuronHighLevelReflection(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint desc_account, uint actions_state,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronHighLevelReflection; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMoreLessEqual :  public CNeuronBaseOCL
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override {return true; }

public:
                     CNeuronMoreLessEqual(void) {};
                    ~CNeuronMoreLessEqual(void) {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFinAgent   :  public CNeuronRelativeCrossAttention
  {
protected:
   CNeuronTransposeOCL  cTransposeState;
   CNeuronLowLevelReflection  cLowLevelReflection[2];
   CNeuronHighLevelReflection cHighLevelReflection;
   CNeuronMoreLessEqual cTools;
   CNeuronPSformer      cMarketIntelligence;
   CNeuronMemory        cMarketMemory;
   CNeuronRelativeCrossAttention cCrossLowLevel;
   CNeuronRelativeCrossAttention cMarketToLowLevel;
   CNeuronRelativeCrossAttention cMarketToTools;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronFinAgent(void) {};
                    ~CNeuronFinAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint account_descr, uint nactions, uint segments,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronFinAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMemoryDistil  :  public CNeuronMemory
  {
protected:
   CNeuronBaseOCL       cConcatenated;
   CNeuronTransposeOCL  cTransposeConc;
   CNeuronConvOCL       cConvolution;
   CNeuronEmbeddingOCL  cDistilMemory;
   CNeuronRelativeCrossAttention cCrossAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMemoryDistil(void) {};
                    ~CNeuronMemoryDistil(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMemoryDistil; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFinConAgent   :  public CNeuronRelativeCrossAttention
  {
protected:
   CNeuronMemoryDistil  cStatesMemory;
   CNeuronMemoryDistil  cActionsMemory;
   CNeuronBaseOCL       caRole[2];
   CNeuronRelativeCrossAttention cStateToRole;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override  { return feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        updateInputWeights(NeuronOCL); }

public:
                     CNeuronFinConAgent(void) {};
                    ~CNeuronFinConAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint stack_size, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronFinConAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFinConManager   :  public CNeuronFinConAgent
  {
protected:
   CNeuronTransposeOCL  cTransposeState;
   CNeuronFinConAgent   caAgents[3];
   CNeuronFinConAgent   caTrAgents[3];
   CNeuronFinConAgent   cRiskAgent;
   CNeuronBaseOCL       cConcatenatedAgents;
   CNeuronBaseOCL       cAccount;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override  {ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronFinConManager(void) {};
                    ~CNeuronFinConManager(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint stack_size, uint account_descr, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronFinConManager; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
   virtual void      TrainMode(bool flag);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMacroHFTHyperAgent  :  public CNeuronSoftMaxOCL
  {
protected:
   CNeuronMemoryDistil  cMemory;
   CNeuronRMAT          cStatePrepare;
   CNeuronTransposeOCL  cTranspose;
   CNeuronConvOCL       cScale;
   CNeuronBaseOCL       cMLP[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMacroHFTHyperAgent(void) {};
                    ~CNeuronMacroHFTHyperAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint agents, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMacroHFTHyperAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMacroHFT   :  public CNeuronBaseOCL
  {
protected:
   CNeuronTransposeOCL  cTranspose;
   CNeuronFinConAgent   caAgents[6];
   CNeuronMacroHFTHyperAgent  cHyperAgent;
   CNeuronBaseOCL       cConcatenated;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMacroHFT(void) {};
                    ~CNeuronMacroHFT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint stack_size, uint nactions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMacroHFT; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMacroHFTvsRiskManager  :  public CResidualConv
  {
protected:
   CNeuronBaseOCL       caAccountProjection[2];
   CNeuronMemoryDistil  cMemoryAccount;
   CNeuronMemoryDistil  cMemoryAction;
   CNeuronRelativeCrossAttention cCrossAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronMacroHFTvsRiskManager(void) {};
                    ~CNeuronMacroHFTvsRiskManager(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint stack_size, uint nactions, uint account_decr,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMacroHFTvsRiskManager; }
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleRelativeSelfAttention   :  public CNeuronRelativeSelfAttention
  {
protected:
   //---
   virtual bool      AttentionOut(void);

public:
                     CNeuronMultiScaleRelativeSelfAttention(void) {};
                    ~CNeuronMultiScaleRelativeSelfAttention(void) {};
   //---
   virtual int       Type(void) override   const   {  return defNeuronMultiScaleRelativeSelfAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleRelativeCrossAttention   :  public CNeuronRelativeCrossAttention
  {
protected:
   virtual bool      AttentionOut(void);

public:
                     CNeuronMultiScaleRelativeCrossAttention(void) {};
                    ~CNeuronMultiScaleRelativeCrossAttention(void) {};
   //---
   virtual int       Type(void) override   const   {  return defNeuronMultiScaleRelativeCrossAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRecursiveAttention  :  public CNeuronMultiScaleRelativeCrossAttention
  {
protected:
   CNeuronMultiScaleRelativeSelfAttention cSelfAttention;
   CNeuronTransposeOCL  cTransposeSA;
   CNeuronConvOCL       cConvolution;
   CNeuronEmbeddingOCL  cHistory;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
   override  { return feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None)
   override  { return calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
   override  { return updateInputWeights(NeuronOCL); }

public:
                     CNeuronRecursiveAttention(void) {};
                    ~CNeuronRecursiveAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint history_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRecursiveAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronLinerAttention   :  public CNeuronBaseOCL
  {
protected:
   uint                 iWindow;
   uint                 iWindowKey;
   uint                 iUnits;
   uint                 iVariables;
   //---
   CNeuronConvOCL       cQuery;
   CNeuronConvOCL       cKey;
   CNeuronTransposeVRCOCL  cKeyT;
   CNeuronBaseOCL       cKeyValue;
   CNeuronBaseOCL       cAttentionOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronLinerAttention(void) {};
                    ~CNeuronLinerAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronLinerAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHidformerTSAgent    :  public CResidualConv
  {
protected:
   CNeuronBaseOCL                caRole[2];
   CNeuronRelativeCrossAttention cStateToRole;
   CNeuronS3                     cShuffle;
   CNeuronRecursiveAttention     cRecursiveState;
   CResidualConv                 cResidualState;
   CNeuronRecursiveAttention     cRecursiveAction;
   CNeuronMultiScaleRelativeCrossAttention   cActionToState;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformerTSAgent(void) {};
                    ~CNeuronHidformerTSAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint stack_size, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformerTSAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHidformerFreqAgent    :  public CResidualConv
  {
protected:
   CNeuronTransposeOCL           cTranspose;
   CNeuronLegendreWaveletsHL     cLegendre;
   CNeuronTransposeRCDOCL        cHLState;
   CNeuronLinerAttention         cAttentionState;
   CResidualConv                 cResidualState;
   CNeuronS3                     cShuffle;
   CNeuronRecursiveAttention     cRecursiveAction;
   CNeuronMultiScaleRelativeCrossAttention cActionToState;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformerFreqAgent(void) {};
                    ~CNeuronHidformerFreqAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint filters, uint units_count,
                          uint heads, uint stack_size, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformerFreqAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHidformer  :  public CNeuronBaseOCL
  {
protected:
   CNeuronTransposeOCL  cTranspose;
   CNeuronHidformerTSAgent caTSAgents[4];
   CNeuronHidformerFreqAgent  caFreqAgents[2];
   CNeuronMacroHFTHyperAgent  cHyperAgent;
   CNeuronBaseOCL       cConcatenated;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformer(void) {};
                    ~CNeuronHidformer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint stack_size, uint nactions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResNeXtBottleneck :  public CNeuronConvOCL
  {
protected:
   CNeuronConvOCL          cProjectionIn;
   CNeuronBatchNormOCL     cNormalizeIn;
   CNeuronTransposeRCDOCL  cTransposeIn;
   CNeuronConvOCL          cFeatureExtraction;
   CNeuronBatchNormOCL     cNormalizeFeature;
   CNeuronTransposeRCDOCL  cTransposeOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;


public:
                     CNeuronResNeXtBottleneck(void) {};
                    ~CNeuronResNeXtBottleneck(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint window,
                          uint step, uint units_count, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtBottleneck;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResNeXtResidual:  public CNeuronConvOCL
  {
protected:
   CNeuronTransposeOCL     cTransposeIn;
   CNeuronConvOCL          cProjectionTime;
   CNeuronBatchNormOCL     cNormalizeTime;
   CNeuronTransposeOCL     cTransposeOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResNeXtResidual(void) {};
                    ~CNeuronResNeXtResidual(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtResidual;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResNeXtBlock :  public CNeuronBaseOCL
  {
protected:
   uint                       iChanelsOut;
   CNeuronResNeXtBottleneck   cBottleneck;
   CNeuronResNeXtResidual     cResidual;
   CBufferFloat               cBuffer;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResNeXtBlock(void) {};
                    ~CNeuronResNeXtBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint window,
                          uint step, uint units_count, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuron2DSSMOCL  :  public CNeuronBaseOCL
  {
protected:
   uint                 iWindowOut;
   uint                 iUnitsOut;
   CNeuronBaseOCL       cHiddenStates;
   CLayer               cProjectionX_Time;
   CLayer               cProjectionX_Variable;
   CNeuronConvOCL       cA;
   CNeuronConvOCL       cB_Time;
   CNeuronConvOCL       cB_Variable;
   CNeuronConvOCL       cC_Time;
   CNeuronConvOCL       cC_Variable;
   CNeuronConvOCL       cDelta_Time;
   CNeuronConvOCL       cDelta_Variable;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForwardSSM2D(void);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradientsSSM2D(void);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuron2DSSMOCL(void)  {};
                    ~CNeuron2DSSMOCL(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint window_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuron2DSSMOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronChimera    :  public CNeuronBaseOCL
  {
protected:
   CNeuron2DSSMOCL    caSSM[2];
   CNeuronConvOCL     cDiscretization;
   CLayer             cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronChimera(void) {};
                    ~CNeuronChimera(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint window_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronChimera; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMoT  :  public CNeuronMHAttentionPooling
  {
protected:
   CNeuronConvOCL             cNodesTokenizer;
   CNeuronConvOCL             cEdgesTokenizer;
   CNeuronConvOCL             cSubGraphsTokenizer;
   CLayer                     cUnitarSubGraphsTokenizer;
   CNeuronBaseOCL             cConcatenate;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMoT(void) {};
                    ~CNeuronMoT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMoT; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronChimeraPlus    :  public CNeuronChimera
  {
protected:
   CNeuron2DSSMOCL    cSSMPlus;
   CLayer             cDiscretizationPlus;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronChimeraPlus(void) {};
                    ~CNeuronChimeraPlus(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint window_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronChimeraPlus; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHypridDecoder :  public CNeuronHidformer
  {
protected:
   CNeuronChimeraPlus   cChimera;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHypridDecoder(void) {};
                    ~CNeuronHypridDecoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint stack_size, uint nactions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHypridDecoder; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAttraos :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cOne;
   CNeuronBaseOCL    cX_norm;
   CNeuronConvOCL    cA;
   CNeuronConvOCL    cX_proj;
   CNeuronBaseOCL    cDelta;
   CNeuronBaseOCL    cB;
   CNeuronBaseOCL    cC;
   CNeuronConvOCL    cD;
   CNeuronBaseOCL    cH;
   CNeuronConvOCL    cDelta_proj;
   CNeuronBaseOCL    cDeltaA;
   CNeuronBaseOCL    cDeltaB;
   CNeuronBaseOCL    cDeltaBX;
   CNeuronBaseOCL    cDeltaH;
   CNeuronBaseOCL    cHS;
   //---
   virtual bool      PScan(void);
   virtual bool      PScanCalcGradient(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAttraos(void) {};
                    ~CNeuronAttraos(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronAttraos; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool       WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTopKGates  :  public CNeuronSoftMaxOCL
  {
protected:
   int               iK;
   CBufferFloat      cbNoise;
   CNeuronConvOCL    cProjection;
   CNeuronBaseOCL    cGates;
   //---
   virtual bool      TopKgates(void);
   virtual bool      TopKgatesGradient(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTopKGates(void) {};
                    ~CNeuronTopKGates(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint gates, uint top_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronTopKGates; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetGates(void) const { return cProjection.GetFilters() / 2; }
   virtual uint      GetUnits(void) const { return cProjection.GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMoE  :  public CNeuronBaseOCL
  {
protected:
   CNeuronTopKGates     cGates;
   CNeuronConvOCL       cXProj;
   CLayer               cExperts;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMoE(void) {};
                    ~CNeuronMoE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint units_count,
                          uint experts, uint top_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronMoE; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual void      TrainMode(bool flag)
     {  bTrain = flag;  cGates.TrainMode(bTrain); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronChanelMask :  public CNeuronBaseOCL
  {
   //---
protected:
   uint              iUnits;
   uint              iFFTdimension;
   CBufferFloat      cbFFTReal;
   CBufferFloat      cbFFTImag;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   virtual bool      Mask(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) { return true; }

public:
                     CNeuronChanelMask(void) {};
                    ~CNeuronChanelMask(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronChanelMask; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool       WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDUET :  public CNeuronTransposeOCL
  {
protected:
   uint                       iWindowKey;
   uint                       iHeads;
   //---
   CNeuronTransposeOCL        cTranspose;
   CNeuronMoE                 cExperts;
   CNeuronConvOCL             cQKV;
   CNeuronBaseOCL             cQ;
   CNeuronBaseOCL             cKV;
   CNeuronChanelMask          cMask;
   CBufferFloat               cbScores;
   CNeuronBaseOCL             cMHAttentionOut;
   CNeuronConvOCL             cPooling;
   CNeuronBaseOCL             cResidual;
   CNeuronMHFeedForward       cFeedForward;
   //---
   virtual bool      AttentionOut(void);
   virtual bool      AttentionInsideGradients(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CNeuronDUET(void) {};
                    ~CNeuronDUET(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint units_out, uint experts, uint top_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronDUET; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   //virtual bool       WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual void      TrainMode(bool flag)
     {  bTrain = flag; cExperts.TrainMode(bTrain); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiWindowsConvOCL    :  public CNeuronConvOCL
  {
protected:
   int               aiWindows[];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronMultiWindowsConvOCL(void) {  activation = SoftPlus;  iWindow = -1; }
                    ~CNeuronMultiWindowsConvOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &windows[], uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMultiWindowsConvOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual int       GetWindow(uint index)
     {
      if(index >= aiWindows.Size())
         return 0;
      return aiWindows[index];
     }
   virtual uint      GetUnits(void) const override { return CNeuronConvOCL::GetUnits() / aiWindows.Size(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAdaBN   :  public CNeuronMoE
  {
public:
                     CNeuronAdaBN(void) {};
                    ~CNeuronAdaBN(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint units_count,
                          uint &bottlenecks[], uint top_k, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronAdaBN; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronComplexConvOCL    :  public CNeuronConvOCL
  {
protected:
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronComplexConvOCL(void)   {  activation = None;   }
                    ~CNeuronComplexConvOCL(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void)   const   {  return defNeuronComplexConvOCL;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronComplexMVMHMaskAttention :  public CNeuronBaseOCL
  {
protected:
   uint                       iWindow;
   uint                       iWindowKey;
   uint                       iHeads;
   uint                       iUnits;
   uint                       iVariables;
   //---
   CNeuronComplexConvOCL      cQKV;
   CNeuronBaseOCL             cQ;
   CNeuronBaseOCL             cKV;
   CNeuronConvOCL             cMask;
   CNeuronBaseOCL             cMHAttentionOut;
   CNeuronComplexConvOCL      cPooling;
   CNeuronBaseOCL             cResidual;
   CNeuronComplexConvOCL      cFeedForward[2];
   //---
   virtual bool               AttentionOut(void);
   virtual bool               AttentionInsideGradients(void);
   virtual bool               SumAndNormalize(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, bool normilize = true, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 0.5f) override;
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;

public:
                     CNeuronComplexMVMHMaskAttention(void) {};
                    ~CNeuronComplexMVMHMaskAttention(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void)   const   {  return defNeuronComplexMVMHMaskAttention;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCATCH   :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL        cTranspose;
   CNeuronBaseOCL             caFreqIn[2];
   CNeuronBaseOCL             cFreqConcat;
   CNeuronComplexConvOCL      caProjection[2];
   CNeuronComplexMVMHMaskAttention  caChannelFusion[2];
   CNeuronComplexConvOCL      caLinearHead[2];
   CNeuronBaseOCL             caFreqOut[2];
   //---
   virtual bool               FFT(CBufferFloat *inp_re, CBufferFloat *inp_im,
                                  CBufferFloat *out_re, CBufferFloat *out_im,
                                  uint variables, bool reverse = false);
   virtual bool               ComplexNormalize(CNeuronBaseOCL *cInputFreqComplex,
         CNeuronBaseOCL *cNormFreqComplex,
         CBufferFloat   *cMeans,
         CBufferFloat   *cVariances,
         int dimension, int variables);
   virtual bool               ComplexNormalizeGradient(CNeuronBaseOCL *cInputFreqComplex,
         CNeuronBaseOCL *cNormFreqComplex,
         CBufferFloat   *cVariances,
         int dimension, int variables);
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;


public:
                     CNeuronCATCH(void) {};
                    ~CNeuronCATCH(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint time_step, uint variables,
                                   uint window, uint step,
                                   uint window_key, uint heads,
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override const   {  return defNeuronCATCH;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSkillsEncoder   :  public CNeuronSoftMaxOCL
  {
protected:
   CNeuronCATCH               cCrossObservAttention;
   CNeuronTransposeOCL        cTranspose;
   CNeuronConvOCL             cSkillsProjection[2];
   CNeuronBaseOCL             cPrevSkillsConcat;
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;

public:
                     CNeuronSkillsEncoder(void) {};
                    ~CNeuronSkillsEncoder(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint time_step, uint variables, uint skills,
                                   uint window, uint step, uint window_key, uint heads,
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override const   {  return defNeuronSkillsEncoder;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHiSSDLowLevelControler:  public CNeuronConvOCL
  {
protected:
   uint                       iTaskSkills;
   uint                       iCommonSkills;
   //---
   CNeuronSkillsEncoder       cTaskSpecificSkillsEncoder;
   CNeuronTransposeOCL        cTranspose;
   CNeuronBaseOCL             cObservAndSkillsConcat;
   CNeuronBatchNormOCL        cNormalizarion;
   CNeuronConvOCL             cActionDecoder[2];
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool               calcInputGradients(CNeuronBaseOCL *NeuronOCL,
         CBufferFloat *SecondInput,
         CBufferFloat *SecondGradient,
         ENUM_ACTIVATION SecondActivation = None
                                                ) override;

public:
                     CNeuronHiSSDLowLevelControler(void) {};
                    ~CNeuronHiSSDLowLevelControler(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint time_step, uint variables,
                                   uint task_skills, uint common_skills, uint n_actions,
                                   uint window, uint step, uint window_key, uint heads,
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override const   {  return defNeuronHiSSDLowLevelControler;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCGLSTMOCL : public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cConcatenateInputs;
   CNeuronConvOCL    cProjection;
   //---
   virtual bool      CSLSTM_feedForward(void);
   virtual bool      CSLSTM_CalcHiddenGradient(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCGLSTMOCL(void) {};
                    ~CNeuronCGLSTMOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint count, uint window, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void) override  const    {  return defNeuronCGLSTMOCL; }
   virtual bool      Clear(void) override;
   virtual CBufferFloat *getLSTMWeights(void) { return cProjection.GetWeightsConv(); }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHProbAttention :  public CResidualConv
  {
protected:
   uint                       iWindow;
   uint                       iWindowKey;
   uint                       iHeads;
   uint                       iUnits;
   uint                       iTopKQuerys;
   uint                       iRandomKeys;
   int                        ibScore;
   //---
   CNeuronConvOCL             cQKV;
   CNeuronBaseOCL             cQ;
   CNeuronBaseOCL             cKV;
   CNeuronBaseOCL             cRandomK;
   CNeuronBaseOCL             cMHAttentionOut;
   CNeuronConvOCL             cPooling;
   CNeuronTransposeOCL        cTranspose[2];
   CNeuronConvOCL             cScaling;
   //---
   virtual bool               RandomKeys(CBufferFloat* indexes, int random, int units, int heads);
   virtual bool               QueryImportance(void);
   virtual bool               TopKIndexes(void);
   //---
   virtual bool               AttentionOut(void);
   virtual bool               AttentionInsideGradients(void);
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;

public:
                     CNeuronMHProbAttention(void) {};
                    ~CNeuronMHProbAttention(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window, uint window_key, uint heads, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void)   const   {  return defNeuronMHProbAttention;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTSPositionEncoder   :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL             cProjection;
   CNeuronBatchNormOCL        cNorm;
   CBufferFloat               cPeriods;
   //---
   virtual bool               AddPE(CBufferFloat *time);
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTSPositionEncoder(void) {};
                    ~CNeuronTSPositionEncoder(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint window, uint units_count, uint &periods[], uint freqs,
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override  const   {  return defNeuronTSPositionEncoder;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint               GetWindow(void) const { return cProjection.GetWindow(); }
   virtual uint               GetWindowOut(void) const { return cProjection.GetFilters(); }
   virtual uint               GetUnits(void) const { return cProjection.GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMamba4CastEmbeding :   public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL             cProjection;
   CNeuronBatchNormOCL        cNorm;
   CNeuronTSPositionEncoder   cProjectionWithTE;
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CMamba4CastEmbeding(void) {};
                    ~CMamba4CastEmbeding(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint window, uint window_out, uint units_count, uint &periods[],
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override  const   {  return defMamba4CastEmbeding;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiWindowsConvWPadOCL    :  public CNeuronConvOCL
  {
protected:
   int               aiWindows[];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronMultiWindowsConvWPadOCL(void) {  activation = SoftPlus;  iWindow = -1; }
                    ~CNeuronMultiWindowsConvWPadOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint &windows[],
                          uint step, uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMultiWindowsConvWPadOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual uint      GetWindow(void) const { return aiWindows[0]; }
   virtual uint      GetWindowsSize(void) const { return aiWindows.Size(); }
   virtual uint      GetWindowOut(void) const { return iWindowOut; }
   virtual uint      GetUnits(void) const { return Neurons() / (iVariables * GetWindowsSize() * iWindowOut); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronConcatDiff:  public CNeuronBaseOCL
  {
protected:
   uint              iUnits;
   uint              iVariables;
   uint              iStep;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override   { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronConcatDiff(void) :  iUnits(0), iVariables(1), iStep(1) {  activation = None; }
                    ~CNeuronConcatDiff(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint step, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override  const   {  return defNeuronConcatDiff;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMantisPatching   :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL     cToVarSeq;
   CNeuronConvOCL          cProjecting;
   CNeuronTransposeVRCOCL  cToVarProjSeq;
   CNeuronConvOCL          cPatchingProj;
   CNeuronProofOCL         cProof;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronMantisPatching(void) {};
                    ~CNeuronMantisPatching(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint count, uint patchs, uint variables,
                          uint embedding_size, uint patch_filters,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void)  override const {  return defNeuronMantisPatching; }
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMantisAttentionUnit :  public CNeuronSoftMaxOCL
  {
protected:
   CNeuronBaseOCL                cClassToken[2];
   CNeuronMVCrossAttentionMLKV   cAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronMantisAttentionUnit(void) {};
                    ~CNeuronMantisAttentionUnit(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint token_size, uint window, uint window_key, uint heads,
                          uint units_count, uint layers, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void)  override const {  return defNeuronMantisAttentionUnit; }
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTimeFoundPatching   :  public CNeuronConvOCL
  {
protected:
   CNeuronTransposeOCL              cToVarSeq;
   CNeuronMultiWindowsConvWPadOCL   cProjecting;
   CNeuronTransposeVRCOCL           cToVarProjSeq;
   CNeuronMultiWindowsConvWPadOCL   cPatchingProj;
   CNeuronMHFeedForward             cPatchsFeedForward;
   CNeuronConvOCL                   cPatchingAgr;
   CNeuronProofOCL                  cProof;
   CNeuronTransposeOCL              cToPatchVarEmb;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronTimeFoundPatching(void) {};
                    ~CNeuronTimeFoundPatching(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint count, uint patchs, uint variables,
                          uint embedding_size, uint patch_filters,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void)  override const {  return defNeuronTimeFoundPatching; }
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTimeFoundTransformerUnit  :  public CNeuronCrossDMHAttention
  {
protected:
   CNeuronMVMHAttentionMLKV   cSelfAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        updateInputWeights(NeuronOCL); }

public:
                     CNeuronTimeFoundTransformerUnit(void) {};
                    ~CNeuronTimeFoundTransformerUnit(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window,
                          uint window_key, uint heads, uint heads_kv, uint units_count,
                          uint layers, uint layers_to_one_kv, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const { return defNeuronTimeFoundTransformerUnit; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSwiGLUOCL  :  public   CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    caProjections[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL)          override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)   override;

public:
                     CNeuronSwiGLUOCL(void) {};
                    ~CNeuronSwiGLUOCL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out, uint units_count,
                          uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const {   return defNeuronSwiGLUOCL; }
   //--- methods for working with files
   virtual bool      Save(int const file_handle)   override;
   virtual bool      Load(int const file_handle)   override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)     override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual uint      GetWindow(void) const      { return caProjections[0].GetWindow(); }
   virtual uint      GetWindowOut(void) const   { return caProjections[0].GetFilters();}
   virtual uint      GetVariables(void) const   { return caProjections[0].GetVariables(); }
   virtual uint      GetUnits(void) const       { return caProjections[0].GetUnits(); }
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMaskMultiWinConv    :  public CNeuronConvOCL
  {
protected:
   uint              iWindowsTotal;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronMaskMultiWinConv(void) {};
                    ~CNeuronMaskMultiWinConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window,
                          uint windows_total, uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMaskMultiWinConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTimeMoESparseExperts   :  public CNeuronBaseOCL
  {
protected:
   CNeuronSwiGLUOCL        cExpertsIn;
   CNeuronSwiGLUOCL        cSharedIn;
   CNeuronMaskMultiWinConv cExpertsOut;
   CNeuronConvOCL          cSharedOut;
   CNeuronTopKGates        cMasks;
   CNeuronConvOCL          cSharedGates;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL)          override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)   override;

public:
                     CNeuronTimeMoESparseExperts(void) {};
                    ~CNeuronTimeMoESparseExperts(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window,
                          uint window_out, uint units_count, uint variables, uint experts,
                          uint topK, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const {   return defNeuronTimeMoESparseExperts; }
   //--- methods for working with files
   virtual bool      Save(int const file_handle)   override;
   virtual bool      Load(int const file_handle)   override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)     override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual uint      GetWindow(void) const      { return cSharedIn.GetWindow();     }
   virtual uint      GetVariables(void) const   { return cSharedIn.GetVariables();  }
   virtual uint      GetUnits(void) const       { return cSharedIn.GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTimeMoEAttention   :  public CNeuronCrossDMHAttention
  {
protected:
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        updateInputWeights(NeuronOCL); }

public:
                     CNeuronTimeMoEAttention(void)  {};
                    ~CNeuronTimeMoEAttention(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint window_cross, uint units_cross,
                          uint heads, uint layers,
                          uint experts, uint experts_dimension, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronTimeMoEAttention; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAdaptConv    :  public CNeuronConvOCL
  {
protected:
   CBufferFloat      bMainFreq;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *out_re, CBufferFloat *out_im, uint variables, bool reverse = false);
   virtual bool      PeriodsFinding(CBufferFloat *inp_re, CBufferFloat *inp_im, CBufferFloat *main_freq, uint variables);
   virtual bool      AdaptiveConvolution(CNeuronBaseOCL *NeuronOCL, CBufferFloat *main_freq);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAdaptConv(void) {};
                    ~CNeuronAdaptConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window,
                          uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronAdaptConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRoPE  :  public CNeuronPositionEncoder
  {
protected:
   uint              iWindow;
   uint              iVariables;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)override;

public:
                     CNeuronRoPE(void) : iWindow(0), iVariables(0) {};
                    ~CNeuronRoPE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint count, uint window, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual int       Type(void) override const {  return defNeuronRoPE; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CParams  :  public   CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cOne;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override          { return FeedForward(); }

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override   { return UpdateInputWeights(); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override   { return true; }

public:
                     CParams(void) {};
                    ~CParams(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual bool      Identity(const int rows, const int cols);
   //---
   virtual bool      FeedForward(void);
   virtual bool      UpdateInputWeights(void);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defParams; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual CBufferFloat   *getWeightsParams(void)        {  return cOne.getWeights();  }                 ///< Get pointer of gradient buffer @return Pointer to object
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronK2VAEEncoder   :  public CNeuronBaseOCL
  {
protected:
   //--- Koopman
   CNeuronTimeMoESparseExperts   cKoopman;
   CNeuronBaseOCL                cKoopmanPred;
   CNeuronBaseOCL                cKoopmanRest;
   //---
   CNeuronTimeMoEAttention cAuxiliaryNet;
   //--- Kalman Filter
   CParams              cB;      // Control input matrix B
   CParams              cF;      // State transition matrix F
   CParams              cH;      // Observation matrix H
   CParams              cQ;      // Learnable covariance matrices Q
   CParams              cR;      // Learnable covariance matrices R
   CNeuronBaseOCL       cP;      // Covariance matrices P
   //---
   CNeuronTransposeOCL  cFT;
   CNeuronTransposeOCL  cHT;
   CNeuronTransposeOCL  cQT;
   CNeuronTransposeOCL  cRT;
   CNeuronTransposeOCL  cPT;
   //---
   CNeuronBaseOCL       cQ_QT;
   CNeuronBaseOCL       cR_RT;
   CNeuronBaseOCL       cXPred;
   CNeuronBaseOCL       cF_P;
   CNeuronBaseOCL       cPPred;
   //---
   CNeuronBaseOCL       cP_HT;
   CNeuronBaseOCL       cH_P_HT;
   matrix<double>       mS, mSGrad;
   CNeuronBaseOCL       cSInv;
   CNeuronBaseOCL       cK;
   CNeuronTransposeOCL  cKT;
   CNeuronBaseOCL       cYPred;
   CNeuronBaseOCL       cDeltY;
   CNeuronBaseOCL       cX;
   CNeuronBaseOCL       cK_H;
   CNeuronBaseOCL       cIdifK_H;
   CNeuronTransposeOCL  cIdifK_HT;
   matrix<double>       mP;
   matrix<double>       mPGrad;
   matrix<double>       mNoise;
   matrix<double>       mKL;
   matrix<double>       mGrad;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronK2VAEEncoder(void) {};
                    ~CNeuronK2VAEEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_cross,
                          uint heads, uint layers, uint scenarios,
                          uint experts, uint experts_dimension, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void)        const                      {  return defNeuronK2VAEEncoder; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronInterpolationAttention :  public CNeuronBaseOCL
  {
protected:
   //---
   uint                 iCount;
   uint                 iDimension;
   //---
   CParams              cW;
   CParams              cA;
   CParams              cGL;
   //---
   CNeuronBaseOCL       cH;
   CNeuronBaseOCL       cAdj;
   CNeuronBaseOCL       cAttention;
   //---
   virtual bool      InterpolationAttention(CNeuronBaseOCL *NeuronOCL);
   virtual bool      InterpolationAttentionGrad(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronInterpolationAttention(void) { activation = None; }
                    ~CNeuronInterpolationAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const {  return defNeuronInterpolationAttention; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAGCN    :  public CNeuronBaseOCL
  {
protected:
   CParams              cEa;
   CNeuronSwiGLUOCL     cWx;
   CNeuronSwiGLUOCL     cWe;
   CNeuronBaseOCL       cWconcat_ex;
   CNeuronConvOCL       cEn;
   CNeuronTransposeOCL  cEnT;
   CNeuronBaseOCL       cEnEnT;
   CNeuronSoftMaxOCL    cAadapt;
   CNeuronBaseOCL       cAadaptX;
   CNeuronBaseOCL       cApreX;
   CNeuronSwiGLUOCL     cWadapt;
   CNeuronSwiGLUOCL     cWpre;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronAGCN(void) {activation = None;}
                    ~CNeuronAGCN(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronAGCN; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   //---
   virtual uint      GetWindow(void) const      { return cWx.GetWindow(); }
   virtual uint      GetUnits(void) const       { return cWx.GetUnits(); }
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGinARCell  :  public CNeuronBaseOCL
  {
protected:
   CNeuronInterpolationAttention cX_IA;
   CNeuronAGCN                   cX_AGCN;
   CNeuronAGCN                   cForgetGate;
   CNeuronAGCN                   cResetGate;
   CNeuronBaseOCL                cContext;
   CBufferFloat                  bTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronGinARCell(void) { activation = None;}
                    ~CNeuronGinARCell(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronGinARCell; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGinAR   :  public CNeuronSwiGLUOCL
  {
protected:
   CParams              cEa;
   CNeuronSwiGLUOCL     cWx;
   CNeuronSwiGLUOCL     cWe;
   CNeuronBaseOCL       cWconcat_ex;
   CNeuronConvOCL       cEn;
   CNeuronTransposeOCL  cEnT;
   CNeuronBaseOCL       cEnEnT;
   CNeuronSoftMaxOCL    cApre;
   CNeuronGinARCell     caCells[4];
   CNeuronBaseOCL       cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGinAR(void) {};
                    ~CNeuronGinAR(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronGinAR; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPeriodNorm :  public CNeuronBaseOCL
  {
protected:
   uint                    iPeriod;
   uint                    iVariables;
   uint                    iCount;
   CNeuronBaseOCL          cMeanSTDevs;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPeriodNorm(void) : iPeriod(-1), iVariables(1) {};
                    ~CNeuronPeriodNorm(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint period, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronPeriodNorm; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   CNeuronBaseOCL*   GetMeanSTDevs(void) { return cMeanSTDevs.AsObject(); }
   virtual uint      GetPeriod(void) const { return iPeriod; }
   virtual uint      GetVariables(void) const { return iVariables; }
   virtual uint      GetUnits(void) const { return iCount; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAdaptSpatialNorm :  public CNeuronBaseOCL
  {
protected:
   uint                    iVariables;
   uint                    iCount;
   //---
   CParams                 cEn;
   CNeuronTransposeOCL     cEnT;
   CNeuronBaseOCL          cEnEnT;
   CNeuronSoftMaxOCL       cAttan;
   CNeuronBaseOCL          cMeanSTDevs;
   //---
   virtual bool      AdaptSpatialNorm(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AdaptSpatialNormGrad(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAdaptSpatialNorm(void) : iCount(0), iVariables(1) {};
                    ~CNeuronAdaptSpatialNorm(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronAdaptSpatialNorm; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   CNeuronBaseOCL*   GetMeanSTDevs(void) { return cMeanSTDevs.AsObject(); }
   virtual uint      GetVariables(void) const { return iVariables; }
   virtual uint      GetUnits(void) const { return iCount; }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSCNNEncoder   :  public CNeuronTransposeOCL
  {
protected:
   CNeuronPeriodNorm       cLongNorm;
   CNeuronTransposeVRCOCL  cSeasonTransp;
   CNeuronPeriodNorm       cSeasonNorm;
   CNeuronTransposeVRCOCL  cUnSeasonTransp;
   CNeuronPeriodNorm       cShortNorm;
   CNeuronAdaptSpatialNorm cAdaptSpatNorm;
   CNeuronBaseOCL          cConcatenated;
   CNeuronSwiGLUOCL        cProjection;
   CNeuronTransposeOCL     cTranspose;
   CNeuronConvOCL          caFusion[2];
   CNeuronBaseOCL          cFusionOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronSCNNEncoder(void) {};
                    ~CNeuronSCNNEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables, uint forecast,
                          uint season_period, uint short_period,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSCNNEncoder; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSCNN    :  public   CNeuronBaseOCL
  {
protected:
   CLayer            cLayers;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronSCNN(void) {};
                    ~CNeuronSCNN(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables, uint forecast,
                          uint season_period, uint short_period, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSCNN; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAttentNorm :  public CNeuronBaseOCL
  {
protected:
   uint                    iPeriod;
   uint                    iVariables;
   uint                    iCount;
   //---
   CParams                 cAttention;
   CNeuronSoftMaxOCL       cSoftMax;
   CNeuronBaseOCL          cMeans;
   CNeuronBaseOCL          cSTDevs;
   //---
   virtual bool      AttentNorm(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentNormGrad(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAttentNorm(void) :  iPeriod(0), iVariables(0), iCount(0) {};
                    ~CNeuronAttentNorm(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint period, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronAttentNorm; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   CNeuronBaseOCL*   GetMeans(void) { return cMeans.AsObject(); }
   CNeuronBaseOCL*   GetSTDevs(void) { return cSTDevs.AsObject(); }
   virtual uint      GetPeriod(void) const { return iPeriod; }
   virtual uint      GetVariables(void) const { return iVariables; }
   virtual uint      GetUnits(void) const { return iCount; }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {  activation = None; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSAttentNorm :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL     cTranspose;
   CNeuronBaseOCL          cAttention;
   CNeuronSoftMaxOCL       cSoftMax;
   CNeuronBaseOCL          cMeans;
   CNeuronBaseOCL          cSTDevs;
   CNeuronBaseOCL          cNorm;
   //---
   virtual bool      AttentNorm(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentNormGrad(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSAttentNorm(void) {};
                    ~CNeuronSAttentNorm(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSAttentNorm; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   CNeuronBaseOCL*   GetMeans(void) { return cMeans.AsObject(); }
   CNeuronBaseOCL*   GetSTDevs(void) { return cSTDevs.AsObject(); }
   virtual uint      GetVariables(void) const { return iWindow; }
   virtual uint      GetUnits(void) const { return iCount; }
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {  activation = None; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPolynomialRegression   :  public CNeuronBaseOCL
  {
protected:
   CNeuronSwiGLUOCL     cProjection;
   CNeuronConvOCL       cConvolution;
   CNeuronConvOCL       cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPolynomialRegression(void) {};
                    ~CNeuronPolynomialRegression(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint window, uint window_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronPolynomialRegression; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetWindowIn(void) const { return cProjection.GetWindow(); }
   virtual uint      GetWindowOut(void) const { return cResidual.GetFilters(); }
   virtual uint      GetUnits(void) const { return cProjection.GetUnits(); }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSSCNNEncoder   :  public CNeuronTransposeOCL
  {
protected:
   CNeuronPeriodNorm       cLongNorm;
   CNeuronConvOCL          cLongExtrapolate;
   CNeuronTransposeOCL     cLongMeanSTDevTransp;
   CNeuronBaseOCL          cLongMeanExtrapolate;
   CNeuronTransposeVRCOCL  cSeasonTransp;
   CNeuronAttentNorm       cSeasonNorm;
   CNeuronTransposeVRCOCL  cUnSeasonTransp;
   CNeuronConvOCL          cSeasonExtrapolate;
   CNeuronConvOCL          cSeasonMeanExtrapolate;
   CNeuronAttentNorm       cShortNorm;
   CNeuronConvOCL          cShortExtrapolate;
   CNeuronConvOCL          cShortMeanExtrapolate;
   CNeuronSAttentNorm      cSpatialNorm;
   CNeuronConvOCL          cSpatialExtrapolate;
   CNeuronConvOCL          cSpatialMeanExtrapolate;
   CNeuronBaseOCL          cConcatenated;
   CNeuronTransposeOCL     cTranspose;
   CNeuronPolynomialRegression   cFusion;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)  override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronSSCNNEncoder(void) {};
                    ~CNeuronSSCNNEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables, uint forecast,
                          uint season_period, uint short_period,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSSCNNEncoder; }
   virtual void      TrainMode(bool flag) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSSCNN    :  public   CNeuronSCNN
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL)  override;

public:
                     CNeuronSSCNN(void) {};
                    ~CNeuronSSCNN(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units_count, uint variables, uint forecast,
                          uint season_period, uint short_period, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSSCNN; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CCircleParams  :  public CNeuronBaseOCL
  {
protected:
   CLayer            cOnes;
   CNeuronBaseOCL    *cCurrent;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override          { return FeedForward(); }

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override   { return UpdateInputWeights(); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override   { return true; }

public:
                     CCircleParams(void) : cCurrent(NULL) {};
                    ~CCircleParams(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, uint period, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      Identity(const int rows, const int cols);
   virtual bool      Zeros(void);
   //---
   virtual bool      FeedForward(void);
   virtual bool      UpdateInputWeights(void);
   virtual bool      SetPosition(int position);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defCircleParams; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual CBufferFloat   *getWeightsParams(void);
   virtual int       GetPeriod(void) const { return cOnes.Total(); }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTQMHA   :  public CNeuronCrossAttention
  {
protected:
   uint              iTimeframe;
   CCircleParams     cParams;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);

public:
                     CNeuronTQMHA(void) {};
                    ~CNeuronTQMHA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_in, uint period, uint timeframe,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronTQMHA;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CChebPolinom   :  public CNeuronBaseOCL
  {
protected:
   uint              iDimension;
   uint              iSteps;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CChebPolinom(void)   {};
                    ~CChebPolinom(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint dimension, uint steps, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defChebPolinom; }
   virtual uint      GetDimension(void) const { return iDimension; }
   virtual uint      GetSteps(void) const { return iSteps; }
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetGrapConv   :  public CNeuronTransposeRCDOCL
  {
protected:
   CNeuronBaseOCL    cX_G;

public:
                     CNeuronHimNetGrapConv(void) {};
                    ~CNeuronHimNetGrapConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint count, uint window, uint cheb_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual bool      FeedForward(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CChebPolinom *Support);
   //---
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void)        const                      {  return defNeuronHimNetGrapConv; }
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual uint      GetCount(void) override const { return iWindow; }
   virtual uint      GetWindow(void) override const { return GetDimension(); }
   virtual uint      GetChebK(void) const { return iCount; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetGCRU    :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL          cInpAndHidden;
   CNeuronHimNetGrapConv   cZ_R;
   CNeuronConvOCL          cZ_R_emb;
   CNeuronBaseOCL          cZe_Re;
   CNeuronBaseOCL          cZ;
   CNeuronBaseOCL          cR;
   CNeuronBaseOCL          cCandidate;
   CNeuronHimNetGrapConv   cHC;
   CNeuronConvOCL          cHC_emb;
   CNeuronBaseOCL          cHCe;
   CBufferFloat            bSupportAccum;

public:
                     CNeuronHimNetGCRU(void) {};
                    ~CNeuronHimNetGCRU(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out,
                          uint cheb_k, uint embed_dim,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL,
                                 CChebPolinom *Support,
                                 CNeuronBaseOCL *Embedding);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support,
                                        CNeuronBaseOCL *Embedding);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support,
                                        CNeuronBaseOCL *Embedding);
   //---
   virtual int       Type(void)   const   {  return defNeuronHimNetGCRU;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual uint      GetCount(void) const { return cZ_R.GetCount(); }
   virtual uint      GetWindowIn(void) const { return cZ_R.GetWindow() - GetWindowOut(); }
   virtual uint      GetWindowOut(void) const { return Neurons() / GetCount(); }
   virtual uint      GetChebK(void) const { return cZ_R.GetChebK(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetTempEncoder   : public CNeuronBaseOCL
  {
protected:
   uint              aTimeframes[2];
   CCircleParams     caEmbeddings[2];
   CNeuronBaseOCL    cConcatEmbeddings;
   CLayer            cGRCUs;
   CBufferFloat      bSupportAccum;

public:
                     CNeuronHimNetTempEncoder(void) {};
                    ~CNeuronHimNetTempEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out, uint cheb_k,
                          uint layers, uint embed_dim, uint period1,
                          uint timeframe1, uint period2, uint timeframe2,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL,
                                 CChebPolinom *Support,
                                 int label);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support);
   //---
   virtual int       Type(void)   const   {  return defNeuronHimNetTempEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetCount(void) const;
   virtual uint      GetWindowIn(void) const;
   virtual uint      GetWindowOut(void) const;
   virtual uint      GetChebK(void) const;
   //---
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetSpatEncoder   :     public CNeuronBaseOCL
  {
protected:
   CParams           cEmbedding;
   CLayer            cGRCUs;
   CBufferFloat      bSupportAccum;
   CBufferFloat      bEmbeddingAccum;

public:
                     CNeuronHimNetSpatEncoder(void) {};
                    ~CNeuronHimNetSpatEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out, uint cheb_k,
                          uint layers, uint embed_dim,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL,
                                 CChebPolinom *Support);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL,
                                        CChebPolinom *Support);
   //---
   virtual int       Type(void)   const   {  return defNeuronHimNetSpatEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetCount(void) const;
   virtual uint      GetWindowIn(void) const;
   virtual uint      GetWindowOut(void) const;
   virtual uint      GetChebK(void) const;
   //---
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetEncoder    :  public CNeuronBaseOCL
  {
protected:
   CParams                    cEmbedding;
   CNeuronTransposeOCL        cEmbeddingT;
   CNeuronBaseOCL             cSupport;
   CNeuronSoftMaxOCL          cNormSupport;
   CChebPolinom               cPolinomSupport;
   CNeuronHimNetTempEncoder   cTempEncoder;
   CNeuronHimNetSpatEncoder   cSpatEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHimNetEncoder(void) {};
                    ~CNeuronHimNetEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out, uint cheb_k,
                          uint layers, uint embed_dim, uint period1,
                          uint timeframe1, uint period2, uint timeframe2,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHimNetEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual bool      Clear(void) override;
   virtual bool      SetGradient(CBufferFloat *buffer, bool delete_prev = true);
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronHimNetDecoder    :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL             cProjection;
   CNeuronTransposeOCL        cProjectionT;
   CNeuronConvOCL             cEmbedding;
   CNeuronBaseOCL             cSupport;
   CNeuronSoftMaxOCL          cNormSupport;
   CChebPolinom               cPolinomSupport;
   CLayer                     cGRCUs;
   CBufferFloat               bSupportAccum;
   CBufferFloat               bEmbeddingAccum;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHimNetDecoder(void) {};
                    ~CNeuronHimNetDecoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out,
                          uint cheb_k, uint layers, uint embed_dim,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronHimNetDecoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSNSMHAttention   :  public CNeuronConvOCL
  {
   float             fSparse;
   //---
   CNeuronBaseOCL    cNeighbors;
   CNeuronBaseOCL    cRamdomCandidates;
   CNeuronConvOCL    cProjection[2];
   CNeuronBaseOCL    cScores;

   //---
   virtual bool      SignificantNeighborsSampling(CNeuronBaseOCL *NeuronOCL);
   virtual bool      SparseMHScores(void);
   virtual bool      SparseMHScoresGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSNSMHAttention(void) {};
                    ~CNeuronSNSMHAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint heads,
                          uint m_units, float sparse,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSNSMHAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual CBufferFloat* GetIndexes(void) { return cNeighbors.getOutput(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFastGConv   :     public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL          cInpAndHidden;
   CNeuronBaseOCL          cNormAttention;
   CNeuronBaseOCL          cInvDiag;
   CNeuronBaseOCL          cAX;
   CNeuronBaseOCL          cAXplusX;
   CNeuronBaseOCL          cNormAXplusX;
   CNeuronConvOCL          cZ_R;
   CNeuronBaseOCL          cZ;
   CNeuronBaseOCL          cR;
   CNeuronBaseOCL          cCandidate;
   CNeuronConvOCL          cHC;
   //---
   virtual bool      RandomWalk(CBufferFloat* data,
                                CBufferFloat* normal,
                                CBufferFloat* inv_diag,
                                const int rows,
                                const int cols
                               );
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)  override { ReturnFalse; };
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None
                                       ) override { ReturnFalse; }
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronFastGConv(void) {};
                    ~CNeuronFastGConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint sparse_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      FeedForward(CNeuronBaseOCL *SourceData, CNeuronSNSMHAttention *SparseAttent);
   virtual bool      CalcInputGradients(CNeuronBaseOCL *SourceData, CNeuronSNSMHAttention *SparseAttent);
   //---
   virtual int       Type(void)   const   {  return defNeuronFastGConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual uint      GetCount(void) const { return (uint)cInvDiag.Neurons(); }
   virtual uint      GetSparseDimension(void) const { return (uint)cNormAttention.Neurons() / GetCount(); }
   virtual uint      GetWindow(void) const { return (uint)Neurons() / GetCount(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSAGDFN  :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL     cTranspose;
   CLayer                  cEmbedding;
   CNeuronSNSMHAttention   cAttention;
   CLayer                  cGCRU;
   CLayer                  cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSAGDFN(void) {};
                    ~CNeuronSAGDFN(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint time_steps, uint variables, uint embedding_dim,
                          uint emb_layers, uint sparse_dimension, uint heads,
                          float sparse, uint gcru_layers,
                          uint forecast, uint forec_layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSAGDFN;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpatialEmbedding   :     public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   uint              iEmbeddingDim;
   CParams           cEmbedding;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpatialEmbedding(void) {};
                    ~CNeuronSpatialEmbedding(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint embed_dim,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSpatialEmbedding;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTempEmbedding    : public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iUnits;
   uint              aiEmbeddingDim[2];
   uint              aiFrames[2];
   uint              aiPeriod[2];
   CParams           caEmbeddings[2];
   //---
   virtual bool      ConcatByLabel(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   virtual bool      ConcatByLabelGrad(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse;}
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse;}
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None
                                       ) override;

public:
                     CNeuronTempEmbedding(void) {};
                    ~CNeuronTempEmbedding(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window,
                          uint embed_dim1, uint period1, uint frame1,
                          uint embed_dim2, uint period2, uint frame2,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronTempEmbedding;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGlobalLocalAttention   :  public CNeuronMHFeedForward
  {
protected:
   CNeuronSNSMHAttention   cMask;
   CNeuronConvOCL          cQ;
   CNeuronConvOCL          cKV;
   CNeuronBaseOCL          cScore;
   CNeuronBaseOCL          cMHAttention;
   CNeuronConvOCL          cW0;
   CNeuronBaseOCL          cResidual;
   //---
   virtual bool      GlobalLocalAttention(void);
   virtual bool      GlobalLocalAttentionGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGlobalLocalAttention(void) {};
                    ~CNeuronGlobalLocalAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint dimension_k,
                          uint heads, uint m_units, float sparse,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronGlobalLocalAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronExtralonger   :  public CNeuronMHAttentionPooling
  {
protected:
   CLayer               cProjectionT;
   CLayer               cTimeModule;
   CLayer               cSpatialModule;
   CLayer               cMixModule;
   CNeuronBaseOCL       cConcatResults;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse;}
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override {ReturnFalse;}
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None
                                       ) override;

public:
                     CNeuronExtralonger(void) {};
                    ~CNeuronExtralonger(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint time_steps_in, uint time_steps_out, uint variables,
                          uint dimension, uint emb_dimension, uint period1,
                          uint frame1, uint period2, uint frame2, uint layers,
                          uint heads, uint dimension_k, uint m_units, float sparse,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronExtralonger;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGraphons   :  public CNeuronBaseOCL
  {
protected:
   CLayer            acProbability;
   CLayer            acDataEmb;
   CParams           cExpertsEmb;
   CNeuronBaseOCL    cGraphs;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGraphons(void) {};
                    ~CNeuronGraphons(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint emb_dimension,
                          uint experts, float dropout,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronGraphons;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGraphAttention   :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    cValue;
   CNeuronGraphons   cGraphs;
   CNeuronSoftMaxOCL cScores;
   CNeuronBaseOCL    cAttention;
   CNeuronBaseOCL    cResidual;
   CNeuronConvOCL    cFeedForward[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGraphAttention(void) {};
                    ~CNeuronGraphAttention(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint emb_dimension,
                          uint experts, float dropout,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronGraphAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSparseSoftMax    :  public CNeuronSoftMaxOCL
  {
protected:
   uint              iDimensionIn;
   CBufferFloat      cIndexes;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSparseSoftMax(void) {};
                    ~CNeuronSparseSoftMax(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint dimension_in, uint dimension_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual int       Type(void) override const { return defNeuronSparseSoftMax; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual CBufferFloat* GetIndexes(void)  { return GetPointer(cIndexes); }
   virtual uint      DimensionOut(void) const {  return uint(Neurons() / iHeads); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSparseGraphAttention   :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL       cValue;
   CNeuronGraphons      cGraphs;
   CNeuronSparseSoftMax cScores;
   CNeuronBaseOCL       cAttention;
   CNeuronBaseOCL       cResidual;
   CNeuronConvOCL       cFeedForward[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSparseGraphAttention(void) {};
                    ~CNeuronSparseGraphAttention(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint experts, float dropout,
                          uint emb_dimension, uint sparse_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronSparseGraphAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGlobLocGraphAtt  :  public CNeuronMHAttentionPooling
  {
protected:
   CNeuronGraphAttention         cGlobal;
   CNeuronSparseGraphAttention   cLocal;
   CNeuronBaseOCL                cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGlobLocGraphAtt(void) {};
                    ~CNeuronGlobLocGraphAtt(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint experts, float dropout,
                          uint emb_dimension, uint sparse_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronGlobLocGraphAtt;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronExtralongerGraph   :  public CNeuronExtralonger
  {
public:
                     CNeuronExtralongerGraph(void) {};
                    ~CNeuronExtralongerGraph(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint time_steps_in, uint time_steps_out, uint variables,
                          uint dimension, uint emb_dimension, uint period1,
                          uint frame1, uint period2, uint frame2, uint layers,
                          uint experts, uint m_units, float sparse,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronExtralongerGraph;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeConv  :  public CNeuronBaseOCL
  {
protected:
   CNeuronConvOCL    cConv;
   CParams           cLevels;
   //---
   virtual bool      FloatToSpike(void);
   virtual bool      FloatToSpikeGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeConv(void) {};
                    ~CNeuronSpikeConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out, uint units_count,
                          uint variables, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual CBufferFloat* GetWeightsConv(void)   {  return cConv.GetWeightsConv();  }
   virtual int       Type(void) override const  {  return defNeuronSpikeConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cConv.GetWindow(); }
   virtual uint      GetFilters(void) const { return cConv.GetFilters(); }
   virtual uint      GetVariables(void) const { return cConv.GetVariables(); }
   virtual uint      GetUnits(void) const { return cConv.GetUnits(); }
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGateLineAttention  :  public CNeuronSpikeConv
  {
protected:
   uint              iDimensionK;
   uint              iVariables;
   uint              iHeads;
   //---
   CNeuronConvOCL    cQKV;
   CNeuronBaseOCL    cQ;
   CNeuronBaseOCL    cK;
   CNeuronBaseOCL    cV;
   CNeuronBaseOCL    cKV;
   CNeuronConvOCL    cGate;
   CNeuronBaseOCL    cState;
   CNeuronBaseOCL    cMHAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGateLineAttention(void) {};
                    ~CNeuronGateLineAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint variables, uint dimension, uint dimension_k, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronGateLineAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual bool      Clear(void) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetVariables(void) const { return iVariables; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSparseAttention   :  public CNeuronSpikeConv
  {
protected:
   CNeuronConvOCL       cValue;
   CNeuronGraphons      cGraphs;
   CNeuronSparseSoftMax cScores;
   CNeuronBaseOCL       cAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeSparseAttention(void) {};
                    ~CNeuronSpikeSparseAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint experts, float dropout,
                          uint emb_dimension, uint sparse_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronSpikeSparseAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMoEConv   :  public CNeuronBaseOCL
  {
protected:
   CLayer            acProbability;
   CParams           cExperts;
   CNeuronBaseOCL    cWeightsConv;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMoEConv(void) {};
                    ~CNeuronMoEConv(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint window_out,
                          uint experts, float dropout, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronMoEConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikingBrain  :  public CNeuronBaseOCL
  {
protected:
   CNeuronSpikeSparseAttention   cTimeAttention;
   CLayer                        caSpatialAttention;
   CNeuronBaseOCL                cResidual;
   CLayer                        caFFN;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikingBrain(void) {};
                    ~CNeuronSpikingBrain(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint experts, float dropout,
                          uint emb_dimension, uint sparse_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const   {  return defNeuronSpikingBrain;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeActivation  :  public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cHV;
   CParams           cLevels;
   CParams           cGates;
   //---
   virtual bool      FloatToSpike(void);
   virtual bool      FloatToSpikeGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeActivation(void) {};
                    ~CNeuronSpikeActivation(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint neurons, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeActivation;   }
   //--- methods for working with files
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeResNeXtBottleneck :  public CNeuronSpikeActivation
  {
protected:
   CLayer            cBottleneck;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;


public:
                     CNeuronSpikeResNeXtBottleneck(void) {};
                    ~CNeuronSpikeResNeXtBottleneck(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint units_in,
                          uint units_out, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeResNeXtBottleneck;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeResNeXtResidual :  public CNeuronSpikeResNeXtBottleneck
  {
public:
                     CNeuronSpikeResNeXtResidual(void) {};
                    ~CNeuronSpikeResNeXtResidual(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeResNeXtResidual;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeResNeXtBlock   :  public CNeuronSpikeActivation
  {
protected:
   CNeuronTransposeOCL           caTranspose[2];
   CNeuronSpikeResNeXtBottleneck caBottleneck[2];
   CNeuronSpikeResNeXtResidual   cResidual;
   CLayer                        cG;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeResNeXtBlock(void) {};
                    ~CNeuronSpikeResNeXtBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint units_in,
                          uint units_out, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeResNeXtBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetChannelsOut(void) const { return caTranspose[1].GetCount(); }
   virtual uint      GetUnitsOut(void) const { return caTranspose[1].GetWindow(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSSAM    :  public CNeuronSpikeActivation
  {
protected:
   uint                 iUnits;
   uint                 iHeads;
   uint                 iDimension;
   bool                 bMaskFuture;
   CNeuronSpikeConv     cQKV;
   CParams              cDiaganalBias;
   int                  ciScore;
   CNeuronBaseOCL       cMHAttentionOut;
   CNeuronSpikeConv     cW0;
   CNeuronBaseOCL       cResidual;
   CNeuronBatchNormOCL  cNorm;
   //---
   virtual bool      SpikeMHAttention(void);
   virtual bool      SpikeMHAttentionGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSSAM(void) : ciScore(INVALID_HANDLE) {};
                    ~CNeuronSSAM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint heads, uint dimension_k,
                          bool mask_future, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSSAM;   }
   //--- methods for working with files
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSSAMResNeXtBlock   :  public CNeuronSpikeResNeXtBlock
  {
protected:
   CNeuronSSAM        caSSAM[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSSAMResNeXtBlock(void) {};
                    ~CNeuronSSAMResNeXtBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint units_in,
                          uint units_out, uint group_size, uint groups,
                          uint heads, uint dimension_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSSAMResNeXtBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTFS    :  public   CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL    cMaskTime;
   CNeuronBaseOCL    cMaskSpatial;
   CNeuronBaseOCL    cConcatenated;
   CNeuronConvOCL    cSpatialDScale;
   CNeuronConvOCL    cTimeDScale;
   CNeuronTransposeVRCOCL caTranspose[2];
   CBufferFloat      cOne;
   //---
   virtual bool      STFS(CNeuronBaseOCL *NeuronOCL);
   virtual bool      STFSGrad(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTFS(void) {};
                    ~CNeuronSTFS(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSTFS;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeConvGRU  :  public CNeuronSpikeActivation
  {
protected:
   CNeuronBaseOCL       cConcatenated[2];
   CNeuronConvOCL       cZR;
   CNeuronBaseOCL       cZ;
   CNeuronBaseOCL       cR;
   CNeuronConvOCL       cHt;
   CNeuronBaseOCL       cH;
   CNeuronBatchNormOCL  cNorm;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeConvGRU(void) {};
                    ~CNeuronSpikeConvGRU(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint chanels_in, uint chanels_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeConvGRU;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual uint      GetChanelsIn(void) const { return cHt.GetWindow() - cHt.GetFilters(); }
   virtual uint      GetChanelsOut(void) const { return cHt.GetFilters(); }
   virtual uint      GetUnits(void) const { return cHt.GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAddToStack :  public CNeuronBaseOCL
  {
protected:
   uint              iStackSize;
   uint              iDimension;
   uint              iVariables;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAddToStack(void) : iStackSize(0), iDimension(0), iVariables(0) {};
                    ~CNeuronAddToStack(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronAddToStack;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual uint      GetStackSize(void) const { return iStackSize; }
   virtual uint      GetDimension(void) const { return iDimension; }
   virtual uint      GetVariables(void) const { return iVariables; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronStackCorrelation :  public CNeuronBaseOCL
  {
protected:
   CNeuronAddToStack cStack;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronStackCorrelation(void) {};
                    ~CNeuronStackCorrelation(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronStackCorrelation;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetStackSize(void) const { return cStack.GetStackSize(); }
   virtual uint      GetDimension(void) const { return cStack.GetDimension(); }
   virtual uint      GetVariables(void) const { return cStack.GetVariables(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTEFlowNetEncoder   :  public CNeuronSpikeConvGRU
  {
protected:
   CLayer            cRecurrentLine;
   CLayer            cFlow;
   CLayer            cCorrelation;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTEFlowNetEncoder(void) {};
                    ~CNeuronSTEFlowNetEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &chanels[], uint &units[], uint group_size,
                          uint groups, uint heads, uint dimension_k, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual CNeuronBaseOCL* GetLayer(uint layer);
   virtual uint      Layers(void)   const { return (cRecurrentLine.Total() + 1) / 3; }
   //--
   virtual int       Type(void)   override const   {  return defNeuronSTEFlowNetEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTEFlowNetResidualBlock :  public CNeuronBaseOCL
  {
protected:
   CLayer            cBlock;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTEFlowNetResidualBlock(void) {};
                    ~CNeuronSTEFlowNetResidualBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels, uint units, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //--
   virtual int       Type(void)   override const   {  return defNeuronSTEFlowNetResidualBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTEFlowNetDecoder   :  public CNeuronSpikeActivation
  {
protected:
   CLayer                        cMainLine;
   CLayer                        cFlow;
   CNeuronSTEFlowNetEncoder*     cEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTEFlowNetDecoder(void) {};
                    ~CNeuronSTEFlowNetDecoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &chanels[], uint &units[], uint group_size,
                          uint groups, uint heads, uint dimension_k,
                          CNeuronSTEFlowNetEncoder* encoder,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //--
   virtual int       Type(void)   override const   {  return defNeuronSTEFlowNetDecoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetEncoder(CNeuronSTEFlowNetEncoder* encoder) { cEncoder = encoder; }
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTEFlowNet  :  public CNeuronSTEFlowNetDecoder
  {
   //---
protected:
   CNeuronSTEFlowNetEncoder         cSTEEncoder;
   CNeuronSTEFlowNetResidualBlock   cSTEResidualBlock;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTEFlowNet(void) {};
                    ~CNeuronSTEFlowNet(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &chanels[], uint &units[], uint group_size,
                          uint groups, uint heads, uint dimension_k, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //--
   virtual int       Type(void)   override const   {  return defNeuronSTEFlowNet;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleStackCorrelation :  public CNeuronStackCorrelation
  {
protected:
   uint                 iLevels;
   CNeuronBaseOCL       cScales;
   //---
   virtual bool      AggregationByTime(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AggregationByTimeGrad(CNeuronBaseOCL *NeuronOCL);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultiScaleStackCorrelation(void) {};
                    ~CNeuronMultiScaleStackCorrelation(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint variables, uint levels,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronMultiScaleStackCorrelation;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetLevels(void) const { return iLevels; }
   virtual uint      GetVariables(void) override const { return cStack.GetVariables() / iLevels; }

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeConvGRU2D  :  public CNeuronSpikeConvGRU
  {
protected:
   CNeuronConvOCL       cProjectionX;
   CNeuronBaseOCL       cConcatenated2[2];
   CNeuronConvOCL       cZR2;
   CNeuronBaseOCL       cZ2;
   CNeuronBaseOCL       cR2;
   CNeuronConvOCL       cHt2;
   CNeuronBaseOCL       cH2;
   CNeuronTransposeOCL  cTranspose[3];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeConvGRU2D(void) {};
                    ~CNeuronSpikeConvGRU2D(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint chanels_in, uint chanels_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeConvGRU2D;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronRAFT :  public CNeuronSpikeActivation
  {
protected:
   CLayer                              cEncoder;
   CLayer                              cContext;
   CLayer                              cCorrelation;
   CNeuronBaseOCL                      cConcatenated;
   CNeuronSpikeConvGRU2D               cGRU;
   CNeuronSpikeResNeXtResidual         cResidual;
   CLayer                              cG;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronRAFT(void) {};
                    ~CNeuronRAFT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &units[], uint &chanels[], uint group_size,
                          uint groups, uint stack_size, uint levels,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronRAFT;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultScalStackCorrelBySegments :  public CNeuronAddToStack
  {
protected:
   CNeuronMultiScaleStackCorrelation cCorrelation;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultScalStackCorrelBySegments(void) {};
                    ~CNeuronMultScalStackCorrelBySegments(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint segments, uint segment_size, uint dimension,
                          uint variables, uint levels,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronMultScalStackCorrelBySegments;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override;
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetLevels(void) const { return cCorrelation.GetLevels(); }
   virtual uint      SetSegmentSize(void) const { return cCorrelation.GetStackSize(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTMAMFE  :  public CNeuronConvOCL
  {
protected:
   CLayer            cEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTMAMFE(void) {};
                    ~CNeuronTMAMFE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &windows[], uint window_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronTMAMFE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTMAMPA  :  public CNeuronBaseOCL
  {
protected:
   CNeuronRelativeCrossAttention cCrossAttention;
   CLayer                        cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTMAMPA(void) {};
                    ~CNeuronTMAMPA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units,
                          uint segments, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronTMAMPA; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronTMA  :  public CNeuronRAFT
  {
protected:
   CNeuronAddToStack cStatesStack;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronTMA(void) {};
                    ~CNeuronTMA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &units[], uint &chanels[], uint group_size,
                          uint groups, uint segments, uint segment_size, uint levels,
                          uint window_key, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronTMA;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSTConvGRU  :  public CNeuronSpikeActivation
  {
protected:
   CNeuronBaseOCL             cConcatenated[2];
   CNeuronMultiWindowsConvOCL cFSM;
   CNeuronBaseOCL             cSM;
   CNeuronConvOCL             cOs;
   CNeuronConvOCL             cOt;
   CNeuronBaseOCL             cFo;
   CNeuronTransposeRCDOCL     cS;
   CNeuronBatchNormOCL        cNorm;
   //---
   virtual bool      GRU(void);
   virtual bool      GRU_grad(void);

   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronSpikeSTConvGRU(void) {};
                    ~CNeuronSpikeSTConvGRU(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &chanels_in[], uint chanels_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeSTConvGRU;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual uint      GetChanels(void) const { return cOt.GetFilters(); }
   virtual uint      GetUnits(void) const { return cOt.GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFERENet :  public CNeuronSpikeConvGRU
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronFERENet(void) {};
                    ~CNeuronFERENet(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &chanels[],
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronFERENet;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CNeuronBaseOCL* GetLayer(int index) const { return cFlow[index]; }
   virtual CNeuronBaseOCL* operator[](const int index) const { return GetLayer(index); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFGDModule  :  public CNeuronBaseOCL
  {
protected:
   CNeuronBatchNormOCL        cNorm;
   CNeuronConvOCL             cD;
   CNeuronBaseOCL             cFlowIn;
   CNeuronConvOCL             cFlow;
   CNeuronBaseOCL             cConcatenated;
   CNeuronMultiWindowsConvOCL cXH;
   //---
   virtual bool      GRU(void);
   virtual bool      GRU_grad(void);

   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second)override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronFGDModule(void) {};
                    ~CNeuronFGDModule(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint units,
                          uint main_dimension, uint flow_dimension, uint dimension_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronFGDModule;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CNeuronConvOCL* GetD(void)  { return cD.AsObject(); }
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronEVMGRFlowNet  :  public CNeuronSpikeActivation
  {
protected:
   CNeuronFERENet                   cEncoder;
   CNeuronSTEFlowNetResidualBlock   cNeck;
   CLayer                           cDecoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronEVMGRFlowNet(void) {};
                    ~CNeuronEVMGRFlowNet(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &chanels[],
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronEVMGRFlowNet;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual bool      Clear(void) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeConvBlock   :  public CNeuronBatchNormOCL
  {
protected:
   CNeuronSpikeActivation  cActivation;
   CNeuronConvOCL          cConv;
   CNeuronAddToStack       cStack;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeConvBlock(void) {};
                    ~CNeuronSpikeConvBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeConvBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cConv.GetWindow();   }
   virtual uint      GetFilters(void) const { return cConv.GetFilters(); }
   virtual uint      GetVariables(void) const { return cConv.GetVariables(); }
   virtual uint      GetUnits(void) const { return cConv.GetUnits(); }
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMSRes   :  public CNeuronBaseOCL
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMSRes(void) {};
                    ~CNeuronMSRes(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMSRes;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   virtual uint      GetWindow(void) const { return ((CNeuronSpikeConvBlock*)cFlow[0]).GetWindow();   }
   virtual uint      GetVariables(void) const { return ((CNeuronSpikeConvBlock*)cFlow[0]).GetVariables(); }
   virtual uint      GetUnits(void) const { return ((CNeuronSpikeConvBlock*)cFlow[0]).GetUnits(); }
   virtual CBufferFloat   *getWeights(void) override { return (!!cFlow[-1] ? cFlow[-1].getWeights() : NULL); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeQKAttention :  public CNeuronBaseOCL
  {
protected:
   CNeuronSpikeConvBlock      cQK;
   CNeuronSpikeActivation     cSpikeQK;
   CNeuronBaseOCL             cQ;
   CNeuronBaseOCL             cK;
   CNeuronMultiWindowsConvOCL cScore;
   CNeuronSpikeActivation     cSpikeScore;
   CLayer                     cProject;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeQKAttention(void) {};
                    ~CNeuronSpikeQKAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint heads, uint dimension_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeQKAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSDSA :  public CNeuronMSRes
  {
protected:
   CNeuronSpikeQKAttention cAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSDSA(void) {};
                    ~CNeuronSDSA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint heads, uint dimension_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSDSA;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSDformerUBlock   :   public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cEncoder;
   CNeuronBaseOCL*   cNeck;
   CNeuronBaseOCL    cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSDformerUBlock(void) {};
                    ~CNeuronSDformerUBlock(void) { DeleteObj(cNeck) };
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &windows[], uint &heads[],
                          uint &layers[], uint dimension_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSDformerUBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   virtual CBufferFloat*   GetStack(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSDformerFlow   :   public CNeuronSDformerUBlock
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSDformerFlow(void) {};
                    ~CNeuronSDformerFlow(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &windows[], uint &heads[],
                          uint &layers[], uint dimension_k,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSDformerFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikePatchStak   :  public CNeuronAddToStack
  {
protected:
   CNeuronSpikeConvBlock   cBlock;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikePatchStak(void) {};
                    ~CNeuronSpikePatchStak(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint patch_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikePatchStak;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpike2DSSMOCL  :  public CNeuronBaseOCL
  {
protected:
   uint                    iWindowOut;
   uint                    iUnitsOut;
   //---
   CNeuronBaseOCL          cHiddenStates;
   CLayer                  cProjectionX_Time;
   CLayer                  cProjectionX_Variable;
   CNeuronSpikeConvBlock   cA;
   CNeuronSpikeConvBlock   cBC_Time;
   CNeuronSpikeConvBlock   cBC_Variable;
   CNeuronBaseOCL          cB_Time;
   CNeuronBaseOCL          cB_Variable;
   CNeuronBaseOCL          cC_Time;
   CNeuronBaseOCL          cC_Variable;
   CNeuronSpikeConvBlock   cDelta_Time;
   CNeuronSpikeConvBlock   cDelta_Variable;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForwardSSM2D(void);
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradientsSSM2D(void);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpike2DSSMOCL(void)  {};
                    ~CNeuronSpike2DSSMOCL(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint window_out, uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSpike2DSSMOCL; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeMamba2D   :  public CNeuronSpikeConvBlock
  {
protected:
   CNeuronSpikeConvBlock   cXZProject;
   CNeuronBaseOCL          cX;
   CNeuronBaseOCL          cZ;
   CNeuronSpike2DSSMOCL    cSSM;
   CNeuronBaseOCL          cZSSM;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeMamba2D(void) {};
                    ~CNeuronSpikeMamba2D(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint inside_dimension, uint units_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSpikeMamba2D; }
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj);
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSTSSM   :  public CNeuronSpikeConvBlock
  {
protected:
   CNeuronSpikePatchStak   cEmbeddings;
   CLayer                  cProjection;
   CLayer                  cSSM;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeSTSSM(void) {};
                    ~CNeuronSpikeSTSSM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension, uint patch_dimension,
                          uint ssm_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeSTSSM;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      Clear(void) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronESTMEncoder   :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronESTMEncoder(void) {};
                    ~CNeuronESTMEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &stack_size[], uint &dimension[], uint patch_dimension,
                          uint ssm_dimension, uint &variables[],
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronESTMEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      Clear(void) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCreateFlow :  public CNeuronSpikePatchStak
  {
protected:
   CNeuronBaseOCL    cState;
   //---
   virtual bool      CalcFlow(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCreateFlow(void) {};
                    ~CNeuronCreateFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint patch_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronCreateFlow;   }
   //--- methods for working with files
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      Clear(void) override;
   //---
   virtual CNeuronBaseOCL* GetState(void) { return cState.AsObject(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronBiDirectCorrelation :  public CNeuronTMAMFE
  {
protected:
   CNeuronBatchNormOCL        cInpNorm;
   CNeuronBaseOCL             cRefFrame;
   CNeuronTransposeOCL        cRefFrameT;
   CNeuronMultiWindowsConvOCL cWarp;
   CNeuronBatchNormOCL        cWarpNorm;
   CNeuronBaseOCL             cCorrelation;
   CNeuronBaseOCL             cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronBiDirectCorrelation(void) {};
                    ~CNeuronBiDirectCorrelation(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint variables, uint window_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronBiDirectCorrelation;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSATMA  :  public CNeuronBaseOCL
  {
protected:
   CLayer                        cWarp;
   CNeuronBaseOCL                cRefFrame;
   CNeuronRelativeCrossAttention cCrossAttention;
   CNeuronConvOCL                cSpatialAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSATMA(void) {};
                    ~CNeuronSATMA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint variables,
                          uint window_key, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronSATMA; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value)  override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronBAT :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer                              cPrepare;
   CLayer                              cEncoder;
   CLayer                              cContext;
   CNeuronBaseOCL                      cConcatenated;
   CNeuronSpikeConvGRU2D               cGRU;
   CNeuronTransposeRCDOCL              cTranspose;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronBAT(void) {};
                    ~CNeuronBAT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint embed_size,
                          uint stack_size, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronBAT;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   virtual CBufferFloat   *getWeights(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSCM   :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cScale;
   CNeuronBaseOCL    cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeSCM(void) {};
                    ~CNeuronSpikeSCM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeSCM;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return (!cScale[0] ? 0 : ((CNeuronSpikeConvBlock*)cScale[0]).GetWindow());   }
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeDepthWiseConv  :  public CNeuronSpikeConvBlock
  {
protected:
   CNeuronTransposeVRCOCL           cTranspose[2];
   CNeuronMultiWindowsConvWPadOCL   cDepthConv;
   CNeuronBatchNormOCL              cNorm;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeDepthWiseConv(void) {};
                    ~CNeuronSpikeDepthWiseConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint depth_window, uint depth_step,
                          uint units, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeDepthWiseConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeFAM   :  public CNeuronBatchNormOCL
  {
protected:
   CNeuronBaseOCL                   cProduct;
   CNeuronSpikeActivation           cActivation;
   CNeuronSpikeDepthWiseConv        cConv;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronSpikeFAM(void) {};
                    ~CNeuronSpikeFAM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeFAM;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cConv.GetWindow();   }
   virtual uint      GetVariables(void) const { return cConv.GetVariables(); }
   virtual uint      GetUnits(void) const { return cConv.GetUnits(); }
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeMDC   :  public CNeuronBatchNormOCL
  {
protected:
   CLayer            cSCMs;
   CLayer            cFAMs;
   CLayer            cAFFs;
   CLayer            cProjects;
   CLayer            cProjectsOut;
   CLayer            cFeatExtracts;
   CNeuronBaseOCL    cConcatOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeMDC(void) {};
                    ~CNeuronSpikeMDC(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeMDC;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMDS   :  public CNeuronBaseOCL
  {
protected:
   uint                    iDimension;
   uint                    iUnits;
   uint                    iVariables;
   CNeuronBaseOCL          cConcat;
   CLayer                  cScores;
   CNeuronTransposeVRCOCL  cTranspose;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronMDS(void) {};
                    ~CNeuronMDS(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMDS;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeADM : public CNeuronMDS
  {
protected:
   CNeuronSpikeConvBlock   cConv;
   CNeuronSpikeMDC         cMDC;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override { return feedForward(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override {  return updateInputWeights(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override { return calcInputGradients(NeuronOCL); }

public:
                     CNeuronSpikeADM(void) {};
                    ~CNeuronSpikeADM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeADM;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronBATvADM :  public CNeuronBAT
  {
public:
                     CNeuronBATvADM(void) {};
                    ~CNeuronBATvADM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint embed_size,
                          uint stack_size, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronBATvADM;   }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCDCSelfCorrector :  public CNeuronBatchNormOCL
  {
protected:
   uint              iUnits;
   uint              iVariables;
   CLayer            cEstimator;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCDCSelfCorrector(void) {};
                    ~CNeuronCDCSelfCorrector(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint &inside_dimensions[],
                          uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronCDCSelfCorrector;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeCDC  :  public CNeuronSpikeConvBlock
  {
protected:
   CNeuronCDCSelfCorrector       cEsimator;
   CNeuronBaseOCL                cFlowCorrector;
   CNeuronRelativeSelfAttention  cFlowCorrelation;
   CNeuronBaseOCL                cFlowMask;
   CNeuronBaseOCL                cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeCDC(void) {};
                    ~CNeuronSpikeCDC(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint &inside_dimensions[],
                          uint window_out, uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeCDC;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeEEMFLow  :   public CNeuronSpikeCDC
  {
protected:
   uint                    iMaxDisplacement;
   int                     ibShifts;
   CLayer                  cPrepare;
   CLayer                  cEncoder[3];
   CNeuronBaseOCL          cCorrelations[3];
   CNeuronBaseOCL          cConcat[3];
   CLayer                  cDecoder[4];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeEEMFLow(void) : ibShifts(INVALID_HANDLE) {};
                    ~CNeuronSpikeEEMFLow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint embed_size,
                          uint stack_size, uint variables,
                          uint max_displacement,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeEEMFLow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   virtual CBufferFloat   *getWeights(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleExcitation   :  public CNeuronBaseOCL
  {
protected:
   CLayer            cExcitation;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultiScaleExcitation(void) {};
                    ~CNeuronMultiScaleExcitation(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint scales, uint bottleneck,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMultiScaleExcitation;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiScaleDifference :  public CNeuronSpikeConvBlock
  {
protected:
   uint              iaShifts[];
   int               ibShifts;
   //---
   CLayer            cBottleneck;
   CLayer            cExcitation;
   CLayer            cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultiScaleDifference(void) : ibShifts(INVALID_HANDLE) {};
                    ~CNeuronMultiScaleDifference(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint bottleneck, uint &shifts[],
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMultiScaleDifference;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCorrelationEncoder  :  public CNeuronSpikeConvBlock
  {
protected:
   uint                    iMaxDisplacement;
   int                     ibShifts;
   CNeuronBaseOCL          cCorrelation;
   CNeuronSpikeConvBlock   cWarpCorrs;
   CNeuronSpikeConvBlock   cWarpInps;
   CLayer                  cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCorrelationEncoder(void) : ibShifts(INVALID_HANDLE) {};
                    ~CNeuronCorrelationEncoder(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units,
                          uint bottleneck, uint max_displacement,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronCorrelationEncoder;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronEDCFlow :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer                        cPrepare;
   CLayer                        cFeatureEncoder[2];
   CLayer                        cConextEncoder;
   CNeuronMultiScaleDifference   cDiffEncoder;
   CNeuronCorrelationEncoder     cCorrelation;
   CLayer                        cCorrelationUpsampling;
   CLayer                        cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronEDCFlow(void) {};
                    ~CNeuronEDCFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_in, uint units_in, uint bottleneck,
                          uint dimension_out, uint units_out,
                          uint &shifts[], uint max_displacement,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronEDCFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual CBufferFloat   *getWeights(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronCreateICEFlow :  public CNeuronCreateFlow
  {
protected:
   CNeuronBaseOCL       cConcat;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronCreateICEFlow(void) {};
                    ~CNeuronCreateICEFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint patch_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronCreateICEFlow;   }
   //--- methods for working with files
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeMFE  :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeMFE(void) {};
                    ~CNeuronSpikeMFE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint &windows[], uint window_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronSpikeMFE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeMHCrossAttention :  public CNeuronMSRes
  {
protected:
   uint                    iHeads;
   //---
   CNeuronSpikeConvBlock   cQ_Embedding;
   CNeuronSpikeConvBlock   cKV_Embedding;
   int                     ibScoreIndex;
   CNeuronBaseOCL          cMHAttentionOut;
   CNeuronSpikeConvBlock   cW0;
   CNeuronBaseOCL          cAttentionOut;
   CNeuronBaseOCL          cContext;
   //---
   virtual bool            Attention(void);
   virtual bool            AttentionGradients(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronSpikeMHCrossAttention(void) : ibScoreIndex(INVALID_HANDLE) {};
                    ~CNeuronSpikeMHCrossAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count, uint window_k, uint units_k,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronSpikeMHCrossAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeMixFusion   :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cMain;
   CNeuronConvOCL    cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeMixFusion(void) {};
                    ~CNeuronSpikeMixFusion(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_in, uint dimension_out,
                          uint units_count, uint depth,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeMixFusion;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cResidual.GetWindow();   }
   //---
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTFlow :  public CNeuronSpikeConvBlock
  {
protected:
   uint                    iMaxDisplacement;
   int                     ibShifts;
   //---
   CNeuronBatchNormOCL           cNorm;
   CNeuronCreateICEFlow          cICE;
   CNeuronSpikePatchStak         cEvents;
   CLayer                        cEncoder;
   CNeuronBaseOCL                cICEEmbedding;
   CNeuronBaseOCL                cEventsEmbedding;
   CNeuronBaseOCL                cContext;
   CNeuronBaseOCL                cICECorrelation;
   CNeuronBaseOCL                cEventsCorrelation;
   CNeuronBaseOCL                cFeatureVsCorrelation;
   CNeuronSpikeMFE               cMoutionEncoder;
   CNeuronTransposeRCDOCL        cTranspose;
   CNeuronSpikeMHCrossAttention  cCrossModalAggregation;
   CNeuronSpikeMixFusion         cMixFusion;
   CLayer                        cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTFlow(void) {};
                    ~CNeuronSTFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_in, uint units_in, uint bottleneck,
                          uint dimension_out, uint units_out, uint max_displacement,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSTFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPSSE :  public CNeuronAddToStack
  {
protected:
   CLayer                  cPrepare;
   //---
   CParams                 cA;
   CNeuronSpikeConvBlock   cBCDxE;
   //---
   CNeuronBaseOCL          cB;
   CNeuronBaseOCL          cC;
   CNeuronBaseOCL          cDx;
   CNeuronBaseOCL          cE;
   CNeuronBaseOCL          cH;
   //---
   CNeuronBaseOCL          cAE;
   CNeuronTransposeVRCOCL  cAET;
   CNeuronBaseOCL          cAED;
   //---
   CNeuronBaseOCL          cAH;
   CNeuronBaseOCL          cBX;
   CNeuronBaseOCL          cHc;
   CNeuronBaseOCL          cCHc;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPSSE(void) {};
                    ~CNeuronPSSE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint patch_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronPSSE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeDepthWiseResidual :  public CNeuronSpikeDepthWiseConv
  {
protected:
   CNeuronSpikeConvBlock         cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeDepthWiseResidual(void) {};
                    ~CNeuronSpikeDepthWiseResidual(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint depth_window, uint depth_step,
                          uint units, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeDepthWiseResidual;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSuperKernelBlock  :  public CNeuronMSRes
  {
protected:
   CLayer            cSKFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeSuperKernelBlock(void) {};
                    ~CNeuronSpikeSuperKernelBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint &depth_window[], uint &depth_step[],
                          uint units, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeSuperKernelBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeGMA   :  public CNeuronBaseOCL
  {
protected:
   uint              iChanels;
   uint              iUnits;
   uint              iVariables;
   //---
   CLayer            cQKV;
   CNeuronBaseOCL    cQ;
   CNeuronBaseOCL    cK;
   CNeuronTransposeVRCOCL  cKT;
   CNeuronBaseOCL    cV;
   CLayer            cScore;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeGMA(void) {};
                    ~CNeuronSpikeGMA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels, uint units, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeGMA;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronETROF   :  public   CNeuronAddToStack
  {
protected:
   CNeuronPSSE                   cPSSE;
   CNeuronBaseOCL                cRefFrame;
   CNeuronTransposeOCL           cRefFrameT;
   CNeuronBaseOCL                cForwardBackward;
   CNeuronBaseOCL                cCorrelation;
   CLayer                        cMixFusion;
   CNeuronSpikeSuperKernelBlock  cContext;
   CLayer                        cMotionEncoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;


public:
                     CNeuronETROF(void) {};
                    ~CNeuronETROF(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension_in, uint dimension_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronETROF;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPSSEFlow   :  public CNeuronSpikeSuperKernelBlock
  {
protected:
   CNeuronETROF                  cETROF;
   CNeuronSpikeSuperKernelBlock  cLocal;
   CLayer                        cGlobal;
   CLayer                        cUpdateFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPSSEFlow(void) {};
                    ~CNeuronPSSEFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint forecast, uint dimension, uint hidden_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronPSSEFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeUSR   :  public CNeuronSpikeADM
  {
protected:
   CNeuronCreateICEFlow    cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSpikeUSR(void) {};
                    ~CNeuronSpikeUSR(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint stack_size, uint dimension,
                          uint patch_dimension, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeUSR;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronInitialFlow   :  public CNeuronBaseOCL
  {
protected:
   CParams           cCandidates;
   CLayer            cScores;
   CNeuronAddToStack cStack;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronInitialFlow(void) {};
                    ~CNeuronInitialFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint window_out,
                          uint candidates, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronInitialFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSpikeSMR   :  public CNeuronMSRes
  {
protected:
   CNeuronSpikeSuperKernelBlock  cUpsampleFlow;
   CNeuronBaseOCL                cConcat;
   CNeuronSpikeConvGRU2D         cConvGRU;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronSpikeSMR(void) {};
                    ~CNeuronSpikeSMR(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint window, uint second_window, uint window_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronSpikeSMR;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   virtual CBufferFloat* GetHidden() const { return cConvGRU.getOutput(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronEVAFlow :  public   CNeuronSpikeSMR
  {
protected:
   CLayer               cPrepare;
   CLayer               cEncoder;
   CNeuronInitialFlow   cInitFlow;
   CLayer               cDecoder;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override
     { return        feedForward(NeuronOCL); }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override
     { return        updateInputWeights(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(NeuronOCL); }

public:
                     CNeuronEVAFlow(void) {};
                    ~CNeuronEVAFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint forecast, uint dimension, uint hidden_dimension,
                          uint layers, uint candidates,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronEVAFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResFlowHTR :  public   CNeuronMSRes
  {
protected:
   CLayer                  cContextProjection;
   CLayer                  cMFProjection;
   CNeuronBaseOCL          cContextVsMF;
   CNeuronSpikeConvGRU2D   cGRU;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronResFlowHTR(void) {};
                    ~CNeuronResFlowHTR(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint units, uint mf_dimension, uint context_size,
                          uint inside_dimension, uint chanels_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResFlowHTR;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResFlow :  public CNeuronSpikeSuperKernelBlock
  {
protected:
   CNeuronPSSE          cPrepare;
   CNeuronEVAFlow       cLTR;
   CLayer               cContext;
   CLayer               cMFE;
   CNeuronBaseOCL       cUpsampleLTR;
   CNeuronBaseOCL       cMFEvsUpLTR;
   CNeuronResFlowHTR    cHTR;
   CNeuronBaseOCL       cUpdatedFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResFlow(void) {};
                    ~CNeuronResFlow(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint hidden_dimension,
                          uint ltr_forecast, uint htr_points_to_segment,
                          uint ltr_layers, uint ltr_candidates,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronResFlow;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronConfGateTailAwareMoE   :  public CNeuronBaseOCL
  {
protected:
   CLayer            cConfidenceGating;
   CLayer            cScore;
   CParams           cTailAwareMoE;
   CLayer            cCGScore;
   CNeuronAddToStack cCGStack;
   CNeuronAddToStack cScoreStack;
   CNeuronBaseOCL    cDensity;
   //---
   virtual bool      MixExpertsPredict(void);
   virtual bool      MixExpertsPredictGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronConfGateTailAwareMoE(void) {};
                    ~CNeuronConfGateTailAwareMoE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units,
                          uint forecast, uint experts,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronConfGateTailAwareMoE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronResFlowLattice :  public CNeuronResFlow
  {
protected:
   CNeuronConfGateTailAwareMoE   cTailAwareMoE;
   CNeuronBaseOCL                cConcat;
   CNeuronSpikeSuperKernelBlock  cAdaptFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResFlowLattice(void) {};
                    ~CNeuronResFlowLattice(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint hidden_dimension,
                          uint ltr_forecast, uint htr_points_to_segment,
                          uint ltr_layers, uint ltr_candidates,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronResFlowLattice;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CFieldAwareParams    :  public CNeuronBaseOCL
  {
protected:
   uint              iFields;
   CParams           cCandidates;
   CParams           cFieldEmbeddings;
   CLayer            cScore;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override          { return FeedForward(); }

   ///\ingroup neuron_base_opt
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override   { return UpdateInputWeights(); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override   { return true; }

public:
                     CFieldAwareParams(void) {};
                    ~CFieldAwareParams(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint params_to_field, uint fields, uint embed_size,
                          uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      FeedForward(void);
   virtual bool      UpdateInputWeights(void);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defFieldAwareParams; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetFields(void) const { return iFields; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFieldAwareConv   :  public CNeuronBaseOCL
  {
protected:
   CFieldAwareParams    cParams;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronFieldAwareConv(void) {};
                    ~CNeuronFieldAwareConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint window_out, uint fields,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronFieldAwareConv; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cParams.Neurons() / Neurons(); }
   virtual uint      GetFields(void) const { return cParams.GetFields(); }
   virtual uint      GetFilters(void) const { return Neurons() / GetFields(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHFAT :  public CNeuronMSRes
  {
protected:
   uint                    iHeads;
   //---
   CNeuronFieldAwareConv   cQ_Embedding;
   CNeuronFieldAwareConv   cKV_Embedding;
   CParams                 cScale;
   int                     ibScoreIndex;
   CNeuronBaseOCL          cMHAttentionOut;
   CNeuronSpikeConvBlock   cW0;
   CNeuronBaseOCL          cAttentionOut;
   //---
   virtual bool            Attention(void);
   virtual bool            AttentionGradients(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override ;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHFAT(void) {};
                    ~CNeuronMHFAT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint fields,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMHFAT;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHCrossFAT :  public CNeuronMHFAT
  {
protected:
   CNeuronBaseOCL    cContext;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   //---
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronMHCrossFAT(void) {};
                    ~CNeuronMHCrossFAT(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads, uint fields,
                          uint window_cross, uint fields_cross,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defNeuronMHCrossFAT;   }
   //--- methods for working with files
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFieldPatternEmbedding    :  public CNeuronBaseOCL
  {
protected:
   uint              iFields;
   uint              iWindow;
   CParams           cCandidates;
   CLayer            cScore;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronFieldPatternEmbedding(void) {};
                    ~CNeuronFieldPatternEmbedding(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint fields, uint embed_size,
                          uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronFieldPatternEmbedding; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetFields(void) const { return iFields; }
   virtual CBufferFloat   *getWeights(void) override { return (!!cScore[-2] ? ((CNeuronConvOCL*)cScore[-2]).GetWeightsConv() : NULL); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronAutoToken  :  public CNeuronBaseOCL
  {
protected:
   uint              iEmbedingSize;
   uint              iTopK;
   uint              iUnits;
   uint              iGroups;
   //---
   CLayer            cDNN;
   CLayer            cW;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronAutoToken(void) {};
                    ~CNeuronAutoToken(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint embed_size,
                          uint units_in, uint groups,
                          uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronAutoToken; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMultiMixAttention   :  public CNeuronBaseOCL
  {
protected:
   CFieldAwareParams    cParams;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMultiMixAttention(void) {};
                    ~CNeuronMultiMixAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint heads,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMultiMixAttention; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual uint      GetHeads(void) { return cParams.GetFields(); }
   virtual uint      GetUnits(void) { return uint(MathSqrt(cParams.Neurons() / GetHeads())); }
   virtual uint      GetDimension(void) { return Neurons() / GetUnits(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSparseMoEConv  :  public CNeuronBaseOCL
  {
protected:
   CNeuronFieldPatternEmbedding  cParams;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSparseMoEConv(void) {};
                    ~CNeuronSparseMoEConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint filters, uint units,
                          uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSparseMoEConv; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cParams.Neurons() / Neurons(); }
   virtual uint      GetFields(void) const { return cParams.GetFields(); }
   virtual uint      GetFilters(void) const { return Neurons() / GetFields(); }
   virtual CBufferFloat   *getWeights(void) override { return cParams.getWeights(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMTmixAttBlock :  public CNeuronBaseOCL
  {
protected:
   CNeuronMultiMixAttention   cAttention;
   CLayer                     cSharedExpert;
   CLayer                     cFieldMoE;
   CLayer                     cSceneMoE;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMTmixAttBlock(void) {};
                    ~CNeuronMTmixAttBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint heads,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMTmixAttBlock; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CCrossMHFlashAttention   :  public CNeuronBaseOCL
  {
protected:
   uint              iQUnits;
   uint              iKVUnits;
   uint              iHeads;
   uint              iDimension;
   bool              bMask;
   CBufferFloat      cLogSumExp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override  { return true; }

public:
                     CCrossMHFlashAttention(void) {};
                    ~CCrossMHFlashAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint heads, bool mask_future,
                          uint q_units, uint kv_units,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const   {  return defCrossMHFlashAttention;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMixFeedForward :  public CNeuronBaseOCL
  {
protected:
   CLayer                     cSharedExpert;
   CLayer                     cFieldMoE;
   CLayer                     cSceneMoE;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMixFeedForward(void) {};
                    ~CNeuronMixFeedForward(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMixFeedForward; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {};
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CBufferFloat   *getWeights(void) override { return (!!cSceneMoE[-1] ? cSceneMoE[-1].getWeights() : NULL); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronOneTrans   :  public CNeuronMixFeedForward
  {
protected:
   CLayer               cPrepare;
   CNeuronBaseOCL       cSequenceLast;
   CNeuronSparseMoEConv cKVSequenceLast;
   CNeuronAddToStack    cStackSequence;
   CNeuronAddToStack    cStackKVSequence;
   CLayer               cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronOneTrans(void) {};
                    ~CNeuronOneTrans(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint units_out,
                          uint heads, uint stack_size, uint layers,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronOneTrans; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMHTHCrossAttention :  public CNeuronMTmixAttBlock
  {
protected:
   uint              iQUnits;
   uint              iXUnits;
   uint              iHeads;
   uint              iXDimension;
   bool              bMask;
   //---
   CBufferFloat      cLogSumExp;
   CLayer            cPrepareQ;
   CLayer            cW0;
   //---
   virtual bool      AttentionOut(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);
   virtual bool      AttentionInsideGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
         CBufferFloat *SecondGradient,
         ENUM_ACTIVATION SecondActivation = None);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronMHTHCrossAttention(void) {};
                    ~CNeuronMHTHCrossAttention(void) {};
   //---
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_q, uint units_q, uint heads,
                          uint dimension_x, uint unit_x,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMHTHCrossAttention; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSTCA :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer               cPrepare;
   CNeuronBaseOCL       cLastSequence;
   CNeuronBaseOCL       cLastNonSequence;
   CNeuronAddToStack    cStackSequence;
   CFieldAwareParams    cScenarios;
   CLayer               cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSTCA(void) {};
                    ~CNeuronSTCA(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint units_out,
                          uint heads, uint scenarios, uint stack_size, uint layers,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSTCA; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronWeightGenerator :  public CNeuronConvOCL
  {
protected:
   CNeuronConvOCL    cInputProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronWeightGenerator(void) {};
                    ~CNeuronWeightGenerator(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint scenarios,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronWeightGenerator; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPCGR :  public CNeuronTransposeRCDOCL
  {
protected:
   uint                    iWindowIn;
   uint                    iInsideDimension;
   uint                    iWindowOut;
   uint                    iQuerys;
   uint                    iScenarios;
   //---
   CNeuronFieldAwareConv   cShared;
   CFieldAwareParams       cScenarios;
   CNeuronBaseOCL          cQueryVsScenarios;
   CNeuronWeightGenerator  cWQscenarios;
   CNeuronBaseOCL          cGeneratedQuery;
   CNeuronTransposeRCDOCL  cQScInToScQIn;
   CFieldAwareParams       cWKshared;
   CNeuronWeightGenerator  cWKscenarios;
   CNeuronTransposeOCL     cWKscT;
   CNeuronBaseOCL          cWKT;
   CNeuronTransposeOCL     cWK;
   CNeuronBaseOCL          cQueryMod;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPCGR(void) {};
                    ~CNeuronPCGR(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint querys, uint scenarios,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronPCGR; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronADS :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer               cPrepare;
   CNeuronBaseOCL       cLastSequence;
   CNeuronBaseOCL       cLastNonSequence;
   CNeuronAddToStack    cStackSequence;
   CNeuronPCGR          cQuerys;
   CLayer               cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronADS(void) {};
                    ~CNeuronADS(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint units_out,
                          uint heads, uint scenarios, uint stack_size, uint layers,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronADS; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPerTokenFFN   :  public CNeuronFieldAwareConv
  {
protected:
   CNeuronFieldAwareConv   cProj;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPerTokenFFN(void) {};
                    ~CNeuronPerTokenFFN(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronPerTokenFFN; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual uint      GetWindow(void) const override { return cProj.GetWindow(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronScenariosToken :  public CNeuronFieldPatternEmbedding
  {
protected:
   CNeuron2DSSMOCL   cScenariosGenerator;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronScenariosToken(void) {};
                    ~CNeuronScenariosToken(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint fields, uint window_out, uint scenarios,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronScenariosToken; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUnifiedTokenizer :  public CNeuronFieldPatternEmbedding
  {
protected:
   CNeuronBatchNormOCL        cNorm;
   CNeuronMultiWindowsConvOCL cProj;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronUnifiedTokenizer(void) {};
                    ~CNeuronUnifiedTokenizer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronUnifiedTokenizer; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDomainAwareAttention :  public CNeuronPerTokenFFN
  {
protected:
   uint              iQUnits;
   uint              iXUnits;
   uint              iHeads;
   uint              iXDimension;
   //---
   CBufferFloat      cLogSumExp;
   CLayer            cPrepareQ;
   CLayer            cW0;
   //---
   virtual bool      AttentionOut(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context);
   virtual bool      AttentionInsideGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
         CBufferFloat *SecondGradient,
         ENUM_ACTIVATION SecondActivation = None);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronDomainAwareAttention(void) {};
                    ~CNeuronDomainAwareAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_q, uint units_q, uint heads,
                          uint dimension_x, uint unit_x,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronDomainAwareAttention; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronFeatureSelfIteration   :  public CNeuronDomainAwareAttention
  {
protected:
   CNeuronAddToStack          cStack;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override
     { return        feedForward(NeuronOCL); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override
     { return        calcInputGradients(prevLayer); }
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override
     { return        updateInputWeights(NeuronOCL); }

public:
                     CNeuronFeatureSelfIteration(void) {};
                    ~CNeuronFeatureSelfIteration(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_q, uint units_q, uint heads, uint stack_size,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronFeatureSelfIteration; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual CNeuronAddToStack*   GetStack(void) { return cStack.AsObject(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDomainFused  :  public CNeuronPerTokenFFN
  {
protected:
   CNeuronBaseOCL       cTasks;
   CNeuronBaseOCL       cScenarios;
   CLayer               cSelector;
   CNeuronBaseOCL       cScenariosToTask;
   CNeuronBaseOCL       cConcat;
   //---
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronDomainFused(void) {};
                    ~CNeuronDomainFused(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint tasks, uint scenarios,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronDomainFused; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMDL :  public CNeuronPerTokenFFN
  {
protected:
   CNeuronUnifiedTokenizer    cPrepare;
   CNeuronBaseOCL             cLastNonSequence;
   CLayer                     cFeatures;
   CNeuronScenariosToken      cScenarios;
   CLayer                     cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMDL(void) {};
                    ~CNeuronMDL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads,
                          uint scenarios, uint stack_size, uint layers,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMDL; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUniMixerTokenizer   :  public CNeuronFieldAwareConv
  {
protected:
   CNeuronUnifiedTokenizer cTokenizer;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronUniMixerTokenizer(void) {};
                    ~CNeuronUniMixerTokenizer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units, uint embed_size,
                          uint blocks, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronUnifiedTokenizer; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronPerTokenSwiGLU   :  public CNeuronPerTokenFFN
  {
protected:
   CNeuronFieldAwareConv   cGate;
   CNeuronBaseOCL          cAfterGate;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronPerTokenSwiGLU(void) {};
                    ~CNeuronPerTokenSwiGLU(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronPerTokenSwiGLU; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUniMixer  :  public CNeuronPerTokenSwiGLU
  {
protected:
   uint              iBlocks;
   //---
   CFieldAwareParams cWb;
   CNeuronBaseOCL    cLocalMixing;
   CParams           cG[2];
   CNeuronBaseOCL    cGlobalMixing[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronUniMixer(void) {};
                    ~CNeuronUniMixer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint blocks,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronUniMixer; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSiameseNorm   : public CNeuronBaseOCL
  {
protected:
   CNeuronBaseOCL       cNormYIn;
   CNeuronBaseOCL       cSumXYIn;
   CNeuronUniMixer      cMixer;
   CNeuronBaseOCL       cYOut;
   //---
   virtual bool         feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool         feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;
   virtual bool         calcInputGradients(CNeuronBaseOCL *prevLayer, CBufferFloat *SecondInput,
                                           CBufferFloat *SecondGradient,
                                           ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool         calcInputGradients(CNeuronBaseOCL *prevLayer) override;
   virtual bool         updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool         updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *Context) override;

public:
                     CNeuronSiameseNorm(void) {};
                    ~CNeuronSiameseNorm(void) {};
   //---
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint blocks,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSiameseNorm; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual CNeuronBaseOCL*  GetYOut(void) { return cYOut.AsObject(); }
   //---
   virtual uint      GetWindow(void) const { return cMixer.GetWindow(); }
   virtual uint      GetFields(void) const { return cMixer.GetFields(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUniMixerBlock :  public CNeuronPerTokenSwiGLU
  {
protected:
   CNeuronUniMixerTokenizer   cPrepare;
   CNeuronBaseOCL             cLastNonSequence;
   CLayer                     cFeatures;
   CNeuronScenariosToken      cScenarios;
   CLayer                     cFlow;
   CLayer                     cOutputProj;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronUniMixerBlock(void) {};
                    ~CNeuronUniMixerBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint blocks,
                          uint scenarios, uint stack_size, uint layers,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronUniMixerBlock; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronINFNetScenarios  :  public CNeuronBaseOCL
  {
protected:
   CLayer            cSequence;
   CLayer            cContext;
   CLayer            cGlobal;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronINFNetScenarios(void) {};
                    ~CNeuronINFNetScenarios(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window_in, uint &fields[], uint window_out,
                          uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronINFNetScenarios; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override {  activation = None; }

   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronINFNetHUBSequence  :   public CNeuronSpikeConvBlock
  {
protected:
   CNeuronSpikeConvBlock   cProjection;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronINFNetHUBSequence(void) {};
                    ~CNeuronINFNetHUBSequence(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units_count, uint hub_size, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronINFNetHUBSequence;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   //---
   virtual uint      GetWindow(void) const { return cProjection.GetWindow();   }
   virtual uint      GetUnits(void) const { return cProjection.GetUnits(); }
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronINFNetBlock :  public CNeuronPerTokenSwiGLU
  {
protected:
   uint                       iDimension;
   uint                       iSequenceUnits;
   uint                       iSequenceVar;
   uint                       iContextUnits;
   uint                       iScenariosUnits;
   CNeuronUniMixerTokenizer   cPrepare;
   CLayer                     cSequence;
   CLayer                     cContext;
   CLayer                     cScenarios;
   CLayer                     cHUBs;
   //---
   virtual bool      INFNetBGU(CNeuronBaseOCL* scale_and_shift, CNeuronBaseOCL* sequence_in, CNeuronBaseOCL* sequence_out,
                               CNeuronBaseOCL* context_in, CNeuronBaseOCL* context_out,
                               CNeuronBaseOCL* scenarios_in, CNeuronBaseOCL* scenarios_out);
   virtual bool      INFNetBGUGrad(CNeuronBaseOCL* scale_and_shift, CNeuronBaseOCL* sequence_in, CNeuronBaseOCL* sequence_out,
                                   CNeuronBaseOCL* context_in, CNeuronBaseOCL* context_out,
                                   CNeuronBaseOCL* scenarios_in, CNeuronBaseOCL* scenarios_out);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronINFNetBlock(void) {};
                    ~CNeuronINFNetBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads, uint stack_size,
                          uint layers, uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronINFNetBlock; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronSequenceTokenizer :  public CNeuronBaseOCL
  {
protected:
   CNeuronBatchNormOCL        cNorm;
   CNeuronSpikeConvBlock      cProj[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronSequenceTokenizer(void) {};
                    ~CNeuronSequenceTokenizer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units,  uint embed_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronSequenceTokenizer; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual CNeuronBatchNormOCL*    GetNorm(void) { return cNorm.AsObject(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDifferenceTokenizer :  public CNeuronBaseOCL
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *second) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronDifferenceTokenizer(void) {};
                    ~CNeuronDifferenceTokenizer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint inpSize,
                          uint tokenSize, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronDifferenceTokenizer; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronReGENTADBackbone :  public CNeuronBaseOCL
  {
protected:
   CLayer                     cSeqToken;
   CLayer                     cTransformer;
   CLayer                     cRecurrent;
   CNeuronDifferenceTokenizer cDifference;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronReGENTADBackbone(void) {};
                    ~CNeuronReGENTADBackbone(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads, uint stack_size,
                          uint layers, uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronReGENTADBackbone; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   virtual CNeuronBatchNormOCL*    GetNorm(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronReGENTADHead  :  public CNeuronRevINDenormOCL
  {
protected:
   CLayer            cFlow;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronReGENTADHead(void) {};
                    ~CNeuronReGENTADHead(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension_in, uint units_in, uint dimension_out,
                          uint units_out, CNeuronBatchNormOCL* norm,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronReGENTADHead; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
   virtual bool      SetNorm(CNeuronBatchNormOCL* norm);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronReGENTAD   :  public CNeuronBaseOCL
  {
protected:
   CNeuronReGENTADBackbone*   cBackbone;
   CLayer                     cHeads[3];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual CNeuronReGENTADBackbone* CreateBackbone(void) { return new CNeuronReGENTADBackbone(); }

public:
                     CNeuronReGENTAD(void) : cBackbone(NULL) {};
                    ~CNeuronReGENTAD(void) { DeleteObj(cBackbone); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads, uint stack_size,
                          uint layers, uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronReGENTAD; }
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual void      SetActivationFunction(ENUM_ACTIVATION value) override { };
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronDictionaryCrossAtt   :  public CNeuronSpikeConvBlock
  {
protected:
   uint                    iDictionarySize;
   uint                    iHeads;
   uint                    iHiddenDimension;
   int                     bufLogSumExp;
   CParams                 cDictionaryKeys;
   CParams                 cDictionaryValues;
   CParams                 cPrototypes;
   CParams                 cLevels;
   CNeuronSpikeConvBlock   cQuerys;
   CNeuronBaseOCL          cAttentionOut;
   CNeuronBaseOCL          cSimilarity;
   CNeuronSpikeConvBlock   cW0;
   CNeuronSpikeConv        cProjSimilarity;
   CNeuronSpikeConvBlock   cFeedForward0;
   //---
   virtual bool      MHCrossAttvsSim(void);
   virtual bool      MHCrossAttvsSimGrad(void);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronDictionaryCrossAtt(void)   : bufLogSumExp(INVALID_HANDLE) {};
                    ~CNeuronDictionaryCrossAtt(void) { if(bufLogSumExp >= 0 && !!OpenCL) OpenCL.BufferFree(bufLogSumExp); }
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint heads, uint Dictionary_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronDictionaryCrossAtt;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual uint      GetWindow(void) const { return cQuerys.GetWindow();   }
   //---
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
   //---
   virtual CNeuronBaseOCL* GetSimilarity(void)  { return cSimilarity.AsObject(); }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGDformerBackbone :  public CNeuronReGENTADBackbone
  {
protected:
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronGDformerBackbone(void) {};
                    ~CNeuronGDformerBackbone(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads, uint stack_size,
                          uint layers, uint embed_size, uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronGDformerBackbone; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronGDformer   :  public CNeuronReGENTAD
  {
protected:
   virtual CNeuronReGENTADBackbone* CreateBackbone(void) override { return new CNeuronGDformerBackbone(); }

public:
                     CNeuronGDformer(void) {};
                    ~CNeuronGDformer(void) {};
   //---
   virtual int       Type(void) override const  {  return defNeuronGDformer; }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMomADTTM   :  public CNeuronBaseOCL
  {
protected:
   uint                 iCandidates;
   uint                 iDecisions;
   uint                 iDimension;
   uint                 iTopK;
   CNeuronBaseOCL       cDistance;
   CNeuronBaseOCL       cMeanDistance;
   CNeuronSparseSoftMax cScores;
   CBufferFloat         cCandidatesGradient;
   //---
   virtual bool      MeanScores(void);
   virtual bool      MeanScoresGradient(void);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override  { return true; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronMomADTTM(void) : iCandidates(0),
                     iDecisions(0),
                     iDimension(0),
                     iTopK(0) {};
                    ~CNeuronMomADTTM(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint latent_dim, uint candidates, uint decisions,
                          uint topK, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMomADTTM; }
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMomADMPI  :  public CNeuronCrossAttention
  {
protected:
   CNeuronAddToStack   cStack;
   CNeuronMomADTTM     cTTM;
   CNeuronCrossAttention cPolicyMarket;
   CNeuronCrossAttention cTTMHistory;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { ReturnFalse; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronMomADMPI(void) { cContext = NULL; };
                    ~CNeuronMomADMPI(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint latent_dim, uint candidates, uint market_dim,
                          uint market_units, uint stack_size, uint topK,
                          uint heads, uint key_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronMomADMPI; }
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUncAD_MSD  :  public CNeuronBaseOCL
  {
protected:
   uint              iDimension;
   uint              iUnits;
   uint              iVariables;
   uint              iQuantiles;
   float             fQuantSize;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return true; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { return (!!NeuronOCL && NeuronOCL.getGradient().Fill(0)); };

public:
                     CNeuronUncAD_MSD(void) {};
                    ~CNeuronUncAD_MSD(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint dimension, uint units, uint variables, uint quantiles,
                          float range, ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronUncAD_MSD;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUncAD_MUE  :  public CNeuronSpikeConvBlock
  {
protected:
   CLayer            cMSDFlow;
   CLayer            cUncertainty;
   CNeuronBaseOCL    cConcatenated;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronUncAD_MUE(void) {};
                    ~CNeuronUncAD_MUE(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint variables,
                          uint stack_size, uint quantiles, float range,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronUncAD_MUE;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj)   override;
   virtual void      TrainMode(bool flag) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUncAD_UGP  :  public CNeuronSpikeMHCrossAttention
  {
protected:
   CNeuronMultiMixAttention      cSelfAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronUncAD_UGP(void) {};
                    ~CNeuronUncAD_UGP(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint context_window,
                          uint context_units, uint heads, uint key_dimension,
                          uint candidates, uint topK,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronUncAD_UGP;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUncAD_UGPL :  public CNeuronSpikeMHCrossAttention
  {
protected:
   CParams                     cMode;
   CNeuronSpikeMHCrossAttention cModeFusion;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;

public:
                     CNeuronUncAD_UGPL(void) {};
                    ~CNeuronUncAD_UGPL(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint units, uint context_window,
                          uint context_units, uint mode_units, uint heads,
                          uint key_dimension,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronUncAD_UGPL;  }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronUncAD : public CNeuronSpikeConvBlock
  {
protected:
   uint                         iInputDim;
   uint                         iAgentUnits;
   uint                         iContextUnits;
   uint                         iEmbedSize;
   uint                         iQuantiles;
   uint                         iCandidates;
   uint                         iTopK;
   //---
   CNeuronUniMixerTokenizer     cTokenizer;
   CNeuronBaseOCL               cAgents;
   CNeuronBaseOCL               cContext;
   CNeuronUncAD_MUE             cMUE;
   CNeuronUncAD_UGP             cUGP;
   CNeuronUncAD_UGPL            cUGPL;
   CNeuronSpikeMHCrossAttention cMapInteraction;
   //--- tau heads
   CNeuronConvOCL               cTauAValues;
   CNeuronConvOCL               cTauAScores;
   CNeuronSparseSoftMax         cTauASoftMax;
   CNeuronBaseOCL               cTauAWeighted;
   CNeuronConvOCL               cTauEValues;
   CNeuronConvOCL               cTauEScores;
   CNeuronSparseSoftMax         cTauESoftMax;
   CNeuronBaseOCL               cTauEWeighted;
   //--- selection/fusion
   CNeuronBaseOCL               cFusionInput;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      TauAForward(void);
   virtual bool      TauEForward(void);
   virtual bool      TauAGradients(void);
   virtual bool      TauEGradients(void);

public:
                     CNeuronUncAD(void) : iInputDim(0),
                     iAgentUnits(0),
                     iContextUnits(0),
                     iEmbedSize(0),
                     iQuantiles(0),
                     iCandidates(0),
                     iTopK(0) {};
                    ~CNeuronUncAD(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads,
                          uint stack_size, uint quantiles, uint embed_size,
                          uint candidates, uint topK, uint latent_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override const  {  return defNeuronUncAD;  }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuronMomAD : public CNeuronUncAD
  {
protected:
   CNeuronMomADMPI              cMPI;
   bool                         bTauELinked;
   //---
   virtual bool      TauEForward(void) override;
   virtual bool      TauEGradients(void) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   bool              LinkTauEToMPI(void);
   bool              DetachTauEFromMPI(void);

public:
                     CNeuronMomAD(void) : bTauELinked(false) {};
                    ~CNeuronMomAD(void) { DetachTauEFromMPI(); };
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint &dimensions[], uint units_s, uint heads,
                          uint stack_size, uint quantiles, uint embed_size,
                          uint candidates, uint topK, uint latent_count,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override const  {  return defNeuronMomAD;  }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   virtual void      TrainMode(bool flag) override;
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual bool      Clear(void) override;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#endif // NEURONET_DECLARATIONS_MQH

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
