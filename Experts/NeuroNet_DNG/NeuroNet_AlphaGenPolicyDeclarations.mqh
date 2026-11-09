//+------------------------------------------------------------------+
//| AlphaGen policy declarations.                                    |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAGEN_POLICY_DECLARATIONS_MQH
#define NEURONET_ALPHAGEN_POLICY_DECLARATIONS_MQH

class CAlphaGenBufferView : public CBufferFloat
  {
public:
   bool                                Bind(COpenCLMy *context, const int index, const int count);
   ~CAlphaGenBufferView(void);
  };

class CNeuronAlphaGenDenseOCL : public CNeuronBaseOCL
  {
protected:
   uint                                m_iRandom;
   CBufferFloat                        m_cWeightGradient, m_cFlags, m_cOnes, m_cScalar;
   int                                 m_iRows;
   virtual float                       GenerateWeight(void) override;

public:
   CNeuronAlphaGenDenseOCL(void);
   bool                                Configure(COpenCLMy *context, const int inputs, const int outputs, const uint seed,
                  const ENUM_ACTIVATION function = None);
   bool                                Geometry(const int inputs, const int outputs);
   bool                                AllocateChecks(void);
   bool                                ForwardDense(CNeuronBaseOCL *source);
   bool                                Copy(CBufferFloat *source);
   bool                                Outer(CNeuronBaseOCL *next);
   CBufferFloat     *WeightGradient(void);
   bool                                Check(CNeuronAlphaGenEvaluatorOCL *evaluator);
   bool                                Apply(void);
   bool                                SaveState(const int handle);
   bool                                LoadState(const int handle, COpenCLMy *context);
  };

class CNeuronAlphaGenPolicyLSTMOCL : public CNeuronAlphaGenLSTMOCL
  {
protected:
   uint                                m_iRandom;
   virtual float                       GenerateWeight(void) override;

public:
   CNeuronAlphaGenPolicyLSTMOCL(void);
   void                                Seed(const uint seed);
   int                                 RecurrentWeightsIndex(void) { return(m_cWeightsLSTM.GetIndex()); }
   bool                                Geometry(const int inputs, const int hidden, const int capacity);
   bool                                Check(CNeuronAlphaGenEvaluatorOCL *evaluator);
   bool                                SaveState(const int handle);
   bool                                LoadState(const int handle, COpenCLMy *context);
  };

class CNeuronAlphaGenPolicyOCL : public CNeuronBaseOCL
  {
protected:
   CBufferFloat                        m_cConfig, m_cSample, m_cAction, m_cMetadata, m_cNoise;
   uint                                m_iRandom;
   bool                                m_bReady;
   int                                 m_iMode;
   float                               Uniform(void);
   int                                 EmbeddingDim(void);
   int                                 SearchHidden(void);
   int                                 TradeHidden(void);
   int                                 MLP(void);
   bool                                Allocate(CBufferFloat &buffer, const int count);
   bool                                Setup(COpenCLMy *context, const float &config[], const uint seed);
   bool                                SequenceMap(CBufferFloat *source, CBufferFloat *output, const int length, const int hidden, const int mode,
                    CBufferFloat *account);
   bool                                Distribution(CNeuronBaseOCL *head, CBufferFloat *mask, const bool categorical, const int mode);
   bool                                NumericSpan(CNeuronAlphaGenEvaluatorOCL *evaluator, CBufferFloat *buffer, const int count);
   bool                                ReLU(CNeuronBaseOCL *layer, const int mode);
   bool                                SaveCommon(const int handle);
   bool                                LoadCommon(const int handle, COpenCLMy *context);

public:
   CNeuronAlphaGenPolicyOCL(void);
   CBufferFloat     *Sample(void);
   CBufferFloat     *Action(void);
   CBufferFloat     *Metadata(void);
  };

class CNeuronAlphaGenSearchPolicyOCL : public CNeuronAlphaGenPolicyOCL
  {
protected:
   CNeuronAlphaGenDenseOCL             m_cEmbedding, m_cMean, m_cHead;
   CNeuronAlphaGenPolicyLSTMOCL        m_cFirst, m_cSecond;
   CBufferFloat                        m_cEmbedded, m_cEmbeddedGradient, m_cFirstGradient;
   CBufferFloat *m_cTokens;
   int                                 m_iLength;
   bool                                Workspace(void);

public:
   CNeuronAlphaGenSearchPolicyOCL(void);
   bool                                Configure(COpenCLMy *context, const float &config[], const uint seed);
   int                                 BatchWorkStride(void);
   bool                                GenerateBatch(const int count, CBufferFloat *results, CBufferFloat *workspace);
   bool                                ForwardPrefix(CBufferFloat *tokens, const int length, CBufferFloat *mask, const int mode = 0);
   bool                                Backward(CNeuronAlphaGenEvaluatorOCL *evaluator);
   bool                                Apply(void);
   bool                                Save(const int handle);
   bool                                Load(const int handle, COpenCLMy *context);
  };

class CNeuronAlphaGenTradingPolicyOCL : public CNeuronAlphaGenPolicyOCL
  {
protected:
   CNeuronAlphaGenPolicyLSTMOCL        m_cRecurrent;
   CNeuronAlphaGenDenseOCL             m_cInput, m_cHidden1, m_cHidden2, m_cHead;
   CBufferFloat                        m_cSignalGradient;
   CBufferFloat *m_cSignal;
   int                                 m_iLength;

public:
   CNeuronAlphaGenTradingPolicyOCL(void);
   bool                                Configure(COpenCLMy *context, const float &config[], const uint seed);
   bool                                Encode(CBufferFloat *signal, const int length, CBufferFloat *anchor = NULL);
   bool                                Heads(CBufferFloat *account, const int mode = 0);
   bool                                Forward(CBufferFloat *signal, const int length, CBufferFloat *anchor, CBufferFloat *account,
                const int mode = 0);
   bool                                Backward(CNeuronAlphaGenEvaluatorOCL *evaluator);
   bool                                Apply(void);
   CBufferFloat     *Anchor(void);
   bool                                CommitAnchor(void);
   bool                                ResetAnchor(void);
   bool                                Save(const int handle);
   bool                                Load(const int handle, COpenCLMy *context);
  };

#endif
