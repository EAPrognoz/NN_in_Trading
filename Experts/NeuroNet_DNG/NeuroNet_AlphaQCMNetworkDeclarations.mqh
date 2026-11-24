//+------------------------------------------------------------------+
//|                         NeuroNet_AlphaQCMNetworkDeclarations.mqh |
//|                                                     NeuroNet_DNG |
//+------------------------------------------------------------------+
#ifndef NEURONET_ALPHAQCM_NETWORK_DECLARATIONS_MQH
#define NEURONET_ALPHAQCM_NETWORK_DECLARATIONS_MQH
#include "AlphaQCM_NetworkContract.mqh"

class CNeuronAlphaQCMEncoderOCL;

enum ENUM_AQC_MATRIX_LAYOUT
  {
   AQC_EMBED_ROWS,
   AQC_LSTM_FIOG_ROWS,
   AQC_DENSE_BIAS_LAST_ROWS
  };

//+------------------------------------------------------------------+
//| SAQCInferenceMatrix - Borrowed row-major FP32 parameter matrix.   |
//+------------------------------------------------------------------+
struct SAQCInferenceMatrix
  {
   int                    buffer;
   int                    width;
   int                    rows;
   ENUM_AQC_MATRIX_LAYOUT layout;
  };

//+------------------------------------------------------------------+
//| SAQCInferenceView - Short-lived parameter view from one encoder.  |
//+------------------------------------------------------------------+
struct SAQCInferenceView
  {
   CNeuronAlphaQCMEncoderOCL *owner;
   COpenCLMy                 *context;
   ulong                     revision;
   ulong                     lifetime;
   int                       length;
   int                       matrix_count;
   bool                      iqn;
   SAQCInferenceMatrix       matrices[6];
  };

class CAlphaQCMRecurrentOwner : public CNeuronAlphaGenPolicyLSTMOCL
  {
public:
   int FirstIndex(void) const { return(m_cFirstMomentumLSTM.GetIndex()); }
   int SecondIndex(void) const { return(m_cSecondMomentumLSTM.GetIndex()); }
  };

struct SAQCMatrix
  {
   int                    weights;
   int                    gradient;
   int                    first;
   int                    second;
   int                    width;
   int                    rows;
   ENUM_AQC_MATRIX_LAYOUT layout;
  };

enum ENUM_AQC_NET_BUFFER
  {
   AQX_EMBED,AQX_TAPE1,AQX_TAPE2,AQX_STATE,AQX_MEAN,AQX_COS,AQX_PHI,AQX_PRODUCT,
   AQX_HIDDEN,AQX_RESULT,AQX_GHIDDEN,AQX_GPRODUCT,AQX_GPHI,AQX_GMEAN,AQX_GINTER,
   AQX_GEMBED,AQX_GATES,AQX_FLAGS,AQX_COUNT
  };

class CNeuronAlphaQCMEncoderOCL : public CNeuronBaseOCL
  {
protected:
   CNeuronAlphaGenDenseOCL  m_embedding;
   CNeuronAlphaGenDenseOCL  m_tau;
   CNeuronAlphaGenDenseOCL  m_hidden;
   CNeuronAlphaGenDenseOCL  m_head;
   CAlphaQCMRecurrentOwner  m_first;
   CAlphaQCMRecurrentOwner  m_second;
   CBufferFloat             m_work[AQX_COUNT];
   int                      m_L;
   int                      m_A;
   int                      m_E;
   int                      m_H;
   int                      m_M;
   int                      m_B;
   int                      m_N;
   int                      m_tokens;
   int                      m_lengths;
   int                      m_taus;
   int                      m_prepared;
   float                    m_position_scale;
   bool                     m_iqn;
   bool                     m_forward_ready;
   bool                     m_gradient_ready;
   bool                     m_bound;
   ulong                    m_revision;
   ulong                    m_viewLifetime;

public:
   CNeuronAlphaQCMEncoderOCL(void);
   ~CNeuronAlphaQCMEncoderOCL(void);
   virtual int Type(void) const override { return(defNeuronAlphaQCMEncoderOCL); }
   bool Configure(COpenCLMy *context,const bool iqn,const int length,const int actions,
                  const int embedding,const int hidden,const int frequencies);
   bool InitializeWeights(const ulong seed,const uint stream);
   bool ExportInferenceViews(SAQCInferenceView &view);
   bool ValidateInferenceView(const SAQCInferenceView &view);
   bool ReserveBatch(const int rows,const int quantiles);
   void ReleaseWorkspace(void);
   bool BindBatch(const int tokens,const int lengths,const int tau,const int rows,const int quantiles,
                  const int first,const float position_scale);
   int Result(void) const { return(m_work[AQX_RESULT].GetIndex()); }
   virtual int getOutputIndex(void) const override { return(Result()); }
   bool BeginUpdate(void);
   bool PrepareGradient(const int gradient);
   bool Backward(const int gradient);
   bool Preflight(const int flags,const int offset,const bool parameters=true);
   bool Apply(void);
   virtual bool WeightsUpdate(CNeuronBaseOCL *source,float fraction) override;
   virtual bool Save(const int handle) override;
   virtual bool Load(const int handle) override;

protected:
   int  MatrixCount(void) const { return(m_iqn?6:5); }
   bool MatrixView(const int id,SAQCMatrix &v);
   bool Geometry(void);
   bool Workspace(const bool backward);
   bool ReserveInput(int &buffer,const long count);
   void ReleaseInput(int &buffer);
   bool Recur(const bool second);
   bool Reverse(const bool second);
   bool DenseRows(const int x,const int w,const int y,const int rows,const int inputs,const int outputs,const int relu);
   bool DenseBackward(const int x,const int dy,const int dx,const int parameter,const int rows,const bool relu);
   bool RunBackward(void);
   virtual bool feedForward(CNeuronBaseOCL *previous) override;
   virtual bool calcInputGradients(CNeuronBaseOCL *previous) override;
   virtual bool updateInputWeights(CNeuronBaseOCL *previous) override;
  };
#endif
