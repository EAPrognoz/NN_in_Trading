//+------------------------------------------------------------------+
//|                                              NetCreatorPanel.mqh |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "..\..\NeuroNet_DNG\NeuroNet.mqh"
#include <Controls\Dialog.mqh>
#include <Controls\Label.mqh>
#include <Controls\ListView.mqh>
#include <Controls\SpinEdit.mqh>
#include <Controls\ComboBox.mqh>
#include <Controls\Button.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
//--- indents and gaps
#define INDENT_LEFT                         (11)      // indent from left (with allowance for border width)
#define INDENT_TOP                          (11)      // indent from top (with allowance for border width)
#define INDENT_RIGHT                        (11)      // indent from right (with allowance for border width)
#define INDENT_BOTTOM                       (11)      // indent from bottom (with allowance for border width)
#define CONTROLS_GAP_X                      (5)       // gap by X coordinate
#define CONTROLS_GAP_Y                      (5)       // gap by Y coordinate
//--- for buttons
#define BUTTON_HEIGHT                       (20)      // size by Y coordinate
//--- for the indication area
#define EDIT_WIDTH                          (100)     // size by X coordinate
#define EDIT_HEIGHT                         (20)      // size by Y coordinate
//--- for group controls
#define LIST_WIDTH                          (280)     // size by X coordinate
#define ADDS_WIDTH                          (200)     // size by X coordinate
//---
#define LABEL_NAME                          "Label"
#define EDIT_NAME                           "Edit"
//---
#define DEFAULT_NEURONS                     (1000)
#define DEFAULT_OPTIMIZATION                ADAM
#define DEFAULT_ACTIVATION                  SIGMOID
#define DEFAULT_PROBABILITY                 (0.3)
//---
#define PANEL_WIDTH                         (2 * LIST_WIDTH + ADDS_WIDTH + \
                                             2 * CONTROLS_GAP_X + \
                                             INDENT_LEFT + INDENT_RIGHT + \
                                             4 * CONTROLS_DIALOG_CLIENT_OFF)
#define PANEL_HEIGHT                        (INDENT_TOP + INDENT_BOTTOM + \
                                             12 * (EDIT_HEIGHT + CONTROLS_GAP_Y) + \
                                             BUTTON_HEIGHT + CONTROLS_GAP_Y + \
                                             4 * CONTROLS_BORDER_WIDTH + \
                                             CONTROLS_DIALOG_CAPTION_HEIGHT)
//---
#define EXTENSION                           ".nnw"
//---
#define KEY_UP                               38
#define KEY_DOWN                             40
#define KEY_DELETE                           46
//---
#define ON_EVENT_CONTROL(event,control,handler)          if(id==(event+CHARTEVENT_CUSTOM) && lparam==control.Id()) { handler(control); return(true); }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNetModify :  public CNet
  {
public:
                     CNetModify(void) {};
                    ~CNetModify(void) {};
   //---
   uint              LayersTotal(void);
   CArrayObj*        GetLayersDiscriptions(void);
   virtual bool      AddLayer(CLayer* new_layer);
   virtual bool      AddLayers(CArrayObj* new_layers);
   virtual CLayer*   GetLayer(uint layer);
   COpenCLMy*        GetOpenCL(void)   {return opencl;}
   void              SetOpenCL(COpenCLMy *OpenCL)  {opencl = OpenCL;}
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint CNetModify::LayersTotal(void)
  {
   if(!layers)
      return 0;
//---
   return layers.Total();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CArrayObj* CNetModify::GetLayersDiscriptions(void)
  {
   CArrayObj* result = new CArrayObj();
   for(uint i = 0; i < LayersTotal(); i++)
     {
      CLayer* layer = layers.At(i);
      if(!layer)
         break;
      CNeuronBaseOCL* neuron = layer.At(0);
      if(!neuron)
         break;
      if(!result.Add(neuron.GetLayerInfo()))
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayer* CNetModify::GetLayer(uint layer)
  {
   if(!layers || LayersTotal() <= layer)
      return NULL;
//---
   return layers.At(layer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetModify::AddLayer(CLayer *new_layer)
  {
   if(!new_layer)
      return false;
   if(!layers)
     {
      layers = new CArrayLayer();
      if(!layers)
         return false;
     }
//---
   return layers.Add(new_layer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetModify::AddLayers(CArrayObj *new_layers)
  {
   if(!new_layers || new_layers.Total() <= 0)
      return false;
//---
   if(!layers || LayersTotal() <= 0)
     {
      Create(new_layers);
      return true;
     }
//---
   CLayerDescription *desc = NULL, *next = NULL, *prev = NULL;
   CLayer *temp;
//---
   int shift = (int)LayersTotal() - 1;
   CLayer* last_layer = layers.At(shift);
   if(!last_layer)
      return false;
//---
   CNeuronBaseOCL* neuron = last_layer.At(0);
   if(!neuron)
      return false;
//---
   desc = neuron.GetLayerInfo();
   next = new_layers.At(0);
   uint outputs = (next == NULL || (next.type != defNeuron && next.type != defNeuronBaseOCL) ? 0 : next.count);
   if(!!next && next.type == defNeuronFQF)
      outputs = next.count * next.window_out;
   if(!neuron.numOutputs(outputs, next.optimization))
      return false;
//---
   int total = new_layers.Total();
   for(int i = 0; i < total; i++)
     {
      if(i == 1 && !!prev)
         delete prev;
      prev = desc;
      desc = new_layers.At(i);
      next = new_layers.At(i + 1);
      outputs = (next == NULL || (next.type != defNeuron && next.type != defNeuronBaseOCL) ? 0 : next.count);
      if(!!next && next.type == defNeuronFQF)
         outputs = next.count * next.window_out;
      temp = new CLayer(outputs);
      uint neurons = (desc.count + (desc.type == defNeuron || desc.type == defNeuronBaseOCL ? 1 : 0));
      CNeuronBaseOCL *neuron_ocl = NULL;
      CNeuronConvOCL *neuron_conv_ocl = NULL;
      CNeuronProofOCL *neuron_proof_ocl = NULL;
      CNeuronAttentionOCL *neuron_attention_ocl = NULL;
      CNeuronMLMHAttentionOCL *neuron_mlattention_ocl = NULL;
      CNeuronDropoutOCL *dropout = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CVAE *vae = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      CNeuronSoftMaxOCL *softmax = NULL;
      CNeuronFQF *fqf = NULL;
      switch(desc.type)
        {
         case defNeuron:
         case defNeuronBaseOCL:
            neuron_ocl = new CNeuronBaseOCL();
            if(CheckPointer(neuron_ocl) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!neuron_ocl.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
              {
               delete neuron_ocl;
               delete temp;
               return false;
              }
            neuron_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_ocl))
              {
               delete neuron_ocl;
               delete temp;
               return false;
              }
            neuron_ocl = NULL;
            break;
         //---
         case defNeuronConvOCL:
            neuron_conv_ocl = new CNeuronConvOCL();
            if(CheckPointer(neuron_conv_ocl) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!neuron_conv_ocl.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.optimization, desc.batch))
              {
               delete neuron_conv_ocl;
               delete temp;
               return false;
              }
            neuron_conv_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_conv_ocl))
              {
               delete neuron_conv_ocl;
               delete temp;
               return false;
              }
            neuron_conv_ocl = NULL;
            break;
         //---
         case defNeuronProofOCL:
            neuron_proof_ocl = new CNeuronProofOCL();
            if(!neuron_proof_ocl)
              {
               delete temp;
               return false;
              }
            if(!neuron_proof_ocl.Init(outputs, 0, opencl, desc.window, desc.step, desc.count, desc.optimization, desc.batch))
              {
               delete neuron_proof_ocl;
               delete temp;
               return false;
              }
            neuron_proof_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_proof_ocl))
              {
               delete neuron_proof_ocl;
               delete temp;
               return false;
              }
            neuron_proof_ocl = NULL;
            break;
         //---
         case defNeuronAttentionOCL:
            neuron_attention_ocl = new CNeuronAttentionOCL();
            if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!neuron_attention_ocl.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
              {
               delete neuron_attention_ocl;
               delete temp;
               return false;
              }
            neuron_attention_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_attention_ocl))
              {
               delete neuron_attention_ocl;
               delete temp;
               return false;
              }
            neuron_attention_ocl = NULL;
            break;
         //---
         case defNeuronMHAttentionOCL:
            neuron_attention_ocl = new CNeuronMHAttentionOCL();
            if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!neuron_attention_ocl.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
              {
               delete neuron_attention_ocl;
               delete temp;
               return false;
              }
            neuron_attention_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_attention_ocl))
              {
               delete neuron_attention_ocl;
               delete temp;
               return false;
              }
            neuron_attention_ocl = NULL;
            break;
         //---
         case defNeuronMLMHAttentionOCL:
            neuron_mlattention_ocl = new CNeuronMLMHAttentionOCL();
            if(CheckPointer(neuron_mlattention_ocl) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!neuron_mlattention_ocl.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
              {
               delete neuron_mlattention_ocl;
               delete temp;
               return false;
              }
            neuron_mlattention_ocl.SetActivationFunction(desc.activation);
            if(!temp.Add(neuron_mlattention_ocl))
              {
               delete neuron_mlattention_ocl;
               delete temp;
               return false;
              }
            neuron_mlattention_ocl = NULL;
            break;
         //---
         case defNeuronDropoutOCL:
            dropout = new CNeuronDropoutOCL();
            if(CheckPointer(dropout) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!dropout.Init(outputs, 0, opencl, desc.count, desc.probability, desc.optimization, desc.batch))
              {
               delete dropout;
               delete temp;
               return false;
              }
            if(!temp.Add(dropout))
              {
               delete dropout;
               delete temp;
               return false;
              }
            dropout = NULL;
            break;
         //---
         case defNeuronBatchNormOCL:
            batch = new CNeuronBatchNormOCL();
            if(CheckPointer(batch) == POINTER_INVALID)
              {
               delete temp;
               return false;
              }
            if(!batch.Init(outputs, 0, opencl, desc.count, desc.batch, desc.optimization))
              {
               delete batch;
               delete temp;
               return false;
              }
            batch.SetActivationFunction(desc.activation);
            if(!temp.Add(batch))
              {
               delete batch;
               delete temp;
               return false;
              }
            batch = NULL;
            break;
         //---
         case defNeuronVAEOCL:
            vae = new CVAE();
            if(!vae)
              {
               delete temp;
               return false;
              }
            if(!vae.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
              {
               delete vae;
               delete temp;
               return false;
              }
            if(!temp.Add(vae))
              {
               delete vae;
               delete temp;
               return false;
              }
            vae = NULL;
            break;
         case defNeuronLSTMOCL:
            lstm = new CNeuronLSTMOCL();
            if(!lstm)
              {
               delete temp;
               return false;
              }
            if(!lstm.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
              {
               delete lstm;
               delete temp;
               return false;
              }
            if(i > 0)
              {
               desc = new_layers.At(i - 1);
               if(!lstm.SetInputs(desc.count))
                 {
                  delete lstm;
                  delete temp;
                  return false;
                 }
              }
            if(!temp.Add(lstm))
              {
               delete lstm;
               delete temp;
               return false;
              }
            lstm = NULL;
            break;
         //---
         case defNeuronSoftMaxOCL:
            softmax = new CNeuronSoftMaxOCL();
            if(!softmax)
              {
               delete temp;
               return false;
              }
            if(!softmax.Init(outputs, 0, opencl, desc.count * desc.step, desc.optimization, desc.batch))
              {
               delete softmax;
               delete temp;
               return false;
              }
            softmax.SetHeads(desc.step);
            if(!temp.Add(softmax))
              {
               delete softmax;
               delete temp;
               return false;
              }
            softmax = NULL;
            break;
         //---
         case defNeuronFQF:
            fqf = new CNeuronFQF();
            if(!fqf)
              {
               delete temp;
               return false;
              }
            if(!fqf.Init(outputs, 0, opencl, desc.count, desc.window_out, prev.count * (prev.type == defNeuronConv ? prev.window_out : 1), desc.optimization, desc.batch))
              {
               delete fqf;
               delete temp;
               return false;
              }
            if(!temp.Add(fqf.AsObject()))
              {
               delete fqf;
               delete temp;
               return false;
              }
            fqf = NULL;
            break;
         //---
         default:
            delete temp;
            return false;
            break;
        }
      //---
      if(!layers.Add(temp))
        {
         delete temp;
         delete layers;
         return false;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNetCreatorPanel : protected CAppDialog
  {
protected:
   //--- pre-trained model
   CNetModify        m_Model;
   CEdit             m_edPTModel;
   CEdit             m_edPTModelLayers;
   CSpinEdit         m_spPTModelLayers;
   CListView         m_lstPTModel;
   CArrayObj*        m_arPTModelDescription;
   bool              b_UseCommonDirectory;

   virtual CLabel*   CreateLabel(const int id, const string text, const int x1, const int y1, const int x2, const int y2);
   virtual bool      CreateEdit(const int id, CEdit& object, const int x1, const int y1, const int x2, const int y2, bool read_only);
   virtual bool      EditReedOnly(CEdit& object, const bool flag);
   virtual string    LayerTypeToString(int type);
   virtual int       LayerDescriptionToString(const CLayerDescription* layer, string& result[]);
   virtual bool      LoadModel(string file_name);
   //--- add layers
   CComboBox         m_cbNewNeuronType;
   CLabel*           m_lbCount;
   CEdit             m_edCount;
   CEdit             m_edWindow;
   CEdit             m_edWindowOut;
   CLabel*           m_lbWindowOut;
   CEdit             m_edStep;
   CLabel*           m_lbStepHeads;
   CEdit             m_edLayers;
   CEdit             m_edBatch;
   CEdit             m_edProbability;
   CComboBox         m_cbActivation;
   CComboBox         m_cbOptimization;
   CButton           m_btAddLayer;
   CButton           m_btDeleteLayer;
   CArrayObj         m_arAddLayers;

   virtual bool      CreateComboBoxType(const int x1, const int y1, const int x2, const int y2);
   virtual bool      CreateComboBoxActivation(const int x1, const int y1, const int x2, const int y2);
   virtual bool      ActivationListMain(void);
   virtual bool      ActivationListEmpty(void);
   virtual bool      CreateComboBoxOptimization(const int x1, const int y1, const int x2, const int y2);
   virtual bool      SetCounts(const uint type);
   //--- new model
   CListView         m_lstNewModel;
   CButton           m_btSave;
   //--- events
   virtual bool      OpenPreTrainedModel(void);
   virtual bool      ChangeNumberOfLayers(void);
   virtual bool      OnChangeListPTModel(void);
   virtual bool      OnChangeListNewModel(void);
   virtual bool      OnChangeNeuronType(void);
   virtual bool      OnClickAddButton(void);
   virtual bool      OnClickDeleteButton(void);
   virtual bool      OnClickSaveButton(void);
   virtual bool      OnChangeWindowStep(void);
   virtual bool      OnEndEdit(CEdit& object);
   virtual bool      OnEndEditProbability(void);

public:
                     CNetCreatorPanel();
                    ~CNetCreatorPanel();
   //--- main application dialog creation and destroy
   virtual bool      Create(const long chart, const string name, const int subwin, const int x1, const int y1, bool common);
   virtual void      Destroy(const int reason = REASON_PROGRAM) override { CAppDialog::Destroy(reason); }
   //--- chart event handler
   void              ChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam);
   virtual bool      OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);

   bool              Run(void) { return CAppDialog::Run();}
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNetCreatorPanel::CNetCreatorPanel()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNetCreatorPanel::~CNetCreatorPanel()
  {
   if(!!m_arPTModelDescription)
      delete m_arPTModelDescription;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::Create(const long chart, const string name, const int subwin, const int x1, const int y1, bool common)
  {
   if(!CAppDialog::Create(chart, name, subwin, x1, y1, x1 + PANEL_WIDTH, y1 + PANEL_HEIGHT))
      return false;
//---
   int lx1 = INDENT_LEFT;
   int ly1 = INDENT_TOP;
   int lx2 = lx1 + LIST_WIDTH;
   int ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(0, "PreTrained model", lx1, ly1, lx2, ly2))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateEdit(0, m_edPTModel, lx1, ly1, lx2, ly2, true))
      return false;
   if(!m_edPTModel.Text("Select file"))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(1, "Layers Total", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(1, m_edPTModelLayers, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edPTModelLayers.Text("0"))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(2, "Transfer Layers", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!m_spPTModelLayers.Create(m_chart_id, "spPTMCopyLayers", m_subwin, lx2 - EDIT_WIDTH, ly1, lx2, ly2))
      return false;
   m_spPTModelLayers.MinValue(0);
   m_spPTModelLayers.MaxValue(0);
   m_spPTModelLayers.Value(0);
   if(!Add(m_spPTModelLayers))
      return false;
//---
   lx1 = INDENT_LEFT;
   lx2 = lx1 + LIST_WIDTH;
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ClientAreaHeight() - INDENT_BOTTOM;
   if(!m_lstPTModel.Create(m_chart_id, "lstPTModel", m_subwin, lx1, ly1, lx2, ly2))
      return false;
   if(!m_lstPTModel.VScrolled(true))
      return false;
   if(!Add(m_lstPTModel))
      return false;
//---
   lx1 = lx2 + CONTROLS_GAP_X;
   lx2 = lx1 + ADDS_WIDTH;
   ly1 = INDENT_TOP;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(3, "Add layer", lx1, ly1, lx2, ly2))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateComboBoxType(lx1, ly1, lx2, ly2))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!(m_lbCount = CreateLabel(4, "Neurons", lx1, ly1, lx1 + EDIT_WIDTH, ly2)))
      return false;
//---
   if(!CreateEdit(2, m_edCount, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edCount.Text((string)DEFAULT_NEURONS))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(5, "Activation", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateComboBoxActivation(lx2 - EDIT_WIDTH, ly1, lx2, ly2))
      return false;
   m_cbActivation.Disable();
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(6, "Optimization", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateComboBoxOptimization(lx2 - EDIT_WIDTH, ly1, lx2, ly2))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(7, "Window", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(3, m_edWindow, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edWindow.Text((string)2))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   m_lbStepHeads = CreateLabel(8, "Step", lx1, ly1, lx1 + EDIT_WIDTH, ly2);
   if(!m_lbStepHeads)
      return false;
//---
   if(!CreateEdit(4, m_edStep, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edStep.Text((string)1))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   m_lbWindowOut = CreateLabel(9, "Window Out", lx1, ly1, lx1 + EDIT_WIDTH, ly2);
   if(!m_lbWindowOut)
      return false;
//---
   if(!CreateEdit(5, m_edWindowOut, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edWindowOut.Text((string)1))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(10, "Layers", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(6, m_edLayers, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edLayers.Text((string)1))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(11, "Batch size", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(7, m_edBatch, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edBatch.Text((string)1))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(12, "Probability", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(8, m_edProbability, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edProbability.Text((string)DEFAULT_PROBABILITY))
      return false;
   if(!OnEndEditProbability())
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + BUTTON_HEIGHT;
   if(!m_btAddLayer.Create(m_chart_id, "btAddLayer", m_subwin, lx1, ly1, lx1 + ADDS_WIDTH / 2, ly2))
      return false;
   if(!m_btAddLayer.Text("ADD LAYER"))
      return false;
   m_btAddLayer.Locking(false);
   if(!Add(m_btAddLayer))
      return false;
//---
   if(!m_btDeleteLayer.Create(m_chart_id, "btDeleteLayer", m_subwin, lx2 - ADDS_WIDTH / 2, ly1, lx2, ly2))
      return false;
   if(!m_btDeleteLayer.Text("DELETE"))
      return false;
   m_btDeleteLayer.Locking(false);
   if(!Add(m_btDeleteLayer))
      return false;
//---
   lx1 = lx2 + CONTROLS_GAP_X;
   lx2 = lx1 + LIST_WIDTH;
   ly1 = INDENT_TOP;
   ly2 = ly1 + EDIT_HEIGHT;
   int width = (LIST_WIDTH - CONTROLS_GAP_X) / 2;
   if(!CreateLabel(99, "New model", lx1, ly1, lx1 + width, ly2))
      return false;
//---
   if(!m_btSave.Create(m_chart_id, "btSave", m_subwin, lx2 - width, ly1, lx2, ly2))
      return false;
   if(!m_btSave.Text("SAVE MODEL"))
      return false;
   m_btSave.Locking(false);
   if(!Add(m_btSave))
      return false;
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   ly2 = ClientAreaHeight() - INDENT_BOTTOM;
   if(!m_lstNewModel.Create(m_chart_id, "lstNewModel", m_subwin, lx1, ly1, lx2, ly2))
      return false;
   if(!m_lstNewModel.VScrolled(true))
      return false;
   if(!Add(m_lstNewModel))
      return false;
//---
   b_UseCommonDirectory = common;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CNetCreatorPanel)
ON_EVENT(ON_CLICK, m_edPTModel, OpenPreTrainedModel)
ON_EVENT(ON_CLICK, m_btAddLayer, OnClickAddButton)
ON_EVENT(ON_CLICK, m_btDeleteLayer, OnClickDeleteButton)
ON_EVENT(ON_CLICK, m_btSave, OnClickSaveButton)
ON_EVENT(ON_CHANGE, m_spPTModelLayers, ChangeNumberOfLayers)
ON_EVENT(ON_CHANGE, m_lstPTModel, OnChangeListPTModel)
ON_EVENT(ON_CHANGE, m_lstNewModel, OnChangeListNewModel)
ON_EVENT(ON_CHANGE, m_cbNewNeuronType, OnChangeNeuronType)
ON_EVENT(ON_END_EDIT, m_edWindow, OnChangeWindowStep)
ON_EVENT(ON_END_EDIT, m_edStep, OnChangeWindowStep)
ON_EVENT(ON_END_EDIT, m_edProbability, OnEndEditProbability)
ON_EVENT_CONTROL(ON_END_EDIT, m_edCount, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edWindowOut, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edLayers, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edBatch, OnEndEdit)
EVENT_MAP_END(CAppDialog)
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OpenPreTrainedModel(void)
  {
   string filenames[];
   if(FileSelectDialog("Выберите файлы для загрузки", NULL,
                       "Neuron Net (*.nnw;*.inv;*.fwd)|*.nnw;*.inv;*.fwd|All files (*.*)|*.*",
                       FSD_FILE_MUST_EXIST | (b_UseCommonDirectory ? FSD_COMMON_FOLDER : NULL), filenames, NULL) > 0)
     {
      if(!LoadModel(filenames[0]))
         return false;
     }
   else
      m_edPTModel.Text("Files not selected");
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::LoadModel(string file_name)
  {
   float error, undefine, forecast;
   datetime time;
   ResetLastError();
   if(!m_Model.Load(file_name, error, undefine, forecast, time, b_UseCommonDirectory))
     {
      m_lstPTModel.ItemsClear();
      m_lstPTModel.ItemAdd("Error of load model", 0);
      m_lstPTModel.ItemAdd(file_name, 1);
      int err = GetLastError();
      if(err == 0)
         m_lstPTModel.ItemAdd("The file is damaged");
      else
         m_lstPTModel.ItemAdd(StringFormat("error id: %d", GetLastError()), 2);
      m_edPTModel.Text("Select file");
      return false;
     }
   m_edPTModel.Text(file_name);
   m_edPTModelLayers.Text((string)m_Model.LayersTotal());
   if(!!m_arPTModelDescription)
      delete m_arPTModelDescription;
   m_arPTModelDescription = m_Model.GetLayersDiscriptions();
   m_lstPTModel.ItemsClear();
   int total = m_arPTModelDescription.Total();
   for(int i = 0; i < total; i++)
     {
      CLayerDescription* temp = m_arPTModelDescription.At(i);
      if(!temp)
         return false;
      //---
      string items[];
      int total_items = LayerDescriptionToString(temp, items);
      if(total_items < 0)
        {
         printf("%s %d Error at layer %d: %d", __FUNCSIG__, __LINE__, i, GetLastError());
         return false;
        }
      if(!m_lstPTModel.AddItem(StringFormat("____ Layer %d ____", i + 1), i + 1))
         return false;
      for(int it = 0; it < total_items; it++)
         if(!m_lstPTModel.AddItem(items[it], i + 1))
            return false;
     }
   m_spPTModelLayers.MaxValue(total);
   m_spPTModelLayers.Value(total);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CNetCreatorPanel::LayerTypeToString(int type)
  {
   string result = "Unknown";
   switch(type)
     {
      case defNeuronBaseOCL:
         result = "Dense";
         break;
      case defNeuronConvOCL:
         result = "Conolution";
         break;
      case defNeuronProofOCL:
         result = "Proof";
         break;
      case defNeuronAttentionOCL:
         result = "Self Attention";
         break;
      case defNeuronMHAttentionOCL:
         result = "Multi-Head Attention";
         break;
      case defNeuronMLMHAttentionOCL:
         result = "Multi-Layer MH Attention";
         break;
      case defNeuronDropoutOCL:
         result = "Dropout";
         break;
      case defNeuronBatchNormOCL:
         result = "Batchnorm";
         break;
      case defNeuronVAEOCL:
         result = "VAE";
         break;
      case defNeuronLSTMOCL:
         result = "LSTM";
         break;
      case defNeuronSoftMaxOCL:
         result = "SoftMax";
         break;
      case defNeuronFQF:
         result = "FQF";
         break;
      default:
         result = StringFormat("%s %#x", result, type);
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CNetCreatorPanel::LayerDescriptionToString(const CLayerDescription *layer, string& result[])
  {
   if(!layer)
      return -1;
//---
   string temp;
   ArrayFree(result);
   switch(layer.type)
     {
      case defNeuronBaseOCL:
         temp = StringFormat("Dense (outputs %d, \activation %s, \optimization %s)", layer.count, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronConvOCL:
         temp = StringFormat("Conolution (outputs %d, \window %d, step %d, window out %d, \activation %s, \optimization %s)", layer.count * layer.window_out, layer.window, layer.step, layer.window_out, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronProofOCL:
         temp = StringFormat("Proof (outputs %d, \window %d, step %d, \optimization %s)", layer.count, layer.window, layer.step, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronAttentionOCL:
         temp = StringFormat("Self Attention (outputs %d, \units %d, window %d, \optimization %d)", layer.count * layer.window, layer.count, layer.window, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronMHAttentionOCL:
         temp = StringFormat("Multi-Head Attention (outputs %d, \units %d, window %d, heads %d, \optimization %s)", layer.count * layer.window, layer.count, layer.window, layer.step, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronMLMHAttentionOCL:
         temp = StringFormat("Multi-Layer MH Attention (outputs %d, \units %d, window %d, key size %d, \heads %d, layers %d, \optimization %s)", layer.count * layer.window, layer.count, layer.window, layer.window_out, layer.step, layer.layers, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronDropoutOCL:
         temp = StringFormat("Dropout (outputs %d, \probability %d, \optimization %s)", layer.count, layer.probability, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronBatchNormOCL:
         temp = StringFormat("Batchnorm (outputs %d, \batch size %d, \optimization %s)", layer.count, layer.batch, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronVAEOCL:
         temp = StringFormat("VAE (outputs %d)", layer.count);
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronLSTMOCL:
         temp = StringFormat("LSTM (outputs %d, \optimization %s)", layer.count, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronSoftMaxOCL:
         temp = StringFormat("SoftMax (outputs %d, heads %d)", layer.count, layer.step);
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      case defNeuronFQF:
         temp = StringFormat("FQF (outputs %d, quantiles %d, \optimization %s))", layer.count, layer.window_out, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
      default:
         temp = StringFormat("Unknown type %#x (outputs %d, \activation %s, \optimization %s)", layer.type, layer.count, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
     }
//---
   return ArraySize(result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::ChangeNumberOfLayers(void)
  {
   int total = m_spPTModelLayers.Value();
   m_lstPTModel.SelectByValue(total);
//---
   m_lstNewModel.ItemsClear();
   for(int i = 0; i < total; i++)
     {
      CLayerDescription* temp = m_arPTModelDescription.At(i);
      if(!temp)
         return false;
      //---
      string items[];
      int total_items = LayerDescriptionToString(temp, items);
      if(total_items < 0)
        {
         printf("%s %d Error at layer %d: %d", __FUNCSIG__, __LINE__, i, GetLastError());
         return false;
        }
      if(!m_lstNewModel.AddItem(StringFormat("____ Layer %d ____", i + 1), i + 1))
         return false;
      for(int it = 0; it < total_items; it++)
         if(!m_lstNewModel.AddItem(items[it], i + 1))
            return false;
     }
//---
   int shift = total;
   total = m_arAddLayers.Total();
   for(int i = 0; i < total; i++)
     {
      CLayerDescription* temp = m_arAddLayers.At(i);
      if(!temp)
         return false;
      //---
      string items[];
      int total_items = LayerDescriptionToString(temp, items);
      if(total_items < 0)
        {
         printf("%s %d Error at layer %d: %d", __FUNCSIG__, __LINE__, i, GetLastError());
         return false;
        }
      if(!m_lstNewModel.AddItem(StringFormat("____ Layer %d ____", shift + i + 1), shift + i + 1))
         return false;
      for(int it = 0; it < total_items; it++)
         if(!m_lstNewModel.AddItem(items[it], shift + i + 1))
            return false;
     }
   m_lstNewModel.SelectByValue(shift + total);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLabel* CNetCreatorPanel::CreateLabel(const int id, const string text, const int x1, const int y1, const int x2, const int y2)
  {
   CLabel *tmp_label = new CLabel();
   if(!tmp_label)
      return NULL;
   if(!tmp_label.Create(m_chart_id, StringFormat("%s%d", LABEL_NAME, id), m_subwin, x1, y1, x2, y2))
     {
      delete tmp_label;
      return NULL;
     }
   if(!tmp_label.Text(text))
     {
      delete tmp_label;
      return NULL;
     }
   if(!Add(tmp_label))
     {
      delete tmp_label;
      return NULL;
     }
//---
   return tmp_label;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::CreateEdit(const int id, CEdit& object, const int x1, const int y1, const int x2, const int y2, bool read_only)
  {
   if(!object.Create(m_chart_id, StringFormat("%s%d", EDIT_NAME, id), m_subwin, x1, y1, x2, y2))
      return false;
   if(!object.TextAlign(ALIGN_RIGHT))
      return false;
   if(!EditReedOnly(object, read_only))
      return false;
   if(!Add(object))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::EditReedOnly(CEdit& object, const bool flag)
  {
   if(!object.ReadOnly(flag))
      return false;
   if(!object.ColorBackground(flag ? CONTROLS_DIALOG_COLOR_CLIENT_BG : CONTROLS_EDIT_COLOR_BG))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::CreateComboBoxType(const int x1, const int y1, const int x2, const int y2)
  {
   if(!m_cbNewNeuronType.Create(m_chart_id, "cbNewNeuronType", m_subwin, x1, y1, x2, y2))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronBaseOCL), defNeuronBaseOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronConvOCL), defNeuronConvOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronProofOCL), defNeuronProofOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronLSTMOCL), defNeuronLSTMOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronAttentionOCL), defNeuronAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronMHAttentionOCL), defNeuronMHAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronMLMHAttentionOCL), defNeuronMLMHAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronDropoutOCL), defNeuronDropoutOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronBatchNormOCL), defNeuronBatchNormOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronVAEOCL), defNeuronVAEOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronSoftMaxOCL), defNeuronSoftMaxOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronFQF), defNeuronFQF))
      return false;
   if(!Add(m_cbNewNeuronType))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::CreateComboBoxActivation(const int x1, const int y1, const int x2, const int y2)
  {
   if(!m_cbActivation.Create(m_chart_id, "cbNewNeuronActivation", m_subwin, x1, y1, x2, y2))
      return false;
   if(!ActivationListMain())
      return false;
   if(!Add(m_cbActivation))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::ActivationListMain(void)
  {
   if(!m_cbActivation.ItemsClear())
      return false;
   for(int i = -1; i < 3; i++)
      if(!m_cbActivation.ItemAdd(EnumToString((ENUM_ACTIVATION)i), i + 2))
         return false;
   if(!m_cbActivation.SelectByValue((int)DEFAULT_ACTIVATION + 2))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::ActivationListEmpty(void)
  {
   if(!m_cbActivation.ItemsClear() || !m_cbActivation.ItemAdd(EnumToString((ENUM_ACTIVATION) - 1), 1))
      return false;
   if(!m_cbActivation.SelectByValue((int) 1))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::CreateComboBoxOptimization(const int x1, const int y1, const int x2, const int y2)
  {
   if(!m_cbOptimization.Create(m_chart_id, "cbNewNeuronOptimization", m_subwin, x1, y1, x2, y2))
      return false;
   for(int i = 0; i < 3; i++)
      if(!m_cbOptimization.ItemAdd(EnumToString((ENUM_OPTIMIZATION)i), i + 2))
         return false;
   if(!m_cbOptimization.SelectByValue((int)DEFAULT_OPTIMIZATION + 2))
      return false;
   if(!Add(m_cbOptimization))
      return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnChangeListPTModel(void)
  {
   long value = m_lstPTModel.Value();
//---
   return m_spPTModelLayers.Value((int)value);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnChangeListNewModel(void)
  {
   long value = m_lstNewModel.Value();
//---
   return m_lstNewModel.SelectByValue(value);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnClickAddButton(void)
  {
   int type = (int)m_cbNewNeuronType.Value();
   if(StringFind(LayerTypeToString(type), "Unknown", 0) >= 0)
     {
      Print("Wrong type of new layer");
      return false;
     }
//---
   int count = (int)StringToInteger(m_edCount.Text());
   if(count <= 0)
     {
      Print("Layer don't have neurons");
      return false;
     }
//---
   CLayerDescription *desc = new CLayerDescription();
   if(!desc)
     {
      Print("Error of create description object: %d", GetLastError());
      return false;
     }
   desc.type = type;
   desc.count = count;
   desc.activation = (ENUM_ACTIVATION)(m_cbActivation.Value() - 2);
   desc.optimization = (ENUM_OPTIMIZATION)(m_cbOptimization.Value() - 2);
   desc.batch = (int)StringToInteger(m_edBatch.Text());
   desc.layers = (int)StringToInteger(m_edLayers.Text());
   string text = m_edProbability.Text();
   StringReplace(text, ",", ".");
   StringReplace(text, " ", "");
   desc.probability = (float)StringToDouble(text);
   desc.step = (int)StringToInteger(m_edStep.Text());
   desc.window = (int)StringToInteger(m_edWindow.Text());
   desc.window_out = (int)StringToInteger(m_edWindowOut.Text());
//---
   if(!m_arAddLayers.Add(desc))
     {
      Print("Error of adding description: %d", GetLastError());
      return false;
     }
//---
   return ChangeNumberOfLayers();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnClickDeleteButton(void)
  {
   int shift = m_spPTModelLayers.Value();
   int position = (int)m_lstNewModel.Value() - 1;
//---
   if(position < 0)
     {
      Print("Layer not select");
      return false;
     }
//---
   position -= shift;
   if(position < 0)
     {
      Print("You can delete only added layers");
      return false;
     }
//---
   if(!m_arAddLayers.Delete(position))
     {
      PrintFormat("Error of remove layer: %d", GetLastError());
      return false;
     }
//---
   return ChangeNumberOfLayers();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnClickSaveButton(void)
  {
   string filenames[];
   if(FileSelectDialog("Выберите файлы для сохранения", NULL,
                       "Neuron Net (*.nnw;*.inv;*.fwd)|*.nnw;*.inv;*.fwd|All files (*.*)|*.*",
                       FSD_WRITE_FILE | (b_UseCommonDirectory ? FSD_COMMON_FOLDER : NULL), filenames, "NewModel.nnw") <= 0)
     {
      Print("File not selected");
      return false;
     }
//---
   string file_name = filenames[0];
   if(StringLen(file_name) - StringLen(EXTENSION) > StringFind(file_name, "."))
      file_name += EXTENSION;
   CNetModify* new_model = new CNetModify();
   if(!new_model)
      return false;
   int total = m_spPTModelLayers.Value();
   if(total > 0 && !m_Model.TrainMode(false))
      return false;
   bool result = true;
   for(int i = 0; i < total && result; i++)
     {
      if(!new_model.AddLayer(m_Model.GetLayer((uint)i)))
         result = false;
     }
//---
   new_model.SetOpenCL(m_Model.GetOpenCL());
   if(result && m_arAddLayers.Total() > 0)
      if(!new_model.AddLayers(GetPointer(m_arAddLayers)))
         result = false;
//---
   if(result && !new_model.Save(file_name, 1.0e37f, 100, 0, 0, b_UseCommonDirectory))
      result = false;
//---
   if(!!new_model)
      delete new_model;
   LoadModel(m_edPTModel.Text());
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNetCreatorPanel::ChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   CAppDialog::ChartEvent(id, lparam, dparam, sparam);
   if(id == CHARTEVENT_KEYDOWN && m_spPTModelLayers.IsVisible())
     {
      switch((int)lparam)
        {
         case KEY_UP:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_spPTModelLayers.Id() + 2, 0.0, m_spPTModelLayers.Name() + "Inc");
            break;
         case KEY_DOWN:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_spPTModelLayers.Id() + 3, 0.0, m_spPTModelLayers.Name() + "Dec");
            break;
         case KEY_DELETE:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_btDeleteLayer.Id(), 0.0, m_btDeleteLayer.Name());
            break;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnChangeNeuronType(void)
  {
   long type = m_cbNewNeuronType.Value();
   switch((int)type)
     {
      case defNeuronBaseOCL:
         if(!EditReedOnly(m_edCount, false) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!ActivationListMain())
            return false;
         break;
      case defNeuronConvOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, false))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!m_lbStepHeads.Text("Step"))
            return false;
         if(!m_lbWindowOut.Text("Window Out"))
            return false;
         if(!ActivationListMain())
            return false;
         if(!SetCounts(defNeuronConvOCL))
            return false;
         break;
      case defNeuronProofOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!m_lbStepHeads.Text("Step"))
            return false;
         if(!SetCounts(defNeuronProofOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronLSTMOCL:
         if(!EditReedOnly(m_edCount, false) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronDropoutOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, false) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!SetCounts(defNeuronDropoutOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronBatchNormOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, false) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!SetCounts(defNeuronBatchNormOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!SetCounts(defNeuronAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronMHAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!m_lbStepHeads.Text("Heads"))
            return false;
         if(!SetCounts(defNeuronMHAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronMLMHAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, false) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, false))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!m_lbStepHeads.Text("Heads"))
            return false;
         if(!m_lbWindowOut.Text("Keys size"))
            return false;
         if(!SetCounts(defNeuronMLMHAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
      case defNeuronVAEOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!ActivationListEmpty())
            return false;
         if(!SetCounts(defNeuronVAEOCL))
            return false;
         break;
      case defNeuronSoftMaxOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbCount.Text("Neurons"))
            return false;
         if(!m_lbStepHeads.Text("Heads"))
            return false;
         if(!m_edStep.Text("1"))
            return false;
         if(!ActivationListEmpty())
            return false;
         if(!SetCounts(defNeuronSoftMaxOCL))
            return false;
         break;
      case defNeuronFQF:
         if(!EditReedOnly(m_edCount, false) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, false))
            return false;
         if(!m_lbWindowOut.Text("Quantiles"))
            return false;
         if(!m_edWindowOut.Text("32"))
            return false;
         if(!m_lbCount.Text("Actions"))
            return false;
         if(!m_edCount.Text("3"))
            return false;
         if(!ActivationListEmpty())
            return false;
         if(!SetCounts(defNeuronFQF))
            return false;
         break;
      default:
         return false;
         break;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::SetCounts(const uint type)
  {
   const uint position = m_arAddLayers.Total();
   CLayerDescription *prev;
   if(position <= 0)
     {
      if(!m_arPTModelDescription || m_spPTModelLayers.Value() <= 0)
         return false;
      prev = m_arPTModelDescription.At(m_spPTModelLayers.Value() - 1);
     }
//---
   else
     {
      if(m_arAddLayers.Total() < (int)position)
         return false;
      prev = m_arAddLayers.At(position - 1);
     }
   if(!prev)
      return false;
//---
   uint outputs = prev.count;
   switch(prev.type)
     {
      case defNeuronAttentionOCL:
      case defNeuronMHAttentionOCL:
      case defNeuronMLMHAttentionOCL:
         outputs *= prev.window;
         break;
      case defNeuronConvOCL:
         outputs *= prev.window_out;
         break;
     }
//---
   if(outputs <= 0)
      return false;
//---
   uint counts = 0;
   int window = (int)StringToInteger(m_edWindow.Text());
   int step = (int)StringToInteger(m_edStep.Text());
   switch(type)
     {
      case defNeuronConvOCL:
      case defNeuronProofOCL:
         if(step <= 0)
            break;
         counts = (outputs - window - 1 + 2 * step) / step;
         break;
      case defNeuronAttentionOCL:
      case defNeuronMHAttentionOCL:
      case defNeuronMLMHAttentionOCL:
         if(window <= 0)
            break;
         counts = (outputs + window - 1) / window;
         break;
      case defNeuronVAEOCL:
         counts = outputs / 2;
         break;
      case defNeuronSoftMaxOCL:
         counts = outputs / step;
         break;
      case defNeuronFQF:
         return true;
      default:
         counts = outputs;
         break;
     }
//---
   return m_edCount.Text((string)counts);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnChangeWindowStep(void)
  {
   if(!OnEndEdit(m_edWindow) || !OnEndEdit(m_edStep))
      return false;
   return SetCounts((uint)m_cbNewNeuronType.Value());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnEndEdit(CEdit& object)
  {
   long value = StringToInteger(object.Text());
   return object.Text((string)fmax(1, value));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNetCreatorPanel::OnEndEditProbability(void)
  {
   double value = StringToDouble(m_edProbability.Text());
   return m_edProbability.Text(DoubleToString(fmax(0, fmin(1, value)), 2));
  }
//+------------------------------------------------------------------+

