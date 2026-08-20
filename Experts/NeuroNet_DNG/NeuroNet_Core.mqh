//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE
//+------------------------------------------------------------------+
//| Logical method implementations generated from NeuroNet.mqh: Core
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CConnection::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
//---
   if(FileWriteDouble(file_handle, weight) <= 0)
      ReturnFalse;
   if(FileWriteDouble(file_handle, deltaWeight) <= 0)
      ReturnFalse;
   if(FileWriteDouble(file_handle, mt) <= 0)
      ReturnFalse;
   if(FileWriteDouble(file_handle, vt) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CConnection::Load(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
//---
   weight = (float)FileReadDouble(file_handle);
   deltaWeight = (float)FileReadDouble(file_handle);
   mt = (float)FileReadDouble(file_handle);
   vt = (float)FileReadDouble(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CArrayCon::CreateElement(int index)
  {
   if(index < 0 || index >= m_data_max)
      ReturnFalse;
//---
   xor128;
   float weigh = (float)(rnd_w / UINT_MAX - 0.5);
   m_data[index] = new CConnection(weigh / 100);
   if(!CheckPointer(m_data[index]) != POINTER_INVALID)
      ReturnFalse;
//---
   return (true);
  }
//+------------------------------------------------------------------+
//|                                                                |
//+------------------------------------------------------------------+
//| float CNeuronBase::lr=0.0000001;   // net learning rate        |
float CNeuronBase::alpha = (float)0.8;     // momentum
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBase::CNeuronBase(void)  :
   outputVal(1),
   gradient(0),
   activation(TANH),
   t(1),
   optimization(SGD)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBase::~CNeuronBase(void)
  {
   DeleteObj(Connections);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBase::Init(uint numOutputs, uint myIndex, ENUM_OPTIMIZATION optimization_type)
  {
   if(CheckPointer(Connections) == POINTER_INVALID)
     {
      Connections = new CArrayCon();
      if(CheckPointer(Connections) == POINTER_INVALID)
         ReturnFalse;
     }
//---
   if(Connections.Reserve(fmax(numOutputs, 1)))
      for(uint c = 0; c < numOutputs; c++)
        {
         if(!Connections.CreateElement(c))
            ReturnFalse;
         Connections.IncreaseTotal();
        }
//---
   m_myIndex = myIndex;
   optimization = optimization_type;
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuron::updateInputWeights(CLayer *&prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   float lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
   int total = prevLayer.Total();
   for(int n = 0; n < total && !IsStopped(); n++)
     {
      CNeuron *neuron = prevLayer.At(n);
      CConnection *con = neuron.Connections.At(m_myIndex);
      if(CheckPointer(con) == POINTER_INVALID)
         continue;
      if(optimization == SGD)
         con.weight += con.deltaWeight = (gradient != 0 ? lr * neuron.getOutputVal() * gradient : 0) + (con.deltaWeight != 0 ? alpha*con.deltaWeight : 0);
      else
        {
         con.mt = b1 * con.mt + (1 - b1) * gradient;
         con.vt = (float)(b2 * con.vt + (1 - b2) * MathPow((double)gradient, 2.0) + 0.00000001);
         con.weight += con.deltaWeight = (float)(lt * con.mt / sqrt(con.vt));
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuron::sumDOW(CLayer *&nextLayer)
  {
   float sum = 0.0;
   int total = nextLayer.Total() - 1;
   for(int n = 0; n < total; n++)
     {
      CConnection *con = Connections.At(n);
      if(CheckPointer(con) == POINTER_INVALID)
         continue;
      float weight = con.weight;
      if(weight != 0)
        {
         CNeuron *neuron = nextLayer.At(n);
         sum += weight * neuron.gradient;
        }
     }
   return sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuron::calcHiddenGradients(CLayer *&nextLayer)
  {
   float targetVal = sumDOW(nextLayer) + outputVal;
   return calcOutputGradients(targetVal);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuron::calcOutputGradients(float targetVal)
  {
   float delta = (targetVal > 1 ? 1 : targetVal < -1 ? -1 : targetVal) - outputVal;
   gradient = (delta != 0 ? delta * activationFunctionDerivative(outputVal) : 0);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuron::feedForward(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || prevLayer.Type() != defLayer)
      ReturnFalse;
//---
   prevVal = outputVal;
   float sum = 0.0;
   int total = prevLayer.Total();
   for(int n = 0; n < total && !IsStopped(); n++)
     {
      CNeuron *temp = prevLayer.At(n);
      float val = temp.getOutputVal();
      if(val != 0)
        {
         CConnection *con = temp.Connections.At(m_myIndex);
         if(CheckPointer(con) == POINTER_INVALID)
            continue;
         sum += val * con.weight;
        }
     }
   outputVal = activationFunction(MathMin(MathMax(sum, -18), 18));
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool COpenCLMy::Initialize(const string program, const bool show_log = true)
  {
   if(!COpenCL::Initialize(program, show_log))
      ReturnFalse;
   uint size = 0;
   if(!CLGetDeviceInfo(GetContext(), CL_DEVICE_MAX_WORK_ITEM_SIZES, uLocalDimension.cdata, size))
      ReturnFalse;
   size = 0;
   if(!CLGetDeviceInfo(GetContext(), CL_DEVICE_MAX_WORK_GROUP_SIZE, uMaxWorkGroup.cdata, size))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong COpenCLMy::GetBufferSize(const int buffer_index) const
  {
//--- check parameters
   if(buffer_index < 0 || buffer_index >= m_buffers_total)
      return(0);
   if(m_buffers[buffer_index] == INVALID_HANDLE)
      return(0);
   if(m_context == INVALID_HANDLE || m_program == INVALID_HANDLE)
      return(0);
//---
   return CLGetInfoInteger(m_buffers[buffer_index], CL_BUFFER_SIZE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLayer::CreateElement(int index)
  {
   if(index >= m_data_max)
      ReturnFalse;
//---
   bool result = false;
   CNeuronBase *temp = NULL;
   CNeuronProof *temp_p = NULL;
   CNeuronBaseOCL *temp_ocl = NULL;
   CNeuronProofOCL *temp_proof_ocl = NULL;
   CNeuronConvOCL *temp_con_ocl = NULL;
   CNeuronAttentionOCL *temp_at_ocl = NULL;
   CNeuronMH2AttentionOCL *temp_mh2at_ocl = NULL;
   CNeuronMLMHAttentionOCL *temp_mlat_ocl = NULL;
   CNeuronDropoutOCL *temp_drop_ocl = NULL;
   CNeuronBatchNormOCL *temp_batch_ocl = NULL;
   CVAE *vae = NULL;
   CNeuronLSTMOCL *lstm = NULL;
   CNeuronSoftMaxOCL *softmax = NULL;
   CNeuronFQF *fqf = NULL;
   CNeuronMultiModel *multi_model = NULL;
   CNeuronConcatenate *concat = NULL;
   CNeuronEmbeddingOCL *emb = NULL;
   CNeuronPositionEncoder *pe = NULL;
   CNeuronTransposeOCL *tr = NULL;
   CNeuronCGConvOCL *cgc = NULL;
   CNeuronDOTOCL *dot = NULL;
   CNeuronFAQOCL *faq = NULL;
   CNeuronCrossAttention *cross = NULL;
   CNeuronGTE *gte = NULL;
   CNeuronCCMROCL *ccmr = NULL;
   CNeuronNODEOCL *node = NULL;
   CNeuronConformer *conf = NULL;
   CNeuronRevINDenormOCL *revin = NULL;
   CNeuronClientOCL *client = NULL;
   CNeuronLearnabledPE *lpe = NULL;
   CNeuronUShapeAttention *usa = NULL;
   CNeuronPatching *patch = NULL;
   CNeuronTiDEOCL *tide = NULL;
   CNeuronFEDW *fedw = NULL;
   CNeuronFITSOCL *fits = NULL;
   CNeuronFreDFOCL *fredf = NULL;
   CNeuronATFNetOCL *atf = NULL;
   CNeuronComplexMLMHAttention *compl_att = NULL;
   CNeuronS3 *s3 = NULL;
   CNeuronMLMHAttentionMLKV *mlkv = NULL;
   CNeuronMLCrossAttentionMLKV *crmlkv = NULL;
   CNeuronCSCMOCL *cscm = NULL;
   CNeuronSPyrAttentionOCL *spyr = NULL;
   CNeuronPLROCL *plr = NULL;
   CNeuronTPMEncoder *tpme = NULL;
   CNeuronTPM *tpm = NULL;
   CNeuronSTNNEncoder *stnne = NULL;
   CNeuronSTNNDecoder *stnnd = NULL;
   CNeuronSparseTSF *sptsf = NULL;
   CNeuronTEMPOOCL *tempo = NULL;
   CNeuronInjectTST *inject = NULL;
   CNeuronMambaOCL *mamba = NULL;
   CNeuronTrajLLMOCL *traj = NULL;
   CNeuronUniTraj *utraj = NULL;
   CNeuronHiVTOCL *hivt = NULL;
   CNeuronPointNetOCL *pn = NULL;
   CNeuronSceneSpecific *ss = NULL;
   CNeuronSEFormer *sef = NULL;
   CNeuronMVMHAttentionMLKV *mv = NULL;
   CNeuronMVCrossAttentionMLKV *mvc = NULL;
   CNeuronSPFormer *spf = NULL;
   CNeuronMAFT *maft = NULL;
   CNeuronGRES *gres = NULL;
   CResidualConv *res = NULL;
   CNeuronRefMask *refm = NULL;
   CNeuronOCM *ocm = NULL;
   CNeuronRelativeSelfAttention *rat = NULL;
   CNeuronRMAT *rmat = NULL;
   CNeuronMHAttentionPooling *atpool = NULL;
   CNeuronMotifs *motif = NULL;
   CNeuronMultiScaleAttention *msat = NULL;
   CNeuronMolformer *molf = NULL;
   CNeuronAMCT *amct = NULL;
   CNeuronRelativeCrossAttention *rcat = NULL;
   CNeuronNAFS *nafs = NULL;
   CNeuronHyperProjection *lorenz = NULL;
   CNeuronHyperboloids *logmap = NULL;
   CNeuronHypDiff *hypdif = NULL;
   CNeuronBaseSAMOCL *sam_base = NULL;
   CNeuronConvSAMOCL *sam_conv = NULL;
   CNeuronPSBlock *ps_block = NULL;
   CNeuronPSformer *psformer = NULL;
   CNeuronMASA *masa = NULL;
   CNeuronMarketObserver *mo = NULL;
   CNeuronRLAgent *rl_ag = NULL;
   CNeuronControlAgent *contr_ag = NULL;
   CNeuronMASAAT *masaat = NULL;
   CNeuronMHConvOCL *mhconv = NULL;
   CNeuronMHFeedForward *mhff = NULL;
   CNeuronDMHAttention  *dmhat = NULL;
   CNeuronCrossDMHAttention  *crdmhat = NULL;
   CNeuronMultitaskStockformer *mtstockf = NULL;
   CNeuronFinMem *fin_mem = NULL;
   CNeuronFinAgent *fin_agent = NULL;
   CNeuronFinConManager *fin_con = NULL;
   CNeuronMacroHFT *mhft = NULL;
   CNeuronMacroHFTHyperAgent *mhft_hyp = NULL;
   CNeuronMacroHFTvsRiskManager *mhft_risk = NULL;
   CNeuronHidformer *hidf = NULL;
   CNeuronResNeXtBlock *resnext = NULL;
   CNeuronChimera *chimera = NULL;
   CNeuronMoT *mot = NULL;
   CNeuronHypridDecoder *hypridDecod = NULL;
   CNeuronAttraos *attraos = NULL;
   CNeuronDUET *duet = NULL;
   CNeuronTransposeRCDOCL *trrcd = NULL;
   CNeuronTransposeVRCOCL *tvrc = NULL;
   CNeuronMultiWindowsConvOCL *mwconv = NULL;
   CNeuronAdaBN *adabn = NULL;
   CNeuronCATCH *catch
         = NULL;
   CNeuronSkillsEncoder *skills = NULL;
   CNeuronHiSSDLowLevelControler *hissd = NULL;
   CNeuronLinerAttention *lineatt = NULL;
   CNeuronCGLSTMOCL *cglstm = NULL;
   CNeuronMHProbAttention *probatt = NULL;
   CNeuronTSPositionEncoder *tspe = NULL;
   CMamba4CastEmbeding* m4cemb = NULL;
   CNeuronMultiWindowsConvWPadOCL *mwconvpad = NULL;
   CNeuronConcatDiff *condiff = NULL;
   CNeuronMantisPatching *mantpatch = NULL;
   CNeuronMantisAttentionUnit *mantatt = NULL;
   CNeuronTimeFoundPatching *tf_patch = NULL;
   CNeuronTimeFoundTransformerUnit *tf_tu = NULL;
   CNeuronSwiGLUOCL *swiglu = NULL;
   CNeuronTimeMoESparseExperts *timemoesp = NULL;
   CNeuronTimeMoEAttention *timemoeatt = NULL;
   CNeuronAdaptConv *ac = NULL;
   CNeuronRoPE *rope = NULL;
   CNeuronK2VAEEncoder *k2vae = NULL;
   CNeuronInterpolationAttention *ia = NULL;
   CNeuronGinARCell *ginar_cell = NULL;
   CNeuronGinAR *ginar = NULL;
   CNeuronPeriodNorm *per_norm = NULL;
   CNeuronAdaptSpatialNorm *adapt_norm = NULL;
   CNeuronSCNNEncoder* scnn_enc = NULL;
   CNeuronSCNN* scnn = NULL;
   CNeuronAttentNorm* atn = NULL;
   CNeuronSAttentNorm* satn = NULL;
   CNeuronPolynomialRegression* pol_regr = NULL;
   CNeuronSSCNNEncoder* sscnn_enc = NULL;
   CNeuronSSCNN* sscnn = NULL;
   CCircleParams* circl_param = NULL;
   CNeuronTQMHA* tqmha = NULL;
   CNeuronHimNetGCRU* gcru = NULL;
   CNeuronHimNetEncoder* hn_enc = NULL;
   CNeuronHimNetDecoder* hn_dec = NULL;
   CNeuronSNSMHAttention* snsatt = NULL;
   CNeuronFastGConv* fgcru = NULL;
   CNeuronSAGDFN* sagdfn = NULL;
   CNeuronSpatialEmbedding* semb = NULL;
   CNeuronTempEmbedding* temb = NULL;
   CNeuronGlobalLocalAttention* glatt = NULL;
   CNeuronExtralonger* exlon = NULL;
   CNeuronGraphAttention* gratt = NULL;
   CNeuronSparseGraphAttention* sgratt = NULL;
   CNeuronGlobLocGraphAtt* glgratt = NULL;
   CNeuronExtralongerGraph* exlongr = NULL;
   CNeuronSpikeConv* spike = NULL;
   CNeuronGateLineAttention* glineatt = NULL;
   CNeuronSpikeSparseAttention* spikeatt = NULL;
   CNeuronMoEConv *moecon = NULL;
   CNeuronSpikingBrain* spikebrain = NULL;
   CNeuronSparseSoftMax* sparsoftmax = NULL;
   CNeuronSpikeActivation* spikeact = NULL;
   CNeuronSpikeResNeXtBlock* spikeresblock = NULL;
   CNeuronSpikeResNeXtBottleneck* spikeresbottleneck = NULL;
   CNeuronSpikeResNeXtResidual* spikeresresid = NULL;
   CNeuronSSAM* ssam = NULL;
   CNeuronSSAMResNeXtBlock* ssamblock = NULL;
   CNeuronSTFS* stfs = NULL;
   CNeuronSpikeConvGRU* sp_conv_gru = NULL;
   CNeuronAddToStack* stack = NULL;
   CNeuronStackCorrelation* st_correl = NULL;
   CNeuronSTEFlowNetEncoder* ste_encod = NULL;
   CNeuronSTEFlowNetResidualBlock* ste_resid = NULL;
   CNeuronSTEFlowNetDecoder* ste_decod = NULL;
   CNeuronSTEFlowNet* ste = NULL;
   CNeuronMultiScaleStackCorrelation* mul_correl = NULL;
   CNeuronSpikeConvGRU2D* sp_conv_gru2D = NULL;
   CNeuronRAFT* raft = NULL;
   CNeuronMultScalStackCorrelBySegments* segm_corr = NULL;
   CNeuronTMAMFE* mfe = NULL;
   CNeuronTMAMPA* mpa = NULL;
   CNeuronTMA* tma = NULL;
   CNeuronFERENet*   fere = NULL;
   CNeuronSpikeSTConvGRU* st_convgru = NULL;
   CNeuronFGDModule* fgd = NULL;
   CNeuronEVMGRFlowNet* evmgr = NULL;
   CNeuronSpikeConvBlock* spike_block = NULL;
   CNeuronMSRes* msres = NULL;
   CNeuronSpikeQKAttention* qkatt = NULL;
   CNeuronSDSA* sdsa = NULL;
   CNeuronSDformerUBlock* sdublock = NULL;
   CNeuronSDformerFlow* sdflow = NULL;
   CNeuronSpikePatchStak* sp_patch_stak = NULL;
   CNeuronSpike2DSSMOCL* sp_ssm = NULL;
   CNeuronSpikeMamba2D* sp_mamba = NULL;
   CNeuronSpikeSTSSM* sp_stssm = NULL;
   CNeuronESTMEncoder* sp_estm = NULL;
   CNeuronCreateFlow* cr_flow = NULL;
   CNeuronBiDirectCorrelation* bi_corr = NULL;
   CNeuronSATMA* satma = NULL;
   CNeuronBAT* bat = NULL;
   CNeuronSpikeSCM* scm = NULL;
   CNeuronSpikeFAM* fam = NULL;
   CNeuronSpikeMDC* mdc = NULL;
   CNeuronMDS* mds = NULL;
   CNeuronSpikeADM* adm = NULL;
   CNeuronCDCSelfCorrector* selfcorrect = NULL;
   CNeuronSpikeCDC* cdc = NULL;
   CNeuronSpikeEEMFLow* eemflow = NULL;
   CNeuronSpikeDepthWiseConv* dw_conv = NULL;
   CNeuronMultiScaleExcitation* excit = NULL;
   CNeuronMultiScaleDifference* ms_diff = NULL;
   CNeuronCorrelationEncoder* corr_enc = NULL;
   CNeuronEDCFlow* edcflow = NULL;
   CNeuronCreateICEFlow* ice = NULL;
   CNeuronSpikeMFE* sp_mfe = NULL;
   CNeuronSpikeMHCrossAttention* sp_crosatt = NULL;
   CNeuronSpikeMixFusion* mix_fusion = NULL;
   CNeuronSTFlow* stflow = NULL;
   CNeuronPSSE* pse = NULL;
   CNeuronSpikeDepthWiseResidual* dwr = NULL;
   CNeuronSpikeSuperKernelBlock* sk_block = NULL;
   CNeuronSpikeGMA* gma = NULL;
   CNeuronETROF* etrof = NULL;
   CNeuronPSSEFlow* pseflow = NULL;
   CNeuronSpikeUSR* usr = NULL;
   CNeuronInitialFlow* init_flow = NULL;
   CNeuronSpikeSMR* smr = NULL;
   CNeuronEVAFlow* eva = NULL;
   CNeuronResFlowHTR* htr = NULL;
   CNeuronResFlow* resflow = NULL;
   CNeuronConfGateTailAwareMoE* tailaware = NULL;
   CNeuronResFlowLattice* lattice = NULL;
   CFieldAwareParams* fapar = NULL;
   CNeuronFieldAwareConv* faconv = NULL;
   CNeuronMHFAT* fat = NULL;
   CNeuronMHCrossFAT* cross_fat = NULL;
   CNeuronFieldPatternEmbedding* faemb = NULL;
   CNeuronAutoToken* autotoken = NULL;
   CNeuronMultiMixAttention* mixattention = NULL;
   CNeuronSparseMoEConv* sparsemoe = NULL;
   CNeuronMTmixAttBlock* mixattenblock = NULL;
   CCrossMHFlashAttention* flash_att = NULL;
   CNeuronMixFeedForward* mixff = NULL;
   CNeuronOneTrans* onetrans = NULL;
   CNeuronMHTHCrossAttention* thcatt = NULL;
   CNeuronSTCA* stca = NULL;
   CNeuronWeightGenerator* wgen = NULL;
   CNeuronPCGR* pcrg = NULL;
   CNeuronADS* ads = NULL;
   CNeuronPerTokenFFN* pertoken = NULL;
   CNeuronScenariosToken* scentoken = NULL;
   CNeuronUnifiedTokenizer* uniftokens = NULL;
   CNeuronDomainAwareAttention* domainatt = NULL;
   CNeuronFeatureSelfIteration* featureinter = NULL;
   CNeuronDomainFused* domainfused = NULL;
   CNeuronMDL* mdl = NULL;
   CNeuronUniMixerTokenizer* umtoken = NULL;
   CNeuronPerTokenSwiGLU* ptswiglu = NULL;
   CNeuronUniMixer* unimix = NULL;
   CNeuronSiameseNorm* siamnorm = NULL;
   CNeuronUniMixerBlock* umblock = NULL;
   CNeuronINFNetScenarios* infnscen = NULL;
   CNeuronINFNetHUBSequence* hub = NULL;
   CNeuronINFNetBlock* infnblock = NULL;
   CNeuronSequenceTokenizer* seq_tok = NULL;
   CNeuronDifferenceTokenizer* diff_tok = NULL;
   CNeuronReGENTADBackbone* regen_backbone = NULL;
   CNeuronReGENTADHead* regen_head = NULL;
   CNeuronReGENTAD* regentad = NULL;
   CNeuronDictionaryCrossAtt* dict_cross = NULL;
   CNeuronGDformerBackbone* gdformer_backbone = NULL;
   CNeuronGDformer* gdformer = NULL;
   CNeuronMomADTTM* momad_ttm = NULL;
   CNeuronMomADMPI* momad_mpi = NULL;
   CNeuronVLADriverRAGMPI* vladriver_rag_mpi = NULL;
   CNeuronUncAD_MSD* uncad_msd = NULL;
   CNeuronUncAD_MUE* uncad_mue = NULL;
   CNeuronUncAD_UGP* uncad_ugp = NULL;
   CNeuronUncAD_UGPL* uncad_ugpl = NULL;
   CNeuronUncAD* uncad = NULL;
   CNeuronMomAD* momad = NULL;
   CNeuronCogDriverData* cogdriver_data = NULL;
   CNeuronCogDriverRankTCM* cogdriver_rank = NULL;
   CNeuronCogDriverForecastHead* cogdriver_forecast = NULL;
   CNeuronCogDriverForecastToken* cogdriver_token = NULL;
   CNeuronScenarioForecast* scenario_forecast = NULL;
   CNeuronScenarioCrossAttention* scenario_cross = NULL;
   CD2Skill* d2_skill = NULL;
   CNeuronRAGMemory* rag_memory = NULL;
//---
   uint windows[] = {1, 1, 1};
   uint mfe_windows[] = {3, 3};
   uint sdflow_win[] = {1, 1, 1, 1, 1};
   uint sdflow_layers[] = {1, 1, 1, 1};
   vector<float> vec = vector<float>::Ones(2);
   if(iFileHandle <= 0)
     {
      temp = new CNeuron();
      if(CheckPointer(temp) == POINTER_INVALID || !temp.Init(iOutputs, index, SGD))
         ReturnFalse;
      result = true;
     }
   else
     {
      int type = FileReadInteger(iFileHandle);
      switch(type)
        {
         case defNeuron:
            temp = new CNeuron();
            if(CheckPointer(temp) == POINTER_INVALID)
               result = false;
            result = temp.Init(iOutputs, index, ADAM);
            break;
         case defNeuronProof:
            temp_p = new CNeuronProof();
            if(CheckPointer(temp_p) == POINTER_INVALID)
               result = false;
            if(temp_p.Init(iOutputs, index, 1, 1, 1, ADAM))
              {
               temp = temp_p;
               result = true;
              }
            DeleteObj(temp_p);
            break;
         case defNeuronConv:
            temp_p = new CNeuronConv();
            if(CheckPointer(temp_p) == POINTER_INVALID)
               result = false;
            if(temp_p.Init(iOutputs, index, 1, 1, 1, ADAM))
              {
               temp = temp_p;
               result = true;
              }
            DeleteObj(temp_p);
            break;
         case defNeuronLSTM:
            temp_p = new CNeuronLSTM();
            if(CheckPointer(temp_p) == POINTER_INVALID)
               result = false;
            if(temp_p.Init(iOutputs, index, 1, 1, 1, ADAM))
              {
               temp = temp_p;
               result = true;
              }
            DeleteObj(temp_p);
            break;
         case defNeuronBaseOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_ocl = new CNeuronBaseOCL();
            if(CheckPointer(temp_ocl) == POINTER_INVALID)
               result = false;
            if(temp_ocl.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = temp_ocl;
               return true;
              }
            DeleteObj(temp_ocl);
            break;
         case defNeuronBaseSAMOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            sam_base = new CNeuronBaseSAMOCL();
            if(CheckPointer(sam_base) == POINTER_INVALID)
               result = false;
            if(sam_base.Init(iOutputs, index, OpenCL, 1, 0, ADAM, 1))
              {
               m_data[index] = sam_base;
               return true;
              }
            DeleteObj(sam_base);
            break;
         case defNeuronProofOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_proof_ocl = new CNeuronProofOCL();
            if(CheckPointer(temp_proof_ocl) == POINTER_INVALID)
               result = false;
            if(temp_proof_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_proof_ocl;
               return true;
              }
         case defNeuronConvOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_con_ocl = new CNeuronConvOCL();
            if(CheckPointer(temp_con_ocl) == POINTER_INVALID)
               result = false;
            if(temp_con_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_con_ocl;
               return true;
              }
            break;
         case defNeuronConvSAMOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            sam_conv = new CNeuronConvSAMOCL();
            if(CheckPointer(sam_conv) == POINTER_INVALID)
               result = false;
            if(sam_conv.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = sam_conv;
               return true;
              }
            break;
         case defNeuronAttentionOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_at_ocl = new CNeuronAttentionOCL();
            if(CheckPointer(temp_at_ocl) == POINTER_INVALID)
               result = false;
            if(temp_at_ocl.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_at_ocl;
               return true;
              }
            break;
         case defNeuronMHAttentionOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_at_ocl = new CNeuronMHAttentionOCL();
            if(CheckPointer(temp_at_ocl) == POINTER_INVALID)
               result = false;
            if(temp_at_ocl.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_at_ocl;
               return true;
              }
            break;
         case defNeuronMLMHAttentionOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_mlat_ocl = new CNeuronMLMHAttentionOCL();
            if(CheckPointer(temp_mlat_ocl) == POINTER_INVALID)
               result = false;
            if(temp_mlat_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = temp_mlat_ocl;
               return true;
              }
            break;
         case defNeuronMLMHSparseAttentionOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_mlat_ocl = new CNeuronMLMHSparseAttention();
            if(CheckPointer(temp_mlat_ocl) == POINTER_INVALID)
               result = false;
            if(temp_mlat_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = temp_mlat_ocl;
               return true;
              }
            break;
         case defNeuronDropoutOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_drop_ocl = new CNeuronDropoutOCL();
            if(CheckPointer(temp_drop_ocl) == POINTER_INVALID)
               result = false;
            if(temp_drop_ocl.Init(iOutputs, index, OpenCL, 1.0f, 0.1f, ADAM, 1))
              {
               m_data[index] = temp_drop_ocl;
               return true;
              }
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_batch_ocl = (type == defNeuronBatchNormWithNoise ? new CNeuronBatchNormWithNoise() : new CNeuronBatchNormOCL());
            if(CheckPointer(temp_batch_ocl) == POINTER_INVALID)
               result = false;
            if(temp_batch_ocl.Init(iOutputs, index, OpenCL, 1, 1, ADAM))
              {
               m_data[index] = temp_batch_ocl;
               return true;
              }
            break;
         case defNeuronVAEOCL:
            if(!OpenCL)
               ReturnFalse;
            vae = new CVAE();
            if(!vae)
               result = false;
            if(vae.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = vae;
               return true;
              }
            break;
         case defNeuronLSTMOCL:
            if(!OpenCL)
               ReturnFalse;
            lstm = new CNeuronLSTMOCL();
            if(!lstm)
               result = false;
            if(lstm.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = lstm;
               return true;
              }
            break;
         case defNeuronSoftMaxOCL:
            if(!OpenCL)
               ReturnFalse;
            softmax = new CNeuronSoftMaxOCL();
            if(!softmax)
               result = false;
            if(softmax.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = softmax;
               return true;
              }
            break;
         case defNeuronFQF:
            if(!OpenCL)
               ReturnFalse;
            fqf = new CNeuronFQF();
            if(!fqf)
               result = false;
            if(fqf.Init(iOutputs, index, OpenCL, 1, 32, 32, ADAM, 1))
              {
               m_data[index] = fqf.AsObject();
               return true;
              }
            break;
         case defNeuronMultiModels:
            if(!OpenCL)
               ReturnFalse;
            multi_model = new CNeuronMultiModel();
            if(!multi_model)
               result = false;
            if(multi_model.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = multi_model;
               return true;
              }
            break;
         case defNeuronConcatenate:
            if(!OpenCL)
               ReturnFalse;
            concat = new CNeuronConcatenate();
            if(!concat)
               result = false;
            if(concat.Init(iOutputs, index, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = concat;
               return true;
              }
            break;
         case defNeuronSoftActorCritic:
            if(!OpenCL)
               ReturnFalse;
            fqf = new CNeuronSoftActorCritic();
            if(!fqf)
               ReturnFalse;
            if(fqf.Init(iOutputs, index, OpenCL, 1, 32, 32, ADAM, 1))
              {
               m_data[index] = fqf.AsObject();
               return true;
              }
            break;
         case defNeuronEmbeddingOCL:
            if(!OpenCL)
               ReturnFalse;
            emb = new CNeuronEmbeddingOCL();
            if(!emb)
               ReturnFalse;
            if(!emb.Init(iOutputs, index, OpenCL, 1, 1, windows))
              {
               DeleteObj(emb);
               ReturnFalse;
              }
            m_data[index] = emb;
            return true;
            break;
         case defNeuronPEOCL:
            if(!OpenCL)
               ReturnFalse;
            pe = new CNeuronPositionEncoder();
            if(!pe)
               ReturnFalse;
            if(!pe.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               DeleteObj(pe);
               ReturnFalse;
              }
            m_data[index] = pe;
            return true;
            break;
         case defNeuronTransposeOCL:
            if(!OpenCL)
               ReturnFalse;
            tr = new CNeuronTransposeOCL();
            if(!tr)
               ReturnFalse;
            if(!tr.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               DeleteObj(tr);
               ReturnFalse;
              }
            m_data[index] = tr;
            return true;
            break;
         case defNeuronMH2AttentionOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_mh2at_ocl = new CNeuronMH2AttentionOCL();
            if(CheckPointer(temp_mh2at_ocl) == POINTER_INVALID)
               result = false;
            if(temp_mh2at_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_mh2at_ocl;
               return true;
              }
            DeleteObj(temp_mh2at_ocl);
            break;
         case defNeuronCGConvOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            cgc = new CNeuronCGConvOCL();
            if(CheckPointer(cgc) == POINTER_INVALID)
               result = false;
            if(cgc.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               m_data[index] = cgc;
               return true;
              }
            DeleteObj(cgc);
            break;
         case defNeuronMFTOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_mlat_ocl = new CNeuronMFTOCL();
            if(CheckPointer(temp_mlat_ocl) == POINTER_INVALID)
               result = false;
            if(temp_mlat_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = temp_mlat_ocl;
               return true;
              }
            DeleteObj(temp_mlat_ocl);
            break;
         case defNeuronXCiTOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            temp_mlat_ocl = new CNeuronXCiTOCL();
            if(CheckPointer(temp_mlat_ocl) == POINTER_INVALID)
               result = false;
            if(temp_mlat_ocl.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = temp_mlat_ocl;
               return true;
              }
            DeleteObj(temp_mlat_ocl);
            break;
         case defNeuronDOTOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            dot = new CNeuronDOTOCL();
            if(CheckPointer(dot) == POINTER_INVALID)
               result = false;
            if(dot.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = dot;
               return true;
              }
            DeleteObj(dot);
            break;
         case defNeuronFAQOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            faq = new CNeuronFAQOCL();
            if(CheckPointer(faq) == POINTER_INVALID)
               result = false;
            if(faq.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = faq;
               return true;
              }
            DeleteObj(faq);
            break;
         case defNeuronCrossAttenOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            cross = new CNeuronCrossAttention();
            if(CheckPointer(cross) == POINTER_INVALID)
               result = false;
            if(cross.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = cross;
               return true;
              }
            DeleteObj(cross);
            break;
         case defNeuronGTE:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            gte = new CNeuronGTE();
            if(CheckPointer(gte) == POINTER_INVALID)
               result = false;
            if(gte.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = gte;
               return true;
              }
            DeleteObj(gte);
            break;
         case defNeuronCCMROCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            ccmr = new CNeuronCCMROCL();
            if(CheckPointer(ccmr) == POINTER_INVALID)
               result = false;
            if(ccmr.Init(iOutputs, index, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = ccmr;
               return true;
              }
            DeleteObj(ccmr);
            break;
         case defNeuronNODEOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            node = new CNeuronNODEOCL();
            if(CheckPointer(node) == POINTER_INVALID)
               result = false;
            if(node.Init(iOutputs, index, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = node;
               return true;
              }
            DeleteObj(node);
            break;
         case defNeuronConformerOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            conf = new CNeuronConformer();
            if(CheckPointer(conf) == POINTER_INVALID)
               result = false;
            if(conf.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = conf;
               return true;
              }
            DeleteObj(conf);
            break;
         case defNeuronRevInDenormOCL:
            if(CheckPointer(OpenCL) == POINTER_INVALID)
               ReturnFalse;
            revin = new CNeuronRevINDenormOCL();
            if(CheckPointer(revin) == POINTER_INVALID)
               result = false;
            if(revin.Init(iOutputs, index, OpenCL, 1, -1, NULL))
              {
               m_data[index] = revin;
               return true;
              }
            DeleteObj(revin);
            break;
         case defNeuronClientOCL:
            client = new CNeuronClientOCL();
            if(!client)
               ReturnFalse;
            if(client.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, windows, ADAM, 1))
              {
               m_data[index] = client;
               return true;
              }
            DeleteObj(client);
            break;
         case defNeuronLearnabledPE:
            lpe = new CNeuronLearnabledPE();
            if(!lpe)
               ReturnFalse;
            if(lpe.Init(iOutputs, index, OpenCL, 1, ADAM, 1))
              {
               m_data[index] = lpe;
               return true;
              }
            DeleteObj(lpe);
            break;
         case defNeuronUShapeAttention:
            usa = new CNeuronUShapeAttention();
            if(!usa)
               ReturnFalse;
            if(usa.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = usa;
               return true;
              }
            DeleteObj(usa);
            break;
         case defNeuronPatchingOCL:
            patch = new CNeuronPatching();
            if(!patch)
               ReturnFalse;
            if(patch.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = patch;
               return true;
              }
            DeleteObj(patch);
            break;
         //---
         case defNeuronTiDEOCL:
            tide = new CNeuronTiDEOCL();
            if(!tide)
               ReturnFalse;
            if(tide.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, windows, ADAM, 1))
              {
               m_data[index] = tide;
               return true;
              }
            DeleteObj(tide);
            break;
         //---
         case defNeuronFEDW:
            fedw = new CNeuronFEDW();
            if(!fedw)
               ReturnFalse;
            if(fedw.Init(iOutputs, 0, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = fedw;
               return true;
              }
            DeleteObj(fedw);
            break;
         //---
         case defNeuronFITSOCL:
            fits = new CNeuronFITSOCL();
            if(!fits)
               ReturnFalse;
            if(fits.Init(iOutputs, 0, OpenCL, 1, 1, 1, 0.1f, ADAM, 1))
              {
               m_data[index] = fits;
               return true;
              }
            DeleteObj(fits);
            break;
         //---
         case defNeuronFreDFOCL:
            fredf = new CNeuronFreDFOCL();
            if(!fredf)
               ReturnFalse;
            if(fredf.Init(iOutputs, 0, OpenCL, 1, 1, 0.8f, false, ADAM, 1))
              {
               m_data[index] = fredf;
               return true;
              }
            DeleteObj(fredf);
            break;
         //---
         case defNeuronComplexMLMHAttentionOCL:
            compl_att = new CNeuronComplexMLMHAttention();
            if(!compl_att)
               ReturnFalse;
            if(compl_att.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = compl_att;
               return true;
              }
            DeleteObj(compl_att);
            break;
         //---
         case defNeuronATFNetOCL:
            atf = new CNeuronATFNetOCL();
            if(!atf)
               ReturnFalse;
            if(atf.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, windows, ADAM, 1))
              {
               m_data[index] = atf;
               return true;
              }
            DeleteObj(atf);
            break;
         //---
         case defNeuronS3:
            s3 = new CNeuronS3();
            if(!s3)
               ReturnFalse;
            if(s3.Init(iOutputs, 0, OpenCL, 1, 1, ADAM, 1))
              {
               m_data[index] = s3;
               return true;
              }
            DeleteObj(s3);
            break;
         //---
         case defNeuronMLMHAttentionMLKV:
            mlkv = new CNeuronMLMHAttentionMLKV();
            if(!mlkv)
               ReturnFalse;
            if(mlkv.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mlkv;
               return true;
              }
            DeleteObj(mlkv);
            break;
         //---
         case defNeuronMVMHAttentionMLKV:
            mv = new CNeuronMVMHAttentionMLKV();
            if(CheckPointer(mv) == POINTER_INVALID)
               result = false;
            if(mv.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mv;
               return true;
              }
            DeleteObj(mv);
            break;
         //---
         case defNeuronMLMHSceneConditionAttention:
            mlkv = new CNeuronMLMHSceneConditionAttention();
            if(!mlkv)
               ReturnFalse;
            if(mlkv.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mlkv;
               return true;
              }
            DeleteObj(mlkv);
            break;
         //---
         case defNeuronMLCrossAttentionMLKV:
            crmlkv = new CNeuronMLCrossAttentionMLKV();
            if(!crmlkv)
               ReturnFalse;
            if(crmlkv.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = crmlkv;
               return true;
              }
            DeleteObj(crmlkv);
            break;
         //---
         case defNeuronMVCrossMHAttentionMLKV:
            mvc = new CNeuronMVCrossAttentionMLKV();
            if(!mvc)
               ReturnFalse;
            if(mvc.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mvc;
               return true;
              }
            DeleteObj(mvc);
            break;
         //---
         case defNeuronCSCMOCL:
            cscm = new CNeuronCSCMOCL();
            if(!cscm)
               ReturnFalse;
            if(cscm.Init(iOutputs, 0, OpenCL, windows, 1, 1, false, ADAM, 1))
              {
               m_data[index] = cscm;
               return true;
              }
            DeleteObj(cscm);
            break;
         //---
         case defNeuronSPyrAttentionMLKV:
            spyr = new CNeuronSPyrAttentionOCL();
            if(!spyr)
               ReturnFalse;
            if(spyr.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 2, ADAM, 1))
              {
               m_data[index] = spyr;
               return true;
              }
            DeleteObj(spyr);
            break;
         //---
         case defNeuronPLROCL:
            plr = new CNeuronPLROCL();
            if(!plr)
               ReturnFalse;
            if(plr.Init(iOutputs, 0, OpenCL, 1, 1, false, ADAM, 1))
              {
               m_data[index] = plr;
               return true;
              }
            DeleteObj(plr);
            break;
         //---
         case defNeuronTPM:
            tpm = new CNeuronTPM();
            if(!tpm)
               ReturnFalse;
            if(tpm.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = tpm;
               return true;
              }
            DeleteObj(tpm);
            break;
         case defNeuronTPMEncoder:
            tpme = new CNeuronTPMEncoder();
            if(!tpme)
               ReturnFalse;
            if(tpme.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = tpme;
               return true;
              }
            DeleteObj(tpme);
            break;
         case defNeuronSTNNEncoder:
            stnne = new CNeuronSTNNEncoder();
            if(!stnne)
               ReturnFalse;
            if(stnne.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = stnne;
               return true;
              }
            DeleteObj(stnne);
            break;
         case defNeuronSTNNDecoder:
            stnnd = new CNeuronSTNNDecoder();
            if(!stnnd)
               ReturnFalse;
            if(stnnd.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = stnnd;
               return true;
              }
            DeleteObj(stnnd);
            break;
         case defNeuronSparseTSF:
            sptsf = new CNeuronSparseTSF();
            if(!sptsf)
               ReturnFalse;
            if(sptsf.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = sptsf;
               return true;
              }
            DeleteObj(sptsf);
            break;
         case defNeuronTEMPOOCL:
            tempo = new CNeuronTEMPOOCL();
            if(!tempo)
               ReturnFalse;
            if(tempo.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = tempo;
               return true;
              }
            DeleteObj(tempo);
            break;
         case defNeuronInjectTST:
            inject = new CNeuronInjectTST();
            if(!inject)
               ReturnFalse;
            if(inject.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = inject;
               return true;
              }
            DeleteObj(inject);
            break;
         case defNeuronMambaOCL:
            mamba = new CNeuronMambaOCL();
            if(!mamba)
               ReturnFalse;
            if(mamba.Init(iOutputs, 0, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mamba;
               return true;
              }
            DeleteObj(mamba);
            break;
         case defNeuronTrajLLMOCL:
            traj = new CNeuronTrajLLMOCL();
            if(!traj)
               ReturnFalse;
            if(traj.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = traj;
               return true;
              }
            DeleteObj(traj);
            break;
         //---
         case defNeuronUniTrajOCL:
            utraj = new CNeuronUniTraj();
            if(!utraj)
               ReturnFalse;
            if(utraj.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = utraj;
               return true;
              }
            DeleteObj(utraj);
            break;
         //---
         case defNeuronHiVTOCL:
            hivt = new CNeuronHiVTOCL();
            if(!hivt)
               ReturnFalse;
            if(hivt.Init(iOutputs, 0, OpenCL, 1, 1, 1, 10, 1, 1, ADAM, 1))
              {
               m_data[index] = hivt;
               return true;
              }
            DeleteObj(hivt);
            break;
         //---
         case defNeuronPointNetOCL:
            pn = new CNeuronPointNetOCL();
            if(!pn)
               ReturnFalse;
            if(pn.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = pn;
               return true;
              }
            DeleteObj(pn);
            break;
         //---
         case defNeuronPointNet2OCL:
            pn = new CNeuronPointNet2OCL();
            if(!pn)
               ReturnFalse;
            if(pn.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = pn;
               return true;
              }
            DeleteObj(pn);
            break;
         //---
         case defNeuronPointFormer:
            pn = new CNeuronPointFormer();
            if(!pn)
               ReturnFalse;
            if(pn.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = pn;
               return true;
              }
            DeleteObj(pn);
            break;
         case defNeuronHyperDet:
            pn = new CNeuronHyperDet();
            if(!pn)
               ReturnFalse;
            if(pn.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, ADAM, 1))
              {
               m_data[index] = pn;
               return true;
              }
            DeleteObj(pn);
            break;
         case defNeuronSceneSpecific:
            ss = new CNeuronSceneSpecific();
            if(!ss)
               ReturnFalse;
            if(ss.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = ss;
               return true;
              }
            DeleteObj(ss);
            break;
         case defNeuronSEFormer:
            sef = new CNeuronSEFormer();
            if(!sef)
               ReturnFalse;
            if(sef.Init(iOutputs, 0, OpenCL, 1, 1, 1, false, 1, 1, ADAM, 1))
              {
               m_data[index] = sef;
               return true;
              }
            DeleteObj(sef);
            break;
         case defNeuronSPFormer:
            spf = new CNeuronSPFormer();
            if(!spf)
               ReturnFalse;
            if(spf.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 20, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = spf;
               return true;
              }
            DeleteObj(spf);
            break;
         case defNeuronMAFT:
            maft = new CNeuronMAFT();
            if(!maft)
               ReturnFalse;
            if(maft.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 20, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = maft;
               return true;
              }
            DeleteObj(maft);
            break;
         case defNeuronGRES:
            gres = new CNeuronGRES();
            if(!gres)
               ReturnFalse;
            if(gres.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 20, 1, 5, 1, 1, ADAM, 2))
              {
               m_data[index] = gres;
               return true;
              }
            DeleteObj(gres);
            break;
         case defResidualConv:
            res = new CResidualConv();
            if(!res)
               ReturnFalse;
            if(res.Init(iOutputs, 0, OpenCL, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = res;
               return true;
              }
            DeleteObj(res);
            break;
         case defNeuronRefMask:
            refm = new CNeuronRefMask();
            if(!refm)
               ReturnFalse;
            if(refm.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = refm;
               return true;
              }
            DeleteObj(refm);
            break;
         case defNeuronOCM:
            ocm = new CNeuronOCM();
            if(!ocm)
               ReturnFalse;
            if(ocm.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = ocm;
               return true;
              }
            DeleteObj(ocm);
            break;
         case defNeuronRelativeSelfAttention:
            rat = new CNeuronRelativeSelfAttention();
            if(!rat)
               ReturnFalse;
            if(rat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = rat;
               return true;
              }
            DeleteObj(rat);
            break;
         case defNeuronRelativeCrossAttention:
            rcat = new CNeuronRelativeCrossAttention();
            if(!rcat)
               ReturnFalse;
            if(rcat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = rcat;
               return true;
              }
            DeleteObj(rcat);
            break;
         case defNeuronRMAT:
            rmat = new CNeuronRMAT();
            if(!rmat)
               ReturnFalse;
            if(rmat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = rmat;
               return true;
              }
            DeleteObj(rmat);
            break;
         case defNeuronMHAttentionPooling:
            atpool = new CNeuronMHAttentionPooling();
            if(!atpool)
               ReturnFalse;
            if(atpool.Init(iOutputs, 0, OpenCL, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = atpool;
               return true;
              }
            DeleteObj(atpool);
            break;
         case defNeuronMotifs:
            motif = new CNeuronMotifs();
            if(!motif)
               ReturnFalse;
            if(motif.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = motif;
               return true;
              }
            DeleteObj(motif);
            break;
         case defNeuronMultiScaleAttention:
            msat = new CNeuronMultiScaleAttention();
            if(!msat)
               ReturnFalse;
            if(msat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = msat;
               return true;
              }
            DeleteObj(msat);
            break;
         case defNeuronMolformer:
            molf = new CNeuronMolformer();
            if(!molf)
               ReturnFalse;
            if(molf.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = molf;
               return true;
              }
            DeleteObj(molf);
            break;
         case defNeuronAMCT:
            amct = new CNeuronAMCT();
            if(!amct)
               ReturnFalse;
            if(amct.Init(iOutputs, 0, OpenCL, 1, 1, 1, 5, 1, 1, ADAM, 2))
              {
               m_data[index] = amct;
               return true;
              }
            DeleteObj(amct);
            break;
         case defNeuronNAFS:
            nafs = new CNeuronNAFS();
            if(!nafs)
               ReturnFalse;
            if(nafs.Init(iOutputs, 0, OpenCL, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = nafs;
               return true;
              }
            DeleteObj(nafs);
            break;
         case defNeuronDiffusion:
            usa = new CNeuronDiffusion();
            if(!usa)
               ReturnFalse;
            if(usa.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = usa;
               return true;
              }
            DeleteObj(usa);
            break;
         case defNeuronHyperProjection:
            lorenz = new CNeuronHyperProjection();
            if(!lorenz)
               ReturnFalse;
            if(lorenz.Init(iOutputs, index, OpenCL, 1, 1, ADAM, 1))
              {
               m_data[index] = lorenz;
               return true;
              }
            DeleteObj(lorenz);
            break;
         case defNeuronHyperboloids:
            logmap = new CNeuronHyperboloids();
            if(!logmap)
               ReturnFalse;
            if(logmap.Init(iOutputs, index, OpenCL, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = logmap;
               return true;
              }
            DeleteObj(logmap);
            break;
         case defNeuronHypDiff:
            hypdif = new CNeuronHypDiff();
            if(!hypdif)
               ReturnFalse;
            if(hypdif.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = hypdif;
               return true;
              }
            DeleteObj(hypdif);
            break;
         case defNeuronPSBlock:
            ps_block = new CNeuronPSBlock();
            if(!ps_block)
               ReturnFalse;
            if(ps_block.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 0.0f, ADAM, 1))
              {
               m_data[index] = ps_block;
               return true;
              }
            DeleteObj(ps_block);
            break;
         case defNeuronPSformer:
            psformer = new CNeuronPSformer();
            if(!psformer)
               ReturnFalse;
            if(psformer.Init(iOutputs, index, OpenCL, 1, 1, 1, 0, ADAM, 1))
              {
               m_data[index] = psformer;
               return true;
              }
            DeleteObj(psformer);
            break;
         case defNeuronMASA:
            masa = new CNeuronMASA();
            if(!masa)
               ReturnFalse;
            if(masa.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, -1, NULL, ADAM, 1))
              {
               m_data[index] = masa;
               return true;
              }
            DeleteObj(masa);
            break;
         case defNeuronMarketObserver:
            mo = new CNeuronMarketObserver();
            if(!mo)
               ReturnFalse;
            if(mo.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mo;
               return true;
              }
            DeleteObj(mo);
            break;
         case defNeuronRLAgent:
            rl_ag = new CNeuronRLAgent();
            if(!rl_ag)
               ReturnFalse;
            if(rl_ag.Init(iOutputs, index, OpenCL, 1, 1, 1, 0, 1, 3, ADAM, 1))
              {
               m_data[index] = rl_ag;
               return true;
              }
            DeleteObj(rl_ag);
            break;
         case defNeuronControlAgent:
            contr_ag = new CNeuronControlAgent();
            if(!contr_ag)
               ReturnFalse;
            if(contr_ag.Init(iOutputs, index, OpenCL, 1, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = contr_ag;
               return true;
              }
            DeleteObj(contr_ag);
            break;
         case defNeuronMASAAT:
            masaat = new CNeuronMASAAT();
            if(!masaat)
               ReturnFalse;
            if(masaat.Init(iOutputs, index, OpenCL, 1, 1, 1, 3, 1, vec, 1, ADAM, 1))
              {
               m_data[index] = masaat;
               return true;
              }
            DeleteObj(masaat);
            break;
         case defNeuronMHConvOCL:
            mhconv = new CNeuronMHConvOCL();
            if(!mhconv)
               ReturnFalse;
            if(mhconv.Init(iOutputs, index, OpenCL, 1, 1, 1, 3, 1, 1, ADAM, 1))
              {
               m_data[index] = mhconv;
               return true;
              }
            DeleteObj(mhconv);
            break;
         case defNeuronMHFeedForward:
            mhff = new CNeuronMHFeedForward();
            if(!mhff)
               ReturnFalse;
            if(mhff.Init(iOutputs, index, OpenCL, 1, 1, 1, 3, 1, ADAM, 1))
              {
               m_data[index] = mhff;
               return true;
              }
            DeleteObj(mhff);
            break;
         case defNeuronDMHAttention:
            dmhat = new CNeuronDMHAttention();
            if(!dmhat)
               ReturnFalse;
            if(dmhat.Init(iOutputs, index, OpenCL, 1, 1, 1, 3, 1, ADAM, 1))
              {
               m_data[index] = dmhat;
               return true;
              }
            DeleteObj(dmhat);
            break;
         case defNeuronCrossDMHAttention:
            crdmhat = new CNeuronCrossDMHAttention();
            if(!crdmhat)
               ReturnFalse;
            if(crdmhat.Init(iOutputs, index, OpenCL, 1, 1, 1, 3, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = crdmhat;
               return true;
              }
            DeleteObj(crdmhat);
            break;
         case defNeuronMultitaskStockformer:
            mtstockf = new CNeuronMultitaskStockformer();
            if(!mtstockf)
               ReturnFalse;
            if(mtstockf.Init(iOutputs, index, OpenCL, 1, 1, 5, 3, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = mtstockf;
               return true;
              }
            DeleteObj(mtstockf);
            break;
         case defNeuronFinMem:
            fin_mem = new CNeuronFinMem();
            if(!fin_mem)
               ReturnFalse;
            if(fin_mem.Init(iOutputs, index, OpenCL, 5, 5, 5, 3, 5, 5, ADAM, 1))
              {
               m_data[index] = fin_mem;
               return true;
              }
            DeleteObj(fin_mem);
            break;
         case defNeuronFinAgent:
            fin_agent = new CNeuronFinAgent();
            if(!fin_agent)
               ReturnFalse;
            if(fin_agent.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 5, 5, 2, ADAM, 1))
              {
               m_data[index] = fin_agent;
               return true;
              }
            DeleteObj(fin_agent);
            break;
         case defNeuronFinConManager:
            fin_con = new CNeuronFinConManager();
            if(!fin_con)
               ReturnFalse;
            if(fin_con.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 5, 5, 6, ADAM, 1))
              {
               m_data[index] = fin_con;
               return true;
              }
            DeleteObj(fin_con);
            break;
         case defNeuronMacroHFT:
            mhft = new CNeuronMacroHFT();
            if(!mhft)
               ReturnFalse;
            if(mhft.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 1, 5, 6, ADAM, 1))
              {
               m_data[index] = mhft;
               return true;
              }
            DeleteObj(mhft);
            break;
         case defNeuronMacroHFTHyperAgent:
            mhft_hyp = new CNeuronMacroHFTHyperAgent();
            if(!mhft_hyp)
               ReturnFalse;
            if(mhft_hyp.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 1, 6, 6, ADAM, 1))
              {
               m_data[index] = mhft_hyp;
               return true;
              }
            DeleteObj(mhft_hyp);
            break;
         case defNeuronMacroHFTvsRiskManager:
            mhft_risk = new CNeuronMacroHFTvsRiskManager();
            if(!mhft_risk)
               ReturnFalse;
            if(mhft_risk.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 6, 6, 6, ADAM, 1))
              {
               m_data[index] = mhft_risk;
               return true;
              }
            DeleteObj(mhft_risk);
            break;
         case defNeuronMultiScaleRelativeSelfAttention:
            rat = new CNeuronMultiScaleRelativeSelfAttention();
            if(!rat)
               ReturnFalse;
            if(rat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = rat;
               return true;
              }
            DeleteObj(rat);
            break;
         case defNeuronMultiScaleRelativeCrossAttention:
            rcat = new CNeuronMultiScaleRelativeCrossAttention();
            if(!rcat)
               ReturnFalse;
            if(rcat.Init(iOutputs, 0, OpenCL, 1, 1, 1, 1, 1, 1, ADAM, 2))
              {
               m_data[index] = rcat;
               return true;
              }
            DeleteObj(rcat);
            break;
         case defNeuronHidformer:
            hidf = new CNeuronHidformer();
            if(!hidf)
               ReturnFalse;
            if(hidf.Init(iOutputs, index, OpenCL, 5, 5, 10, 3, 1, 5, 6, ADAM, 1))
              {
               m_data[index] = hidf;
               return true;
              }
            DeleteObj(hidf);
            break;
         case defNeuronResNeXtBlock:
            resnext = new CNeuronResNeXtBlock();
            if(!resnext)
               ReturnFalse;
            if(resnext.Init(iOutputs, index, OpenCL, 5, 5, 3, 2, 10, 4, 32, ADAM, 1))
              {
               m_data[index] = resnext;
               return true;
              }
            DeleteObj(resnext);
            break;
         case defNeuronChimera:
            chimera = new CNeuronChimera();
            if(!chimera)
               ReturnFalse;
            if(chimera.Init(iOutputs, index, OpenCL, 5, 5, 3, 3, ADAM, 1))
              {
               m_data[index] = chimera;
               return true;
              }
            DeleteObj(chimera);
            break;
         case defNeuronMoT:
            mot = new CNeuronMoT();
            if(!mot)
               ReturnFalse;
            if(mot.Init(iOutputs, index, OpenCL, 5, 5, ADAM, 1))
              {
               m_data[index] = mot;
               return true;
              }
            DeleteObj(mot);
            break;
         case defNeuronHypridDecoder:
            hypridDecod = new CNeuronHypridDecoder();
            if(!hypridDecod)
               ReturnFalse;
            if(hypridDecod.Init(iOutputs, index, OpenCL, 5, 5, 3, 3, 1, 5, 6, ADAM, 1))
              {
               m_data[index] = hypridDecod;
               return true;
              }
            DeleteObj(hypridDecod);
            break;
         case defNeuronAttraos:
            attraos = new CNeuronAttraos();
            if(!attraos)
               ReturnFalse;
            if(attraos.Init(iOutputs, index, OpenCL, 5, 5, 10, ADAM, 1))
              {
               m_data[index] = attraos;
               return true;
              }
            DeleteObj(attraos);
            break;
         case defNeuronDUET:
            duet = new CNeuronDUET();
            if(!duet)
               ReturnFalse;
            if(duet.Init(iOutputs, index, OpenCL, 5, 5, 10, 2, 10, 4, 2, ADAM, 1))
              {
               m_data[index] = duet;
               return true;
              }
            DeleteObj(duet);
            break;
         case defNeuronTransposeRCDOCL:
            trrcd = new CNeuronTransposeRCDOCL();
            if(!trrcd)
               ReturnFalse;
            if(trrcd.Init(iOutputs, index, OpenCL, 5, 5, 10, ADAM, 1))
              {
               m_data[index] = trrcd;
               return true;
              }
            DeleteObj(trrcd);
            break;
         case defNeuronTransposeVRCOCL:
            tvrc = new CNeuronTransposeVRCOCL();
            if(!tvrc)
               ReturnFalse;
            if(tvrc.Init(iOutputs, index, OpenCL, 5, 5, 10, ADAM, 1))
              {
               m_data[index] = tvrc;
               return true;
              }
            DeleteObj(tvrc);
            break;
         case defNeuronMultiWindowsConvOCL:
            mwconv = new CNeuronMultiWindowsConvOCL();
            if(!mwconv)
               ReturnFalse;
            if(mwconv.Init(iOutputs, index, OpenCL, windows, 5, 5, 1, ADAM, 1))
              {
               m_data[index] = mwconv;
               return true;
              }
            DeleteObj(mwconv);
            break;
         case defNeuronAdaBN:
            adabn = new CNeuronAdaBN();
            if(!adabn)
               ReturnFalse;
            if(adabn.Init(iOutputs, index, OpenCL, 5, 5, 10, windows, 2, 1, ADAM, 1))
              {
               m_data[index] = adabn;
               return true;
              }
            DeleteObj(adabn);
            break;
         case defNeuronCATCH:
            catch
                  = new CNeuronCATCH();
            if(!catch)
               ReturnFalse;
            if(catch.Init(iOutputs, index, OpenCL, 8, 5, 4, 1, 2, 1, ADAM, 1))
              {
               m_data[index] = catch;
               return true;
              }
            DeleteObj(catch);
            break;
         case defNeuronSkillsEncoder:
            skills = new CNeuronSkillsEncoder();
            if(!skills)
               ReturnFalse;
            if(skills.Init(iOutputs, index, OpenCL, 8, 5, 4, 4, 1, 2, 2, ADAM, 1))
              {
               m_data[index] = skills;
               return true;
              }
            DeleteObj(skills);
            break;
         case defNeuronHiSSDLowLevelControler:
            hissd = new CNeuronHiSSDLowLevelControler();
            if(!hissd)
               ReturnFalse;
            if(hissd.Init(iOutputs, index, OpenCL, 8, 5, 4, 4, 1, 2, 1, 1, 2, ADAM, 1))
              {
               m_data[index] = hissd;
               return true;
              }
            DeleteObj(hissd);
            break;
         case defNeuronLinerAttention:
            lineatt = new CNeuronLinerAttention();
            if(!lineatt)
               ReturnFalse;
            if(lineatt.Init(iOutputs, index, OpenCL, 3, 3, 4, 1, ADAM, 1))
              {
               m_data[index] = lineatt;
               return true;
              }
            DeleteObj(lineatt);
            break;
         case defNeuronCGLSTMOCL:
            cglstm = new CNeuronCGLSTMOCL();
            if(!cglstm)
               ReturnFalse;
            if(cglstm.Init(iOutputs, index, OpenCL, 8, 3, 1, ADAM, 1))
              {
               m_data[index] = cglstm;
               return true;
              }
            DeleteObj(cglstm);
            break;
         case defNeuronMHProbAttention:
            probatt = new CNeuronMHProbAttention();
            if(!probatt)
               ReturnFalse;
            if(probatt.Init(iOutputs, index, OpenCL, 3, 3, 1, 1, ADAM, 1))
              {
               m_data[index] = probatt;
               return true;
              }
            DeleteObj(probatt);
            break;
         case defNeuronTSPositionEncoder:
            tspe = new CNeuronTSPositionEncoder();
            if(!tspe)
               ReturnFalse;
            if(tspe.Init(iOutputs, index, OpenCL, 3, 3, windows, 1, ADAM, 1))
              {
               m_data[index] = tspe;
               return true;
              }
            DeleteObj(tspe);
            break;
         case defMamba4CastEmbeding:
            m4cemb = new CMamba4CastEmbeding();
            if(!m4cemb)
               ReturnFalse;
            if(m4cemb.Init(iOutputs, index, OpenCL, 3, 9, 3, windows, ADAM, 1))
              {
               m_data[index] = m4cemb;
               return true;
              }
            DeleteObj(m4cemb);
            break;
         case defNeuronMultiWindowsConvWPadOCL:
            mwconvpad = new CNeuronMultiWindowsConvWPadOCL();
            if(!mwconvpad)
               ReturnFalse;
            if(mwconvpad.Init(iOutputs, index, OpenCL, windows, 1, 5, 1, 1, ADAM, 1))
              {
               m_data[index] = mwconvpad;
               return true;
              }
            DeleteObj(mwconvpad);
            break;
         case defNeuronConcatDiff:
            condiff = new CNeuronConcatDiff();
            if(!condiff)
               ReturnFalse;
            if(condiff.Init(iOutputs, index, OpenCL, 12, 1, 2, ADAM, 1))
              {
               m_data[index] = condiff;
               return true;
              }
            DeleteObj(condiff);
            break;
         case defNeuronMantisPatching:
            mantpatch = new CNeuronMantisPatching();
            if(!mantpatch)
               ReturnFalse;
            if(mantpatch.Init(iOutputs, index, OpenCL, 12, 3, 2, 5, 5, ADAM, 1))
              {
               m_data[index] = mantpatch;
               return true;
              }
            DeleteObj(mantpatch);
            break;
         case defNeuronMantisAttentionUnit:
            mantatt = new CNeuronMantisAttentionUnit();
            if(!mantatt)
               ReturnFalse;
            if(mantatt.Init(iOutputs, index, OpenCL, 12, 3, 2, 5, 5, 1, 5, ADAM, 1))
              {
               m_data[index] = mantatt;
               return true;
              }
            DeleteObj(mantatt);
            break;
         case defNeuronTimeFoundPatching:
            tf_patch = new CNeuronTimeFoundPatching();
            if(!tf_patch)
               ReturnFalse;
            if(tf_patch.Init(iOutputs, index, OpenCL, 12, 3, 2, 5, 5, ADAM, 1))
              {
               m_data[index] = tf_patch;
               return true;
              }
            DeleteObj(tf_patch);
            break;
         case defNeuronTimeFoundTransformerUnit:
            tf_tu = new CNeuronTimeFoundTransformerUnit();
            if(!tf_tu)
               ReturnFalse;
            if(tf_tu.Init(iOutputs, index, OpenCL, 3, 3, 2, 2, 5, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = tf_tu;
               return true;
              }
            DeleteObj(tf_tu);
            break;
         case defNeuronSwiGLUOCL:
            swiglu = new CNeuronSwiGLUOCL();
            if(!swiglu)
               ReturnFalse;
            if(swiglu.Init(iOutputs, index, OpenCL, 3, 3, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = swiglu;
               return true;
              }
            DeleteObj(swiglu);
            break;
         case defNeuronTimeMoESparseExperts:
            timemoesp = new CNeuronTimeMoESparseExperts();
            if(!timemoesp)
               ReturnFalse;
            if(timemoesp.Init(iOutputs, index, OpenCL, 3, 3, 2, 2, 5, 2, ADAM, 1))
              {
               m_data[index] = timemoesp;
               return true;
              }
            DeleteObj(timemoesp);
            break;
         case defNeuronTimeMoEAttention:
            timemoeatt = new CNeuronTimeMoEAttention();
            if(!timemoeatt)
               ReturnFalse;
            if(timemoeatt.Init(iOutputs, index, OpenCL, 3, 3, 2, 3, 5, 1, 1, 5, 2, 2, ADAM, 1))
              {
               m_data[index] = timemoeatt;
               return true;
              }
            DeleteObj(timemoeatt);
            break;
         case defNeuronAdaptConv:
            ac = new CNeuronAdaptConv();
            if(!ac)
               ReturnFalse;
            if(ac.Init(iOutputs, index, OpenCL, 3, 3, 2, 3, ADAM, 1))
              {
               m_data[index] = ac;
               return true;
              }
            DeleteObj(ac);
            break;
         case defNeuronRoPE:
            rope = new CNeuronRoPE();
            if(!rope)
               ReturnFalse;
            if(rope.Init(iOutputs, index, OpenCL, 3, 4, 1, ADAM, 1))
              {
               m_data[index] = rope;
               return true;
              }
            DeleteObj(rope);
            break;
         case defNeuronK2VAEEncoder:
            k2vae = new CNeuronK2VAEEncoder();
            if(!k2vae)
               ReturnFalse;
            if(k2vae.Init(iOutputs, index, OpenCL, 3, 3, 5, 1, 1, 2, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = k2vae;
               return true;
              }
            DeleteObj(k2vae);
            break;
         case defNeuronInterpolationAttention:
            ia = new CNeuronInterpolationAttention();
            if(!ia)
               ReturnFalse;
            if(ia.Init(iOutputs, index, OpenCL, 3, 3, ADAM, 1))
              {
               m_data[index] = ia;
               return true;
              }
            DeleteObj(ia);
            break;
         case defNeuronGinARCell:
            ginar_cell = new CNeuronGinARCell();
            if(!ginar_cell)
               ReturnFalse;
            if(ginar_cell.Init(iOutputs, index, OpenCL, 3, 3, ADAM, 1))
              {
               m_data[index] = ginar_cell;
               return true;
              }
            DeleteObj(ginar_cell);
            break;
         case defNeuronGinAR:
            ginar = new CNeuronGinAR();
            if(!ginar)
               ReturnFalse;
            if(ginar.Init(iOutputs, index, OpenCL, 3, 3, ADAM, 1))
              {
               m_data[index] = ginar;
               return true;
              }
            DeleteObj(ginar);
            break;
         case defNeuronPeriodNorm:
            per_norm = new CNeuronPeriodNorm();
            if(!per_norm)
               ReturnFalse;
            if(per_norm.Init(iOutputs, index, OpenCL, 3, 3, 1, ADAM, 1))
              {
               m_data[index] = per_norm;
               return true;
              }
            DeleteObj(per_norm);
            break;
         case defNeuronAdaptSpatialNorm:
            adapt_norm = new CNeuronAdaptSpatialNorm();
            if(!adapt_norm)
               ReturnFalse;
            if(adapt_norm.Init(iOutputs, index, OpenCL, 3, 1, ADAM, 1))
              {
               m_data[index] = adapt_norm;
               return true;
              }
            DeleteObj(adapt_norm);
            break;
         case defNeuronSCNNEncoder:
            scnn_enc = new CNeuronSCNNEncoder();
            if(!scnn_enc)
               ReturnFalse;
            if(scnn_enc.Init(iOutputs, index, OpenCL, 3, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = scnn_enc;
               return true;
              }
            DeleteObj(scnn_enc);
            break;
         case defNeuronSCNN:
            scnn = new CNeuronSCNN();
            if(!scnn)
               ReturnFalse;
            if(scnn.Init(iOutputs, index, OpenCL, 3, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = scnn;
               return true;
              }
            DeleteObj(scnn);
            break;
         case defNeuronAttentNorm:
            atn = new CNeuronAttentNorm();
            if(!atn)
               ReturnFalse;
            if(atn.Init(iOutputs, index, OpenCL, 3, 3, 1, ADAM, 1))
              {
               m_data[index] = atn;
               return true;
              }
            DeleteObj(atn);
            break;
         case defNeuronSAttentNorm:
            satn = new CNeuronSAttentNorm();
            if(!satn)
               ReturnFalse;
            if(satn.Init(iOutputs, index, OpenCL, 3, 3, ADAM, 1))
              {
               m_data[index] = satn;
               return true;
              }
            DeleteObj(satn);
            break;
         case defNeuronPolynomialRegression:
            pol_regr = new CNeuronPolynomialRegression();
            if(!pol_regr)
               ReturnFalse;
            if(pol_regr.Init(iOutputs, index, OpenCL, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = pol_regr;
               return true;
              }
            DeleteObj(pol_regr);
            break;
         case defNeuronSSCNNEncoder:
            sscnn_enc = new CNeuronSSCNNEncoder();
            if(!sscnn_enc)
               ReturnFalse;
            if(sscnn_enc.Init(iOutputs, index, OpenCL, 3, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = sscnn_enc;
               return true;
              }
            DeleteObj(sscnn_enc);
            break;
         case defNeuronSSCNN:
            sscnn = new CNeuronSSCNN();
            if(!sscnn)
               ReturnFalse;
            if(sscnn.Init(iOutputs, index, OpenCL, 3, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = sscnn;
               return true;
              }
            DeleteObj(sscnn);
            break;
         case defCircleParams:
            circl_param = new CCircleParams();
            if(!circl_param)
               ReturnFalse;
            if(circl_param.Init(iOutputs, index, OpenCL, 3, 1, ADAM, 1))
              {
               m_data[index] = circl_param;
               return true;
              }
            DeleteObj(circl_param);
            break;
         case defNeuronTQMHA:
            tqmha = new CNeuronTQMHA();
            if(!tqmha)
               ReturnFalse;
            if(tqmha.Init(iOutputs, index, OpenCL, 3, 1, 1, 5, 3, 1, ADAM, 1))
              {
               m_data[index] = tqmha;
               return true;
              }
            DeleteObj(tqmha);
            break;
         case defNeuronHimNetGCRU:
            gcru = new CNeuronHimNetGCRU();
            if(!gcru)
               ReturnFalse;
            if(gcru.Init(iOutputs, index, OpenCL, 3, 1, 1, 2, 1, ADAM, 1))
              {
               m_data[index] = gcru;
               return true;
              }
            DeleteObj(gcru);
            break;
         case defNeuronHimNetEncoder:
            hn_enc = new CNeuronHimNetEncoder();
            if(!hn_enc)
               ReturnFalse;
            if(hn_enc.Init(iOutputs, index, OpenCL, 3, 1, 1, 2, 1, 2, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = hn_enc;
               return true;
              }
            DeleteObj(hn_enc);
            break;
         case defNeuronHimNetDecoder:
            hn_dec = new CNeuronHimNetDecoder();
            if(!hn_dec)
               ReturnFalse;
            if(hn_dec.Init(iOutputs, index, OpenCL, 3, 1, 1, 2, 1, 2, ADAM, 1))
              {
               m_data[index] = hn_dec;
               return true;
              }
            DeleteObj(hn_dec);
            break;
         case defNeuronSNSMHAttention:
            snsatt = new CNeuronSNSMHAttention();
            if(!snsatt)
               ReturnFalse;
            if(snsatt.Init(iOutputs, index, OpenCL, 3, 3, 1, 2, 0.5f, ADAM, 1))
              {
               m_data[index] = snsatt;
               return true;
              }
            DeleteObj(snsatt);
            break;
         case defNeuronFastGConv:
            fgcru = new CNeuronFastGConv();
            if(!fgcru)
               ReturnFalse;
            if(fgcru.Init(iOutputs, index, OpenCL, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = fgcru;
               return true;
              }
            DeleteObj(fgcru);
            break;
         case defNeuronSAGDFN:
            sagdfn = new CNeuronSAGDFN();
            if(!sagdfn)
               ReturnFalse;
            if(sagdfn.Init(iOutputs, index, OpenCL, 3, 2, 1, 1, 1, 1, 0.5f, 1, 3, 1, ADAM, 1))
              {
               m_data[index] = sagdfn;
               return true;
              }
            DeleteObj(sagdfn);
            break;
         case defNeuronSpatialEmbedding:
            semb = new CNeuronSpatialEmbedding();
            if(!semb)
               ReturnFalse;
            if(semb.Init(iOutputs, index, OpenCL, 3, 2, 2, ADAM, 1))
              {
               m_data[index] = semb;
               return true;
              }
            DeleteObj(semb);
            break;
         case defNeuronTempEmbedding:
            temb = new CNeuronTempEmbedding();
            if(!temb)
               ReturnFalse;
            if(temb.Init(iOutputs, index, OpenCL, 3, 2, 1, 1, 1, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = temb;
               return true;
              }
            DeleteObj(temb);
            break;
         case defNeuronGlobalLocalAttention:
            glatt = new CNeuronGlobalLocalAttention();
            if(!glatt)
               ReturnFalse;
            if(glatt.Init(iOutputs, index, OpenCL, 3, 2, 1, 1, 1, 0.5f, ADAM, 1))
              {
               m_data[index] = glatt;
               return true;
              }
            DeleteObj(glatt);
            break;
         case defNeuronExtralonger:
            exlon = new CNeuronExtralonger();
            if(!exlon)
               ReturnFalse;
            if(exlon.Init(iOutputs, index, OpenCL, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0.5f, ADAM, 1))
              {
               m_data[index] = exlon;
               return true;
              }
            DeleteObj(exlon);
            break;
         case defNeuronExtralongerGraph:
            exlongr = new CNeuronExtralongerGraph();
            if(!exlongr)
               ReturnFalse;
            if(exlongr.Init(iOutputs, index, OpenCL, 3, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 0.5f, ADAM, 1))
              {
               m_data[index] = exlongr;
               return true;
              }
            DeleteObj(exlongr);
            break;
         case defNeuronGraphAttention:
            gratt = new CNeuronGraphAttention();
            if(!gratt)
               ReturnFalse;
            if(gratt.Init(iOutputs, index, OpenCL, 3, 1, 2, 2, 0.5f, ADAM, 1))
              {
               m_data[index] = gratt;
               return true;
              }
            DeleteObj(gratt);
            break;
         case defNeuronSparseGraphAttention:
            sgratt = new CNeuronSparseGraphAttention();
            if(!sgratt)
               ReturnFalse;
            if(sgratt.Init(iOutputs, index, OpenCL, 3, 1, 2, 0.5f, 2, 1, ADAM, 1))
              {
               m_data[index] = sgratt;
               return true;
              }
            DeleteObj(sgratt);
            break;
         case defNeuronGlobLocGraphAtt:
            glgratt = new CNeuronGlobLocGraphAtt();
            if(!glgratt)
               ReturnFalse;
            if(glgratt.Init(iOutputs, index, OpenCL, 3, 1, 2, 0.5f, 2, 1, ADAM, 1))
              {
               m_data[index] = glgratt;
               return true;
              }
            DeleteObj(glgratt);
            break;
         case defNeuronSpikeConv:
            spike = new CNeuronSpikeConv();
            if(!spike)
               ReturnFalse;
            if(spike.Init(iOutputs, index, OpenCL, 3, 1, 2, 1, 1, ADAM, 1))
              {
               m_data[index] = spike;
               return true;
              }
            DeleteObj(spike);
            break;
         case defNeuronGateLineAttention:
            glineatt = new CNeuronGateLineAttention();
            if(!glineatt)
               ReturnFalse;
            if(glineatt.Init(iOutputs, index, OpenCL, 3, 1, 2, 1, ADAM, 1))
              {
               m_data[index] = glineatt;
               return true;
              }
            DeleteObj(glineatt);
            break;
         case defNeuronSpikeSparseAttention:
            spikeatt = new CNeuronSpikeSparseAttention();
            if(!spikeatt)
               ReturnFalse;
            if(spikeatt.Init(iOutputs, index, OpenCL, 3, 1, 2, 0.5f, 1, 1, ADAM, 1))
              {
               m_data[index] = spikeatt;
               return true;
              }
            DeleteObj(spikeatt);
            break;
         case defNeuronMoEConv:
            moecon = new CNeuronMoEConv();
            if(!moecon)
               ReturnFalse;
            if(moecon.Init(iOutputs, index, OpenCL, 3, 1, 2, 2, 0.5f, 1, ADAM, 1))
              {
               m_data[index] = moecon;
               return true;
              }
            DeleteObj(moecon);
            break;
         case defNeuronSpikingBrain:
            spikebrain = new CNeuronSpikingBrain();
            if(!spikebrain)
               ReturnFalse;
            if(spikebrain.Init(iOutputs, index, OpenCL, 3, 1, 2, 0.5f, 1, 1, ADAM, 1))
              {
               m_data[index] = spikebrain;
               return true;
              }
            DeleteObj(spikebrain);
            break;
         case defNeuronSparseSoftMax:
            sparsoftmax = new CNeuronSparseSoftMax();
            if(!sparsoftmax)
               ReturnFalse;
            if(sparsoftmax.Init(iOutputs, index, OpenCL, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = sparsoftmax;
               return true;
              }
            DeleteObj(sparsoftmax);
            break;
         case defNeuronSpikeActivation:
            spikeact = new CNeuronSpikeActivation();
            if(!spikeact)
               ReturnFalse;
            if(spikeact.Init(iOutputs, index, OpenCL, 3, ADAM, 1))
              {
               m_data[index] = spikeact;
               return true;
              }
            DeleteObj(spikeact);
            break;
         case defNeuronSpikeResNeXtBlock:
            spikeresblock = new CNeuronSpikeResNeXtBlock();
            if(!spikeresblock)
               ReturnFalse;
            if(spikeresblock.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, 2, 3, ADAM, 1))
              {
               m_data[index] = spikeresblock;
               return true;
              }
            DeleteObj(spikeresblock);
            break;
         case defNeuronSpikeResNeXtBottleneck:
            spikeresbottleneck = new CNeuronSpikeResNeXtBottleneck();
            if(!spikeresbottleneck)
               ReturnFalse;
            if(spikeresbottleneck.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, 2, 3, ADAM, 1))
              {
               m_data[index] = spikeresbottleneck;
               return true;
              }
            DeleteObj(spikeresbottleneck);
            break;
         case defNeuronSpikeResNeXtResidual:
            spikeresresid = new CNeuronSpikeResNeXtResidual();
            if(!spikeresresid)
               ReturnFalse;
            if(spikeresresid.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = spikeresresid;
               return true;
              }
            DeleteObj(spikeresresid);
            break;
         case defNeuronSSAM:
            ssam = new CNeuronSSAM();
            if(!ssam)
               ReturnFalse;
            if(ssam.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, true, ADAM, 1))
              {
               m_data[index] = ssam;
               return true;
              }
            DeleteObj(ssam);
            break;
         case defNeuronSSAMResNeXtBlock:
            ssamblock = new CNeuronSSAMResNeXtBlock();
            if(!ssamblock)
               ReturnFalse;
            if(ssamblock.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, 3, 3, 1, 3, ADAM, 1))
              {
               m_data[index] = ssamblock;
               return true;
              }
            DeleteObj(ssamblock);
            break;
         case defNeuronSTFS:
            stfs = new CNeuronSTFS();
            if(!stfs)
               ReturnFalse;
            if(stfs.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = stfs;
               return true;
              }
            DeleteObj(stfs);
            break;
         case defNeuronSpikeConvGRU:
            sp_conv_gru = new CNeuronSpikeConvGRU();
            if(!sp_conv_gru)
               ReturnFalse;
            if(sp_conv_gru.Init(iOutputs, index, OpenCL, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = sp_conv_gru;
               return true;
              }
            DeleteObj(sp_conv_gru);
            break;
         case defNeuronAddToStack:
            stack = new CNeuronAddToStack();
            if(!stack)
               ReturnFalse;
            if(stack.Init(iOutputs, index, OpenCL, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = stack;
               return true;
              }
            DeleteObj(stack);
            break;
         case defNeuronStackCorrelation:
            st_correl = new CNeuronStackCorrelation();
            if(!st_correl)
               ReturnFalse;
            if(st_correl.Init(iOutputs, index, OpenCL, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = st_correl;
               return true;
              }
            DeleteObj(st_correl);
            break;
         case defNeuronSTEFlowNetEncoder:
            ste_encod = new CNeuronSTEFlowNetEncoder();
            if(!ste_encod)
               ReturnFalse;
            if(ste_encod.Init(iOutputs, index, OpenCL, windows, windows, 3, 3, 1, 3, 3, ADAM, 1))
              {
               m_data[index] = ste_encod;
               return true;
              }
            DeleteObj(ste_encod);
            break;
         case defNeuronSTEFlowNet:
            ste = new CNeuronSTEFlowNet();
            if(!ste)
               ReturnFalse;
            if(ste.Init(iOutputs, index, OpenCL, windows, windows, 3, 3, 1, 3, 3, ADAM, 1))
              {
               m_data[index] = ste;
               return true;
              }
            DeleteObj(ste);
            break;
         case defNeuronSTEFlowNetDecoder:
            ste_decod = new CNeuronSTEFlowNetDecoder();
            if(!ste_decod)
               ReturnFalse;
            if(ste_decod.Init(iOutputs, index, OpenCL, windows, windows, 3, 3, 1, 3, NULL, ADAM, 1))
              {
               m_data[index] = ste_decod;
               return true;
              }
            DeleteObj(ste_decod);
            break;
         case defNeuronSTEFlowNetResidualBlock:
            ste_resid = new CNeuronSTEFlowNetResidualBlock();
            if(!ste_resid)
               ReturnFalse;
            if(ste_resid.Init(iOutputs, index, OpenCL, 3, 3, 3, 1, ADAM, 1))
              {
               m_data[index] = ste_resid;
               return true;
              }
            DeleteObj(ste_resid);
            break;
         case defNeuronMultiScaleStackCorrelation:
            mul_correl = new CNeuronMultiScaleStackCorrelation();
            if(!mul_correl)
               ReturnFalse;
            if(mul_correl.Init(iOutputs, index, OpenCL, 3, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = mul_correl;
               return true;
              }
            DeleteObj(mul_correl);
            break;
         case defNeuronSpikeConvGRU2D:
            sp_conv_gru2D = new CNeuronSpikeConvGRU2D();
            if(!sp_conv_gru2D)
               ReturnFalse;
            if(sp_conv_gru2D.Init(iOutputs, index, OpenCL, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = sp_conv_gru2D;
               return true;
              }
            DeleteObj(sp_conv_gru2D);
            break;
         case defNeuronRAFT:
            raft = new CNeuronRAFT();
            if(!raft)
               ReturnFalse;
            if(raft.Init(iOutputs, index, OpenCL, windows, windows, 3, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = raft;
               return true;
              }
            DeleteObj(raft);
            break;
         case defNeuronMultScalStackCorrelBySegments:
            segm_corr = new CNeuronMultScalStackCorrelBySegments();
            if(!segm_corr)
               ReturnFalse;
            if(segm_corr.Init(iOutputs, index, OpenCL, 2, 2, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = segm_corr;
               return true;
              }
            DeleteObj(segm_corr);
            break;
         case defNeuronTMAMFE:
            mfe = new CNeuronTMAMFE();
            if(!mfe)
               ReturnFalse;
            if(mfe.Init(iOutputs, index, OpenCL, 2, mfe_windows, 3, ADAM, 1))
              {
               m_data[index] = mfe;
               return true;
              }
            DeleteObj(mfe);
            break;
         case defNeuronTMAMPA:
            mpa = new CNeuronTMAMPA();
            if(!mpa)
               ReturnFalse;
            if(mpa.Init(iOutputs, index, OpenCL, 2, 2, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = mpa;
               return true;
              }
            DeleteObj(mpa);
            break;
         case defNeuronTMA:
            tma = new CNeuronTMA();
            if(!tma)
               ReturnFalse;
            if(tma.Init(iOutputs, index, OpenCL, windows, windows, 3, 3, 3, 2, 1, 2, 1, ADAM, 1))
              {
               m_data[index] = tma;
               return true;
              }
            DeleteObj(tma);
            break;
         case defNeuronSpikeSTConvGRU:
            st_convgru = new CNeuronSpikeSTConvGRU();
            if(!st_convgru)
               ReturnFalse;
            if(st_convgru.Init(iOutputs, index, OpenCL, 3,  mfe_windows, 3, ADAM, 1))
              {
               m_data[index] = st_convgru;
               return true;
              }
            DeleteObj(st_convgru);
            break;
         case defNeuronFERENet:
            fere = new CNeuronFERENet();
            if(!fere)
               ReturnFalse;
            if(fere.Init(iOutputs, index, OpenCL, 3, windows, ADAM, 1))
              {
               m_data[index] = fere;
               return true;
              }
            DeleteObj(fere);
            break;
         case defNeuronFGDModule:
            fgd = new CNeuronFGDModule();
            if(!fgd)
               ReturnFalse;
            if(fgd.Init(iOutputs, index, OpenCL, 3, 3, 3, 3, ADAM, 1))
              {
               m_data[index] = fgd;
               return true;
              }
            DeleteObj(fgd);
            break;
         case defNeuronEVMGRFlowNet:
            evmgr = new CNeuronEVMGRFlowNet();
            if(!evmgr)
               ReturnFalse;
            if(evmgr.Init(iOutputs, index, OpenCL, 3, windows, ADAM, 1))
              {
               m_data[index] = evmgr;
               return true;
              }
            DeleteObj(evmgr);
            break;
         case defNeuronSpikeConvBlock:
            spike_block = new CNeuronSpikeConvBlock();
            if(!spike_block)
               ReturnFalse;
            //--- Serialized SpikeConvBlock restores all nested holders in Load().
            //--- A default Init() creates incompatible temporary shapes first.
            spike_block.AttachOpenCL(OpenCL);
            m_data[index] = spike_block;
            return true;
            break;
         case defNeuronMSRes:
            msres = new CNeuronMSRes();
            if(!msres)
               ReturnFalse;
            if(msres.Init(iOutputs, index, OpenCL, 3, 1, 1, ADAM, 1))
              {
               m_data[index] = msres;
               return true;
              }
            DeleteObj(msres);
            break;
         case defNeuronSpikeQKAttention:
            qkatt = new CNeuronSpikeQKAttention();
            if(!qkatt)
               ReturnFalse;
            if(qkatt.Init(iOutputs, index, OpenCL, 3, 1, 1, 2, ADAM, 1))
              {
               m_data[index] = qkatt;
               return true;
              }
            DeleteObj(qkatt);
            break;
         case defNeuronSDSA:
            sdsa = new CNeuronSDSA();
            if(!sdsa)
               ReturnFalse;
            if(sdsa.Init(iOutputs, index, OpenCL, 3, 1, 1, 2, ADAM, 1))
              {
               m_data[index] = sdsa;
               return true;
              }
            DeleteObj(sdsa);
            break;
         case defNeuronSDformerUBlock:
            sdublock = new CNeuronSDformerUBlock();
            if(!sdublock)
               ReturnFalse;
            if(sdublock.Init(iOutputs, index, OpenCL, 3, windows, mfe_windows, mfe_windows, 2, ADAM, 1))
              {
               m_data[index] = sdublock;
               return true;
              }
            DeleteObj(sdublock);
            break;
         case defNeuronSDformerFlow:
            sdflow = new CNeuronSDformerFlow();
            if(!sdflow)
               ReturnFalse;
            if(sdflow.Init(iOutputs, index, OpenCL, 3, sdflow_win, mfe_windows, sdflow_layers, 2, ADAM, 1))
              {
               m_data[index] = sdflow;
               return true;
              }
            DeleteObj(sdflow);
            break;
         case defNeuronSpikePatchStak:
            sp_patch_stak = new CNeuronSpikePatchStak();
            if(!sp_patch_stak)
               ReturnFalse;
            if(sp_patch_stak.Init(iOutputs, index, OpenCL, 3, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = sp_patch_stak;
               return true;
              }
            DeleteObj(sp_patch_stak);
            break;
         case defNeuronSpike2DSSMOCL:
            sp_ssm = new CNeuronSpike2DSSMOCL();
            if(!sp_ssm)
               ReturnFalse;
            if(sp_ssm.Init(iOutputs, index, OpenCL, 3, 3, 2, 2, ADAM, 1))
              {
               m_data[index] = sp_ssm;
               return true;
              }
            DeleteObj(sp_ssm);
            break;
         case defNeuronSpikeMamba2D:
            sp_mamba = new CNeuronSpikeMamba2D();
            if(!sp_mamba)
               ReturnFalse;
            if(sp_mamba.Init(iOutputs, index, OpenCL, 3, 3, 2, ADAM, 1))
              {
               m_data[index] = sp_mamba;
               return true;
              }
            DeleteObj(sp_mamba);
            break;
         case defNeuronSpikeSTSSM:
            sp_stssm = new CNeuronSpikeSTSSM();
            if(!sp_stssm)
               ReturnFalse;
            if(sp_stssm.Init(iOutputs, index, OpenCL, 3, 2, 2, 2, 2, ADAM, 1))
              {
               m_data[index] = sp_stssm;
               return true;
              }
            DeleteObj(sp_stssm);
            break;
         case defNeuronESTMEncoder:
            sp_estm = new CNeuronESTMEncoder();
            if(!sp_estm)
               ReturnFalse;
            if(sp_estm.Init(iOutputs, index, OpenCL, mfe_windows, windows, 2, 2, windows, ADAM, 1))
              {
               m_data[index] = sp_estm;
               return true;
              }
            DeleteObj(sp_estm);
            break;
         case defNeuronCreateFlow:
            cr_flow = new CNeuronCreateFlow();
            if(!cr_flow)
               ReturnFalse;
            if(cr_flow.Init(iOutputs, index, OpenCL, 5, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = cr_flow;
               return true;
              }
            DeleteObj(cr_flow);
            break;
         case defNeuronBiDirectCorrelation:
            bi_corr = new CNeuronBiDirectCorrelation();
            if(!bi_corr)
               ReturnFalse;
            if(bi_corr.Init(iOutputs, index, OpenCL, 2, 5, 1, 2, ADAM, 1))
              {
               m_data[index] = bi_corr;
               return true;
              }
            DeleteObj(bi_corr);
            break;
         case defNeuronSATMA:
            satma = new CNeuronSATMA();
            if(!satma)
               ReturnFalse;
            if(satma.Init(iOutputs, index, OpenCL, 2, 5, 1, 2, 1, ADAM, 1))
              {
               m_data[index] = satma;
               return true;
              }
            DeleteObj(satma);
            break;
         case defNeuronBAT:
            bat = new CNeuronBAT();
            if(!bat)
               ReturnFalse;
            if(bat.Init(iOutputs, index, OpenCL, 2, 2, 5, 1, ADAM, 1))
              {
               m_data[index] = bat;
               return true;
              }
            DeleteObj(bat);
            break;
         case defNeuronBATvADM:
            bat = new CNeuronBATvADM();
            if(!bat)
               ReturnFalse;
            if(bat.Init(iOutputs, index, OpenCL, 2, 2, 5, 1, ADAM, 1))
              {
               m_data[index] = bat;
               return true;
              }
            DeleteObj(bat);
            break;
         case defNeuronSpikeSCM:
            scm = new CNeuronSpikeSCM;
            if(!scm)
               ReturnFalse;
            if(scm.Init(iOutputs, index, OpenCL, 3, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = scm;
               return true;
              }
            DeleteObj(scm);
            break;
         case defNeuronSpikeFAM:
            fam = new CNeuronSpikeFAM;
            if(!fam)
               ReturnFalse;
            if(fam.Init(iOutputs, index, OpenCL, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = fam;
               return true;
              }
            DeleteObj(fam);
            break;
         case defNeuronSpikeMDC:
            mdc = new CNeuronSpikeMDC;
            if(!mdc)
               ReturnFalse;
            if(mdc.Init(iOutputs, index, OpenCL, 3, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = mdc;
               return true;
              }
            DeleteObj(mdc);
            break;
         case defNeuronMDS:
            mds = new CNeuronMDS;
            if(!mds)
               ReturnFalse;
            if(mds.Init(iOutputs, index, OpenCL, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = mds;
               return true;
              }
            DeleteObj(mds);
            break;
         case defNeuronSpikeADM:
            adm = new CNeuronSpikeADM;
            if(!adm)
               ReturnFalse;
            if(adm.Init(iOutputs, index, OpenCL, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = adm;
               return true;
              }
            DeleteObj(adm);
            break;
         case defNeuronCDCSelfCorrector:
            selfcorrect = new CNeuronCDCSelfCorrector;
            if(!selfcorrect)
               ReturnFalse;
            if(selfcorrect.Init(iOutputs, index, OpenCL, 3, windows, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = selfcorrect;
               return true;
              }
            DeleteObj(selfcorrect);
            break;
         case defNeuronSpikeCDC:
            cdc = new CNeuronSpikeCDC;
            if(!cdc)
               ReturnFalse;
            if(cdc.Init(iOutputs, index, OpenCL, 3, windows, 2, 1, 1, ADAM, 1))
              {
               m_data[index] = cdc;
               return true;
              }
            DeleteObj(cdc);
            break;
         case defNeuronSpikeEEMFLow:
            eemflow = new CNeuronSpikeEEMFLow;
            if(!eemflow)
               ReturnFalse;
            if(eemflow.Init(iOutputs, index, OpenCL, 3, 2, 5, 1, 1, ADAM, 1))
              {
               m_data[index] = eemflow;
               return true;
              }
            DeleteObj(eemflow);
            break;
         case defNeuronSpikeDepthWiseConv:
            dw_conv = new CNeuronSpikeDepthWiseConv;
            if(!dw_conv)
               ReturnFalse;
            if(dw_conv.Init(iOutputs, index, OpenCL, 3, 2, 3, 1, 5, 1, ADAM, 1))
              {
               m_data[index] = dw_conv;
               return true;
              }
            DeleteObj(dw_conv);
            break;
         case defNeuronMultiScaleExcitation:
            excit = new CNeuronMultiScaleExcitation;
            if(!excit)
               ReturnFalse;
            if(excit.Init(iOutputs, index, OpenCL, 3, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = excit;
               return true;
              }
            DeleteObj(excit);
            break;
         case defNeuronMultiScaleDifference:
            ms_diff = new CNeuronMultiScaleDifference;
            if(!ms_diff)
               ReturnFalse;
            if(ms_diff.Init(iOutputs, index, OpenCL, 3, 2, 5, windows, ADAM, 1))
              {
               m_data[index] = ms_diff;
               return true;
              }
            DeleteObj(ms_diff);
            break;
         case defNeuronCorrelationEncoder:
            corr_enc = new CNeuronCorrelationEncoder;
            if(!corr_enc)
               ReturnFalse;
            if(corr_enc.Init(iOutputs, index, OpenCL, 3, 2, 5, 2, ADAM, 1))
              {
               m_data[index] = corr_enc;
               return true;
              }
            DeleteObj(corr_enc);
            break;
         case defNeuronEDCFlow:
            edcflow = new CNeuronEDCFlow;
            if(!edcflow)
               ReturnFalse;
            if(edcflow.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 5, windows, 2, ADAM, 1))
              {
               m_data[index] = edcflow;
               return true;
              }
            DeleteObj(edcflow);
            break;
         case defNeuronCreateICEFlow:
            ice = new CNeuronCreateICEFlow;
            if(!ice)
               ReturnFalse;
            if(ice.Init(iOutputs, index, OpenCL, 3, 5, 5, 1, ADAM, 1))
              {
               m_data[index] = ice;
               return true;
              }
            DeleteObj(ice);
            break;
         case defNeuronSpikeMFE:
            sp_mfe = new CNeuronSpikeMFE;
            if(!sp_mfe)
               ReturnFalse;
            if(sp_mfe.Init(iOutputs, index, OpenCL, 3, mfe_windows, 5, ADAM, 1))
              {
               m_data[index] = sp_mfe;
               return true;
              }
            DeleteObj(sp_mfe);
            break;
         case defNeuronSpikeMHCrossAttention:
            sp_crosatt = new CNeuronSpikeMHCrossAttention;
            if(!sp_crosatt)
               ReturnFalse;
            if(sp_crosatt.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 5, 2, ADAM, 1))
              {
               m_data[index] = sp_crosatt;
               return true;
              }
            DeleteObj(sp_crosatt);
            break;
         case defNeuronSpikeMixFusion:
            mix_fusion = new CNeuronSpikeMixFusion;
            if(!mix_fusion)
               ReturnFalse;
            if(mix_fusion.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, ADAM, 1))
              {
               m_data[index] = mix_fusion;
               return true;
              }
            DeleteObj(mix_fusion);
            break;
         case defNeuronSTFlow:
            stflow = new CNeuronSTFlow;
            if(!stflow)
               ReturnFalse;
            if(stflow.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 5, 2, ADAM, 1))
              {
               m_data[index] = stflow;
               return true;
              }
            DeleteObj(stflow);
            break;
         case defNeuronPSSE:
            pse = new CNeuronPSSE;
            if(!pse)
               ReturnFalse;
            if(pse.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, ADAM, 1))
              {
               m_data[index] = pse;
               return true;
              }
            DeleteObj(pse);
            break;
         case defNeuronSpikeDepthWiseResidual:
            dwr = new CNeuronSpikeDepthWiseResidual;
            if(!dwr)
               ReturnFalse;
            if(dwr.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 5, 2, ADAM, 1))
              {
               m_data[index] = dwr;
               return true;
              }
            DeleteObj(dwr);
            break;
         case defNeuronSpikeSuperKernelBlock:
            sk_block = new CNeuronSpikeSuperKernelBlock;
            if(!sk_block)
               ReturnFalse;
            if(sk_block.Init(iOutputs, index, OpenCL, 3, 5, mfe_windows, mfe_windows, 5, 2, ADAM, 1))
              {
               m_data[index] = sk_block;
               return true;
              }
            DeleteObj(sk_block);
            break;
         case defNeuronSpikeGMA:
            gma = new CNeuronSpikeGMA;
            if(!gma)
               ReturnFalse;
            if(gma.Init(iOutputs, index, OpenCL, 3, 5, 2, ADAM, 1))
              {
               m_data[index] = gma;
               return true;
              }
            DeleteObj(gma);
            break;
         case defNeuronETROF:
            etrof = new CNeuronETROF;
            if(!etrof)
               ReturnFalse;
            if(etrof.Init(iOutputs, index, OpenCL, 3, 5, 5, ADAM, 1))
              {
               m_data[index] = etrof;
               return true;
              }
            DeleteObj(etrof);
            break;
         case defNeuronPSSEFlow:
            pseflow = new CNeuronPSSEFlow;
            if(!pseflow)
               ReturnFalse;
            if(pseflow.Init(iOutputs, index, OpenCL, 3, 5, 5, ADAM, 1))
              {
               m_data[index] = pseflow;
               return true;
              }
            DeleteObj(pseflow);
            break;
         case defNeuronSpikeUSR:
            usr = new CNeuronSpikeUSR;
            if(!usr)
               ReturnFalse;
            if(usr.Init(iOutputs, index, OpenCL, 3, 5, 5, 1, ADAM, 1))
              {
               m_data[index] = usr;
               return true;
              }
            DeleteObj(usr);
            break;
         case defNeuronInitialFlow:
            init_flow = new CNeuronInitialFlow;
            if(!init_flow)
               ReturnFalse;
            if(init_flow.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 5, ADAM, 1))
              {
               m_data[index] = init_flow;
               return true;
              }
            DeleteObj(init_flow);
            break;
         case defNeuronSpikeSMR:
            smr = new CNeuronSpikeSMR;
            if(!smr)
               ReturnFalse;
            if(smr.Init(iOutputs, index, OpenCL, 3, 5, 5, 5, ADAM, 1))
              {
               m_data[index] = smr;
               return true;
              }
            DeleteObj(smr);
            break;
         case defNeuronEVAFlow:
            eva = new CNeuronEVAFlow;
            if(!eva)
               ReturnFalse;
            if(eva.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, ADAM, 1))
              {
               m_data[index] = eva;
               return true;
              }
            DeleteObj(eva);
            break;
         case defNeuronResFlowHTR:
            htr = new CNeuronResFlowHTR;
            if(!htr)
               ReturnFalse;
            if(htr.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, ADAM, 1))
              {
               m_data[index] = htr;
               return true;
              }
            DeleteObj(htr);
            break;
         case defNeuronResFlow:
            resflow = new CNeuronResFlow;
            if(!resflow)
               ReturnFalse;
            if(resflow.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, 2, ADAM, 1))
              {
               m_data[index] = resflow;
               return true;
              }
            DeleteObj(resflow);
            break;
         case defNeuronConfGateTailAwareMoE:
            tailaware = new CNeuronConfGateTailAwareMoE;
            if(!tailaware)
               ReturnFalse;
            if(tailaware.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, ADAM, 1))
              {
               m_data[index] = tailaware;
               return true;
              }
            DeleteObj(tailaware);
            break;
         case defNeuronResFlowLattice:
            lattice = new CNeuronResFlowLattice;
            if(!lattice)
               ReturnFalse;
            if(lattice.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, 2, ADAM, 1))
              {
               m_data[index] = lattice;
               return true;
              }
            DeleteObj(lattice);
            break;
         case defFieldAwareParams:
            fapar = new CFieldAwareParams;
            if(!fapar)
               ReturnFalse;
            if(fapar.Init(iOutputs, index, OpenCL, 4, 5, 2, 3, 1, ADAM, 1))
              {
               m_data[index] = fapar;
               return true;
              }
            DeleteObj(fapar);
            break;
         case defNeuronFieldAwareConv:
            faconv = new CNeuronFieldAwareConv;
            if(!faconv)
               ReturnFalse;
            if(faconv.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = faconv;
               return true;
              }
            DeleteObj(faconv);
            break;
         case defNeuronMHFAT:
            fat = new CNeuronMHFAT;
            if(!fat)
               ReturnFalse;
            if(fat.Init(iOutputs, index, OpenCL, 3, 5, 2, 3, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = fat;
               return true;
              }
            DeleteObj(fat);
            break;
         case defNeuronMHCrossFAT:
            cross_fat = new CNeuronMHCrossFAT;
            if(!cross_fat)
               ReturnFalse;
            if(cross_fat.Init(iOutputs, index, OpenCL, 3, 5, 2, 3, 2, 2, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = cross_fat;
               return true;
              }
            DeleteObj(cross_fat);
            break;
         case defNeuronFieldPatternEmbedding:
            faemb = new CNeuronFieldPatternEmbedding;
            if(!faemb)
               ReturnFalse;
            if(faemb.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, ADAM, 1))
              {
               m_data[index] = faemb;
               return true;
              }
            DeleteObj(faemb);
            break;
         case defNeuronAutoToken:
            autotoken = new CNeuronAutoToken;
            if(!autotoken)
               ReturnFalse;
            if(autotoken.Init(iOutputs, index, OpenCL, 3, 5, 5, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = autotoken;
               return true;
              }
            DeleteObj(autotoken);
            break;
         case defNeuronMultiMixAttention:
            mixattention = new CNeuronMultiMixAttention;
            if(!mixattention)
               ReturnFalse;
            if(mixattention.Init(iOutputs, index, OpenCL, 4, 5, 2, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = mixattention;
               return true;
              }
            DeleteObj(mixattention);
            break;
         case defNeuronSparseMoEConv:
            sparsemoe = new CNeuronSparseMoEConv;
            if(!sparsemoe)
               ReturnFalse;
            if(sparsemoe.Init(iOutputs, index, OpenCL, 3, 5, 5, 2, 1, ADAM, 1))
              {
               m_data[index] = sparsemoe;
               return true;
              }
            DeleteObj(sparsemoe);
            break;
         case defNeuronMTmixAttBlock:
            mixattenblock = new CNeuronMTmixAttBlock;
            if(!mixattenblock)
               ReturnFalse;
            if(mixattenblock.Init(iOutputs, index, OpenCL, 4, 5, 2, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = mixattenblock;
               return true;
              }
            DeleteObj(mixattenblock);
            break;
         case defCrossMHFlashAttention:
            flash_att = new CCrossMHFlashAttention;
            if(!flash_att)
               ReturnFalse;
            if(flash_att.Init(iOutputs, index, OpenCL, 4, 2, false, 3, 2, ADAM, 1))
              {
               m_data[index] = flash_att;
               return true;
              }
            DeleteObj(flash_att);
            break;
         case defNeuronMixFeedForward:
            mixff = new CNeuronMixFeedForward;
            if(!mixff)
               ReturnFalse;
            if(mixff.Init(iOutputs, index, OpenCL, 4, 5, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = mixff;
               return true;
              }
            DeleteObj(mixff);
            break;
         case defNeuronOneTrans:
            onetrans = new CNeuronOneTrans;
            if(!onetrans)
               ReturnFalse;
            if(onetrans.Init(iOutputs, index, OpenCL, windows, 1, 2, 2, 2, 1, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = onetrans;
               return true;
              }
            DeleteObj(onetrans);
            break;
         case defNeuronSTCA:
            stca = new CNeuronSTCA;
            if(!stca)
               ReturnFalse;
            if(stca.Init(iOutputs, index, OpenCL, windows, 1, 2, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = stca;
               return true;
              }
            DeleteObj(stca);
            break;
         case defNeuronMHTHCrossAttention:
            thcatt = new CNeuronMHTHCrossAttention;
            if(!thcatt)
               ReturnFalse;
            if(thcatt.Init(iOutputs, index, OpenCL, 6, 1, 2, 2, 2, 1, 3, 2, ADAM, 1))
              {
               m_data[index] = thcatt;
               return true;
              }
            DeleteObj(thcatt);
            break;
         case defNeuronWeightGenerator:
            wgen = new CNeuronWeightGenerator;
            if(!wgen)
               ReturnFalse;
            if(wgen.Init(iOutputs, index, OpenCL, 3, 2, 2, ADAM, 1))
              {
               m_data[index] = wgen;
               return true;
              }
            DeleteObj(wgen);
            break;
         case defNeuronPCGR:
            pcrg = new CNeuronPCGR;
            if(!pcrg)
               ReturnFalse;
            if(pcrg.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 2, 10, 2, ADAM, 1))
              {
               m_data[index] = pcrg;
               return true;
              }
            DeleteObj(pcrg);
            break;
         case defNeuronADS:
            ads = new CNeuronADS;
            if(!ads)
               ReturnFalse;
            if(ads.Init(iOutputs, index, OpenCL, windows, 1, 2, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = ads;
               return true;
              }
            DeleteObj(ads);
            break;
         case defNeuronPerTokenFFN:
            pertoken = new CNeuronPerTokenFFN;
            if(!pertoken)
               ReturnFalse;
            if(pertoken.Init(iOutputs, index, OpenCL, 5, 1, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = pertoken;
               return true;
              }
            DeleteObj(pertoken);
            break;
         case defNeuronScenariosToken:
            scentoken = new CNeuronScenariosToken;
            if(!scentoken)
               ReturnFalse;
            if(scentoken.Init(iOutputs, index, OpenCL, 5, 4, 5, 2, 3, 2, 1, ADAM, 1))
              {
               m_data[index] = scentoken;
               return true;
              }
            DeleteObj(scentoken);
            break;
         case defNeuronUnifiedTokenizer:
            uniftokens = new CNeuronUnifiedTokenizer;
            if(!uniftokens)
               ReturnFalse;
            if(uniftokens.Init(iOutputs, index, OpenCL, windows, 3, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = uniftokens;
               return true;
              }
            DeleteObj(uniftokens);
            break;
         case defNeuronDomainAwareAttention:
            domainatt = new CNeuronDomainAwareAttention;
            if(!domainatt)
               ReturnFalse;
            if(domainatt.Init(iOutputs, index, OpenCL, 8, 1, 2, 8, 2, 10, 2, 1, ADAM, 1))
              {
               m_data[index] = domainatt;
               return true;
              }
            DeleteObj(domainatt);
            break;
         case defNeuronFeatureSelfIteration:
            featureinter = new CNeuronFeatureSelfIteration;
            if(!featureinter)
               ReturnFalse;
            if(featureinter.Init(iOutputs, index, OpenCL, 8, 1, 2, 5, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = featureinter;
               return true;
              }
            DeleteObj(featureinter);
            break;
         case defNeuronDomainFused:
            domainfused = new CNeuronDomainFused;
            if(!domainfused)
               ReturnFalse;
            if(domainfused.Init(iOutputs, index, OpenCL, 8, 1, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = domainfused;
               return true;
              }
            DeleteObj(domainfused);
            break;
         case defNeuronMDL:
            mdl = new CNeuronMDL;
            if(!mdl)
               ReturnFalse;
            if(mdl.Init(iOutputs, index, OpenCL, windows, 1, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = mdl;
               return true;
              }
            DeleteObj(mdl);
            break;
         case defNeuronUniMixerTokenizer:
            umtoken = new CNeuronUniMixerTokenizer;
            if(!umtoken)
               ReturnFalse;
            if(umtoken.Init(iOutputs, index, OpenCL, windows, 1, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = umtoken;
               return true;
              }
            DeleteObj(umtoken);
            break;
         case defNeuronPerTokenSwiGLU:
            ptswiglu = new CNeuronPerTokenSwiGLU;
            if(!ptswiglu)
               ReturnFalse;
            if(ptswiglu.Init(iOutputs, index, OpenCL, 2, 2, 5, 3, 2, ADAM, 1))
              {
               m_data[index] = ptswiglu;
               return true;
              }
            DeleteObj(ptswiglu);
            break;
         case defNeuronUniMixer:
            unimix = new CNeuronUniMixer;
            if(!unimix)
               ReturnFalse;
            if(unimix.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = unimix;
               return true;
              }
            DeleteObj(unimix);
            break;
         case defNeuronSiameseNorm:
            siamnorm = new CNeuronSiameseNorm;
            if(!siamnorm)
               ReturnFalse;
            if(siamnorm.Init(iOutputs, index, OpenCL, 2, 4, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = siamnorm;
               return true;
              }
            DeleteObj(siamnorm);
            break;
         case defNeuronUniMixerBlock:
            umblock = new CNeuronUniMixerBlock;
            if(!umblock)
               ReturnFalse;
            if(umblock.Init(iOutputs, index, OpenCL, windows, 2, 2, 3, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = umblock;
               return true;
              }
            DeleteObj(umblock);
            break;
         case defNeuronINFNetScenarios:
            infnscen = new CNeuronINFNetScenarios;
            if(!infnscen)
               ReturnFalse;
            if(infnscen.Init(iOutputs, index, OpenCL, 3, windows, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = infnscen;
               return true;
              }
            DeleteObj(infnscen);
            break;
         case defNeuronINFNetHUBSequence:
            hub = new CNeuronINFNetHUBSequence;
            if(!hub)
               ReturnFalse;
            if(hub.Init(iOutputs, index, OpenCL, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = hub;
               return true;
              }
            DeleteObj(hub);
            break;
         case defNeuronINFNetBlock:
            infnblock = new CNeuronINFNetBlock;
            if(!infnblock)
               ReturnFalse;
            if(infnblock.Init(iOutputs, index, OpenCL, windows, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = infnblock;
               return true;
              }
            DeleteObj(infnblock);
            break;
         case defNeuronSequenceTokenizer:
            seq_tok = new CNeuronSequenceTokenizer;
            if(!seq_tok)
               ReturnFalse;
            if(seq_tok.Init(iOutputs, index, OpenCL, 2, 2, 2, ADAM, 1))
              {
               m_data[index] = seq_tok;
               return true;
              }
            DeleteObj(seq_tok);
            break;
         case defNeuronDifferenceTokenizer:
            diff_tok = new CNeuronDifferenceTokenizer;
            if(!diff_tok)
               ReturnFalse;
            if(diff_tok.Init(iOutputs, index, OpenCL, 5, 2, ADAM, 1))
              {
               m_data[index] = diff_tok;
               return true;
              }
            DeleteObj(diff_tok);
            break;
         case defNeuronDictionaryCrossAtt:
            dict_cross = new CNeuronDictionaryCrossAtt;
            if(!dict_cross)
               ReturnFalse;
            if(dict_cross.Init(iOutputs, index, OpenCL, 2, 2, 2, 10, ADAM, 1))
              {
               m_data[index] = dict_cross;
               return true;
              }
            DeleteObj(dict_cross);
            break;
         case defNeuronUncAD_MSD:
            uncad_msd = new CNeuronUncAD_MSD;
            if(!uncad_msd)
               ReturnFalse;
            if(uncad_msd.Init(iOutputs, index, OpenCL, 2, 20, 2, 2, 1.0f, ADAM, 1))
              {
               m_data[index] = uncad_msd;
               return true;
              }
            DeleteObj(uncad_msd);
            break;
         case defNeuronUncAD_MUE:
            uncad_mue = new CNeuronUncAD_MUE;
            if(!uncad_mue)
               ReturnFalse;
            if(uncad_mue.Init(iOutputs, index, OpenCL, 2, 2, 2, 20, 2, 1.0f, ADAM, 1))
              {
               m_data[index] = uncad_mue;
               return true;
              }
            DeleteObj(uncad_mue);
            break;
         case defNeuronUncAD_UGP:
            uncad_ugp = new CNeuronUncAD_UGP;
            if(!uncad_ugp)
               ReturnFalse;
            if(uncad_ugp.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 2, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = uncad_ugp;
               return true;
              }
            DeleteObj(uncad_ugp);
            break;
         case defNeuronUncAD_UGPL:
            uncad_ugpl = new CNeuronUncAD_UGPL;
            if(!uncad_ugpl)
               ReturnFalse;
            if(uncad_ugpl.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 2, 2, 2, ADAM, 1))
              {
               m_data[index] = uncad_ugpl;
               return true;
              }
            DeleteObj(uncad_ugpl);
            break;
         case defNeuronUncAD:
            uncad = new CNeuronUncAD;
            if(!uncad)
               ReturnFalse;
            if(uncad.Init(iOutputs, index, OpenCL, windows, 2, 2, 20, 2, 2, 3, 2, 8, ADAM, 1))
              {
               m_data[index] = uncad;
               return true;
              }
            DeleteObj(uncad);
            break;
         case defNeuronMomAD:
            momad = new CNeuronMomAD;
            if(!momad)
               ReturnFalse;
            if(momad.Init(iOutputs, index, OpenCL, windows, 2, 2, 20, 2, 2, 3, 2, 8, ADAM, 1))
              {
               m_data[index] = momad;
               return true;
              }
            DeleteObj(momad);
            break;
         case defNeuronCogDriverData:
            cogdriver_data = new CNeuronCogDriverData;
            if(!cogdriver_data)
               ReturnFalse;
            if(cogdriver_data.Init(iOutputs, index, OpenCL, 2, 2, 20, 2, 2, 1.0f, ADAM, 1))
              {
               m_data[index] = cogdriver_data;
               return true;
              }
            DeleteObj(cogdriver_data);
            break;
         case defNeuronCogDriverRankTCM:
            cogdriver_rank = new CNeuronCogDriverRankTCM;
            if(!cogdriver_rank)
               ReturnFalse;
            if(cogdriver_rank.Init(iOutputs, index, OpenCL, 8, 8, 2, 4, 2, ADAM, 1))
              {
               m_data[index] = cogdriver_rank;
               return true;
              }
            DeleteObj(cogdriver_rank);
            break;
         case defNeuronCogDriverForecastHead:
            cogdriver_forecast = new CNeuronCogDriverForecastHead;
            if(!cogdriver_forecast)
               ReturnFalse;
            if(cogdriver_forecast.Init(iOutputs, index, OpenCL, 8, 4, 2, 2, 2, ADAM, 1))
              {
               m_data[index] = cogdriver_forecast;
               return true;
              }
            DeleteObj(cogdriver_forecast);
            break;
         case defNeuronCogDriverForecastToken:
            cogdriver_token = new CNeuronCogDriverForecastToken;
            if(!cogdriver_token)
               ReturnFalse;
            if(cogdriver_token.Init(iOutputs, index, OpenCL, 8, 4, ADAM, 1))
              {
               m_data[index] = cogdriver_token;
               return true;
              }
            DeleteObj(cogdriver_token);
            break;
         case defNeuronScenarioForecast:
            //--- Serialized layers reconstruct their exact contract in
            //--- CNeuronScenarioForecast::Load().  Init requires a descriptor
            //--- and is therefore intentionally reserved for CNet::Create().
            scenario_forecast = new CNeuronScenarioForecast;
            if(!scenario_forecast)
               ReturnFalse;
            scenario_forecast.AttachOpenCL(OpenCL);
            m_data[index] = scenario_forecast;
            return true;
            break;
         case defNeuronScenarioCrossAttention:
            scenario_cross = new CNeuronScenarioCrossAttention;
            if(!scenario_cross)
               ReturnFalse;
            //--- Load reconstructs the serialized MSRes and scenario holders.
            //--- Initializing defaults here creates shared temporary buffers;
            //--- their slots can be released while the serialized graph loads.
            scenario_cross.AttachOpenCL(OpenCL);
            m_data[index] = scenario_cross;
            return true;
            break;
         case defNeuronRAGMemory:
            rag_memory = new CNeuronRAGMemory;
            if(!rag_memory)
               ReturnFalse;
            rag_memory.AttachOpenCL(OpenCL);
            m_data[index] = rag_memory;
            return true;
            break;
         case defNeuronD2Skill:
            d2_skill = new CD2Skill;
            if(!d2_skill)
               ReturnFalse;
            d2_skill.AttachOpenCL(OpenCL);
            m_data[index] = d2_skill;
            return true;
            break;
         case defNeuronReGENTADBackbone:
            regen_backbone = new CNeuronReGENTADBackbone;
            if(!regen_backbone)
               ReturnFalse;
            if(regen_backbone.Init(iOutputs, index, OpenCL, windows, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = regen_backbone;
               return true;
              }
            DeleteObj(regen_backbone);
            break;
         case defNeuronReGENTADHead:
            regen_head = new CNeuronReGENTADHead;
            if(!regen_head)
               ReturnFalse;
            if(regen_head.Init(iOutputs, index, OpenCL, 3, 2, 2, 2, NULL, ADAM, 1))
              {
               m_data[index] = regen_head;
               return true;
              }
            DeleteObj(regen_head);
            break;
         case defNeuronReGENTAD:
            regentad = new CNeuronReGENTAD;
            if(!regentad)
               ReturnFalse;
            if(regentad.Init(iOutputs, index, OpenCL, windows, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = regentad;
               return true;
              }
            DeleteObj(regentad);
            break;
         case defNeuronGDformerBackbone:
            gdformer_backbone = new CNeuronGDformerBackbone;
            if(!gdformer_backbone)
               ReturnFalse;
            if(gdformer_backbone.Init(iOutputs, index, OpenCL, windows, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = gdformer_backbone;
               return true;
              }
            DeleteObj(gdformer_backbone);
            break;
         case defNeuronGDformer:
            gdformer = new CNeuronGDformer;
            if(!gdformer)
               ReturnFalse;
            if(gdformer.Init(iOutputs, index, OpenCL, windows, 2, 2, 10, 2, 2, 3, 2, ADAM, 1))
              {
               m_data[index] = gdformer;
               return true;
              }
            DeleteObj(gdformer);
            break;
         case defNeuronMomADTTM:
            momad_ttm = new CNeuronMomADTTM;
            if(!momad_ttm)
               ReturnFalse;
            if(momad_ttm.Init(iOutputs, index, OpenCL, 2, 2, 2, 1, ADAM, 1))
              {
               m_data[index] = momad_ttm;
               return true;
              }
            DeleteObj(momad_ttm);
            break;
         case defNeuronMomADMPI:
            momad_mpi = new CNeuronMomADMPI;
            if(!momad_mpi)
               ReturnFalse;
            if(momad_mpi.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 2, 1, 1, 1, ADAM, 1))
              {
               m_data[index] = momad_mpi;
               return true;
              }
            DeleteObj(momad_mpi);
            break;
         case defNeuronVLADriverRAGMPI:
            vladriver_rag_mpi = new CNeuronVLADriverRAGMPI;
            if(!vladriver_rag_mpi)
               ReturnFalse;
            if(vladriver_rag_mpi.Init(iOutputs, index, OpenCL, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, ADAM, 1))
              {
               m_data[index] = vladriver_rag_mpi;
               return true;
              }
            DeleteObj(vladriver_rag_mpi);
            break;
         default:
            result = false;
            break;
        }
     }
   if(result)
      m_data[index] = temp;
   else
      DeleteObj(temp);
//---
   return (result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CArrayLayer::CreateElement(uint neurons,  uint outputs)
  {
   if(neurons <= 0)
      ReturnFalse;
//---
   CLayer *layer = new CLayer(outputs);
   if(!CheckPointer(layer) != POINTER_INVALID)
      ReturnFalse;
//---
   if(!layer.Reserve(neurons + 1))
      ReturnFalse;
   for(uint i = 0; i <= neurons; i++)
     {
      if(!layer.CreateElement(i))
         ReturnFalse;
      layer.IncreaseTotal();
     }
//---
   return (Add(layer));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBase::feedForward(CObject *&SourceObject)
  {
   bool result = false;
//---
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      return result;
//---
   CLayer *temp_l;
   CNeuronProof *temp_n;
   switch(SourceObject.Type())
     {
      case defLayer:
         temp_l = SourceObject;
         result = feedForward(temp_l);
         break;
      case defNeuronConv:
      case defNeuronProof:
      case defNeuronLSTM:
         temp_n = SourceObject;
         result = feedForward(temp_n.getOutputLayer());
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBase::updateInputWeights(CObject *&SourceObject)
  {
   bool result = false;
//---
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      return result;
//---
   CLayer *temp_l;
   CNeuronProof *temp_n;
   switch(SourceObject.Type())
     {
      case defLayer:
         temp_l = SourceObject;
         result = updateInputWeights(temp_l);
         break;
      case defNeuronConv:
      case defNeuronProof:
      case defNeuronLSTM:
         temp_n = SourceObject;
         temp_l = temp_n.getOutputLayer();
         result = updateInputWeights(temp_l);
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::feedForward(CLayer *prevLayer)
  {
   bool result = false;
//---
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      return result;
//---
   int total = prevLayer.Total() - iWindow + 1;
   CNeuron *temp;
   CConnection *con;
   result = true;
   for(int i = 0; (i < total && result); i += iStep)
     {
      float sum = 0;
      for(int j = 0; (j < iWindow && result); j++)
        {
         temp = prevLayer.At(i + j);
         con = Connections.At(j);
         if(CheckPointer(temp) == POINTER_INVALID || CheckPointer(con) == POINTER_INVALID)
            ReturnFalse;
         float val = temp.getOutputVal();
         sum += val * con.weight;
        }
      temp = OutputLayer.At(i / iStep);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      temp.setOutputVal(activationFunction(sum));
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuronConv::activationFunction(float x)
  {
   if(x >= 0)
      return x;
   return param * x;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBase::calcHiddenGradients(CObject *&TargetObject)
  {
   bool result = false;
//---
   if(CheckPointer(TargetObject) == POINTER_INVALID)
      return result;
//---
   CLayer *temp_l;
   CNeuronProof *temp_n;
   switch(TargetObject.Type())
     {
      case defLayer:
         temp_l = TargetObject;
         result = calcHiddenGradients(temp_l);
         break;
      case defNeuronConv:
      case defNeuronProof:
      case defNeuronLSTM:
         switch(Type())
           {
            case defNeuron:
               temp_n = TargetObject;
               result = temp_n.calcInputGradients(GetPointer(this), m_myIndex);
               break;
            case defNeuronLSTM:
               temp_n = TargetObject;
               temp_l = getOutputLayer();
               if(!temp_n.calcInputGradients(temp_l))
                 {
                  result = false;
                  break;
                 }
               result = calcHiddenGradients(temp_l);
               break;
            default:
               //temp_n=GetPointer(this);
               temp_l =/*temp_n*/getOutputLayer();
               temp_n = TargetObject;
               result = temp_n.calcInputGradients(temp_l);
               break;
           }
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::calcHiddenGradients(CLayer *&nextLayer)
  {
   if(CheckPointer(nextLayer) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID || OutputLayer.Total() <= 0)
      ReturnFalse;
//---
   gradient = 0;
   int total = OutputLayer.Total();
   CNeuron *temp;
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      temp.setGradient(temp.sumDOW(nextLayer)*activationFunctionDerivative(temp.getOutputVal()));
     }
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuronConv::activationFunctionDerivative(float x)
  {
   if(x >= 0)
      return 1;
   return param;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::updateInputWeights(CLayer *&prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   CConnection *con;
   float lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
   for(int n = 0; n < iWindow && !IsStopped(); n++)
     {
      con = Connections.At(n);
      if(CheckPointer(con) == POINTER_INVALID)
         continue;
      float delta = 0;
      int total_i = OutputLayer.Total();
      CNeuron *prev, *out;
      for(int i = 0; i < total_i; i++)
        {
         prev = prevLayer.At(n * iStep + i);
         out = OutputLayer.At(total_i - i - 1);
         if(CheckPointer(prev) == POINTER_INVALID || CheckPointer(out) == POINTER_INVALID)
            continue;
         delta += prev.getOutputVal() * out.getGradient();
        }
      if(optimization == SGD)
         con.weight += con.deltaWeight = (delta != 0 ? lr*delta : 0) + (con.deltaWeight != 0 ? alpha*con.deltaWeight : 0);
      else
        {
         con.mt = b1 * con.mt + (1 - b1) * delta;
         con.vt = (float)(b2 * con.vt + (1 - b2) * MathPow((double)delta, 2.0) + 0.00000001);
         con.weight += con.deltaWeight = (float)(lt * con.mt / sqrt(con.vt));
         t++;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::Init(uint numOutputs, uint myIndex, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type)
  {
   iWindow = window;
   iStep = step;
   if(!CNeuronBase::Init(window, myIndex, optimization_type))
      ReturnFalse;
   OutputLayer = new CLayer(numOutputs);
   if(CheckPointer(OutputLayer) == POINTER_INVALID)
      ReturnFalse;
   if(OutputLayer.Reserve(units_count))
      for(int i = 0; i < units_count; i++)
        {
         if(!OutputLayer.CreateElement(i))
            ReturnFalse;
         OutputLayer.IncreaseTotal();
        }
//---
   if(Type() == defNeuronProof)
     {
      if(CheckPointer(Connections) != POINTER_INVALID)
         Connections.Clear();
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronProof::~CNeuronProof(void)
  {
   DeleteObj(OutputLayer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::feedForward(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   int total = prevLayer.Total() - iWindow + 1;
   CNeuron *temp;
   for(int i = 0; i <= total; i += iStep)
     {
      float sum = 0;
      for(int j = 0; j < iWindow; j++)
        {
         temp = prevLayer.At(i + j);
         if(CheckPointer(temp) == POINTER_INVALID)
            continue;
         sum += temp.getOutputVal();
        }
      temp = OutputLayer.At(i / iStep);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      temp.setOutputVal(sum / iWindow);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::calcHiddenGradients(CLayer *&nextLayer)
  {
   if(CheckPointer(nextLayer) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID || OutputLayer.Total() <= 0)
      ReturnFalse;
//---
   gradient = 0;
   int total = OutputLayer.Total();
   CNeuron *temp;
   for(int i = 0; i < total; i++)
     {
      temp = OutputLayer.At(i);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      temp.setGradient(temp.sumDOW(nextLayer));
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::calcInputGradients(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID  || CheckPointer(prevLayer.At(0)) == POINTER_INVALID)
      ReturnFalse;
//---
   if(prevLayer.At(0).Type() != defNeuron)
     {
      CNeuronProof *temp = prevLayer.At(m_myIndex);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      prevLayer = temp.getOutputLayer();
      if(CheckPointer(prevLayer) == POINTER_INVALID)
         ReturnFalse;
     }
//---
   CNeuronBase *prevNeuron, *outputNeuron;
   int total = prevLayer.Total();
   for(int i = 0; i < total; i++)
     {
      prevNeuron = prevLayer.At(i);
      if(CheckPointer(prevNeuron) == POINTER_INVALID)
         continue;
      float prev_gradient = 0;
      int start = i - iWindow + iStep;
      start = (start - start % iStep) / iStep;
      float stop = (float)((i - i % iStep) / iStep + 1);
      for(int out = (int)fmax(0, start); out < (int)fmin(OutputLayer.Total(), stop); out++)
        {
         outputNeuron = OutputLayer.At(out);
         if(CheckPointer(outputNeuron) == POINTER_INVALID)
            continue;
         prev_gradient += outputNeuron.getGradient() / iWindow;
        }
      prevNeuron.setGradient(prev_gradient);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::calcInputGradients(CNeuronBase *prevNeuron, uint index)
  {
   if(CheckPointer(prevNeuron) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   if(prevNeuron.Type() != defNeuron)
     {
      CNeuronProof *temp = prevNeuron;
      return calcInputGradients(temp.getOutputLayer());
     }
//---
   CNeuronBase *outputNeuron;
   float prev_gradient = 0;
   int start = (int)index - iWindow + iStep;
   start = (start - start % iStep) / iStep;
   float stop = (float)((index - index % iStep) / iStep + 1);
   for(int out = (int)fmax(0, start); out < (int)fmin(OutputLayer.Total(), stop); out++)
     {
      outputNeuron = OutputLayer.At(out);
      if(CheckPointer(outputNeuron) == POINTER_INVALID)
         continue;
      prev_gradient += outputNeuron.getGradient() / iWindow;
     }
   prevNeuron.setGradient(prev_gradient);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::calcInputGradients(CLayer *prevLayer)
  {
   if(CheckPointer(prevLayer) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   if(prevLayer.At(0).Type() != defNeuron)
     {
      CNeuronProof *temp = prevLayer.At(m_myIndex);
      if(CheckPointer(temp) == POINTER_INVALID)
         ReturnFalse;
      prevLayer = temp.getOutputLayer();
      if(CheckPointer(prevLayer) == POINTER_INVALID)
         ReturnFalse;
     }
//---
   CNeuronBase *prevNeuron, *outputNeuron;
   CConnection *con;
   int total = prevLayer.Total();
   for(int i = 0; i < total; i++)
     {
      prevNeuron = prevLayer.At(i);
      if(CheckPointer(prevNeuron) == POINTER_INVALID)
         continue;
      float prev_gradient = 0;
      int start = i - iWindow + iStep;
      start = (start - start % iStep) / iStep;
      float stop = (float)((i - i % iStep) / iStep + 1);
      for(int out = (int)fmax(0, start); out < (int)fmin(OutputLayer.Total(), stop); out++)
        {
         outputNeuron = OutputLayer.At(out);
         int c = ((int)fmin(OutputLayer.Total(), stop) - out - 1) * iStep + i % iStep;
         con = Connections.At(c);
         if(CheckPointer(outputNeuron) == POINTER_INVALID || CheckPointer(con) == POINTER_INVALID)
            continue;
         prev_gradient += outputNeuron.getGradient() * prevNeuron.activationFunctionDerivative(prevNeuron.getOutputVal()) * con.weight;
        }
      prevNeuron.setGradient(prev_gradient);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::calcInputGradients(CNeuronBase *prevNeuron, uint index)
  {
   if(CheckPointer(prevNeuron) == POINTER_INVALID || CheckPointer(OutputLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   if(prevNeuron.Type() != defNeuron)
     {
      CNeuronProof *temp = prevNeuron;
      return calcInputGradients(temp.getOutputLayer());
     }
//---
   CNeuronBase *outputNeuron;
   CConnection *con;
   float prev_gradient = 0;
   int start = (int)index - iWindow + iStep;
   start = (start - start % iStep) / iStep;
   float stop = (float)((index - index % iStep) / iStep + 1);
   for(int out = (int)fmax(0, start); out < (int)fmin(OutputLayer.Total(), stop); out++)
     {
      outputNeuron = OutputLayer.At(out);
      int c = (int)(((int)fmin(OutputLayer.Total(), stop) - out - 1) * iStep + index % iStep);
      con = Connections.At(c);
      if(CheckPointer(outputNeuron) == POINTER_INVALID || CheckPointer(con) == POINTER_INVALID)
         continue;
      prev_gradient += outputNeuron.getGradient() * activationFunctionDerivative(outputNeuron.getOutputVal()) * con.weight;
     }
   prevNeuron.setGradient(prev_gradient);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBase::Save(int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, Type()) < INT_VALUE)
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)activation, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)optimization, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, t, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return Connections.Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription::CLayerDescription(void)   :  type(defNeuron),
   count(0),
   window(1),
   step(1),
   layers(1),
   activation(TANH),
   optimization(ADAM),
   probability((float)0.1),
   batch(1e+4),
   variables(1)
  {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLayerDescription::Copy(CLayerDescription *source)
  {
   if(!source)
      ReturnFalse;
//---
   type = source.type;
   count = source.count;
   window = source.window;
   window_out = source.window_out;
   step = source.step;
   layers = source.layers;
   batch = source.batch;
   activation = source.activation;
   optimization = source.optimization;
   probability = source.probability;
   ArrayCopy(windows, source.windows);
   ArrayCopy(units, source.units);
   ArrayCopy(heads, source.heads);
   radius = source.radius;
   variables = source.variables;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNet::recentAverageSmoothingFactor = 10000.0; // Number of training samples to average over
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Create(CArrayObj *Description)
  {
   recentAverageError = 0;
   backPropCount = 0;
   if(!Description)
      return false;
//---
   int total = Description.Total();
   if(total <= 0)
      ReturnFalse;
//---
   if(!layers)
      layers = new CArrayLayer();
   if(!layers)
      ReturnFalse;
   layers.FreeMode(true);
   layers.Clear();
//---
   CLayer *temp;
   CLayerDescription *desc = NULL, *next = NULL, *prev = NULL;
   CNeuronBase *neuron = NULL;
   CNeuronProof *neuron_p = NULL;
   uint output_count = 0;
   uint temp_count = 0;
//---
   next = Description.At(1);
   if(next.type == defNeuron || next.type >= defNeuronBaseOCL)
     {
      if(!opencl && !OpenCLInit())
        {
         DeleteObj(opencl);
         bOwnOpenCL = false;
        }
     }
   else
     {
      DeleteObj(opencl);
     }
//---
   for(int i = 0; i < total; i++)
     {
      prev = desc;
      desc = Description.At(i);
      if((i + 1) < total)
        {
         next = Description.At(i + 1);
         if(!next)
            ReturnFalse;
        }
      else
         next = NULL;
      uint outputs = (next == NULL || (next.type != defNeuron && next.type != defNeuronBaseOCL &&
                                       next.type != defNeuronBaseSAMOCL) ? 0 : next.count);
      if(!!next && (next.type == defNeuronFQF || next.type == defNeuronSoftActorCritic))
         outputs = next.count * next.window_out;
      temp = new CLayer(outputs);
      uint neurons = (desc.count + (desc.type == defNeuron || desc.type == defNeuronBaseOCL  || desc.type == defNeuronBaseSAMOCL ? 1 : 0));
      if(!!opencl)
        {
         CNeuronBaseOCL *neuron_ocl = NULL;
         CNeuronConvOCL *neuron_conv_ocl = NULL;
         CNeuronProofOCL *neuron_proof_ocl = NULL;
         CNeuronAttentionOCL *neuron_attention_ocl = NULL;
         CNeuronMLMHAttentionOCL *neuron_mlattention_ocl = NULL;
         CNeuronMLMHSparseAttention *neuron_sparseattention = NULL;
         CNeuronMH2AttentionOCL *neuron_mh2attention_ocl = NULL;
         CNeuronDropoutOCL *dropout = NULL;
         CNeuronBatchNormOCL *batch = NULL;
         CVAE *vae = NULL;
         CNeuronLSTMOCL *lstm = NULL;
         CNeuronSoftMaxOCL *softmax = NULL;
         CNeuronFQF *fqf = NULL;
         CNeuronMultiModel *multi_model = NULL;
         CNeuronConcatenate *concat = NULL;
         CNeuronEmbeddingOCL *emb = NULL;
         CNeuronPositionEncoder *pe = NULL;
         CNeuronTransposeOCL *tr = NULL;
         CNeuronCGConvOCL *cgc = NULL;
         CNeuronXCiTOCL *xcit = NULL;
         CNeuronDOTOCL *dot = NULL;
         CNeuronFAQOCL *faq = NULL;
         CNeuronCrossAttention *cross = NULL;
         CNeuronGTE *gte = NULL;
         CNeuronCCMROCL *ccmr = NULL;
         CNeuronNODEOCL *node = NULL;
         CNeuronConformer *conf = NULL;
         CNeuronRevINDenormOCL *revin = NULL;
         CNeuronClientOCL  *client = NULL;
         CNeuronLearnabledPE  *lpe = NULL;
         CNeuronUShapeAttention *usa = NULL;
         CNeuronPatching *patch = NULL;
         CNeuronTiDEOCL *tide = NULL;
         CNeuronFEDW *fedw = NULL;
         CNeuronFITSOCL *fits = NULL;
         CNeuronFreDFOCL *fredf = NULL;
         CNeuronComplexMLMHAttention *comp_att = NULL;
         CNeuronATFNetOCL *atf = NULL;
         CNeuronS3 *s3 = NULL;
         CNeuronMLMHAttentionMLKV *mlkv = NULL;
         CNeuronMLCrossAttentionMLKV *crmlkv = NULL;
         CNeuronCSCMOCL *cscm = NULL;
         CNeuronSPyrAttentionOCL *spyr = NULL;
         CNeuronPLROCL  *plr = NULL;
         CNeuronTPMEncoder *tpme = NULL;
         CNeuronTPM *tpm = NULL;
         CNeuronSTNNEncoder *stnne = NULL;
         CNeuronSTNNDecoder *stnnd = NULL;
         CNeuronSparseTSF *sptsf = NULL;
         CNeuronTEMPOOCL *tempo = NULL;
         CNeuronInjectTST *inject = NULL;
         CNeuronMambaOCL *mamba = NULL;
         CNeuronTrajLLMOCL *traj = NULL;
         CNeuronUniTraj *utraj = NULL;
         CNeuronHiVTOCL *hivt = NULL;
         CNeuronPointNetOCL *pn = NULL;
         CNeuronSceneSpecific *ss = NULL;
         CNeuronSEFormer *sef = NULL;
         CNeuronSPFormer *spf = NULL;
         CNeuronMAFT *maft = NULL;
         CNeuronGRES *gres = NULL;
         CNeuronRefMask *refm = NULL;
         CNeuronRelativeSelfAttention *rat = NULL;
         CNeuronRMAT *rmat = NULL;
         CNeuronMHAttentionPooling *atpool = NULL;
         CNeuronMotifs *motif = NULL;
         CNeuronMultiScaleAttention *msat = NULL;
         CNeuronMolformer *molf = NULL;
         CNeuronAMCT *amct = NULL;
         CNeuronNAFS *nafs = NULL;
         CNeuronHyperProjection *lorentz = NULL;
         CNeuronHyperboloids *logmap = NULL;
         CNeuronHypDiff *hypdiff = NULL;
         CNeuronBaseSAMOCL *sam_base = NULL;
         CNeuronConvSAMOCL *sam_conv = NULL;
         CNeuronPSBlock *ps_block = NULL;
         CNeuronPSformer *psformer = NULL;
         CNeuronMASA *masa = NULL;
         CNeuronMarketObserver *mo = NULL;
         CNeuronRLAgent *rl_agent = NULL;
         CNeuronControlAgent *contr_ag = NULL;
         CNeuronMASAAT *masaat = NULL;
         CNeuronMHConvOCL *mhconv = NULL;
         CNeuronMHFeedForward *mhff = NULL;
         CNeuronDMHAttention  *dmhat = NULL;
         CNeuronCrossDMHAttention  *crdmhat = NULL;
         CNeuronMultitaskStockformer *mtstockf = NULL;
         CNeuronFinMem *fin_mem = NULL;
         CNeuronFinAgent *fin_agent = NULL;
         CNeuronFinConManager *fin_con = NULL;
         CNeuronMacroHFT *mhft = NULL;
         CNeuronMacroHFTHyperAgent *mhft_hyp = NULL;
         CNeuronMacroHFTvsRiskManager *mhft_risk = NULL;
         CNeuronHidformer *hidf = NULL;
         CNeuronResNeXtBlock *resnext = NULL;
         CNeuronChimera *chimera = NULL;
         CNeuronMoT *mot = NULL;
         CNeuronHypridDecoder *hypridDecod = NULL;
         CNeuronAttraos *attraos = NULL;
         CNeuronDUET *duet = NULL;
         CNeuronMultiWindowsConvOCL *mwconv = NULL;
         CNeuronAdaBN *adabn = NULL;
         CNeuronTransposeRCDOCL *trcd = NULL;
         CNeuronTransposeVRCOCL *tvrc = NULL;
         CNeuronCATCH *catch
               = NULL;
         CNeuronSkillsEncoder *skills = NULL;
         CNeuronHiSSDLowLevelControler *hissd = NULL;
         CNeuronLinerAttention *lineatt = NULL;
         CNeuronCGLSTMOCL *cglstm = NULL;
         CNeuronMHProbAttention *probatt = NULL;
         CNeuronTSPositionEncoder *tspe = NULL;
         CMamba4CastEmbeding* m4cemb = NULL;
         CNeuronMultiWindowsConvWPadOCL *mwconvpad = NULL;
         CNeuronConcatDiff *condiff = NULL;
         CNeuronMantisPatching *mantpatch = NULL;
         CNeuronMantisAttentionUnit *mantatt = NULL;
         CNeuronTimeFoundPatching *tf_patch = NULL;
         CNeuronTimeFoundTransformerUnit *tf_tu = NULL;
         CNeuronSwiGLUOCL *swiglu = NULL;
         CNeuronTimeMoESparseExperts *timemoesp = NULL;
         CNeuronTimeMoEAttention *timemoeatt = NULL;
         CNeuronAdaptConv *ac = NULL;
         CNeuronRoPE *rope = NULL;
         CNeuronK2VAEEncoder *k2vae = NULL;
         CNeuronInterpolationAttention *ia = NULL;
         CNeuronGinARCell *ginar_cell = NULL;
         CNeuronGinAR *ginar = NULL;
         CNeuronPeriodNorm *per_norm = NULL;
         CNeuronAdaptSpatialNorm *adapt_norm = NULL;
         CNeuronSCNNEncoder* scnn_enc = NULL;
         CNeuronSCNN* scnn = NULL;
         CNeuronAttentNorm* atn = NULL;
         CNeuronSAttentNorm* satn = NULL;
         CNeuronPolynomialRegression* pol_regr = NULL;
         CNeuronSSCNNEncoder* sscnn_enc = NULL;
         CNeuronSSCNN* sscnn = NULL;
         CNeuronTQMHA* tqmha = NULL;
         CNeuronHimNetEncoder* hn_enc = NULL;
         CNeuronHimNetDecoder* hn_dec = NULL;
         CNeuronSAGDFN* sagdfn = NULL;
         CNeuronSpatialEmbedding* semb = NULL;
         CNeuronTempEmbedding* temb = NULL;
         CNeuronGlobalLocalAttention* glatt = NULL;
         CNeuronExtralonger* exlon = NULL;
         CNeuronExtralongerGraph* exlongr = NULL;
         CNeuronSpikeConv* spike = NULL;
         CNeuronGateLineAttention* glineatt = NULL;
         CNeuronSpikeSparseAttention* spikeatt = NULL;
         CNeuronMoEConv *moecon = NULL;
         CNeuronSpikingBrain* spikebrain = NULL;
         CNeuronSparseSoftMax* sparsoftmax = NULL;
         CNeuronSpikeActivation* spikeact = NULL;
         CNeuronSpikeResNeXtBlock* spikeresblock = NULL;
         CNeuronSpikeResNeXtBottleneck* spikeresbottleneck = NULL;
         CNeuronSpikeResNeXtResidual* spikeresresid = NULL;
         CNeuronSSAM* ssam = NULL;
         CNeuronSSAMResNeXtBlock* ssamblock = NULL;
         CNeuronSTFS* stfs = NULL;
         CNeuronSpikeConvGRU* sp_conv_gru = NULL;
         CNeuronAddToStack* stack = NULL;
         CNeuronStackCorrelation* st_correl = NULL;
         CNeuronSTEFlowNetEncoder* ste_encod = NULL;
         CNeuronSTEFlowNetResidualBlock* ste_resid = NULL;
         CNeuronSTEFlowNetDecoder* ste_decod = NULL;
         CNeuronSTEFlowNet* ste = NULL;
         CNeuronMultiScaleStackCorrelation* mul_correl = NULL;
         CNeuronSpikeConvGRU2D* sp_conv_gru2D = NULL;
         CNeuronRAFT* raft = NULL;
         CNeuronMultScalStackCorrelBySegments* segm_corr = NULL;
         CNeuronTMAMFE* mfe = NULL;
         CNeuronTMAMPA* mpa = NULL;
         CNeuronTMA* tma = NULL;
         CNeuronFERENet*   fere = NULL;
         CNeuronSpikeSTConvGRU* st_convgru = NULL;
         CNeuronFGDModule* fgd = NULL;
         CNeuronEVMGRFlowNet* evmgr = NULL;
         CNeuronSpikeConvBlock* spike_block = NULL;
         CNeuronMSRes* msres = NULL;
         CNeuronSpikeQKAttention* qkatt = NULL;
         CNeuronSDSA* sdsa = NULL;
         CNeuronSDformerUBlock* sdublock = NULL;
         CNeuronSDformerFlow* sdflow = NULL;
         CNeuronSpikePatchStak* sp_patch_stak = NULL;
         CNeuronSpike2DSSMOCL* sp_ssm = NULL;
         CNeuronSpikeMamba2D* sp_mamba = NULL;
         CNeuronSpikeSTSSM* sp_stssm = NULL;
         CNeuronESTMEncoder* sp_estm = NULL;
         CNeuronCreateFlow* cr_flow = NULL;
         CNeuronBiDirectCorrelation* bi_corr = NULL;
         CNeuronSATMA* satma = NULL;
         CNeuronBAT* bat = NULL;
         CNeuronSpikeSCM* scm = NULL;
         CNeuronSpikeFAM* fam = NULL;
         CNeuronSpikeMDC* mdc = NULL;
         CNeuronMDS* mds = NULL;
         CNeuronSpikeADM* adm = NULL;
         CNeuronCDCSelfCorrector* selfcorrect = NULL;
         CNeuronSpikeCDC* cdc = NULL;
         CNeuronSpikeEEMFLow* eemflow = NULL;
         CNeuronSpikeDepthWiseConv* dw_conv = NULL;
         CNeuronMultiScaleExcitation* excit = NULL;
         CNeuronMultiScaleDifference* ms_diff = NULL;
         CNeuronCorrelationEncoder* corr_enc = NULL;
         CNeuronEDCFlow* edcflow = NULL;
         CNeuronCreateICEFlow* ice = NULL;
         CNeuronSpikeMFE* sp_mfe = NULL;
         CNeuronSpikeMHCrossAttention* sp_crosatt = NULL;
         CNeuronSpikeMixFusion* mix_fusion = NULL;
         CNeuronSTFlow* stflow = NULL;
         CNeuronPSSE* pse = NULL;
         CNeuronSpikeDepthWiseResidual* dwr = NULL;
         CNeuronSpikeSuperKernelBlock* sk_block = NULL;
         CNeuronSpikeGMA* gma = NULL;
         CNeuronETROF* etrof = NULL;
         CNeuronPSSEFlow* pseflow = NULL;
         CNeuronSpikeUSR* usr = NULL;
         CNeuronInitialFlow* init_flow = NULL;
         CNeuronSpikeSMR* smr = NULL;
         CNeuronEVAFlow* eva = NULL;
         CNeuronResFlowHTR* htr = NULL;
         CNeuronResFlow* resflow = NULL;
         CNeuronConfGateTailAwareMoE* tailaware = NULL;
         CNeuronResFlowLattice* lattice = NULL;
         CNeuronFieldAwareConv* faconv = NULL;
         CNeuronMHFAT* fat = NULL;
         CNeuronMHCrossFAT* cross_fat = NULL;
         CNeuronFieldPatternEmbedding* faemb = NULL;
         CNeuronAutoToken* autotoken = NULL;
         CNeuronMultiMixAttention* mixattention = NULL;
         CNeuronSparseMoEConv* sparsemoe = NULL;
         CNeuronMTmixAttBlock* mixattenblock = NULL;
         CCrossMHFlashAttention* flash_att = NULL;
         CNeuronMixFeedForward* mixff = NULL;
         CNeuronOneTrans* onetrans = NULL;
         CNeuronMHTHCrossAttention* thcatt = NULL;
         CNeuronSTCA* stca = NULL;
         CNeuronWeightGenerator* wgen = NULL;
         CNeuronPCGR* pcrg = NULL;
         CNeuronADS* ads = NULL;
         CNeuronPerTokenFFN* pertoken = NULL;
         CNeuronScenariosToken* scentoken = NULL;
         CNeuronUnifiedTokenizer* uniftokens = NULL;
         CNeuronDomainAwareAttention* domainatt = NULL;
         CNeuronFeatureSelfIteration* featureinter = NULL;
         CNeuronDomainFused* domainfused = NULL;
         CNeuronMDL* mdl = NULL;
         CNeuronUniMixerTokenizer* umtoken = NULL;
         CNeuronPerTokenSwiGLU* ptswiglu = NULL;
         CNeuronUniMixer* unimix = NULL;
         CNeuronSiameseNorm* siamnorm = NULL;
         CNeuronUniMixerBlock* umblock = NULL;
         CNeuronINFNetScenarios* infnscen = NULL;
         CNeuronINFNetHUBSequence* hub = NULL;
         CNeuronINFNetBlock* infnblock = NULL;
         CNeuronSequenceTokenizer* seq_tok = NULL;
         CNeuronDifferenceTokenizer* diff_tok = NULL;
         CNeuronReGENTADBackbone* regen_backbone = NULL;
         CNeuronReGENTADHead* regen_head = NULL;
         CNeuronReGENTAD* regentad = NULL;
         CNeuronGDformerBackbone* gdformer_backbone = NULL;
         CNeuronGDformer* gdformer = NULL;
         CNeuronMomADTTM* momad_ttm = NULL;
         CNeuronMomADMPI* momad_mpi = NULL;
         CNeuronVLADriverRAGMPI* vladriver_rag_mpi = NULL;
         CNeuronUncAD_MUE* uncad_mue = NULL;
         CNeuronUncAD_UGP* uncad_ugp = NULL;
         CNeuronUncAD_UGPL* uncad_ugpl = NULL;
         CNeuronUncAD* uncad = NULL;
         CNeuronMomAD* momad = NULL;
         CNeuronCogDriverData* cogdriver_data = NULL;
         CNeuronCogDriverRankTCM* cogdriver_rank = NULL;
         CNeuronCogDriverForecastHead* cogdriver_forecast = NULL;
         CNeuronCogDriverForecastToken* cogdriver_token = NULL;
         CNeuronScenarioForecast* scenario_forecast = NULL;
         CNeuronScenarioCrossAttention* scenario_cross = NULL;
         CD2Skill* d2_skill = NULL;
         CNeuronRAGMemory* rag_memory = NULL;
         //---
         switch(desc.type)
           {
            case defNeuron:
            case defNeuronBase:
            case defNeuronBaseOCL:
               neuron_ocl = new CNeuronBaseOCL();
               if(CheckPointer(neuron_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_ocl.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(neuron_ocl);
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_ocl))
                 {
                  DeleteObj(neuron_ocl);
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_ocl = NULL;
               break;
            case defNeuronBaseSAMOCL:
               sam_base = new CNeuronBaseSAMOCL();
               if(CheckPointer(sam_base) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sam_base.Init(outputs, 0, opencl, desc.count, desc.probability, desc.optimization, desc.batch))
                 {
                  delete sam_base;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               sam_base.SetActivationFunction(desc.activation);
               if(!temp.Add(sam_base))
                 {
                  delete sam_base;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               sam_base = NULL;
               break;
            //---
            case defNeuronConvOCL:
               neuron_conv_ocl = new CNeuronConvOCL();
               if(CheckPointer(neuron_conv_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_conv_ocl.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete neuron_conv_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_conv_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_conv_ocl))
                 {
                  delete neuron_conv_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_conv_ocl = NULL;
               break;
            //---
            case defNeuronConvSAMOCL:
               sam_conv = new CNeuronConvSAMOCL();
               if(CheckPointer(sam_conv) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sam_conv.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, 1, desc.probability, desc.optimization, desc.batch))
                 {
                  delete sam_conv;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               sam_conv.SetActivationFunction(desc.activation);
               if(!temp.Add(sam_conv))
                 {
                  delete sam_conv;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               sam_conv = NULL;
               break;
            //---
            case defNeuronProofOCL:
               neuron_proof_ocl = new CNeuronProofOCL();
               if(!neuron_proof_ocl)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_proof_ocl.Init(outputs, 0, opencl, desc.window, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_proof_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_proof_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_proof_ocl))
                 {
                  delete neuron_proof_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_proof_ocl = NULL;
               break;
            //---
            case defNeuronAttentionOCL:
               neuron_attention_ocl = new CNeuronAttentionOCL();
               if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_attention_ocl.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_attention_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_attention_ocl))
                 {
                  delete neuron_attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_attention_ocl = NULL;
               break;
            //---
            case defNeuronMHAttentionOCL:
               neuron_attention_ocl = new CNeuronMHAttentionOCL();
               if(CheckPointer(neuron_attention_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_attention_ocl.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_attention_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_attention_ocl))
                 {
                  delete neuron_attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_attention_ocl = NULL;
               break;
            //---
            case defNeuronMLMHAttentionOCL:
               neuron_mlattention_ocl = new CNeuronMLMHAttentionOCL();
               if(CheckPointer(neuron_mlattention_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_mlattention_ocl.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete neuron_mlattention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mlattention_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_mlattention_ocl))
                 {
                  delete neuron_mlattention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mlattention_ocl = NULL;
               break;
            //---
            case defNeuronMLMHSparseAttentionOCL:
               neuron_sparseattention = new CNeuronMLMHSparseAttention();
               if(CheckPointer(neuron_sparseattention) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_sparseattention.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete neuron_sparseattention;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_sparseattention.SetActivationFunction(desc.activation);
               neuron_sparseattention.Sparse(desc.probability);
               if(!temp.Add(neuron_sparseattention))
                 {
                  delete neuron_mlattention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_sparseattention = NULL;
               break;
            //---
            case defNeuronMFTOCL:
               neuron_mlattention_ocl = new CNeuronMFTOCL();
               if(CheckPointer(neuron_mlattention_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_mlattention_ocl.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete neuron_mlattention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mlattention_ocl.SetActivationFunction(desc.activation);
               if(!temp.Add(neuron_mlattention_ocl))
                 {
                  delete neuron_mlattention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mlattention_ocl = NULL;
               break;
            case defNeuronMH2AttentionOCL:
               neuron_mh2attention_ocl = new CNeuronMH2AttentionOCL();
               if(CheckPointer(neuron_mh2attention_ocl) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!neuron_mh2attention_ocl.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  delete neuron_mh2attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mh2attention_ocl.SetActivationFunction(None);
               if(!temp.Add(neuron_mh2attention_ocl))
                 {
                  delete neuron_mh2attention_ocl;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neuron_mh2attention_ocl = NULL;
               break;
            //---
            case defNeuronXCiTOCL:
               xcit = new CNeuronXCiTOCL();
               if(CheckPointer(xcit) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!xcit.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete xcit;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temp.Add(xcit))
                 {
                  delete xcit;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               xcit = NULL;
               break;
            //---
            case defNeuronDropoutOCL:
               dropout = new CNeuronDropoutOCL();
               if(CheckPointer(dropout) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!dropout.Init(outputs, 0, opencl, desc.count, desc.probability, desc.optimization, desc.batch))
                 {
                  delete dropout;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temp.Add(dropout))
                 {
                  delete dropout;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               dropout = NULL;
               break;
            //---
            case defNeuronBatchNormOCL:
            case defNeuronBatchNormWithNoise:
               batch = (desc.type == defNeuronBatchNormWithNoise ? new CNeuronBatchNormWithNoise() : new CNeuronBatchNormOCL());
               if(CheckPointer(batch) == POINTER_INVALID)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!batch.Init(outputs, 0, opencl, desc.count, desc.batch, desc.optimization))
                 {
                  delete batch;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               batch.SetActivationFunction(desc.activation);
               if(!temp.Add(batch))
                 {
                  delete batch;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               batch = NULL;
               break;
            //---
            case defNeuronVAEOCL:
               vae = new CVAE();
               if(!vae)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!vae.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  delete vae;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temp.Add(vae))
                 {
                  delete vae;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               vae = NULL;
               break;
            case defNeuronLSTMOCL:
               lstm = new CNeuronLSTMOCL();
               if(!lstm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               desc.layers = MathMax(desc.layers, 1);
               if(!lstm.Init(outputs, 0, opencl, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  delete lstm;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!!prev)
                  if(!lstm.SetInputs(prev.count / desc.layers))
                    {
                     delete lstm;
                     DeleteObj(temp);
                     ReturnFalse;
                    }
               if(!temp.Add(lstm))
                 {
                  delete lstm;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               lstm = NULL;
               break;
            //---
            case defNeuronSoftMaxOCL:
               softmax = new CNeuronSoftMaxOCL();
               if(!softmax)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!softmax.Init(outputs, 0, opencl, desc.count * desc.step, desc.optimization, desc.batch))
                 {
                  delete softmax;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               softmax.SetHeads(desc.step);
               if(!temp.Add(softmax))
                 {
                  delete softmax;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               softmax = NULL;
               break;
            //---
            case defNeuronFQF:
               fqf = new CNeuronFQF();
               if(!fqf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fqf.Init(outputs, 0, opencl, desc.count, desc.window_out, prev.count * (prev.type == defNeuronConv ? prev.window_out : 1), desc.optimization, desc.batch))
                 {
                  delete fqf;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temp.Add(fqf.AsObject()))
                 {
                  delete fqf;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               fqf = NULL;
               break;
            case defNeuronMultiModels:
               multi_model = new CNeuronMultiModel();
               if(!multi_model)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!multi_model.Init(desc.window, 0, opencl, desc.count, ADAM, desc.step))
                 {
                  delete multi_model;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               multi_model.SetActivationFunction(desc.activation);
               if(!temp.Add(multi_model))
                 {
                  delete multi_model;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               multi_model = NULL;
               break;
            case defNeuronConcatenate:
               concat = new CNeuronConcatenate();
               if(!concat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!concat.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, ADAM, desc.batch))
                 {
                  delete concat;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               concat.SetActivationFunction(desc.activation);
               if(!temp.Add(concat))
                 {
                  delete concat;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               concat = NULL;
               break;
            //---
            case defNeuronSoftActorCritic:
               fqf = new CNeuronSoftActorCritic();
               if(!fqf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fqf.Init(outputs, 0, opencl, desc.count, desc.window_out, prev.count * (prev.type == defNeuronConv ? prev.window_out : 1), desc.optimization, desc.batch))
                 {
                  delete fqf;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               fqf.SetActivationFunction(desc.activation);
               if(!temp.Add(fqf.AsObject()))
                 {
                  delete fqf;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               fqf = NULL;
               break;
            //---
            case defNeuronEmbeddingOCL:
               emb = new CNeuronEmbeddingOCL();
               if(!emb)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!emb.Init(outputs, 0, opencl, desc.count, desc.window_out, desc.windows))
                 {
                  DeleteObj(temp);
                  delete emb;
                  ReturnFalse;
                 }
               if(!temp.Add(emb))
                 {
                  DeleteObj(temp);
                  delete emb;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPEOCL:
               pe = new CNeuronPositionEncoder();
               if(!pe)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!pe.Init(outputs, 0, opencl, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pe;
                  ReturnFalse;
                 }
               if(!temp.Add(pe))
                 {
                  DeleteObj(temp);
                  delete pe;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTransposeOCL:
               tr = new CNeuronTransposeOCL();
               if(!tr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tr.Init(outputs, 0, opencl, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tr;
                  ReturnFalse;
                 }
               if(!temp.Add(tr))
                 {
                  DeleteObj(temp);
                  delete tr;
                  ReturnFalse;
                 }
               break;
            case defNeuronCGConvOCL:
               cgc = new CNeuronCGConvOCL();
               if(!cgc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!cgc.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete cgc;
                  ReturnFalse;
                 }
               if(!temp.Add(cgc))
                 {
                  DeleteObj(temp);
                  delete cgc;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronDOTOCL:
               dot = new CNeuronDOTOCL();
               if(!dot)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               neurons = prev.count * MathMax((prev.type == defNeuronConv ? prev.window_out : prev.window), 1) / desc.count;
               if(!dot.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, neurons, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete dot;
                  ReturnFalse;
                 }
               if(!temp.Add(dot))
                 {
                  DeleteObj(temp);
                  delete dot;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFAQOCL:
               faq = new CNeuronFAQOCL();
               if(!faq)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!faq.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.units[0], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete faq;
                  ReturnFalse;
                 }
               if(!temp.Add(faq))
                 {
                  DeleteObj(temp);
                  delete faq;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronCrossAttenOCL:
               cross = new CNeuronCrossAttention();
               if(!cross)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!cross.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.step, desc.units[0], desc.windows[1], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete cross;
                  ReturnFalse;
                 }
               if(!temp.Add(cross))
                 {
                  DeleteObj(temp);
                  delete cross;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronGTE:
               gte = new CNeuronGTE();
               if(!gte)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!gte.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete gte;
                  ReturnFalse;
                 }
               if(!temp.Add(gte))
                 {
                  DeleteObj(temp);
                  delete gte;
                  ReturnFalse;
                 }
               break;
            case defNeuronCCMROCL:
               ccmr = new CNeuronCCMROCL();
               if(!ccmr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ccmr.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ccmr;
                  ReturnFalse;
                 }
               if(!temp.Add(ccmr))
                 {
                  DeleteObj(temp);
                  delete ccmr;
                  ReturnFalse;
                 }
               break;
            case defNeuronNODEOCL:
               node = new CNeuronNODEOCL();
               if(!node)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!node.Init(outputs, 0, opencl, desc.window, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete node;
                  ReturnFalse;
                 }
               if(!temp.Add(node))
                 {
                  DeleteObj(temp);
                  delete node;
                  ReturnFalse;
                 }
               break;
            case defNeuronConformerOCL:
               conf = new CNeuronConformer();
               if(!conf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!conf.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.layers, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete conf;
                  ReturnFalse;
                 }
               if(!temp.Add(conf))
                 {
                  DeleteObj(temp);
                  delete conf;
                  ReturnFalse;
                 }
               break;
            case defNeuronRevInDenormOCL:
               if(int(desc.layers) >= layers.Total())
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!(((CLayer *)layers.At(desc.layers)).At(0).Type() == defNeuronBatchNormOCL ||
                    ((CLayer *)layers.At(desc.layers)).At(0).Type() == defNeuronBatchNormWithNoise))
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               revin = new CNeuronRevINDenormOCL();
               if(!revin)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!revin.Init(outputs, 0, opencl, desc.count, desc.layers, ((CLayer *)layers.At(desc.layers)).At(0)))
                 {
                  DeleteObj(temp);
                  DeleteObj(revin);
                  ReturnFalse;
                 }
               if(!temp.Add(revin))
                 {
                  DeleteObj(temp);
                  DeleteObj(revin);
                  ReturnFalse;
                 }
               break;
            case defNeuronClientOCL:
               client = new CNeuronClientOCL();
               if(!client)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!client.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.layers, desc.count, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete client;
                  ReturnFalse;
                 }
               if(!temp.Add(client))
                 {
                  DeleteObj(temp);
                  delete client;
                  ReturnFalse;
                 }
               break;
            case defNeuronLearnabledPE:
               lpe = new CNeuronLearnabledPE();
               if(!lpe)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!lpe.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete lpe;
                  ReturnFalse;
                 }
               if(!temp.Add(lpe))
                 {
                  DeleteObj(temp);
                  delete lpe;
                  ReturnFalse;
                 }
               break;
            case defNeuronUShapeAttention:
               usa = new CNeuronUShapeAttention();
               if(!usa)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!usa.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.batch, desc.optimization, 1))
                 {
                  DeleteObj(temp);
                  delete usa;
                  ReturnFalse;
                 }
               if(!temp.Add(usa))
                 {
                  DeleteObj(temp);
                  delete usa;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPatchingOCL:
               patch = new CNeuronPatching();
               if(!patch)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!patch.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete patch;
                  ReturnFalse;
                 }
               patch.SetActivationFunction(desc.activation);
               if(!temp.Add(patch))
                 {
                  DeleteObj(temp);
                  delete patch;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTiDEOCL:
               tide = new CNeuronTiDEOCL();
               if(!tide)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tide.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tide;
                  ReturnFalse;
                 }
               if(!temp.Add(tide))
                 {
                  DeleteObj(temp);
                  delete tide;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFEDW:
               fedw = new CNeuronFEDW();
               if(!fedw)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fedw.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fedw;
                  ReturnFalse;
                 }
               if(!temp.Add(fedw))
                 {
                  DeleteObj(temp);
                  delete fedw;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFITSOCL:
               fits = new CNeuronFITSOCL();
               if(!fits)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fits.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fits;
                  ReturnFalse;
                 }
               if(!temp.Add(fits))
                 {
                  DeleteObj(temp);
                  delete fits;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFreDFOCL:
               fredf = new CNeuronFreDFOCL();
               if(!fredf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fredf.Init(outputs, 0, opencl, desc.window, desc.count, desc.probability, bool(desc.step), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fredf;
                  ReturnFalse;
                 }
               if(!temp.Add(fredf))
                 {
                  DeleteObj(temp);
                  delete fredf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronComplexMLMHAttentionOCL:
               comp_att = new CNeuronComplexMLMHAttention();
               if(!comp_att)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!comp_att.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete comp_att;
                  ReturnFalse;
                 }
               if(!temp.Add(comp_att))
                 {
                  DeleteObj(temp);
                  delete comp_att;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronATFNetOCL:
               atf = new CNeuronATFNetOCL();
               if(!atf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!atf.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.layers, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete atf;
                  ReturnFalse;
                 }
               if(!temp.Add(atf))
                 {
                  DeleteObj(temp);
                  delete atf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronS3:
               s3 = new CNeuronS3();
               if(!s3)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!s3.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete s3;
                  ReturnFalse;
                 }
               if(!temp.Add(s3))
                 {
                  DeleteObj(temp);
                  delete s3;
                  ReturnFalse;
                 }
               break;
            case defNeuronMLMHAttentionMLKV:
               mlkv = new CNeuronMLMHAttentionMLKV();
               if(!mlkv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mlkv.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mlkv;
                  ReturnFalse;
                 }
               if(!temp.Add(mlkv))
                 {
                  DeleteObj(temp);
                  delete mlkv;
                  ReturnFalse;
                 }
               break;
            case defNeuronMLMHSceneConditionAttention:
               mlkv = new CNeuronMLMHSceneConditionAttention();
               if(!mlkv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mlkv.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mlkv;
                  ReturnFalse;
                 }
               if(!temp.Add(mlkv))
                 {
                  DeleteObj(temp);
                  delete mlkv;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMLCrossAttentionMLKV:
               crmlkv = new CNeuronMLCrossAttentionMLKV();
               if(!crmlkv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!crmlkv.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.heads[0], desc.windows[1], desc.heads[1], desc.units[0], desc.units[1], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete crmlkv;
                  ReturnFalse;
                 }
               if(!temp.Add(crmlkv))
                 {
                  DeleteObj(temp);
                  delete crmlkv;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronCSCMOCL:
               cscm = new CNeuronCSCMOCL();
               if(!cscm)
                  ReturnFalse;
               if(!cscm.Init(outputs, 0, opencl, desc.windows, desc.window, desc.count, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  delete cscm;
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temp.Add(cscm))
                 {
                  DeleteObj(temp);
                  delete cscm;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSPyrAttentionMLKV:
               spyr = new CNeuronSPyrAttentionOCL();
               if(!spyr)
                  ReturnFalse;
               if(!spyr.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete spyr;
                  ReturnFalse;
                 }
               if(!temp.Add(spyr))
                 {
                  DeleteObj(temp);
                  delete spyr;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPLROCL:
               plr = new CNeuronPLROCL();
               if(!plr)
                  ReturnFalse;
               if(!plr.Init(outputs, 0, opencl, desc.window, desc.count, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete plr;
                  ReturnFalse;
                 }
               if(!temp.Add(plr))
                 {
                  DeleteObj(temp);
                  delete plr;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTPM:
               tpm = new CNeuronTPM();
               if(!tpm)
                  ReturnFalse;
               if(!tpm.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tpm;
                  ReturnFalse;
                 }
               if(!temp.Add(tpm))
                 {
                  DeleteObj(temp);
                  delete tpm;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTPMEncoder:
               tpme = new CNeuronTPMEncoder();
               if(!tpme)
                  ReturnFalse;
               if(!tpme.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tpme;
                  ReturnFalse;
                 }
               if(!temp.Add(tpme))
                 {
                  DeleteObj(temp);
                  delete tpme;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSTNNEncoder:
               stnne = new CNeuronSTNNEncoder();
               if(!stnne)
                  ReturnFalse;
               if(!stnne.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete stnne;
                  ReturnFalse;
                 }
               if(!temp.Add(stnne))
                 {
                  DeleteObj(temp);
                  delete stnne;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSTNNDecoder:
               stnnd = new CNeuronSTNNDecoder();
               if(!stnnd)
                  ReturnFalse;
               if(!stnnd.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.heads[0], desc.windows[1], desc.heads[1], desc.units[0], desc.units[1], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete stnnd;
                  ReturnFalse;
                 }
               if(!temp.Add(stnnd))
                 {
                  DeleteObj(temp);
                  delete stnnd;
                  ReturnFalse;
                 }
               break;
            case defNeuronSparseTSF:
               sptsf = new CNeuronSparseTSF();
               if(!sptsf)
                  ReturnFalse;
               if(!sptsf.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete sptsf;
                  ReturnFalse;
                 }
               if(!temp.Add(sptsf))
                 {
                  DeleteObj(temp);
                  delete sptsf;
                  ReturnFalse;
                 }
               break;
            case defNeuronTEMPOOCL:
               tempo = new CNeuronTEMPOOCL();
               if(!tempo)
                  ReturnFalse;
               if(!tempo.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tempo;
                  ReturnFalse;
                 }
               if(!temp.Add(tempo))
                 {
                  DeleteObj(temp);
                  delete tempo;
                  ReturnFalse;
                 }
               break;
            case defNeuronInjectTST:
               inject = new CNeuronInjectTST();
               if(!inject)
                  ReturnFalse;
               if(!inject.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.step, desc.heads[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete inject;
                  ReturnFalse;
                 }
               if(!temp.Add(inject))
                 {
                  DeleteObj(temp);
                  delete inject;
                  ReturnFalse;
                 }
               break;
            case defNeuronMambaOCL:
               mamba = new CNeuronMambaOCL();
               if(!mamba)
                  ReturnFalse;
               if(!mamba.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mamba;
                  ReturnFalse;
                 }
               if(!temp.Add(mamba))
                 {
                  DeleteObj(temp);
                  delete mamba;
                  ReturnFalse;
                 }
               break;
            case defNeuronTrajLLMOCL:
               traj = new CNeuronTrajLLMOCL();
               if(!traj)
                  ReturnFalse;
               if(!traj.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete traj;
                  ReturnFalse;
                 }
               if(!temp.Add(traj))
                 {
                  DeleteObj(temp);
                  delete traj;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronUniTrajOCL:
               utraj = new CNeuronUniTraj();
               if(!utraj)
                  ReturnFalse;
               if(!utraj.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.layers, desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete utraj;
                  ReturnFalse;
                 }
               if(!temp.Add(utraj))
                 {
                  DeleteObj(temp);
                  delete utraj;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronHiVTOCL:
               hivt = new CNeuronHiVTOCL();
               if(!hivt)
                  ReturnFalse;
               if(!hivt.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.step, desc.count, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete hivt;
                  ReturnFalse;
                 }
               if(!temp.Add(hivt))
                 {
                  DeleteObj(temp);
                  delete hivt;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPointNetOCL:
               pn = new CNeuronPointNetOCL();
               if(!pn)
                  ReturnFalse;
               if(!pn.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               if(!temp.Add(pn))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPointNet2OCL:
               pn = new CNeuronPointNet2OCL();
               if(!pn)
                  ReturnFalse;
               if(!pn.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               if(!temp.Add(pn))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPointFormer:
               pn = new CNeuronPointFormer();
               if(!pn)
                  ReturnFalse;
               if(!pn.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               if(!temp.Add(pn))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               break;
            case defNeuronHyperDet:
               pn = new CNeuronHyperDet();
               if(!pn)
                  ReturnFalse;
               if(!pn.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, (bool)desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               if(!temp.Add(pn))
                 {
                  DeleteObj(temp);
                  delete pn;
                  ReturnFalse;
                 }
               break;
            case defNeuronSceneSpecific:
               ss = new CNeuronSceneSpecific();
               if(!ss)
                  ReturnFalse;
               if(!ss.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.units[0], desc.units[1], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ss;
                  ReturnFalse;
                 }
               if(!temp.Add(ss))
                 {
                  DeleteObj(temp);
                  delete ss;
                  ReturnFalse;
                 }
               break;
            case defNeuronSEFormer:
               sef = new CNeuronSEFormer();
               if(!sef)
                  ReturnFalse;
               if(!sef.Init(outputs, 0, opencl, desc.windows[0], desc.units[0], desc.window_out, (bool)desc.step, desc.units[1], desc.windows[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete sef;
                  ReturnFalse;
                 }
               if(!temp.Add(sef))
                 {
                  DeleteObj(temp);
                  delete sef;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSPFormer:
               spf = new CNeuronSPFormer();
               if(!spf)
                  ReturnFalse;
               if(!spf.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.heads[0], desc.windows[1], desc.units[1], desc.heads[1], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete spf;
                  ReturnFalse;
                 }
               if(!temp.Add(spf))
                 {
                  DeleteObj(temp);
                  delete spf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMAFT:
               maft = new CNeuronMAFT();
               if(!maft)
                  ReturnFalse;
               if(!maft.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.heads[0], desc.windows[1], desc.units[1], desc.heads[1], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete maft;
                  ReturnFalse;
                 }
               if(!temp.Add(maft))
                 {
                  DeleteObj(temp);
                  delete maft;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronGRES:
               gres = new CNeuronGRES();
               if(!gres)
                  ReturnFalse;
               if(!gres.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.heads[0], desc.windows[1], desc.units[1], desc.heads[1], desc.windows[2], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete gres;
                  ReturnFalse;
                 }
               if(!temp.Add(gres))
                 {
                  DeleteObj(temp);
                  delete gres;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronRefMask:
               refm = new CNeuronRefMask();
               if(!refm)
                  ReturnFalse;
               if(!refm.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.step, desc.windows[1], desc.units[1], desc.units[2], desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete refm;
                  ReturnFalse;
                 }
               if(!temp.Add(refm))
                 {
                  DeleteObj(temp);
                  delete refm;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronRelativeSelfAttention:
               rat = new CNeuronRelativeSelfAttention();
               if(!rat)
                  ReturnFalse;
               if(!rat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete rat;
                  ReturnFalse;
                 }
               if(!temp.Add(rat))
                 {
                  DeleteObj(temp);
                  delete rat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronRMAT:
               rmat = new CNeuronRMAT();
               if(!rmat)
                  ReturnFalse;
               if(!rmat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete rmat;
                  ReturnFalse;
                 }
               if(!temp.Add(rmat))
                 {
                  DeleteObj(temp);
                  delete rmat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMHAttentionPooling:
               atpool = new CNeuronMHAttentionPooling();
               if(!atpool)
                  ReturnFalse;
               if(!atpool.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete atpool;
                  ReturnFalse;
                 }
               if(!temp.Add(atpool))
                 {
                  DeleteObj(temp);
                  delete atpool;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMotifs:
               motif = new CNeuronMotifs();
               if(!motif)
                  ReturnFalse;
               if(!motif.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete motif;
                  ReturnFalse;
                 }
               if(!temp.Add(motif))
                 {
                  DeleteObj(temp);
                  delete motif;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMultiScaleAttention:
               msat = new CNeuronMultiScaleAttention();
               if(!msat)
                  ReturnFalse;
               if(!msat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete msat;
                  ReturnFalse;
                 }
               if(!temp.Add(msat))
                 {
                  DeleteObj(temp);
                  delete msat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMolformer:
               molf = new CNeuronMolformer();
               if(!molf)
                  ReturnFalse;
               if(!molf.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.heads[0], desc.layers, desc.windows[1], desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete molf;
                  ReturnFalse;
                 }
               if(!temp.Add(molf))
                 {
                  DeleteObj(temp);
                  delete molf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronAMCT:
               amct = new CNeuronAMCT();
               if(!amct)
                  ReturnFalse;
               if(!amct.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.units[1], desc.units[0], desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete amct;
                  ReturnFalse;
                 }
               if(!temp.Add(amct))
                 {
                  DeleteObj(temp);
                  delete amct;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronNAFS:
               nafs = new CNeuronNAFS();
               if(!nafs)
                  ReturnFalse;
               if(!nafs.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete nafs;
                  ReturnFalse;
                 }
               if(!temp.Add(nafs))
                 {
                  DeleteObj(temp);
                  delete nafs;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronDiffusion:
               usa = new CNeuronDiffusion();
               if(!usa)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!usa.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.count, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete usa;
                  ReturnFalse;
                 }
               if(!temp.Add(usa))
                 {
                  DeleteObj(temp);
                  delete usa;
                  ReturnFalse;
                 }
            //---
            case defNeuronHyperProjection:
               lorentz = new CNeuronHyperProjection();
               if(!lorentz)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!lorentz.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete lorentz;
                  ReturnFalse;
                 }
               if(!temp.Add(lorentz))
                 {
                  DeleteObj(temp);
                  delete lorentz;
                  ReturnFalse;
                 }
            //---
            case defNeuronHyperboloids:
               logmap = new CNeuronHyperboloids();
               if(!logmap)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!logmap.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete logmap;
                  ReturnFalse;
                 }
               if(!temp.Add(logmap))
                 {
                  DeleteObj(temp);
                  delete logmap;
                  ReturnFalse;
                 }
            //---
            case defNeuronHypDiff:
               hypdiff = new CNeuronHypDiff();
               if(!hypdiff)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hypdiff.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.heads[0], desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete hypdiff;
                  ReturnFalse;
                 }
               if(!temp.Add(hypdiff))
                 {
                  DeleteObj(temp);
                  delete hypdiff;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPSBlock:
               ps_block = new CNeuronPSBlock();
               if(!ps_block)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ps_block.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ps_block;
                  ReturnFalse;
                 }
               if(!temp.Add(ps_block))
                 {
                  DeleteObj(temp);
                  delete ps_block;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronPSformer:
               psformer = new CNeuronPSformer();
               if(!psformer)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!psformer.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete psformer;
                  ReturnFalse;
                 }
               if(!temp.Add(psformer))
                 {
                  DeleteObj(temp);
                  delete psformer;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMASA:
               masa = new CNeuronMASA();
               if(!masa)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!masa.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.heads[0], desc.units[0],
                             desc.windows[1], desc.window, desc.probability, desc.units[1],
                             desc.windows[2], desc.heads[1], desc.units[2], desc.step,
                             (desc.step >= 0 ? ((CLayer*)layers.At(desc.step)).At(0) : NULL),
                             desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete masa;
                  ReturnFalse;
                 }
               if(!temp.Add(masa))
                 {
                  DeleteObj(temp);
                  delete masa;
                  ReturnFalse;
                 }
               masa.SetActivationFunction(desc.activation);
               break;
            //---
            case defNeuronMarketObserver:
               mo = new CNeuronMarketObserver();
               if(!mo)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mo.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.layers, desc.windows[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mo;
                  ReturnFalse;
                 }
               if(!temp.Add(mo))
                 {
                  DeleteObj(temp);
                  delete mo;
                  ReturnFalse;
                 }
               mo.SetActivationFunction(desc.activation);
               break;
            //---
            case defNeuronRLAgent:
               rl_agent = new CNeuronRLAgent();
               if(!rl_agent)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!rl_agent.Init(outputs, 0, opencl, desc.windows[0], desc.count, desc.window_out, desc.probability, desc.layers, desc.windows[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete rl_agent;
                  ReturnFalse;
                 }
               if(!temp.Add(rl_agent))
                 {
                  DeleteObj(temp);
                  delete rl_agent;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronControlAgent:
               contr_ag = new CNeuronControlAgent();
               if(!contr_ag)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!contr_ag.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.step, desc.windows[1], desc.units[1], desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete contr_ag;
                  ReturnFalse;
                 }
               if(!temp.Add(contr_ag))
                 {
                  DeleteObj(temp);
                  delete contr_ag;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMASAAT:
               masaat = new CNeuronMASAAT();
               if(!masaat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!masaat.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.step, desc.count, desc.layers, desc.radius, desc.windows[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete masaat;
                  ReturnFalse;
                 }
               if(!temp.Add(masaat))
                 {
                  DeleteObj(temp);
                  delete masaat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMHConvOCL:
               mhconv = new CNeuronMHConvOCL();
               if(!mhconv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mhconv.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.heads[0], desc.heads[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mhconv;
                  ReturnFalse;
                 }
               if(!temp.Add(mhconv))
                 {
                  DeleteObj(temp);
                  delete mhconv;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMHFeedForward:
               mhff = new CNeuronMHFeedForward();
               if(!mhff)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mhff.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.heads[0], desc.heads[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mhff;
                  ReturnFalse;
                 }
               if(!temp.Add(mhff))
                 {
                  DeleteObj(temp);
                  delete mhff;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronDMHAttention:
               dmhat = new CNeuronDMHAttention();
               if(!dmhat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!dmhat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete dmhat;
                  ReturnFalse;
                 }
               if(!temp.Add(dmhat))
                 {
                  DeleteObj(temp);
                  delete dmhat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronCrossDMHAttention:
               crdmhat = new CNeuronCrossDMHAttention();
               if(!crdmhat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!crdmhat.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0], desc.windows[1], desc.units[1], desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete crdmhat;
                  ReturnFalse;
                 }
               if(!temp.Add(crdmhat))
                 {
                  DeleteObj(temp);
                  delete crdmhat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMultitaskStockformer:
               mtstockf = new CNeuronMultitaskStockformer();
               if(!mtstockf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mtstockf.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.layers, desc.windows[2], desc.windows[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mtstockf;
                  ReturnFalse;
                 }
               if(!temp.Add(mtstockf))
                 {
                  DeleteObj(temp);
                  delete mtstockf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFinMem:
               fin_mem = new CNeuronFinMem();
               if(!fin_mem)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fin_mem.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fin_mem;
                  ReturnFalse;
                 }
               if(!temp.Add(fin_mem))
                 {
                  DeleteObj(temp);
                  delete fin_mem;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFinAgent:
               fin_agent = new CNeuronFinAgent();
               if(!fin_agent)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fin_agent.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fin_agent;
                  ReturnFalse;
                 }
               if(!temp.Add(fin_agent))
                 {
                  DeleteObj(temp);
                  delete fin_agent;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronFinConManager:
               fin_con = new CNeuronFinConManager();
               if(!fin_con)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fin_con.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete fin_con;
                  ReturnFalse;
                 }
               if(!temp.Add(fin_con))
                 {
                  DeleteObj(temp);
                  delete fin_con;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMacroHFT:
               mhft = new CNeuronMacroHFT();
               if(!mhft)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mhft.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.layers, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mhft;
                  ReturnFalse;
                 }
               if(!temp.Add(mhft))
                 {
                  DeleteObj(temp);
                  delete mhft;
                  ReturnFalse;
                 }
               break;
            case defNeuronMacroHFTHyperAgent:
               mhft_hyp = new CNeuronMacroHFTHyperAgent();
               if(!mhft_hyp)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mhft_hyp.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.layers, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mhft_hyp;
                  ReturnFalse;
                 }
               if(!temp.Add(mhft_hyp))
                 {
                  DeleteObj(temp);
                  delete mhft_hyp;
                  ReturnFalse;
                 }
               break;
            case defNeuronMacroHFTvsRiskManager:
               mhft_risk = new CNeuronMacroHFTvsRiskManager();
               if(!mhft_risk)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mhft_risk.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mhft_risk;
                  ReturnFalse;
                 }
               if(!temp.Add(mhft_risk))
                 {
                  DeleteObj(temp);
                  delete mhft_risk;
                  ReturnFalse;
                 }
               break;
            case defNeuronMultiScaleRelativeSelfAttention:
               rat = new CNeuronMultiScaleRelativeSelfAttention();
               if(!rat)
                  ReturnFalse;
               if(!rat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete rat;
                  ReturnFalse;
                 }
               if(!temp.Add(rat))
                 {
                  DeleteObj(temp);
                  delete rat;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronHidformer:
               hidf = new CNeuronHidformer();
               if(!hidf)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hidf.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step, desc.layers, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete hidf;
                  ReturnFalse;
                 }
               if(!temp.Add(hidf))
                 {
                  DeleteObj(temp);
                  delete hidf;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronResNeXtBlock:
               resnext = new CNeuronResNeXtBlock();
               if(!resnext)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!resnext.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.window, desc.step, desc.units[0], desc.units[1], desc.units[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete resnext;
                  ReturnFalse;
                 }
               if(!temp.Add(resnext))
                 {
                  DeleteObj(temp);
                  delete resnext;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronChimera:
               chimera = new CNeuronChimera();
               if(!chimera)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!chimera.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.units[0], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete chimera;
                  ReturnFalse;
                 }
               if(!temp.Add(chimera))
                 {
                  DeleteObj(temp);
                  delete chimera;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMoT:
               mot = new CNeuronMoT();
               if(!mot)
                  ReturnFalse;
               if(!mot.Init(outputs, 0, opencl, desc.window, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mot;
                  ReturnFalse;
                 }
               if(!temp.Add(mot))
                 {
                  DeleteObj(temp);
                  delete mot;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronHypridDecoder:
               hypridDecod = new CNeuronHypridDecoder();
               if(!hypridDecod)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hypridDecod.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.count, desc.step,
                                    desc.layers, desc.windows[1], desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete hypridDecod;
                  ReturnFalse;
                 }
               if(!temp.Add(hypridDecod))
                 {
                  DeleteObj(temp);
                  delete hypridDecod;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronAttraos:
               attraos = new CNeuronAttraos();
               if(!attraos)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!attraos.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete attraos;
                  ReturnFalse;
                 }
               if(!temp.Add(attraos))
                 {
                  DeleteObj(temp);
                  delete attraos;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronDUET:
               duet = new CNeuronDUET();
               if(!duet)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!duet.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.units[0], desc.step, desc.units[1], desc.units[2], desc.units[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete duet;
                  ReturnFalse;
                 }
               if(!temp.Add(duet))
                 {
                  DeleteObj(temp);
                  delete duet;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMultiWindowsConvOCL:
               mwconv = new CNeuronMultiWindowsConvOCL();
               if(!mwconv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mwconv.Init(outputs, 0, opencl, desc.windows, desc.window_out, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mwconv;
                  ReturnFalse;
                 }
               if(!temp.Add(mwconv))
                 {
                  DeleteObj(temp);
                  delete mwconv;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronAdaBN:
               adabn = new CNeuronAdaBN();
               if(!adabn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!adabn.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.windows, desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete adabn;
                  ReturnFalse;
                 }
               if(!temp.Add(adabn))
                 {
                  DeleteObj(temp);
                  delete adabn;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTransposeRCDOCL:
               trcd = new CNeuronTransposeRCDOCL();
               if(!trcd)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!trcd.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete trcd;
                  ReturnFalse;
                 }
               if(!temp.Add(trcd))
                 {
                  DeleteObj(temp);
                  delete trcd;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTransposeVRCOCL:
               tvrc = new CNeuronTransposeVRCOCL();
               if(!tvrc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tvrc.Init(outputs, 0, opencl, desc.step, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tvrc;
                  ReturnFalse;
                 }
               if(!temp.Add(tvrc))
                 {
                  DeleteObj(temp);
                  delete tvrc;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronCATCH:
               catch
                     = new CNeuronCATCH();
               if(!catch)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!catch.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.step, desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete catch;
                  ReturnFalse;
                 }
               if(!temp.Add(catch))
                 {
                  DeleteObj(temp);
                  delete catch;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSkillsEncoder:
               skills = new CNeuronSkillsEncoder();
               if(!skills)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!skills.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.window, desc.step, desc.window_out, desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete skills;
                  ReturnFalse;
                 }
               if(!temp.Add(skills))
                 {
                  DeleteObj(temp);
                  delete skills;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronHiSSDLowLevelControler:
               hissd = new CNeuronHiSSDLowLevelControler();
               if(!hissd)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hissd.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.windows[2], desc.windows[3], desc.window, desc.step, desc.window_out, desc.windows[4], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete hissd;
                  ReturnFalse;
                 }
               hissd.SetActivationFunction(desc.activation);
               if(!temp.Add(hissd))
                 {
                  DeleteObj(temp);
                  delete hissd;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronLinerAttention:
               lineatt = new CNeuronLinerAttention();
               if(!lineatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!lineatt.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete lineatt;
                  ReturnFalse;
                 }
               lineatt.SetActivationFunction(None);
               if(!temp.Add(lineatt))
                 {
                  DeleteObj(temp);
                  delete lineatt;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronCGLSTMOCL:
               cglstm = new CNeuronCGLSTMOCL();
               if(!cglstm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!cglstm.Init(outputs, 0, opencl, desc.count, desc.window, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete cglstm;
                  ReturnFalse;
                 }
               cglstm.SetActivationFunction(None);
               if(!temp.Add(cglstm))
                 {
                  DeleteObj(temp);
                  delete cglstm;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMHProbAttention:
               probatt = new CNeuronMHProbAttention();
               if(!probatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!probatt.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete probatt;
                  ReturnFalse;
                 }
               if(!temp.Add(probatt))
                 {
                  DeleteObj(temp);
                  delete probatt;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTSPositionEncoder:
               tspe = new CNeuronTSPositionEncoder();
               if(!tspe)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tspe.Init(outputs, 0, opencl, desc.window, desc.count, desc.windows, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tspe;
                  ReturnFalse;
                 }
               if(!temp.Add(tspe))
                 {
                  DeleteObj(temp);
                  delete tspe;
                  ReturnFalse;
                 }
               break;
            //---
            case defMamba4CastEmbeding:
               m4cemb = new CMamba4CastEmbeding();
               if(!m4cemb)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!m4cemb.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete m4cemb;
                  ReturnFalse;
                 }
               if(!temp.Add(m4cemb))
                 {
                  DeleteObj(temp);
                  delete m4cemb;
                  ReturnFalse;
                 }
               break;
            case defNeuronMultiWindowsConvWPadOCL:
               mwconvpad = new CNeuronMultiWindowsConvWPadOCL();
               if(!mwconvpad)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mwconvpad.Init(outputs, 0, opencl, desc.windows, desc.step, desc.window_out, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mwconvpad;
                  ReturnFalse;
                 }
               if(!temp.Add(mwconvpad))
                 {
                  DeleteObj(temp);
                  delete mwconvpad;
                  ReturnFalse;
                 }
               break;
            case defNeuronConcatDiff:
               condiff = new CNeuronConcatDiff();
               if(!condiff)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!condiff.Init(outputs, 0, opencl, desc.count, desc.step, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete condiff;
                  ReturnFalse;
                 }
               if(!temp.Add(condiff))
                 {
                  DeleteObj(temp);
                  delete condiff;
                  ReturnFalse;
                 }
               break;
            case defNeuronMantisPatching:
               mantpatch = new CNeuronMantisPatching();
               if(!mantpatch)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mantpatch.Init(outputs, 0, opencl, desc.count, desc.step, desc.layers, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mantpatch;
                  ReturnFalse;
                 }
               if(!temp.Add(mantpatch))
                 {
                  DeleteObj(temp);
                  delete mantpatch;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMantisAttentionUnit:
               mantatt = new CNeuronMantisAttentionUnit();
               if(!mantatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mantatt.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.window_out, desc.step, desc.count, desc.layers, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete mantatt;
                  ReturnFalse;
                 }
               if(!temp.Add(mantatt))
                 {
                  DeleteObj(temp);
                  delete mantatt;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTimeFoundPatching:
               tf_patch = new CNeuronTimeFoundPatching();
               if(!tf_patch)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tf_patch.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tf_patch;
                  ReturnFalse;
                 }
               if(!temp.Add(tf_patch))
                 {
                  DeleteObj(temp);
                  delete tf_patch;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTimeFoundTransformerUnit:
               tf_tu = new CNeuronTimeFoundTransformerUnit();
               if(!tf_tu)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tf_tu.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.heads[0], desc.heads[1], desc.count, desc.layers, desc.step, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete tf_tu;
                  ReturnFalse;
                 }
               if(!temp.Add(tf_tu))
                 {
                  DeleteObj(temp);
                  delete tf_tu;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSwiGLUOCL:
               swiglu = new CNeuronSwiGLUOCL();
               if(!swiglu)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!swiglu.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete swiglu;
                  ReturnFalse;
                 }
               if(!temp.Add(swiglu))
                 {
                  DeleteObj(temp);
                  delete swiglu;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTimeMoESparseExperts:
               timemoesp = new CNeuronTimeMoESparseExperts();
               if(!timemoesp)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!timemoesp.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete timemoesp;
                  ReturnFalse;
                 }
               if(!temp.Add(timemoesp))
                 {
                  DeleteObj(temp);
                  delete timemoesp;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronTimeMoEAttention:
               timemoeatt = new CNeuronTimeMoEAttention();
               if(!timemoeatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!timemoeatt.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.units[0],
                                   desc.windows[1], desc.units[1], desc.step, desc.layers, desc.units[2],
                                   desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete timemoeatt;
                  ReturnFalse;
                 }
               if(!temp.Add(timemoeatt))
                 {
                  DeleteObj(temp);
                  delete timemoeatt;
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronAdaptConv:
               ac = new CNeuronAdaptConv();
               if(!ac)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ac.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count,
                           desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ac;
                  ReturnFalse;
                 }
               if(!temp.Add(ac))
                 {
                  DeleteObj(temp);
                  delete ac;
                  ReturnFalse;
                 }
               break;
            case defNeuronRoPE:
               rope = new CNeuronRoPE();
               if(!rope)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!rope.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete rope;
                  ReturnFalse;
                 }
               if(!temp.Add(rope))
                 {
                  DeleteObj(temp);
                  delete rope;
                  ReturnFalse;
                 }
               break;
            case defNeuronK2VAEEncoder:
               k2vae = new CNeuronK2VAEEncoder();
               if(!k2vae)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!k2vae.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.units[0], desc.step, desc.layers, desc.units[1], desc.units[2], desc.windows[2], desc.units[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete k2vae;
                  ReturnFalse;
                 }
               if(!temp.Add(k2vae))
                 {
                  DeleteObj(temp);
                  delete k2vae;
                  ReturnFalse;
                 }
               break;
            case defNeuronInterpolationAttention:
               ia = new CNeuronInterpolationAttention();
               if(!ia)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ia.Init(outputs, 0, opencl, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ia;
                  ReturnFalse;
                 }
               if(!temp.Add(ia))
                 {
                  DeleteObj(temp);
                  delete ia;
                  ReturnFalse;
                 }
               break;
            case defNeuronGinARCell:
               ginar_cell = new CNeuronGinARCell();
               if(!ginar_cell)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ginar_cell.Init(outputs, 0, opencl, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ginar_cell;
                  ReturnFalse;
                 }
               if(!temp.Add(ginar_cell))
                 {
                  DeleteObj(temp);
                  delete ginar_cell;
                  ReturnFalse;
                 }
               break;
            case defNeuronGinAR:
               ginar = new CNeuronGinAR();
               if(!ginar)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ginar.Init(outputs, 0, opencl, desc.count, desc.window, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete ginar;
                  ReturnFalse;
                 }
               if(!temp.Add(ginar))
                 {
                  DeleteObj(temp);
                  delete ginar;
                  ReturnFalse;
                 }
               break;
            case defNeuronPeriodNorm:
               per_norm = new CNeuronPeriodNorm();
               if(!per_norm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!per_norm.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete per_norm;
                  ReturnFalse;
                 }
               if(!temp.Add(per_norm))
                 {
                  DeleteObj(temp);
                  delete per_norm;
                  ReturnFalse;
                 }
               break;
            case defNeuronAdaptSpatialNorm:
               adapt_norm = new CNeuronAdaptSpatialNorm();
               if(!adapt_norm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!adapt_norm.Init(outputs, 0, opencl, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete adapt_norm;
                  ReturnFalse;
                 }
               if(!temp.Add(adapt_norm))
                 {
                  DeleteObj(temp);
                  delete adapt_norm;
                  ReturnFalse;
                 }
               break;
            case defNeuronSCNNEncoder:
               scnn_enc = new CNeuronSCNNEncoder();
               if(!scnn_enc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!scnn_enc.Init(outputs, 0, opencl, desc.windows[0], desc.variables, desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete scnn_enc;
                  ReturnFalse;
                 }
               if(!temp.Add(scnn_enc))
                 {
                  DeleteObj(temp);
                  delete scnn_enc;
                  ReturnFalse;
                 }
               break;
            case defNeuronSCNN:
               scnn = new CNeuronSCNN();
               if(!scnn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!scnn.Init(outputs, 0, opencl, desc.windows[0], desc.variables, desc.windows[1], desc.windows[2], desc.windows[3], desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete scnn;
                  ReturnFalse;
                 }
               if(!temp.Add(scnn))
                 {
                  DeleteObj(temp);
                  delete scnn;
                  ReturnFalse;
                 }
               break;
            case defNeuronAttentNorm:
               atn = new CNeuronAttentNorm();
               if(!atn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!atn.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete atn;
                  ReturnFalse;
                 }
               if(!temp.Add(atn))
                 {
                  DeleteObj(temp);
                  delete atn;
                  ReturnFalse;
                 }
               break;
            case defNeuronSAttentNorm:
               satn = new CNeuronSAttentNorm();
               if(!satn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!satn.Init(outputs, 0, opencl, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete satn;
                  ReturnFalse;
                 }
               if(!temp.Add(satn))
                 {
                  DeleteObj(temp);
                  delete satn;
                  ReturnFalse;
                 }
               break;
            case defNeuronPolynomialRegression:
               pol_regr = new CNeuronPolynomialRegression();
               if(!pol_regr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!pol_regr.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete pol_regr;
                  ReturnFalse;
                 }
               if(!temp.Add(pol_regr))
                 {
                  DeleteObj(temp);
                  delete pol_regr;
                  ReturnFalse;
                 }
               break;
            case defNeuronSSCNNEncoder:
               sscnn_enc = new CNeuronSSCNNEncoder();
               if(!sscnn_enc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sscnn_enc.Init(outputs, 0, opencl, desc.windows[0], desc.variables, desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete sscnn_enc;
                  ReturnFalse;
                 }
               if(!temp.Add(sscnn_enc))
                 {
                  DeleteObj(temp);
                  delete sscnn_enc;
                  ReturnFalse;
                 }
               break;
            case defNeuronSSCNN:
               sscnn = new CNeuronSSCNN();
               if(!sscnn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sscnn.Init(outputs, 0, opencl, desc.windows[0], desc.variables, desc.windows[1], desc.windows[2], desc.windows[3], desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  delete sscnn;
                  ReturnFalse;
                 }
               if(!temp.Add(sscnn))
                 {
                  DeleteObj(temp);
                  delete sscnn;
                  ReturnFalse;
                 }
               break;
            case defNeuronTQMHA:
               tqmha = new CNeuronTQMHA();
               if(!tqmha)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tqmha.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.step,
                              desc.count, desc.windows[2], desc.windows[3],
                              desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(tqmha);
                  ReturnFalse;
                 }
               if(!temp.Add(tqmha))
                 {
                  DeleteObj(temp);
                  DeleteObj(tqmha);
                  ReturnFalse;
                 }
               break;
            case defNeuronHimNetEncoder:
               hn_enc = new CNeuronHimNetEncoder();
               if(!hn_enc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hn_enc.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.step,
                               desc.layers, desc.windows[2], desc.windows[3], desc.windows[4],
                               desc.windows[5], desc.windows[6], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(hn_enc);
                  ReturnFalse;
                 }
               if(!temp.Add(hn_enc))
                 {
                  DeleteObj(temp);
                  DeleteObj(hn_enc);
                  ReturnFalse;
                 }
               break;
            case defNeuronHimNetDecoder:
               hn_dec = new CNeuronHimNetDecoder();
               if(!hn_dec)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!hn_dec.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.step,
                               desc.layers, desc.windows[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(hn_dec);
                  ReturnFalse;
                 }
               if(!temp.Add(hn_dec))
                 {
                  DeleteObj(temp);
                  DeleteObj(hn_dec);
                  ReturnFalse;
                 }
               break;
            case defNeuronSAGDFN:
               sagdfn = new CNeuronSAGDFN();
               if(!sagdfn)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sagdfn.Init(outputs, 0, opencl, desc.count, desc.window, desc.windows[0],
                               desc.units[0], desc.windows[1], desc.step, desc.probability,
                               desc.units[1], desc.windows[2], desc.units[2], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sagdfn);
                  ReturnFalse;
                 }
               if(!temp.Add(sagdfn))
                 {
                  DeleteObj(temp);
                  DeleteObj(sagdfn);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpatialEmbedding:
               semb = new CNeuronSpatialEmbedding();
               if(!semb)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!semb.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out,
                             desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(semb);
                  ReturnFalse;
                 }
               if(!temp.Add(semb))
                 {
                  DeleteObj(temp);
                  DeleteObj(semb);
                  ReturnFalse;
                 }
               break;
            case defNeuronTempEmbedding:
               temb = new CNeuronTempEmbedding();
               if(!temb)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!temb.Init(outputs, 0, opencl, desc.count, desc.window, desc.windows[0],
                             desc.units[0], desc.windows[1], desc.windows[2], desc.units[1],
                             desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(temb);
                  ReturnFalse;
                 }
               if(!temp.Add(temb))
                 {
                  DeleteObj(temp);
                  DeleteObj(temb);
                  ReturnFalse;
                 }
               break;
            case defNeuronGlobalLocalAttention:
               glatt = new CNeuronGlobalLocalAttention();
               if(!glatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!glatt.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out,
                              desc.step, desc.variables, desc.probability,
                              desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(glatt);
                  ReturnFalse;
                 }
               if(!temp.Add(glatt))
                 {
                  DeleteObj(temp);
                  DeleteObj(glatt);
                  ReturnFalse;
                 }
               break;
            case defNeuronExtralonger:
               exlon = new CNeuronExtralonger();
               if(!exlon)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!exlon.Init(outputs, 0, opencl, desc.units[0], desc.units[1], desc.window,
                              desc.window_out, desc.windows[0], desc.units[2], desc.windows[1],
                              desc.units[3], desc.windows[2], desc.layers, desc.step,
                              desc.windows[3], desc.units[4], desc.probability,
                              desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(exlon);
                  ReturnFalse;
                 }
               if(!temp.Add(exlon))
                 {
                  DeleteObj(temp);
                  DeleteObj(exlon);
                  ReturnFalse;
                 }
               break;
            case defNeuronExtralongerGraph:
               exlongr = new CNeuronExtralongerGraph();
               if(!exlongr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!exlongr.Init(outputs, 0, opencl, desc.units[0], desc.units[1], desc.window,
                                desc.window_out, desc.windows[0], desc.units[2], desc.windows[1],
                                desc.units[3], desc.windows[2], desc.layers, desc.step,
                                desc.units[4], desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(exlongr);
                  ReturnFalse;
                 }
               if(!temp.Add(exlongr))
                 {
                  DeleteObj(temp);
                  DeleteObj(exlongr);
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSpikeConv:
               spike = new CNeuronSpikeConv();
               if(!spike)
                 {
                  DeleteObj(temb);
                  ReturnFalse;
                 }
               if(!spike.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out,
                              desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spike);
                  ReturnFalse;
                 }
               if(!temp.Add(spike))
                 {
                  DeleteObj(temp);
                  DeleteObj(spike);
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronGateLineAttention:
               glineatt = new CNeuronGateLineAttention();
               if(!glineatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!glineatt.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out,
                                 desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(glineatt);
                  ReturnFalse;
                 }
               if(!temp.Add(glineatt))
                 {
                  DeleteObj(temp);
                  DeleteObj(glineatt);
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSpikeSparseAttention:
               spikeatt = new CNeuronSpikeSparseAttention();
               if(!spikeatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikeatt.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables,
                                 desc.probability, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeatt);
                  ReturnFalse;
                 }
               if(!temp.Add(spikeatt))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeatt);
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronMoEConv:
               moecon = new CNeuronMoEConv();
               if(!moecon)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!moecon.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out,
                               desc.variables, desc.probability, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(moecon);
                  ReturnFalse;
                 }
               if(!temp.Add(moecon))
                 {
                  DeleteObj(temp);
                  DeleteObj(moecon);
                  ReturnFalse;
                 }
               break;
            //---
            case defNeuronSpikingBrain:
               spikebrain = new CNeuronSpikingBrain();
               if(!spikebrain)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikebrain.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables,
                                   desc.probability, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikebrain);
                  ReturnFalse;
                 }
               if(!temp.Add(spikebrain))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikebrain);
                  ReturnFalse;
                 }
               break;
            case defNeuronSparseSoftMax:
               sparsoftmax = new CNeuronSparseSoftMax();
               if(!sparsoftmax)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sparsoftmax.Init(outputs, 0, opencl, desc.count, desc.window,
                                    desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sparsoftmax);
                  ReturnFalse;
                 }
               if(!temp.Add(sparsoftmax))
                 {
                  DeleteObj(temp);
                  DeleteObj(sparsoftmax);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeActivation:
               spikeact = new CNeuronSpikeActivation();
               if(!spikeact)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikeact.Init(outputs, 0, opencl, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeact);
                  ReturnFalse;
                 }
               if(!temp.Add(spikeact))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeact);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeResNeXtBlock:
               spikeresblock = new CNeuronSpikeResNeXtBlock();
               if(!spikeresblock)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikeresblock.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1],
                                      desc.units[0], desc.units[1], desc.windows[2], desc.units[2],
                                      desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresblock);
                  ReturnFalse;
                 }
               if(!temp.Add(spikeresblock))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresblock);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeResNeXtBottleneck:
               spikeresbottleneck = new CNeuronSpikeResNeXtBottleneck();
               if(!spikeresbottleneck)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikeresbottleneck.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1],
                                           desc.units[0], desc.units[1], desc.windows[2], desc.units[2],
                                           desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresbottleneck);
                  ReturnFalse;
                 }
               if(!temp.Add(spikeresbottleneck))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresbottleneck);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeResNeXtResidual:
               spikeresresid = new CNeuronSpikeResNeXtResidual();
               if(!spikeresresid)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spikeresresid.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1],
                                      desc.units[0], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresresid);
                  ReturnFalse;
                 }
               if(!temp.Add(spikeresresid))
                 {
                  DeleteObj(temp);
                  DeleteObj(spikeresresid);
                  ReturnFalse;
                 }
               break;
            case defNeuronSSAM:
               ssam = new CNeuronSSAM();
               if(!ssam)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ssam.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out,
                             (bool)desc.probability, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ssam);
                  ReturnFalse;
                 }
               if(!temp.Add(ssam))
                 {
                  DeleteObj(temp);
                  DeleteObj(ssam);
                  ReturnFalse;
                 }
               break;
            case defNeuronSSAMResNeXtBlock:
               ssamblock = new CNeuronSSAMResNeXtBlock();
               if(!ssamblock)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ssamblock.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1],
                                  desc.units[0], desc.units[1], desc.windows[2], desc.units[2],
                                  desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ssamblock);
                  ReturnFalse;
                 }
               if(!temp.Add(ssamblock))
                 {
                  DeleteObj(temp);
                  DeleteObj(ssamblock);
                  ReturnFalse;
                 }
               break;
            case defNeuronSTFS:
               stfs = new CNeuronSTFS();
               if(!stfs)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!stfs.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.units[0],
                             desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(stfs);
                  ReturnFalse;
                 }
               stfs.SetActivationFunction(desc.activation);
               if(!temp.Add(stfs))
                 {
                  DeleteObj(temp);
                  DeleteObj(stfs);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeConvGRU:
               sp_conv_gru = new CNeuronSpikeConvGRU();
               if(!sp_conv_gru)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_conv_gru.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_conv_gru);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_conv_gru))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_conv_gru);
                  ReturnFalse;
                 }
               break;
            case defNeuronAddToStack:
               stack = new CNeuronAddToStack();
               if(!stack)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!stack.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(stack);
                  ReturnFalse;
                 }
               if(!temp.Add(stack))
                 {
                  DeleteObj(temp);
                  DeleteObj(stack);
                  ReturnFalse;
                 }
               break;
            case defNeuronStackCorrelation:
               st_correl = new CNeuronStackCorrelation();
               if(!st_correl)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!st_correl.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(st_correl);
                  ReturnFalse;
                 }
               st_correl.SetActivationFunction(desc.activation);
               if(!temp.Add(st_correl))
                 {
                  DeleteObj(temp);
                  DeleteObj(st_correl);
                  ReturnFalse;
                 }
               break;
            case defNeuronSTEFlowNetEncoder:
               ste_encod = new CNeuronSTEFlowNetEncoder();
               if(!ste_encod)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ste_encod.Init(outputs, 0, opencl, desc.windows, desc.units, desc.window, desc.variables,
                                  desc.step, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_encod);
                  ReturnFalse;
                 }
               if(!temp.Add(ste_encod))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_encod);
                  ReturnFalse;
                 }
               break;
            case defNeuronSTEFlowNet:
               ste = new CNeuronSTEFlowNet();
               if(!ste)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ste.Init(outputs, 0, opencl, desc.windows, desc.units, desc.window, desc.variables,
                            desc.step, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste);
                  ReturnFalse;
                 }
               if(!temp.Add(ste))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste);
                  ReturnFalse;
                 }
               break;
            case defNeuronSTEFlowNetDecoder:
               ste_decod = new CNeuronSTEFlowNetDecoder();
               if(!ste_decod)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ste_decod.Init(outputs, 0, opencl, desc.windows, desc.units, desc.window, desc.variables,
                                  desc.step, desc.window_out, ((CLayer*)layers.At(desc.layers))[0], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_decod);
                  ReturnFalse;
                 }
               if(!temp.Add(ste_decod))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_decod);
                  ReturnFalse;
                 }
               break;
            case defNeuronSTEFlowNetResidualBlock:
               ste_resid = new CNeuronSTEFlowNetResidualBlock();
               if(!ste_resid)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ste_resid.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_resid);
                  ReturnFalse;
                 }
               if(!temp.Add(ste_resid))
                 {
                  DeleteObj(temp);
                  DeleteObj(ste_resid);
                  ReturnFalse;
                 }
               break;
            case defNeuronMultiScaleStackCorrelation :
               mul_correl = new CNeuronMultiScaleStackCorrelation();
               if(!mul_correl)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mul_correl.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(mul_correl);
                  ReturnFalse;
                 }
               if(!temp.Add(mul_correl))
                 {
                  DeleteObj(temp);
                  DeleteObj(mul_correl);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeConvGRU2D:
               sp_conv_gru2D = new CNeuronSpikeConvGRU2D();
               if(!sp_conv_gru2D)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_conv_gru2D.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_conv_gru2D);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_conv_gru2D))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_conv_gru2D);
                  ReturnFalse;
                 }
               break;
            case defNeuronRAFT:
               raft = new CNeuronRAFT();
               if(!raft)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!raft.Init(outputs, 0, opencl, desc.units, desc.windows, desc.window, desc.step, desc.count, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(raft);
                  ReturnFalse;
                 }
               if(!temp.Add(raft))
                 {
                  DeleteObj(temp);
                  DeleteObj(raft);
                  ReturnFalse;
                 }
               break;
            case defNeuronMultScalStackCorrelBySegments:
               segm_corr = new CNeuronMultScalStackCorrelBySegments();
               if(!segm_corr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!segm_corr.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.variables, desc.layers, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(segm_corr);
                  ReturnFalse;
                 }
               if(!temp.Add(segm_corr))
                 {
                  DeleteObj(temp);
                  DeleteObj(segm_corr);
                  ReturnFalse;
                 }
               break;
            case defNeuronTMAMFE:
               mfe = new CNeuronTMAMFE();
               if(!mfe)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!mfe.Init(outputs, 0, opencl, desc.count, desc.windows, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(mfe);
                  ReturnFalse;
                 }
               if(!temp.Add(mfe))
                 {
                  DeleteObj(temp)
                  DeleteObj(mfe)
                  ReturnFalse;
                 }
               break;
            case defNeuronTMAMPA:
               mpa = new CNeuronTMAMPA();
               if(!mpa)
                 {
                  DeleteObj(temp)
                  ReturnFalse;
                 }
               if(!mpa.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp)
                  DeleteObj(mpa)
                  ReturnFalse;
                 }
               if(!temp.Add(mpa))
                 {
                  DeleteObj(temp)
                  DeleteObj(mpa)
                  ReturnFalse;
                 }
               break;
            case defNeuronTMA:
               tma = new CNeuronTMA();
               if(!tma)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!tma.Init(outputs, 0, opencl, desc.units, desc.windows, desc.window, desc.count, desc.heads[0], desc.heads[1], desc.layers, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(tma);
                  ReturnFalse;
                 }
               if(!temp.Add(tma))
                 {
                  DeleteObj(temp);
                  DeleteObj(tma);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeSTConvGRU:
               st_convgru = new CNeuronSpikeSTConvGRU();
               if(!st_convgru)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!st_convgru.Init(outputs, 0, opencl, desc.count, desc.windows, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(st_convgru);
                  ReturnFalse;
                 }
               if(!temp.Add(st_convgru))
                 {
                  DeleteObj(temp);
                  DeleteObj(st_convgru);
                  ReturnFalse;
                 }
               break;
            case defNeuronFERENet:
               fere = new CNeuronFERENet();
               if(!fere)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fere.Init(outputs, 0, opencl, desc.count, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(fere);
                  ReturnFalse;
                 }
               if(!temp.Add(fere))
                 {
                  DeleteObj(temp);
                  DeleteObj(fere);
                  ReturnFalse;
                 }
               break;
            case defNeuronFGDModule:
               fgd = new CNeuronFGDModule();
               if(!fgd)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!fgd.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(fgd);
                  ReturnFalse;
                 }
               if(!temp.Add(fgd))
                 {
                  DeleteObj(temp);
                  DeleteObj(fgd);
                  ReturnFalse;
                 }
               break;
            case defNeuronEVMGRFlowNet:
               evmgr = new CNeuronEVMGRFlowNet();
               if(!evmgr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!evmgr.Init(outputs, 0, opencl, desc.count, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(evmgr);
                  ReturnFalse;
                 }
               if(!temp.Add(evmgr))
                 {
                  DeleteObj(temp);
                  DeleteObj(evmgr);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeConvBlock:
               spike_block = new CNeuronSpikeConvBlock();
               if(!spike_block)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!spike_block.Init(outputs, 0, opencl, desc.window, desc.step, desc.window_out, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(spike_block);
                  ReturnFalse;
                 }
               if(!temp.Add(spike_block))
                 {
                  DeleteObj(temp);
                  DeleteObj(spike_block);
                  ReturnFalse;
                 }
               break;
            case defNeuronMSRes:
               msres = new CNeuronMSRes();
               if(!msres)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!msres.Init(outputs, 0, opencl, desc.count, desc.window, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(msres);
                  ReturnFalse;
                 }
               if(!temp.Add(msres))
                 {
                  DeleteObj(temp);
                  DeleteObj(msres);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeQKAttention:
               qkatt = new CNeuronSpikeQKAttention();
               if(!qkatt)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!qkatt.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(qkatt);
                  ReturnFalse;
                 }
               if(!temp.Add(qkatt))
                 {
                  DeleteObj(temp);
                  DeleteObj(qkatt);
                  ReturnFalse;
                 }
               break;
            case defNeuronSDSA:
               sdsa = new CNeuronSDSA();
               if(!sdsa)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sdsa.Init(outputs, 0, opencl, desc.count, desc.window, desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdsa);
                  ReturnFalse;
                 }
               if(!temp.Add(sdsa))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdsa);
                  ReturnFalse;
                 }
               break;
            case defNeuronSDformerUBlock:
               sdublock = new CNeuronSDformerUBlock();
               if(!sdublock)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sdublock.Init(outputs, 0, opencl, desc.count, desc.windows, desc.heads, desc.units, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdublock);
                  ReturnFalse;
                 }
               if(!temp.Add(sdublock))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdublock);
                  ReturnFalse;
                 }
               break;
            case defNeuronSDformerFlow:
               sdflow = new CNeuronSDformerFlow();
               if(!sdflow)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sdflow.Init(outputs, 0, opencl, desc.count, desc.windows, desc.heads, desc.units, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdflow);
                  ReturnFalse;
                 }
               if(!temp.Add(sdflow))
                 {
                  DeleteObj(temp);
                  DeleteObj(sdflow);
                  ReturnFalse;
                 }
            case defNeuronSpikePatchStak:
               sp_patch_stak = new CNeuronSpikePatchStak();
               if(!sp_patch_stak)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_patch_stak.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_patch_stak);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_patch_stak))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_patch_stak);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpike2DSSMOCL:
               sp_ssm = new CNeuronSpike2DSSMOCL();
               if(!sp_ssm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_ssm.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.units[0], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_ssm);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_ssm))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_ssm);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeMamba2D:
               sp_mamba = new CNeuronSpikeMamba2D();
               if(!sp_mamba)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_mamba.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_mamba);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_mamba))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_mamba);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeSTSSM:
               sp_stssm = new CNeuronSpikeSTSSM();
               if(!sp_stssm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_stssm.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.windows[2], desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_stssm);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_stssm))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_stssm);
                  ReturnFalse;
                 }
               break;
            case defNeuronESTMEncoder:
               sp_estm = new CNeuronESTMEncoder();
               if(!sp_estm)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!sp_estm.Init(outputs, 0, opencl, desc.heads, desc.windows, desc.window, desc.window_out, desc.units, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_estm);
                  ReturnFalse;
                 }
               if(!temp.Add(sp_estm))
                 {
                  DeleteObj(temp);
                  DeleteObj(sp_estm);
                  ReturnFalse;
                 }
               break;
            case defNeuronCreateFlow:
               cr_flow = new CNeuronCreateFlow();
               if(!cr_flow)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!cr_flow.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(cr_flow);
                  ReturnFalse;
                 }
               if(!temp.Add(cr_flow))
                 {
                  DeleteObj(temp);
                  DeleteObj(cr_flow);
                  ReturnFalse;
                 }
               break;
            case defNeuronBiDirectCorrelation:
               bi_corr = new CNeuronBiDirectCorrelation();
               if(!bi_corr)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!bi_corr.Init(outputs, 0, opencl, desc.window, desc.count, desc.variables, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(bi_corr);
                  ReturnFalse;
                 }
               if(!temp.Add(bi_corr))
                 {
                  DeleteObj(temp);
                  DeleteObj(bi_corr);
                  ReturnFalse;
                 }
               break;
            case defNeuronSATMA:
               satma = new CNeuronSATMA();
               if(!satma)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!satma.Init(outputs, 0, opencl, desc.window, desc.count, desc.variables, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(satma);
                  ReturnFalse;
                 }
               if(!temp.Add(satma))
                 {
                  DeleteObj(temp);
                  DeleteObj(satma);
                  ReturnFalse;
                 }
               break;
            case defNeuronBAT:
               bat = new CNeuronBAT();
               if(!bat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!bat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(bat);
                  ReturnFalse;
                 }
               if(!temp.Add(bat))
                 {
                  DeleteObj(temp);
                  DeleteObj(bat);
                  ReturnFalse;
                 }
               break;
            case defNeuronBATvADM:
               bat = new CNeuronBATvADM();
               if(!bat)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!bat.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(bat);
                  ReturnFalse;
                 }
               if(!temp.Add(bat))
                 {
                  DeleteObj(temp);
                  DeleteObj(bat);
                  ReturnFalse;
                 }
               break;
            case defNeuronCDCSelfCorrector:
               selfcorrect = new CNeuronCDCSelfCorrector();
               if(!selfcorrect)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!selfcorrect.Init(outputs, 0, opencl, desc.window, desc.windows, desc.window_out,
                                    desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(selfcorrect);
                  ReturnFalse;
                 }
               if(!temp.Add(selfcorrect))
                 {
                  DeleteObj(temp);
                  DeleteObj(selfcorrect);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeCDC:
               cdc = new CNeuronSpikeCDC();
               if(!cdc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!cdc.Init(outputs, 0, opencl, desc.window, desc.windows, desc.window_out,
                            desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(cdc);
                  ReturnFalse;
                 }
               if(!temp.Add(cdc))
                 {
                  DeleteObj(temp);
                  DeleteObj(cdc);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeEEMFLow:
               eemflow = new CNeuronSpikeEEMFLow();
               if(!eemflow)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!eemflow.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count,
                                desc.variables, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(eemflow);
                  ReturnFalse;
                 }
               if(!temp.Add(eemflow))
                 {
                  DeleteObj(temp);
                  DeleteObj(eemflow);
                  ReturnFalse;
                 }
               break;
            case defNeuronSpikeDepthWiseConv:
               dw_conv = new CNeuronSpikeDepthWiseConv();
               if(!dw_conv)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!dw_conv.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.layers, desc.step,
                                desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(dw_conv);
                  ReturnFalse;
                 }
               if(!temp.Add(dw_conv))
                 {
                  DeleteObj(temp);
                  DeleteObj(dw_conv);
                  ReturnFalse;
                 }
               break;
            case defNeuronMultiScaleExcitation:
               excit = new CNeuronMultiScaleExcitation();
               if(!excit)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!excit.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(excit);
                  ReturnFalse;
                 }
               if(!temp.Add(excit))
                 {
                  DeleteObj(temp);
                  DeleteObj(excit);
                  ReturnFalse;
                 }
               break;
            case defNeuronMultiScaleDifference:
               ms_diff = new CNeuronMultiScaleDifference();
               if(!ms_diff)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!ms_diff.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.windows, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(ms_diff);
                  ReturnFalse;
                 }
               if(!temp.Add(ms_diff))
                 {
                  DeleteObj(temp);
                  DeleteObj(ms_diff);
                  ReturnFalse;
                 }
               break;
            case defNeuronCorrelationEncoder:
               corr_enc = new CNeuronCorrelationEncoder();
               if(!corr_enc)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!corr_enc.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(corr_enc);
                  ReturnFalse;
                 }
               if(!temp.Add(corr_enc))
                 {
                  DeleteObj(temp);
                  DeleteObj(corr_enc);
                  ReturnFalse;
                 }
               break;
            case defNeuronEDCFlow:
               edcflow = new CNeuronEDCFlow();
               if(!edcflow)
                 {
                  DeleteObj(temp);
                  ReturnFalse;
                 }
               if(!edcflow.Init(outputs, 0, opencl, desc.windows[0], desc.units[0], desc.window_out,
                                desc.windows[1], desc.units[1], desc.heads, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObj(edcflow);
                  ReturnFalse;
                 }
               if(!temp.Add(edcflow))
                 {
                  DeleteObj(temp);
                  DeleteObj(edcflow);
                  ReturnFalse;
                 }
               break;
            case defNeuronCreateICEFlow:
               ice = new CNeuronCreateICEFlow();
               if(!ice)
                  DeleteObjAndFalse(temp);
               if(!ice.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ice);
                 }
               if(!temp.Add(ice))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ice);
                 }
               break;
            case defNeuronSpikeMFE:
               sp_mfe = new CNeuronSpikeMFE();
               if(!sp_mfe)
                  DeleteObjAndFalse(temp);
               if(!sp_mfe.Init(outputs, 0, opencl, desc.count, desc.windows, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sp_mfe);
                 }
               if(!temp.Add(sp_mfe))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sp_mfe);
                 }
               break;
            case defNeuronSpikeMHCrossAttention:
               sp_crosatt = new CNeuronSpikeMHCrossAttention();
               if(!sp_crosatt)
                  DeleteObjAndFalse(temp);
               if(!sp_crosatt.Init(outputs, 0, opencl, desc.windows[0], desc.window_out, desc.step,
                                   desc.units[0], desc.windows[1], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sp_crosatt);
                 }
               if(!temp.Add(sp_crosatt))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sp_crosatt);
                 }
               break;
            case defNeuronSpikeMixFusion:
               mix_fusion = new CNeuronSpikeMixFusion();
               if(!mix_fusion)
                  DeleteObjAndFalse(temp);
               if(!mix_fusion.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mix_fusion);
                 }
               if(!temp.Add(mix_fusion))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mix_fusion);
                 }
               break;
            case defNeuronSTFlow:
               stflow = new CNeuronSTFlow();
               if(!stflow)
                  DeleteObjAndFalse(temp);
               if(!stflow.Init(outputs, 0, opencl, desc.windows[0], desc.units[0], desc.window_out,
                               desc.windows[1], desc.units[1], desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(stflow);
                 }
               if(!temp.Add(stflow))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(stflow);
                 }
               break;
            case defNeuronPSSE:
               pse = new CNeuronPSSE();
               if(!pse)
                  DeleteObjAndFalse(temp);
               if(!pse.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out,
                            desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pse);
                 }
               if(!temp.Add(pse))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pse);
                 }
               break;
            case defNeuronSpikeDepthWiseResidual:
               dwr = new CNeuronSpikeDepthWiseResidual();
               if(!dwr)
                  DeleteObjAndFalse(temp);
               if(!dwr.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.layers,
                            desc.step, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(dwr);
                 }
               if(!temp.Add(dwr))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(dwr);
                 }
               break;
            case defNeuronSpikeSuperKernelBlock:
               sk_block = new CNeuronSpikeSuperKernelBlock();
               if(!sk_block)
                  DeleteObjAndFalse(temp);
               if(!sk_block.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.windows, desc.heads, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sk_block);
                 }
               if(!temp.Add(sk_block))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sk_block);
                 }
               break;
            case defNeuronSpikeGMA:
               gma = new CNeuronSpikeGMA();
               if(!gma)
                  DeleteObjAndFalse(temp);
               if(!gma.Init(outputs, 0, opencl, desc.window, desc.count, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gma);
                 }
               if(!temp.Add(gma))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gma);
                 }
               break;
            case defNeuronETROF:
               etrof = new CNeuronETROF();
               if(!etrof)
                  DeleteObjAndFalse(temp);
               if(!etrof.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(etrof);
                 }
               if(!temp.Add(etrof))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(etrof);
                 }
               break;
            case defNeuronPSSEFlow:
               pseflow = new CNeuronPSSEFlow();
               if(!pseflow)
                  DeleteObjAndFalse(temp);
               if(!pseflow.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pseflow);
                 }
               if(!temp.Add(pseflow))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pseflow);
                 }
               break;
            case defNeuronSpikeUSR:
               usr = new CNeuronSpikeUSR();
               if(!usr)
                  DeleteObjAndFalse(temp);
               if(!usr.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(usr);
                 }
               if(!temp.Add(usr))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(usr);
                 }
               break;
            case defNeuronInitialFlow:
               init_flow = new CNeuronInitialFlow();
               if(!init_flow)
                  DeleteObjAndFalse(temp);
               if(!init_flow.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.variables, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(init_flow);
                 }
               if(!temp.Add(init_flow))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(init_flow);
                 }
               break;
            case defNeuronSpikeSMR:
               smr = new CNeuronSpikeSMR();
               if(!smr)
                  DeleteObjAndFalse(temp);
               if(!smr.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(smr);
                 }
               if(!temp.Add(smr))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(smr);
                 }
               break;
            case defNeuronEVAFlow:
               eva = new CNeuronEVAFlow();
               if(!eva)
                  DeleteObjAndFalse(temp);
               if(!eva.Init(outputs, 0, opencl, desc.count, desc.window, desc.window_out, desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(eva);
                 }
               if(!temp.Add(eva))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(eva);
                 }
               break;
            case defNeuronResFlowHTR:
               htr = new CNeuronResFlowHTR();
               if(!htr)
                  DeleteObjAndFalse(temp);
               if(!htr.Init(outputs, 0, opencl, desc.count, desc.windows[0], desc.windows[1], desc.windows[2], desc.windows[3], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(htr);
                 }
               if(!temp.Add(htr))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(htr);
                 }
               break;
            case defNeuronResFlow:
               resflow = new CNeuronResFlow();
               if(!resflow)
                  DeleteObjAndFalse(temp);
               if(!resflow.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables,
                                desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(resflow);
                 }
               if(!temp.Add(resflow))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(resflow);
                 }
               break;
            //---
            case defNeuronConfGateTailAwareMoE:
               tailaware = new CNeuronConfGateTailAwareMoE();
               if(!tailaware)
                  DeleteObjAndFalse(temp);
               if(!tailaware.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out,
                                  desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(tailaware);
                 }
               if(!temp.Add(tailaware))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(tailaware);
                 }
               break;
            case defNeuronResFlowLattice:
               lattice = new CNeuronResFlowLattice();
               if(!lattice)
                  DeleteObjAndFalse(temp);
               if(!lattice.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count, desc.variables,
                                desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(lattice);
                 }
               if(!temp.Add(lattice))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(lattice);
                 }
               break;
            case defNeuronFieldAwareConv:
               faconv = new CNeuronFieldAwareConv();
               if(!faconv)
                  DeleteObjAndFalse(temp);
               if(!faconv.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.count, desc.windows[2],
                               desc.variables, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(faconv);
                 }
               if(!temp.Add(faconv))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(faconv);
                 }
               faconv.SetActivationFunction(desc.activation);
               break;
            case defNeuronMHFAT:
               fat = new CNeuronMHFAT();
               if(!fat)
                  DeleteObjAndFalse(temp);
               if(!fat.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.step, desc.count, desc.windows[2],
                            desc.layers, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(fat);
                 }
               if(!temp.Add(fat))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(fat);
                 }
               break;
            case defNeuronMHCrossFAT:
               cross_fat = new CNeuronMHCrossFAT();
               if(!cross_fat)
                  DeleteObjAndFalse(temp);
               if(!cross_fat.Init(outputs, 0, opencl, desc.windows[0], desc.windows[1], desc.step, desc.units[0],
                                  desc.windows[2], desc.units[1], desc.windows[3], desc.layers, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cross_fat);
                 }
               if(!temp.Add(cross_fat))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cross_fat);
                 }
               break;
            case defNeuronFieldPatternEmbedding:
               faemb = new CNeuronFieldPatternEmbedding();
               if(!faemb)
                  DeleteObjAndFalse(temp);
               if(!faemb.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out,
                              desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(faemb);
                 }
               if(!temp.Add(faemb))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(faemb);
                 }
               faemb.SetActivationFunction(desc.activation);
               break;
            case defNeuronAutoToken:
               autotoken = new CNeuronAutoToken();
               if(!autotoken)
                  DeleteObjAndFalse(temp);
               if(!autotoken.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.units[0], desc.units[1],
                                  desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(autotoken);
                 }
               if(!temp.Add(autotoken))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(autotoken);
                 }
               break;
            case defNeuronMultiMixAttention:
               mixattention = new CNeuronMultiMixAttention();
               if(!mixattention)
                  DeleteObjAndFalse(temp);
               if(!mixattention.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.window_out,
                                     desc.layers, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixattention);
                 }
               if(!temp.Add(mixattention))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixattention);
                 }
               break;
            case defNeuronSparseMoEConv:
               sparsemoe = new CNeuronSparseMoEConv();
               if(!sparsemoe)
                  DeleteObjAndFalse(temp);
               if(!sparsemoe.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.count,
                                  desc.layers, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sparsemoe);
                 }
               if(!temp.Add(sparsemoe))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(sparsemoe);
                 }
               sparsemoe.SetActivationFunction(desc.activation);
               break;
            case defNeuronMTmixAttBlock:
               mixattenblock = new CNeuronMTmixAttBlock();
               if(!mixattenblock)
                  DeleteObjAndFalse(temp);
               if(!mixattenblock.Init(outputs, 0, opencl, desc.window, desc.count, desc.step,
                                      desc.window_out, desc.layers, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixattenblock);
                 }
               if(!temp.Add(mixattenblock))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixattenblock);
                 }
               break;
            case defCrossMHFlashAttention:
               flash_att = new CCrossMHFlashAttention();
               if(!flash_att)
                  DeleteObjAndFalse(temp);
               if(!flash_att.Init(outputs, 0, opencl, desc.window, desc.step, (bool)desc.probability,
                                  desc.units[0], desc.units[1], desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(flash_att);
                 }
               if(!temp.Add(flash_att))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(flash_att);
                 }
               break;
            case defNeuronMixFeedForward:
               mixff = new CNeuronMixFeedForward();
               if(!mixff)
                  DeleteObjAndFalse(temp);
               if(!mixff.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out,
                              desc.variables, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixff);
                 }
               if(!temp.Add(mixff))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mixff);
                 }
               break;
            case defNeuronOneTrans:
               onetrans = new CNeuronOneTrans();
               if(!onetrans)
                  DeleteObjAndFalse(temp);
               if(!onetrans.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.units[1], desc.step, desc.count,
                                 desc.layers, desc.window_out, desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(onetrans);
                 }
               if(!temp.Add(onetrans))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(onetrans);
                 }
               break;
            case defNeuronMHTHCrossAttention:
               thcatt = new CNeuronMHTHCrossAttention();
               if(!thcatt)
                  DeleteObjAndFalse(temp);
               if(!thcatt.Init(outputs, 0, opencl, desc.windows[0], desc.units[0], desc.step, desc.windows[1], desc.units[1], desc.window_out,
                               desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(thcatt);
                 }
               if(!temp.Add(thcatt))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(thcatt);
                 }
               break;
            case defNeuronSTCA:
               stca = new CNeuronSTCA();
               if(!stca)
                  DeleteObjAndFalse(temp);
               if(!stca.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.units[1], desc.step, desc.units[2], desc.count,
                             desc.layers, desc.window_out, desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(stca);
                 }
               if(!temp.Add(stca))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(stca);
                 }
               break;
            case defNeuronWeightGenerator:
               wgen = new CNeuronWeightGenerator();
               if(!wgen)
                  DeleteObjAndFalse(temp);
               if(!wgen.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.step, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(wgen);
                 }
               if(!temp.Add(wgen))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(wgen);
                 }
               break;
            case defNeuronPCGR:
               pcrg = new CNeuronPCGR();
               if(!pcrg)
                  DeleteObjAndFalse(temp);
               if(!pcrg.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.units[0], desc.units[1], desc.step,
                             desc.units[2], uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pcrg);
                 }
               if(!temp.Add(pcrg))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pcrg);
                 }
               break;
            case defNeuronADS:
               ads = new CNeuronADS();
               if(!ads)
                  DeleteObjAndFalse(temp);
               if(!ads.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.units[1], desc.step, desc.units[2], desc.count,
                            desc.layers, desc.window_out, desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ads);
                 }
               if(!temp.Add(ads))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ads);
                 }
               break;
            case defNeuronPerTokenFFN:
               pertoken = new CNeuronPerTokenFFN();
               if(!pertoken)
                  DeleteObjAndFalse(temp);
               if(!pertoken.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out,
                                 desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pertoken);
                 }
               if(!temp.Add(pertoken))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(pertoken);
                 }
               break;
            case defNeuronScenariosToken:
               scentoken = new CNeuronScenariosToken();
               if(!scentoken)
                  DeleteObjAndFalse(temp);
               if(!scentoken.Init(outputs, 0, opencl, desc.window, desc.units[0], desc.window_out, desc.units[1],
                                  desc.window_out, desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scentoken);
                 }
               if(!temp.Add(scentoken))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scentoken);
                 }
               break;
            case defNeuronUnifiedTokenizer:
               uniftokens = new CNeuronUnifiedTokenizer();
               if(!uniftokens)
                  DeleteObjAndFalse(temp);
               if(!uniftokens.Init(outputs, 0, opencl, desc.windows, desc.count, desc.window_out,
                                   desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uniftokens);
                 }
               if(!temp.Add(uniftokens))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uniftokens);
                 }
               break;
            case defNeuronDomainAwareAttention:
               domainatt = new CNeuronDomainAwareAttention();
               if(!domainatt)
                  DeleteObjAndFalse(temp);
               if(!domainatt.Init(outputs, 0, opencl, desc.windows[0], desc.units[0], desc.step,
                                  desc.windows[1], desc.units[1], desc.window_out, desc.variables,
                                  uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(domainatt);
                 }
               if(!temp.Add(domainatt))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(domainatt);
                 }
               break;
            case defNeuronFeatureSelfIteration:
               featureinter = new CNeuronFeatureSelfIteration();
               if(!featureinter)
                  DeleteObjAndFalse(temp);
               if(!featureinter.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.layers,
                                     desc.window_out, desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(featureinter);
                 }
               if(!temp.Add(featureinter))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(featureinter);
                 }
               break;
            case defNeuronDomainFused:
               domainfused = new CNeuronDomainFused();
               if(!domainfused)
                  DeleteObjAndFalse(temp);
               if(!domainfused.Init(outputs, 0, opencl, desc.window, desc.units[0], desc.units[1],
                                    desc.window_out, desc.variables, uint(desc.probability),
                                    desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(domainfused);
                 }
               if(!temp.Add(domainfused))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(domainfused);
                 }
               break;
            case defNeuronMDL:
               mdl = new CNeuronMDL();
               if(!mdl)
                  DeleteObjAndFalse(temp);
               if(!mdl.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step, desc.units[1],
                            desc.count, desc.layers, desc.window_out, desc.variables,
                            uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mdl);
                 }
               if(!temp.Add(mdl))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(mdl);
                 }
               break;
            case defNeuronUniMixerTokenizer:
               umtoken = new CNeuronUniMixerTokenizer();
               if(!umtoken)
                  DeleteObjAndFalse(temp);
               if(!umtoken.Init(outputs, 0, opencl, desc.windows, desc.count, desc.window_out, desc.step,
                                desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(umtoken);
                 }
               if(!temp.Add(umtoken))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(umtoken);
                 }
               break;
            case defNeuronPerTokenSwiGLU:
               ptswiglu = new CNeuronPerTokenSwiGLU();
               if(!ptswiglu)
                  DeleteObjAndFalse(temp);
               if(!ptswiglu.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.variables,
                                 uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ptswiglu);
                 }
               if(!temp.Add(ptswiglu))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(ptswiglu);
                 }
               break;
            case defNeuronUniMixer:
               unimix = new CNeuronUniMixer();
               if(!unimix)
                  DeleteObjAndFalse(temp);
               if(!unimix.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.window_out, desc.variables,
                               uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(unimix);
                 }
               if(!temp.Add(unimix))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(unimix);
                 }
               break;
            case defNeuronSiameseNorm:
               siamnorm = new CNeuronSiameseNorm();
               if(!siamnorm)
                  DeleteObjAndFalse(temp);
               if(!siamnorm.Init(outputs, 0, opencl, desc.window, desc.count, desc.step, desc.window_out, desc.variables,
                                 uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(siamnorm);
                 }
               if(!temp.Add(siamnorm))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(siamnorm);
                 }
               break;
            case defNeuronUniMixerBlock:
               umblock = new CNeuronUniMixerBlock();
               if(!umblock)
                  DeleteObjAndFalse(temp);
               if(!umblock.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step, desc.units[1],
                                desc.count, desc.layers, desc.window_out, desc.variables,
                                uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(umblock);
                 }
               if(!temp.Add(umblock))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(umblock);
                 }
               break;
            case defNeuronINFNetScenarios:
               infnscen = new CNeuronINFNetScenarios();
               if(!infnscen)
                  DeleteObjAndFalse(temp);
               if(!infnscen.Init(outputs, 0, opencl, desc.window, desc.units, desc.window_out, desc.step,
                                 desc.variables, uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(infnscen);
                 }
               if(!temp.Add(infnscen))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(infnscen);
                 }
               break;
            case defNeuronINFNetHUBSequence:
               hub = new CNeuronINFNetHUBSequence();
               if(!hub)
                  DeleteObjAndFalse(temp);
               if(!hub.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out, desc.variables, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(hub);
                 }
               if(!temp.Add(hub))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(hub);
                 }
               break;
            case defNeuronINFNetBlock:
               infnblock = new CNeuronINFNetBlock();
               if(!infnblock)
                  DeleteObjAndFalse(temp);
               if(!infnblock.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                                  desc.units[1], desc.layers, desc.window_out, desc.variables,
                                  uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(infnblock);
                 }
               if(!temp.Add(infnblock))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(infnblock);
                 }
               break;
            case defNeuronSequenceTokenizer:
               seq_tok = new CNeuronSequenceTokenizer();
               if(!seq_tok)
                  DeleteObjAndFalse(temp);
               if(!seq_tok.Init(outputs, 0, opencl, desc.window, desc.count, desc.window_out,
                                desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(seq_tok);
                 }
               if(!temp.Add(seq_tok))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(seq_tok);
                 }
               break;
            case defNeuronDifferenceTokenizer:
               diff_tok = new CNeuronDifferenceTokenizer();
               if(!diff_tok)
                  DeleteObjAndFalse(temp);
               if(!diff_tok.Init(outputs, 0, opencl, desc.window, desc.window_out, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(diff_tok);
                 }
               if(!temp.Add(diff_tok))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(diff_tok);
                 }
               break;
            case defNeuronReGENTADBackbone:
               regen_backbone = new CNeuronReGENTADBackbone();
               if(!regen_backbone)
                  DeleteObjAndFalse(temp);
               if(!regen_backbone.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                                       desc.units[1], desc.layers, desc.window_out, desc.variables,
                                       uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regen_backbone);
                 }
               if(!temp.Add(regen_backbone))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regen_backbone);
                 }
               break;
            case defNeuronReGENTADHead:
               regen_head = new CNeuronReGENTADHead();
               if(!regen_head)
                  DeleteObjAndFalse(temp);
               if(!regen_head.Init(outputs, 0, opencl, desc.window, desc.units[0], desc.window_out,
                                   desc.units[1], NULL, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regen_head);
                 }
               if(!temp.Add(regen_head))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regen_head);
                 }
               break;
            case defNeuronReGENTAD:
               regentad = new CNeuronReGENTAD();
               if(!regentad)
                  DeleteObjAndFalse(temp);
               if(!regentad.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                                 desc.units[1], desc.layers, desc.window_out, desc.variables,
                                 uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regentad);
                 }
               if(!temp.Add(regentad))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(regentad);
                 }
               break;
            case defNeuronGDformerBackbone:
               gdformer_backbone = new CNeuronGDformerBackbone();
               if(!gdformer_backbone)
                  DeleteObjAndFalse(temp);
               if(!gdformer_backbone.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                                          desc.units[1], desc.layers, desc.window_out, desc.variables,
                                          uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gdformer_backbone);
                 }
               if(!temp.Add(gdformer_backbone))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gdformer_backbone);
                 }
               break;
            case defNeuronGDformer:
               gdformer = new CNeuronGDformer();
               if(!gdformer)
                  DeleteObjAndFalse(temp);
               if(!gdformer.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                                 desc.units[1], desc.layers, desc.window_out, desc.variables,
                                 uint(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gdformer);
                 }
               if(!temp.Add(gdformer))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(gdformer);
                 }
               break;
            case defNeuronMomADTTM:
               if(desc.units.Size() < 2)
                  DeleteObjAndFalse(temp);
               momad_ttm = new CNeuronMomADTTM();
               if(!momad_ttm)
                  DeleteObjAndFalse(temp);
               if(!momad_ttm.Init(outputs, 0, opencl, desc.window, desc.units[0],
                                  desc.units[1], uint(desc.probability),
                                  desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad_ttm);
                 }
               if(!temp.Add(momad_ttm))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad_ttm);
                 }
               break;
            case defNeuronMomADMPI:
               if(desc.units.Size() < 3)
                  DeleteObjAndFalse(temp);
               momad_mpi = new CNeuronMomADMPI();
               if(!momad_mpi)
                  DeleteObjAndFalse(temp);
               if(!momad_mpi.Init(outputs, 0, opencl, desc.window, desc.units[0],
                                  desc.units[1], desc.units[2], desc.count,
                                  uint(desc.probability), desc.step, desc.window_out,
                                  desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad_mpi);
                 }
               if(!temp.Add(momad_mpi))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad_mpi);
                 }
               break;
            case defNeuronVLADriverRAGMPI:
               if(desc.units.Size() < 3)
                  DeleteObjAndFalse(temp);
               vladriver_rag_mpi = new CNeuronVLADriverRAGMPI();
               if(!vladriver_rag_mpi)
                  DeleteObjAndFalse(temp);
               if(!vladriver_rag_mpi.Init(outputs, 0, opencl, desc.window, desc.units[0],
                                          desc.units[1], desc.units[2], desc.count,
                                          uint(desc.probability), desc.step, desc.window_out,
                                          desc.layers, desc.variables,
                                          desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(vladriver_rag_mpi);
                 }
               if(!temp.Add(vladriver_rag_mpi))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(vladriver_rag_mpi);
                 }
               break;
            case defNeuronUncAD_MUE:
               uncad_mue = new CNeuronUncAD_MUE();
               if(!uncad_mue)
                  DeleteObjAndFalse(temp);
               if(!uncad_mue.Init(outputs, 0, opencl, desc.window, desc.window_out,
                                  desc.count, desc.units[0], desc.units[1],
                                  float(desc.probability), desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_mue);
                 }
               if(!temp.Add(uncad_mue))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_mue);
                 }
               break;
            case defNeuronCogDriverData:
               if(desc.units.Size() < 3)
                  DeleteObjAndFalse(temp);
               cogdriver_data = new CNeuronCogDriverData();
               if(!cogdriver_data)
                  DeleteObjAndFalse(temp);
               if(!cogdriver_data.Init(outputs, 0, opencl, desc.window,
                                       desc.count, desc.units[0],
                                       desc.units[1], desc.units[2],
                                       float(desc.probability),
                                       desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_data);
                 }
               if(!temp.Add(cogdriver_data))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_data);
                 }
               break;
            case defNeuronCogDriverRankTCM:
               if(desc.units.Size() < 2 || desc.variables == 0)
                  DeleteObjAndFalse(temp);
               cogdriver_rank = new CNeuronCogDriverRankTCM();
               if(!cogdriver_rank)
                  DeleteObjAndFalse(temp);
               if(!cogdriver_rank.Init(outputs, 0, opencl, desc.window,
                                       desc.count, desc.variables, desc.units[0], desc.units[1],
                                       desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_rank);
                 }
               if(!temp.Add(cogdriver_rank))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_rank);
                 }
               break;
            case defNeuronCogDriverForecastHead:
               if(desc.units.Size() < 2 || desc.variables == 0)
                  DeleteObjAndFalse(temp);
               cogdriver_forecast = new CNeuronCogDriverForecastHead();
               if(!cogdriver_forecast)
                  DeleteObjAndFalse(temp);
               if(!cogdriver_forecast.Init(outputs, 0, opencl, desc.window,
                                           desc.count, desc.variables, desc.units[0], desc.units[1],
                                           desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_forecast);
                 }
               if(!temp.Add(cogdriver_forecast))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_forecast);
                 }
               break;
            case defNeuronCogDriverForecastToken:
               cogdriver_token = new CNeuronCogDriverForecastToken();
               if(!cogdriver_token)
                  DeleteObjAndFalse(temp);
               if(!cogdriver_token.Init(outputs, 0, opencl, desc.window,
                                        desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_token);
                 }
               if(!temp.Add(cogdriver_token))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(cogdriver_token);
                 }
               break;
            case defNeuronScenarioForecast:
               scenario_forecast = new CNeuronScenarioForecast();
               if(!scenario_forecast)
                  DeleteObjAndFalse(temp);
               if(!scenario_forecast.Init(outputs, 0, opencl, desc.count,
                                          desc.variables, desc.window_out, desc.window,
                                          desc.optimization, desc.batch,
                                          (desc.layers > 0 ? desc.layers : 1)))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scenario_forecast);
                 }
               if(!temp.Add(scenario_forecast))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scenario_forecast);
                 }
               break;
            case defNeuronScenarioCrossAttention:
               scenario_cross = new CNeuronScenarioCrossAttention();
               if(!scenario_cross)
                  DeleteObjAndFalse(temp);
               if(desc.variables != 3 && desc.variables != 5)
                  DeleteObjAndFalse(temp);
               if(!scenario_cross.Init(outputs, 0, opencl, desc.count,
                                       desc.window_out, desc.window, desc.variables == 5,
                                       desc.optimization, desc.batch,
                                       (desc.layers > 0 ? desc.layers : 1),
                                       (desc.step > 0 ? desc.step : 1),
                                       uint(desc.probability > 0 ? desc.probability : 1)))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scenario_cross);
                 }
               if(!temp.Add(scenario_cross))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(scenario_cross);
                 }
               break;
            case defNeuronRAGMemory:
               rag_memory = new CNeuronRAGMemory();
               if(!rag_memory)
                  DeleteObjAndFalse(temp);
               if(!rag_memory.Init(outputs, 0, opencl, desc))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(rag_memory);
                 }
               if(!temp.Add(rag_memory))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(rag_memory);
                 }
               break;
            case defNeuronD2Skill:
               d2_skill = new CD2Skill();
               if(!d2_skill)
                  DeleteObjAndFalse(temp);
               if(desc.variables != 3)
                  DeleteObjAndFalse(temp);
               if(!d2_skill.Init(outputs, 0, opencl, desc.count,
                                 desc.window_out, desc.window, false,
                                 desc.optimization, desc.batch,
                                 (desc.layers > 0 ? desc.layers : 1),
                                 (desc.step > 0 ? desc.step : 1),
                                 uint(desc.probability > 0 ? desc.probability : 1)))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(d2_skill);
                 }
               if(!temp.Add(d2_skill))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(d2_skill);
                 }
               break;
            case defNeuronUncAD_UGP:
               uncad_ugp = new CNeuronUncAD_UGP();
               if(!uncad_ugp)
                  DeleteObjAndFalse(temp);
               if(!uncad_ugp.Init(outputs, 0, opencl, desc.window, desc.count,
                                  desc.units[0], desc.units[1], desc.step,
                                  desc.window_out, desc.variables, uint(desc.probability),
                                  desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_ugp);
                 }
               if(!temp.Add(uncad_ugp))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_ugp);
                 }
               break;
            case defNeuronUncAD_UGPL:
               uncad_ugpl = new CNeuronUncAD_UGPL();
               if(!uncad_ugpl)
                  DeleteObjAndFalse(temp);
               if(!uncad_ugpl.Init(outputs, 0, opencl, desc.window, desc.count,
                                   desc.units[0], desc.units[1], desc.variables,
                                   desc.step, desc.window_out, desc.optimization,
                                   desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_ugpl);
                 }
               if(!temp.Add(uncad_ugpl))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad_ugpl);
                 }
               break;
            case defNeuronUncAD:
               uncad = new CNeuronUncAD();
               if(!uncad)
                  DeleteObjAndFalse(temp);
               if(!uncad.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                              desc.units[1], desc.units[2], desc.window_out, desc.variables,
                              uint(desc.probability), desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad);
                 }
               if(!temp.Add(uncad))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(uncad);
                 }
               break;
            case defNeuronMomAD:
               momad = new CNeuronMomAD();
               if(!momad)
                  DeleteObjAndFalse(temp);
               if(!momad.Init(outputs, 0, opencl, desc.windows, desc.units[0], desc.step,
                              desc.units[1], desc.units[2], desc.window_out, desc.variables,
                              uint(desc.probability), desc.count, desc.optimization, desc.batch))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad);
                 }
               if(!temp.Add(momad))
                 {
                  DeleteObj(temp);
                  DeleteObjAndFalse(momad);
                 }
               break;
            //---
            default:
               ReturnFalse;
               break;
           }
        }
      else
         for(uint n = 0; n < neurons; n++)
           {
            switch(desc.type)
              {
               case defNeuron:
                  neuron = new CNeuron();
                  if(CheckPointer(neuron) == POINTER_INVALID)
                    {
                     DeleteObj(temp);
                     DeleteObj(layers);
                     ReturnFalse;
                    }
                  neuron.Init(outputs, n, desc.optimization);
                  neuron.SetActivationFunction(desc.activation);
                  break;
               case defNeuronConv:
                  neuron_p = new CNeuronConv();
                  if(CheckPointer(neuron_p) == POINTER_INVALID)
                    {
                     DeleteObj(temp);
                     DeleteObj(layers);
                     ReturnFalse;
                    }
                  if(CheckPointer(prev) != POINTER_INVALID)
                    {
                     if(prev.type == defNeuron)
                       {
                        temp_count = (int)((prev.count - desc.window) % desc.step);
                        output_count = (int)((prev.count - desc.window - temp_count) / desc.step + (temp_count == 0 ? 1 : 2));
                       }
                     else
                        if(n == 0)
                          {
                           temp_count = (int)((output_count - desc.window) % desc.step);
                           output_count = (int)((output_count - desc.window - temp_count) / desc.step + (temp_count == 0 ? 1 : 2));
                          }
                    }
                  if(neuron_p.Init(outputs, n, desc.window, desc.step, output_count, desc.optimization))
                     neuron = neuron_p;
                  break;
               case defNeuronProof:
                  neuron_p = new CNeuronProof();
                  if(CheckPointer(neuron_p) == POINTER_INVALID)
                    {
                     DeleteObj(temp);
                     DeleteObj(layers);
                     ReturnFalse;
                    }
                  if(CheckPointer(prev) != POINTER_INVALID)
                    {
                     if(prev.type == defNeuron)
                       {
                        temp_count = (int)((prev.count - desc.window) % desc.step);
                        output_count = (int)((prev.count - desc.window - temp_count) / desc.step + (temp_count == 0 ? 1 : 2));
                       }
                     else
                        if(n == 0)
                          {
                           temp_count = (int)((output_count - desc.window) % desc.step);
                           output_count = (int)((output_count - desc.window - temp_count) / desc.step + (temp_count == 0 ? 1 : 2));
                          }
                    }
                  if(neuron_p.Init(outputs, n, desc.window, desc.step, output_count, desc.optimization))
                     neuron = neuron_p;
                  break;
               case defNeuronLSTM:
                  neuron_p = new CNeuronLSTM();
                  if(CheckPointer(neuron_p) == POINTER_INVALID)
                    {
                     DeleteObj(temp);
                     DeleteObj(layers);
                     ReturnFalse;
                    }
                  output_count = (next != NULL ? next.window : desc.step);
                  if(neuron_p.Init(outputs, n, desc.window, 1, output_count, desc.optimization))
                     neuron = neuron_p;
                  break;
              }
            if(!temp.Add(neuron))
              {
               DeleteObj(temp);
               DeleteObj(layers);
               ReturnFalse;
              }
            neuron = NULL;
           }
      if(!layers.Add(temp))
        {
         DeleteObj(temp);
         DeleteObj(layers);
         ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::feedForward(CArrayFloat *inputVals, int window = 1, bool tem = true, CBufferFloat *SecondInput = NULL)
  {
   if(CheckPointer(layers) == POINTER_INVALID || CheckPointer(inputVals) == POINTER_INVALID || layers.Total() <= 1)
      ReturnFalse;
//---
   CLayer *previous = NULL;
   CLayer *current = layers.At(0);
   int total = MathMin(current.Total(), inputVals.Total());
   CNeuronBase *neuron = NULL;
   if(CheckPointer(opencl) == POINTER_INVALID)
     {
      for(int i = 0; i < total; i++)
        {
         neuron = current.At(i);
         if(CheckPointer(neuron) == POINTER_INVALID)
            ReturnFalse;
         int pos = i;
         int d = 0;
         if(window > 1)
           {
            d = i % window;
            pos = (i - d) / window;
           }
         neuron.setOutputVal(inputVals.At(i) + (float)(tem ? (d % 2 == 0 ? sin(pos / MathPow(10000.0, (2.0 * d + 1.0) / (window + 1.0))) : cos(pos / MathPow(10000.0, (2.0 * d + 1.0) / (window + 1.0)))) : 0));
         if(tem)
            inputVals.Update(i, neuron.getOutputVal());
        }
     }
   else
     {
      CNeuronBaseOCL *neuron_ocl = current.At(0);
      CBufferFloat *inputs = neuron_ocl.getOutput();
      if(tem)
        {
         int total_data = inputVals.Total();
         for(int d = 0; d < total_data; d++)
           {
            int pos = d;
            int dim = 0;
            if(window > 1)
              {
               dim = d % window;
               pos = (d - dim) / window;
              }
            float value = (float)(pos / MathPow(10000.0, (2.0 * dim + 1.0) / (window + 1.0)));
            value = (float)(inputVals.At(d) + (dim % 2 == 0 ? sin(value) : cos(value)));
            if(!inputVals.Update(d, value))
               ReturnFalse;
           }
        }
      if(!inputs.Fill(inputVals))
         ReturnFalse;
      //---
      if(!!SecondInput)
        {
         if(SecondInput.GetIndex() < 0)
            if(!SecondInput.BufferCreate(opencl))
               ReturnFalse;
        }
     }
//---
   CObject *temp = NULL;
   for(int l = 1; l < layers.Total(); l++)
     {
      previous = current;
      current = layers.At(l);
      if(CheckPointer(current) == POINTER_INVALID)
         ReturnFalse;
      //---
      if(CheckPointer(opencl) != POINTER_INVALID)
        {
         CNeuronBaseOCL *current_ocl = current.At(0);
         if(!current_ocl.FeedForward(previous.At(0), SecondInput))
            ReturnFalse;
         continue;
        }
      //---
      total = current.Total() % 5;
      if(current.At(0).Type() == defNeuron)
         total--;
      //---
      for(int n = 0; n < total; n++)
        {
         neuron = current.At(n);
         if(CheckPointer(neuron) == POINTER_INVALID)
            ReturnFalse;
         if(previous.At(0).Type() == defNeuron)
           {
            temp = previous;
            if(!neuron.feedForward(temp))
               ReturnFalse;
            continue;
           }
         if(neuron.Type() == defNeuron)
           {
            if(n == 0)
              {
               CLayer *temp_l = new CLayer(total);
               if(CheckPointer(temp_l) == POINTER_INVALID)
                  ReturnFalse;
               CNeuronProof *proof = NULL;
               for(int p = 0; p < previous.Total(); p++)
                 {
                  proof = previous.At(p);
                  if(CheckPointer(proof) == POINTER_INVALID)
                     ReturnFalse;
                  temp_l.AddArray(proof.getOutputLayer());
                 }
               temp = temp_l;
              }
            if(!neuron.feedForward(temp))
               ReturnFalse;
            if(n == total - 1)
              {
               CLayer *temp_l = temp;
               temp_l.FreeMode(false);
               temp_l.Shutdown();
               DeleteObj(temp_l);
              }
            continue;
           }
         temp = previous.At(n);
         if(CheckPointer(temp) == POINTER_INVALID)
            ReturnFalse;
         if(!neuron.feedForward(temp))
            ReturnFalse;
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::backProp(CArrayFloat *targetVals, CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL)
  {
   if(CheckPointer(targetVals) == POINTER_INVALID || CheckPointer(layers) == POINTER_INVALID)
      ReturnFalse;
   if(CheckPointer(opencl) != POINTER_INVALID)
     {
      if(!backPropOCL(targetVals, SecondInput, SecondGradient, false))
         ReturnFalse;
      return true;
     }
//---
   CLayer *outputLayer = layers.At(layers.Total() - 1);
   if(CheckPointer(outputLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   float error = 0.0;
   int total = outputLayer.Total() - 1;
   for(int n = 0; n < total && !IsStopped(); n++)
     {
      CNeuron *neuron = outputLayer.At(n);
      float target = targetVals.At(n);
      float delta = (target > 1 ? 1 : target < -1 ? -1 : target) - neuron.getOutputVal();
      error += delta * delta;
      neuron.calcOutputGradients(targetVals.At(n));
     }
   error /= (float)total;
   error = (float)sqrt(error);
   backPropCount++;
   recentAverageError += (error - recentAverageError) / fmin(recentAverageSmoothingFactor, backPropCount);
//---
   CNeuronBase *neuron = NULL;
   CObject *temp = NULL;
   for(int layerNum = layers.Total() - 2; layerNum > 0; layerNum--)
     {
      CLayer *hiddenLayer = layers.At(layerNum);
      CLayer *nextLayer = layers.At(layerNum + 1);
      total = hiddenLayer.Total();
      for(int n = 0; n < total && !IsStopped(); ++n)
        {
         neuron = hiddenLayer.At(n);
         if(nextLayer.At(0).Type() == defNeuron)
           {
            temp = nextLayer;
            neuron.calcHiddenGradients(temp);
            continue;
           }
         if(neuron.Type() == defNeuron)
           {
            float g = 0;
            for(int i = 0; i < nextLayer.Total(); i++)
              {
               temp = nextLayer.At(i);
               neuron.calcHiddenGradients(temp);
               g += neuron.getGradient();
              }
            neuron.setGradient(g);
            continue;
           }
         temp = nextLayer.At(n);
         neuron.calcHiddenGradients(temp);
        }
     }
//---
   for(int layerNum = layers.Total() - 1; layerNum > 0; layerNum--)
     {
      CLayer *layer = layers.At(layerNum);
      CLayer *prevLayer = layers.At(layerNum - 1);
      total = layer.Total() - (layer.At(0).Type() == defNeuron ? 1 : 0);
      int n_conv = 0;
      for(int n = 0; n < total && !IsStopped(); n++)
        {
         neuron = layer.At(n);
         if(CheckPointer(neuron) == POINTER_INVALID)
            ReturnFalse;
         if(neuron.Type() == defNeuronProof)
            continue;
         switch(prevLayer.At(0).Type())
           {
            case defNeuron:
               temp = prevLayer;
               neuron.updateInputWeights(temp);
               break;
            case defNeuronConv:
            case defNeuronProof:
            case defNeuronLSTM:
               if(neuron.Type() == defNeuron)
                 {
                  for(n_conv = 0; n_conv < prevLayer.Total(); n_conv++)
                    {
                     temp = prevLayer.At(n_conv);
                     neuron.updateInputWeights(temp);
                    }
                 }
               else
                 {
                  temp = prevLayer.At(n);
                  neuron.updateInputWeights(temp);
                 }
               break;
            default:
               temp = NULL;
               break;
           }
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::backPropOCL(CBufferFloat *targetVals, CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL, bool layer_by_layer = false, ENUM_ACTIVATION SecondActivation = None)
  {
   if(CheckPointer(targetVals) == POINTER_INVALID || CheckPointer(layers) == POINTER_INVALID || CheckPointer(opencl) == POINTER_INVALID)
      ReturnFalse;
   CLayer *currentLayer = layers.At(layers.Total() - 1);
   if(CheckPointer(currentLayer) == POINTER_INVALID)
      ReturnFalse;
//---
   float error = 0.0;
   int total = targetVals.Total();
   CNeuronBaseOCL *neuron = currentLayer.At(0);
   if(neuron.Type() != defNeuronSoftMaxOCL
      && neuron.Type() != defNeuronMantisAttentionUnit
     )
     {
      vector<float> target, output;
      if(neuron.getOutputVal(output) < neuron.Neurons())
         ReturnFalse;
      if(output.HasNan() > 0)
        {
         printf("%s - %d -> Output has NaN", __FUNCTION__, __LINE__);
         ReturnFalse;
        }
      targetVals.GetData(target);
      if(target.Size() > output.Size())
        {
         if(!target.Resize(output.Size()))
            ReturnFalse;
        }
      error = output.Loss(target, LOSS_MSE);
     }
//---
   if(!neuron.calcOutputGradients(targetVals, error))
      ReturnFalse;
//---
   if(neuron.TrainMode())
     {
      if(!MathIsValidNumber(error))
         error = getRecentAverageError();
      backPropCount++;
      recentAverageError += (error - recentAverageError) / fmin(recentAverageSmoothingFactor, (float)backPropCount);
     }
//---
   if(layer_by_layer)
     {
      total = layers.Total();
      for(int layer_update = 1; layer_update < total; layer_update++)
        {
         for(int layerNum = total - 2; layerNum > layer_update - 1 && !IsStopped(); layerNum--)
           {
            CLayer *nextLayer = currentLayer;
            currentLayer = layers.At(layerNum);
            neuron = currentLayer.At(0);
            if(!neuron.CalcHiddenGradients(nextLayer.At(0), SecondInput, SecondGradient, SecondActivation))
               ReturnFalse;
           }
         //---
         CLayer *prevLayer = layers.At(layer_update - 1);
         currentLayer = layers.At(layer_update);
         neuron = currentLayer.At(0);
         if(!neuron.UpdateInputWeights(prevLayer.At(0), SecondInput))
            ReturnFalse;
         //---
         for(int layerNum = layer_update; layerNum < total; layerNum++)
           {
            currentLayer = layers.At(layerNum);
            if(CheckPointer(currentLayer) == POINTER_INVALID)
               ReturnFalse;
            //---
            CNeuronBaseOCL *current_ocl = currentLayer.At(0);
            if(!current_ocl.FeedForward(prevLayer.At(0), SecondInput))
               ReturnFalse;
            prevLayer = currentLayer;
           }
        }
      return true;
     }
//--- Calc Hidden Gradients
   total = layers.Total();
   for(int layerNum = total - 2; layerNum >= 0 && !IsStopped(); layerNum--)
     {
      CLayer *nextLayer = currentLayer;
      currentLayer = layers.At(layerNum);
      neuron = currentLayer.At(0);
      if(!neuron.CalcHiddenGradients(nextLayer.At(0), SecondInput, SecondGradient, SecondActivation))
         ReturnFalse;
     }
//---
   CLayer *prevLayer = layers.At(total - 1);
   for(int layerNum = total - 1; layerNum > 0 && !IsStopped(); layerNum--)
     {
      currentLayer = prevLayer;
      prevLayer = layers.At(layerNum - 1);
      neuron = currentLayer.At(0);
      if(neuron.TrainMode())
         if(!neuron.UpdateInputWeights(prevLayer.At(0), SecondInput))
            ReturnFalse;
     }
//---
   bool result = false;
   for(int layerNum = 0; (layerNum < layers.Total() && !result && !IsStopped()); layerNum++)
     {
      currentLayer = layers.At(layerNum);
      CNeuronBaseOCL *temp = currentLayer.At(0);
      CNeuronConvOCL *conv = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      if(!temp.TrainMode())
        {
         if(!!temp.getGradient())
            temp.getGradient().BufferRead();
         if(layerNum == layers.Total() - 1)
            result = true;
         continue;
        }
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            if(!!conv.GetWeightsConv() && conv.GetWeightsConv().BufferRead())
               result = true;
            else
               if(!!temp.getWeights() && temp.getWeights().BufferRead())
                  result = true;
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            if(!!batch.getBatchOptions() && batch.getBatchOptions().BufferRead())
               result = true;
            if(!result && !!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            if(!!lstm.getLSTMWeights() && lstm.getLSTMWeights().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         default:
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
        }
     }
//---
   if(!!SecondGradient)
      SecondGradient.BufferRead();
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNet::getResults(CBufferFloat *&resultVals)
  {
   if(CheckPointer(resultVals) == POINTER_INVALID)
     {
      resultVals = new CBufferFloat();
      if(CheckPointer(resultVals) == POINTER_INVALID)
         return;
     }
//---
   resultVals.Clear();
   if(CheckPointer(layers) == POINTER_INVALID || layers.Total() <= 0)
      return;
//---
   CLayer *output = layers.At(layers.Total() - 1);
   if(CheckPointer(output) == POINTER_INVALID)
      return;
//---
   if(CheckPointer(opencl) != POINTER_INVALID && output.At(0).Type() >= defNeuronBaseOCL)
     {
      CNeuronBaseOCL *temp = output.At(0);
      temp.getOutputVal(resultVals);
      return;
     }
   CNeuronBase *neuron = NULL;
   CLayer *temp = NULL;
   int total = output.Total();
   if(output.At(0).Type() == defNeuron)
      total--;
//---
   for(int i = 0; i < total; i++)
     {
      neuron = output.At(i);
      if(CheckPointer(neuron) == POINTER_INVALID)
         continue;
      if(neuron.Type() == defNeuron)
        {
         resultVals.Add(neuron.getOutputVal());
         continue;
        }
      CNeuronProof *n = neuron;
      temp = n.getOutputLayer();
      for(int ii = 0; ii < temp.Total(); ii++)
        {
         neuron = temp.At(ii);
         if(CheckPointer(neuron) == POINTER_INVALID)
            continue;
         resultVals.Add(neuron.getOutputVal());
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Save(string file_name, float error, float undefine, float forecast, datetime time, bool common = true)
  {
//if(MQLInfoInteger(MQL_OPTIMIZATION) || MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_FORWARD) || MQLInfoInteger(MQL_OPTIMIZATION))
//   return true;
   if(file_name == NULL || !layers)
      ReturnFalse;
//---
   int handle = FileOpen(file_name, (common ? FILE_COMMON : 0) | FILE_BIN | FILE_WRITE);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
//---
   if(FileWriteDouble(handle, error) <= 0 || FileWriteDouble(handle, undefine) <= 0 || FileWriteDouble(handle, forecast) <= 0 || FileWriteLong(handle, (long)time) <= 0)
     {
      FileClose(handle);
      ReturnFalse;
     }
   bool result = layers.Save(handle);
   FileFlush(handle);
   FileClose(handle);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || !layers)
      ReturnFalse;
//---
   if(FileWriteDouble(file_handle, (double)recentAverageError) <= 0 || FileWriteDouble(file_handle, 0) <= 0 || FileWriteDouble(file_handle, 0) <= 0 || FileWriteLong(file_handle, (long)0) <= 0)
      ReturnFalse;
//---
   return layers.Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Load(string file_name, float &error, float &undefine, float &forecast, datetime &time, bool common = true)
  {
//if(MQLInfoInteger(MQL_OPTIMIZATION) || MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_FORWARD) || MQLInfoInteger(MQL_OPTIMIZATION))
//   ReturnFalse;
//---
   if(file_name == NULL)
      ReturnFalse;
//---
   Print(file_name);
   int handle = FileOpen(file_name, (common ? FILE_COMMON : 0) | FILE_BIN | FILE_READ | FILE_SHARE_READ);
   if(handle == INVALID_HANDLE)
      ReturnFalse;
//---
   error = (float)FileReadDouble(handle);
   undefine = (float)FileReadDouble(handle);
   forecast = (float)FileReadDouble(handle);
   time = (datetime)FileReadLong(handle);
   recentAverageError = fmax(error, 0);
//---
   if(CheckPointer(layers) != POINTER_INVALID)
      layers.Clear();
   else
      layers = new CArrayLayer();
   int i = 0, num;
//---
   if(!opencl && !OpenCLInit())
      ReturnFalse;
//--- check
//--- read and check start marker - 0xFFFFFFFFFFFFFFFF
   long temp = FileReadLong(handle);
   if(temp == -1)
     {
      //--- read and check array type
      if(FileReadInteger(handle, INT_VALUE) != layers.Type())
        {
         FileClose(handle);
         ReturnFalse;
        }
     }
   else
     {
      FileClose(handle);
      ReturnFalse;
     }
//--- read array length
   num = FileReadInteger(handle, INT_VALUE);
//--- read array
   if(num != 0)
     {
      for(i = 0; i < num; i++)
        {
         //--- create new element
         CLayer *Layer = new CLayer(0, handle, opencl);
         if(!Layer.Load(handle))
           {
            DeleteObj(Layer);
            break;
           }
         if(Layer.At(0).Type() == defNeuronRevInDenormOCL)
           {
            CNeuronRevINDenormOCL *revin = Layer.At(0);
            int l = revin.GetNormLayer();
            if(!layers.At(l))
              {
               DeleteObj(Layer);
               break;
              }
            CNeuronBaseOCL *neuron = ((CLayer *)layers.At(l)).At(0);
            if(!(neuron.Type() == defNeuronBatchNormOCL ||
                 neuron.Type() == defNeuronBatchNormWithNoise))
              {
               DeleteObj(Layer);
               break;
              }
            if(!revin.Init(revin.getConnections(), 0, opencl, revin.Neurons(), l, neuron))
              {
               DeleteObj(Layer);
               break;
              }
           }
         if(Layer.At(0).Type() == defNeuronMASA)
           {
            CNeuronMASA *masa = Layer.At(0);
            int l = masa.GetNormLayer();
            if(!layers.At(l))
              {
               DeleteObj(Layer);
               break;
              }
            CNeuronBaseOCL *neuron = ((CLayer *)layers.At(l)).At(0);
            if(!(neuron.Type() == defNeuronBatchNormOCL) ||
               (neuron.Type() == defNeuronBatchNormWithNoise))
              {
               DeleteObj(Layer);
               break;
              }
            if(!masa.SetNormLayer(l, neuron))
              {
               DeleteObj(Layer);
               break;
              }
           }
         if(!layers.Add(Layer))
           {
            DeleteObj(Layer);
            break;
           }
        }
     }
   FileClose(handle);
//--- result
   return (layers.Total() == num);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE || FileIsEnding(file_handle))
      ReturnFalse;
//---
     {
      recentAverageError = (float)FileReadDouble(file_handle);
      float temp = (float)fmax(FileReadDouble(file_handle), 0);
      temp = (float)FileReadDouble(file_handle);
      temp = (float)FileReadLong(file_handle);
     }
//---
   if(CheckPointer(layers) != POINTER_INVALID)
      layers.Clear();
   else
      layers = new CArrayLayer();
   int i = 0, num;
//---
   if(CheckPointer(opencl) == POINTER_INVALID)
     {
      if(!OpenCLInit())
         ReturnFalse;
     }
//--- check
//--- read and check start marker - 0xFFFFFFFFFFFFFFFF
   long temp = FileReadLong(file_handle);
   if(temp == -1)
     {
      //--- read and check array type
      if(FileReadInteger(file_handle, INT_VALUE) != layers.Type())
         return(false);
     }
   else
     {
      while((temp = FileReadLong(file_handle)) != -1 && !FileIsEnding(file_handle))
        {
         Sleep(0);
        }
      if(temp != -1)
         return(false);
     }
//--- read array length
   num = FileReadInteger(file_handle, INT_VALUE);
//--- read array
   if(num != 0)
     {
      for(i = 0; i < num; i++)
        {
         //--- create new element
         CLayer *Layer = new CLayer(0, file_handle, opencl);
         if(!Layer.Load(file_handle))
            break;
         if(Layer.At(0).Type() == defNeuronRevInDenormOCL)
           {
            CNeuronRevINDenormOCL *revin = Layer.At(0);
            int l = revin.GetNormLayer();
            if(!layers.At(l))
              {
               DeleteObj(Layer);
               break;
              }
            CNeuronBaseOCL *neuron = ((CLayer *)layers.At(l)).At(0);
            if(!(neuron.Type() == defNeuronBatchNormOCL) ||
               (neuron.Type() == defNeuronBatchNormWithNoise))
              {
               DeleteObj(Layer);
               break;
              }
            if(!revin.Init(revin.getConnections(), 0, opencl, revin.Neurons(), l, neuron))
              {
               DeleteObj(Layer);
               break;
              }
           }
         if(!layers.Add(Layer))
            break;
        }
     }
   FileClose(file_handle);
//--- result
   return (layers.Total() == num);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::TrainMode(bool flag)
  {
   if(CheckPointer(layers) == POINTER_INVALID || layers.Total() <= 1)
      ReturnFalse;
//---
   CLayer *current = NULL;
//---
   CObject *temp = NULL;
   for(int l = 0; l < layers.Total(); l++)
     {
      current = layers.At(l);
      if(CheckPointer(current) == POINTER_INVALID || CheckPointer(current.At(0)) == POINTER_INVALID)
         ReturnFalse;
      //---
      if(CheckPointer(opencl) != POINTER_INVALID)
        {
         CNeuronBaseOCL *current_ocl = current.At(0);
         current_ocl.TrainMode(flag);
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::Save(const int file_handle)
  {
   if(!CNeuronBase::Save(file_handle) || !OutputLayer.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iWindow, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, iStep, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProof::Load(const int file_handle)
  {
   if(!CNeuronBase::Load(file_handle) || !OutputLayer.Load(file_handle))
      ReturnFalse;
   iWindow = FileReadInteger(file_handle, INT_VALUE);
   iStep = FileReadInteger(file_handle, INT_VALUE);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::Save(const int file_handle)
  {
   if(!CNeuronProof::Save(file_handle))
      ReturnFalse;
   if(FileWriteDouble(file_handle, param) < 8)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConv::Load(const int file_handle)
  {
   if(!CNeuronProof::Load(file_handle))
      ReturnFalse;
   param = (float)FileReadDouble(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuronBase::activationFunction(float x)
  {
   switch(activation)
     {
      case TANH:
         return TanhFunction(x);
         break;
      case SIGMOID:
         return SigmoidFunction(x);
         break;
     }
//---
   return x;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuronBase::activationFunctionDerivative(float x)
  {
   switch(activation)
     {
      case TANH:
         return TanhFunctionDerivative(x);
         break;
      case SIGMOID:
         return SigmoidFunctionDerivative(x);
         break;
     }
//---
   return 1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
int COpenCLMy::AddBufferFromArray(T &data[], const uint data_array_offset, const uint data_array_count, const uint flags)
  {
   int result = INVALID_HANDLE;
   for(int i = 0; i < m_buffers_total; i++)
     {
      if(m_buffers[i] != INVALID_HANDLE)
         continue;
      result = i;
      break;
     }
//---
   if(result == INVALID_HANDLE)
     {
      if(ArrayResize(m_buffers, m_buffers_total + 1) > 0)
        {
         m_buffers_total = ArraySize(m_buffers);
         result = m_buffers_total - 1;
         m_buffers[result] = INVALID_HANDLE;
        }
      else
        {
         static bool first_table_failure = true;
         if(first_table_failure)
           {
            first_table_failure = false;
            PrintFormat("COpenCLMy::AddBufferFromArray table allocation failed: slots=%d elements=%u bytes=%I64d error=%d",
                        m_buffers_total, data_array_count, long(data_array_count)*sizeof(T), GetLastError());
           }
         return result;
        }
     }
//---
   if(!BufferFromArray(result, data, data_array_offset, data_array_count, flags))
     {
      static bool first_device_failure = true;
      if(first_device_failure)
        {
         first_device_failure = false;
         PrintFormat("COpenCLMy::AddBufferFromArray device allocation failed: slot=%d elements=%u bytes=%I64d flags=%u error=%d",
                     result, data_array_count, long(data_array_count)*sizeof(T), flags, GetLastError());
        }
      return INVALID_HANDLE;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int COpenCLMy::AddBuffer(uint size_in_bytes, const uint flags)
  {
   int result = INVALID_HANDLE;
   for(int i = 0; i < m_buffers_total; i++)
     {
      if(m_buffers[i] != INVALID_HANDLE)
         continue;
      result = i;
      break;
     }
//---
   if(result < 0)
     {
      if(ArrayResize(m_buffers, m_buffers_total + 1) > 0)
        {
         m_buffers_total = ArraySize(m_buffers);
         result = m_buffers_total - 1;
         m_buffers[result] = INVALID_HANDLE;
        }
      else
         return result;
     }
//---
   if(!BufferCreate(result, size_in_bytes, flags))
      return INVALID_HANDLE;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayer::CLayer(uint outputs = 0, int handle = -1, COpenCLMy *opencl = NULL)
  {
   iOutputs = outputs;
   iFileHandle = handle;
   OpenCL = opencl;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLayer::Load(const int file_handle)
  {
   iFileHandle = file_handle;
   if(!CArrayObj::Load(file_handle))
     {
      for(int i = 0; i < int(m_data.Size()); i++)
        {
         DeleteObj(m_data[i]);
        }
      ReturnFalse;
     }
   if(CheckPointer(m_data[0]) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *ocl = NULL;
   CNeuronBase    *cpu = NULL;
   if(m_data[0].Type() >= defNeuronBaseOCL)
     {
      ocl = m_data[0];
      iOutputs = ocl.getConnections();
     }
   else
     {
      cpu = m_data[0];
      iOutputs = cpu.getConnections().Total();
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLayer::SetOpenCL(COpenCLMy *obj)
  {
   if(OpenCL == obj)
      return;
   OpenCL = obj;
//---
   if(!obj || Total() <= 0)
      return;
   for(int i = 0; i < Total(); i++)
     {
      CNeuronBaseOCL *neuron = m_data[i];
      if(!neuron)
         continue;
      neuron.SetOpenCL(obj);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLayer::TrainMode(bool flag)
  {
   for(int i = 0; i < Total(); i++)
     {
      CNeuronBaseOCL *neuron = m_data[i];
      if(!neuron)
         continue;
      neuron.TrainMode(flag);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNet::~CNet(void)
  {
   DeleteObj(layers);
   if(bOwnOpenCL && CheckPointer(opencl) != POINTER_INVALID)
     {
      opencl.Shutdown();
      delete opencl;
     }
   opencl = NULL;
   bOwnOpenCL = false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat::CBufferFloat(void)  : m_myIndex(INVALID_HANDLE)
  {
   OpenCL = NULL;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat::~CBufferFloat(void)
  {
   if(CheckPointer(OpenCL) != POINTER_INVALID && m_myIndex >= 0)
     {
      if(OpenCL.BufferFree(m_myIndex))
        {
         m_myIndex = -1;
         OpenCL = NULL;
        }
     }
   Shutdown();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferCreate(COpenCL *opencl)
  {
   if(CheckPointer(opencl) == POINTER_INVALID)
      ReturnFalse;
//--- Empty buffers are valid for parameter-free layers and transient holders.
//--- Do not ask OpenCL to allocate a zero-byte device object.
   if(m_data_total <= 0)
     {
      if(CheckPointer(OpenCL) != POINTER_INVALID && m_myIndex >= 0)
         OpenCL.BufferFree(m_myIndex);
      m_myIndex = INVALID_HANDLE;
      OpenCL = NULL;
      return(true);
     }
//---
   if(opencl == OpenCL && m_myIndex >= 0)
     {
      const ulong expected_bytes = (ulong)m_data_total * sizeof(float);
      const ulong actual_bytes = ((COpenCLMy*)opencl).GetBufferSize(m_myIndex);
      if(actual_bytes == expected_bytes && BufferWrite())
         return true;
     }
//---
   if(CheckPointer(OpenCL) != POINTER_INVALID && m_myIndex >= 0)
     {
      OpenCL.BufferFree(m_myIndex);
      m_myIndex = -1;
      OpenCL = NULL;
     }
//---
   const int index = ((COpenCLMy*)opencl).AddBufferFromArray(m_data, 0, m_data_total, CL_MEM_READ_WRITE);
   if(index < 0)
     {
      static bool first_failure = true;
      if(first_failure)
        {
         first_failure = false;
         PrintFormat("CBufferFloat::BufferCreate allocation: floats=%d bytes=%I64d error=%d",
                     m_data_total, long(m_data_total)*sizeof(float), GetLastError());
        }
      ReturnFalse;
     }
   m_myIndex = index;
   OpenCL = opencl;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferFree(void)
  {
   if(CheckPointer(OpenCL) != POINTER_INVALID && m_myIndex >= 0)
     {
      if(OpenCL.BufferFree(m_myIndex))
        {
         m_myIndex = INVALID_HANDLE;
         OpenCL = NULL;
         return true;
        }
     }
//---
   return false;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferRead(void)
  {
   if(m_data_total <= 0)
      return(true);
   if(CheckPointer(OpenCL) == POINTER_INVALID || m_myIndex < 0)
      ReturnFalse;
//---
   if(!OpenCL.BufferRead(m_myIndex, m_data, 0, 0, m_data_total))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferWrite(void)
  {
   if(m_data_total <= 0)
      return(true);
   if(CheckPointer(OpenCL) == POINTER_INVALID || m_myIndex < 0)
      ReturnFalse;
//---
   if(!OpenCL.BufferWrite(m_myIndex, m_data, 0, 0, m_data_total))
      ReturnFalse;
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferInit(uint count, float value)
  {
   if(m_data_total != count && !!OpenCL)
      BufferFree();
   if(m_data_max < (int)count && !Reserve((int)count - m_data_total))
      ReturnFalse;
   m_data_total = (int)fmin(ArrayInitialize(m_data, value), count);
//---
   return m_data_total == count;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
int CBufferFloat::GetData(T &values[])
  {
   if(!!OpenCL && !BufferRead())
      ReturnFalse;
   return ArrayCopy(values, m_data, 0, 0, m_data_total);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
int CBufferFloat::GetData(vector<T> &values)
  {
   if(!!OpenCL && !BufferRead())
      ReturnFalse;
   values.Assign(m_data);
   if(!values.Resize(m_data_total))
      ReturnFalse;
   return (int)values.Size();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
int CBufferFloat::GetData(matrix<T> &values)
  {
   if(!!OpenCL && !BufferRead())
      ReturnFalse;
   ulong r = values.Rows();
   ulong c = values.Cols();
   for(ulong ir = 0; ir < r; ir++)
     {
      for(ulong ic = 0; ic < c; ic++)
        {
         int pos = int(ir * c + ic);
         if(pos >= m_data_total)
            break;
         values[ir, ic] = (T)m_data[pos];
        }
     }
   return MathMin(int(r * c), m_data_total);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CBufferFloat::GetData(CArrayFloat *values)
  {
   if(!values)
      return -1;
   if(!BufferRead())
      return -1;
   if(!values.AssignArray(GetPointer(this)))
      return -1;
   return m_data_total;
  }
//+------------------------------------------------------------------+
//| Assignment (copying) of another array                            |
//+------------------------------------------------------------------+
template<typename T>
bool CBufferFloat::AssignArray(const T &src[])
  {
   int num = ArraySize(src);
//--- check/reserve elements of array
   Clear();
   if(m_data_max < num)
     {
      if(!Reserve(num))
         return(false);
     }
   else
      Resize(num);
//--- copy array
   for(int i = 0; i < num; i++)
     {
      m_data[i] = float(src[i]);
      m_data_total++;
     }
   m_sort_mode = -1;
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Assignment (copying) of another array                            |
//+------------------------------------------------------------------+
bool CBufferFloat::AssignArray(const CArrayFloat *obj)
  {
   if(!obj)
      ReturnFalse;
   int num = obj.Total();
//--- check/reserve elements of array
   Clear();
   if(m_data_max < num)
     {
      if(!Reserve(num))
         return(false);
     }
   else
      Resize(num);
//--- copy array
   for(int i = 0; i < num; i++)
     {
      m_data[i] = (float)obj.At(i);
      m_data_total++;
     }
   m_sort_mode = -1;
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Assignment (copying) of another array                            |
//+------------------------------------------------------------------+
bool CBufferFloat::AssignArray(const matrix<float> &src)
  {
   int num = (int)(src.Rows() * src.Cols());
//--- check/reserve elements of array
   Clear();
   if(m_data_max < num)
     {
      if(!Reserve(num))
         return(false);
     }
   else
      Resize(num);
//--- copy array
   for(int i = 0; i < num; i++)
     {
      m_data[i] = src.Flat(i);
      m_data_total++;
     }
   m_sort_mode = -1;
//---
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Assignment (copying) of another array                            |
//+------------------------------------------------------------------+
bool CBufferFloat::AssignArray(const vector<float> &src)
  {
   int num = (int)(src.Size());
//--- check/reserve elements of array
   Clear();
   if(m_data_max < num)
     {
      if(!Reserve(num))
         return(false);
     }
   else
      Resize(num);
//--- copy array
   for(int i = 0; i < num; i++)
     {
      m_data[i] = src[i];
      m_data_total++;
     }
   m_sort_mode = -1;
//---
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Adding (copying) of another array                                |
//+------------------------------------------------------------------+
bool CBufferFloat::AddArray(const vector<float> &src)
  {
   int num = (int)(src.Size());
//--- check/reserve elements of array
   if(m_data_max < num + m_data_total)
     {
      if(!Reserve(num + m_data_total))
         return(false);
     }
   else
      Resize(num + m_data_total);
//--- copy array
   for(int i = 0; i < num; i++)
      m_data[m_data_total + i] = src[i];
   m_data_total += num;
   m_sort_mode = -1;
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//| Adding (copying) of another array                                |
//+------------------------------------------------------------------+
template<typename T>
bool CBufferFloat::AddArray(const T &src[])
  {
   int num = ArraySize(src);
//--- check/reserve elements of array
   if(m_data_max < num + m_data_total)
     {
      if(!Reserve(num + m_data_total))
         return(false);
     }
   else
      Resize(num + m_data_total);
//--- copy array
   for(int i = 0; i < num; i++)
      m_data[m_data_total + i] = (float)src[i];
   m_data_total += num;
   m_sort_mode = -1;
//--- successful
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::Save(const int file_handle)
  {
   if(CheckPointer(OpenCL) != POINTER_INVALID && m_myIndex >= 0)
      if(!BufferRead())
         ReturnFalse;
//---
   return CArrayFloat::Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::Load(const int file_handle)
  {
   if(!CArrayFloat::Load(file_handle))
      ReturnFalse;
   if(m_data_total == 0 || !OpenCL)
      return true;
//---
   for(int i = 0; i < m_data_total; i++)
     {
      switch(MathClassify(m_data[i]))
        {
         case FP_INFINITE:
         case FP_NAN:
            m_data[i] = 0;
            break;
        }
     }
   COpenCLMy *temp = OpenCL;
   BufferFree();
   return BufferCreate(temp);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::Fill(const float value)
  {
   ArrayFill(m_data, 0, m_data_total, value);
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::Fill(const CBufferFloat *obj)
  {
   if(!obj)
      ReturnFalse;
//---
   int total = obj.Total();
   for(int i = 0; i < fmin(total, m_data_total); i++)
      m_data[i] = obj.At(i);
   for(int i = total; i < m_data_total; i++)
      m_data[i] = 0;
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool CBufferFloat::Fill(const matrix<T> &matr)
  {
//---
   int total = int(matr.Rows() * matr.Cols());
   for(int i = 0; i < fmin(total, m_data_total); i++)
      m_data[i] = float(matr.Flat(i));
   for(int i = total; i < m_data_total; i++)
      m_data[i] = 0;
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool CBufferFloat::Fill(const vector<T> &vec)
  {
//---
   int total = int(vec.Size());
   for(int i = 0; i < fmin(total, m_data_total); i++)
      m_data[i] = float(vec[i]);
   for(int i = total; i < m_data_total; i++)
      m_data[i] = 0;
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::Random(const float min_value, const float max_value)
  {
//---
   const float delta = max_value - min_value;
   for(int i = 0; i < m_data_total; i++)
      m_data[i] = float(::MathRand()) / 32767.0f * delta + min_value;
   if(!!OpenCL)
     {
      if(!BufferWrite())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBaseOCL::CNeuronBaseOCL(void)   :  alpha(momentum),
   activation(None),
   optimization(ADAM),
   t(1),
   bTrain(true)
  {
   OpenCL = NULL;
   Output = new CBufferFloat();
   PrevOutput = new CBufferFloat();
   Weights = new CBufferFloat();
   DeltaWeights = new CBufferFloat();
   Gradient = new CBufferFloat();
   FirstMomentum = new CBufferFloat();
   SecondMomentum = new CBufferFloat();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBaseOCL::~CNeuronBaseOCL(void)
  {
   DeleteObj(Output);
   DeleteObj(PrevOutput);
   DeleteObj(Weights);
   DeleteObj(DeltaWeights);
   DeleteObj(Gradient);
   DeleteObj(FirstMomentum);
   DeleteObj(SecondMomentum);
   OpenCL = NULL;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(CheckPointer(open_cl) == POINTER_INVALID || numNeurons <= 0)
      ReturnFalse;
   OpenCL = open_cl;
   optimization = optimization_type;
   iBatch = batch;
//---
   if(CheckPointer(Output) == POINTER_INVALID)
     {
      Output = new CBufferFloat();
      if(CheckPointer(Output) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!Output.BufferInit(numNeurons, 1.0))
      ReturnFalse;
   if(!Output.BufferCreate(OpenCL))
      ReturnFalse;
   prev_output.Init(numNeurons);
//---
   if(CheckPointer(PrevOutput) == POINTER_INVALID)
     {
      PrevOutput = new CBufferFloat();
      if(CheckPointer(PrevOutput) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!PrevOutput.BufferInit(numNeurons, 1.0))
      ReturnFalse;
   if(!PrevOutput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(Gradient) == POINTER_INVALID)
     {
      Gradient = new CBufferFloat();
      if(CheckPointer(Gradient) == POINTER_INVALID)
         ReturnFalse;
     }
   if(!Gradient.BufferInit(numNeurons, 0.0))
      ReturnFalse;
   if(!Gradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(numOutputs > 0)
     {
      if(CheckPointer(Weights) == POINTER_INVALID)
        {
         Weights = new CBufferFloat();
         if(CheckPointer(Weights) == POINTER_INVALID)
            ReturnFalse;
        }
      int count = (int)((numNeurons + 1) * numOutputs);
      if(!Weights.Reserve(count))
         ReturnFalse;
      float k = (float)(1 / sqrt(numNeurons + 1));
      for(int i = 0; i < count; i++)
        {
         if(!Weights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
            ReturnFalse;
        }
      if(!Weights.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(optimization == SGD)
        {
         if(CheckPointer(DeltaWeights) == POINTER_INVALID)
           {
            DeltaWeights = new CBufferFloat();
            if(CheckPointer(DeltaWeights) == POINTER_INVALID)
               ReturnFalse;
           }
         if(!DeltaWeights.BufferInit(count, 0))
            ReturnFalse;
         if(!DeltaWeights.BufferCreate(OpenCL))
            ReturnFalse;
         DeleteObj(FirstMomentum);
         DeleteObj(SecondMomentum);
        }
      else
        {
         DeleteObj(DeltaWeights);
         //---
         if(CheckPointer(FirstMomentum) == POINTER_INVALID)
           {
            FirstMomentum = new CBufferFloat();
            if(CheckPointer(FirstMomentum) == POINTER_INVALID)
               ReturnFalse;
           }
         if(!FirstMomentum.BufferInit(count, 0))
            ReturnFalse;
         if(!FirstMomentum.BufferCreate(OpenCL))
            ReturnFalse;
         //---
         if(CheckPointer(SecondMomentum) == POINTER_INVALID)
           {
            SecondMomentum = new CBufferFloat();
            if(CheckPointer(SecondMomentum) == POINTER_INVALID)
               ReturnFalse;
           }
         if(!SecondMomentum.BufferInit((optimization == ADAM_MINI ? numOutputs : count), 0))
            ReturnFalse;
         if(!SecondMomentum.BufferCreate(OpenCL))
            ReturnFalse;
        }
     }
   else
     {
      DeleteObj(Weights);
      DeleteObj(DeltaWeights);
      DeleteObj(FirstMomentum);
      DeleteObj(SecondMomentum);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::FeedForward(CObject *SourceObject, CBufferFloat *SecondInput = NULL)
  {
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = SourceObject;
   return feedForward(temp, SecondInput);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::FeedForwardSecond(CObject *SourceObject, CNeuronBaseOCL *SecondNeuron)
  {
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = SourceObject;
   return feedForwardSecond(temp, SecondNeuron);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   const int kernel = def_k_FeedForward;
   setBuffer(kernel, def_k_ff_matrix_w, NeuronOCL.getWeightsIndex())
   setBuffer(kernel, def_k_ff_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ff_matrix_o, Output.GetIndex())
   setArgument(kernel, def_k_ff_inputs, NeuronOCL.Neurons())
   setArgument(kernel, def_k_ff_activation, (int)activation)
//---
   uint global_work_offset[2] = { 0, 0 };
   uint global_work_size[2] = { Output.Total(),
                                (uint)MathMin((NeuronOCL.Neurons() + 3) / 4, OpenCL.GetMaxLocalSize(1))
                              };
   uint local_work_size[2] = { 1, global_work_size[1] };
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::calcHiddenGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2];
   global_work_size[0] = Neurons();
   global_work_size[1] = (uint)MathMin((NeuronOCL.Neurons() + 3) / 4, OpenCL.GetMaxLocalSize(1));
   uint local_work_size[2] = { 1, global_work_size[1] };;
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_w, getWeightsIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_g, NeuronOCL.getGradientIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_o, getOutputIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_ig, getGradientIndex())
   setArgument(def_k_CalcHiddenGradient, def_k_chg_outputs, NeuronOCL.Neurons())
   setArgument(def_k_CalcHiddenGradient, def_k_chg_activation, Activation())
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecuteLoc(def_k_CalcHiddenGradient, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = NeuronOCL.Neurons();
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_w, NeuronOCL.getWeightsIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_g, getGradientIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_o, NeuronOCL.getOutputIndex())
   setBuffer(def_k_CalcHiddenGradient, def_k_chg_matrix_ig, NeuronOCL.getGradientIndex())
   setArgument(def_k_CalcHiddenGradient, def_k_chg_outputs, Neurons())
   setArgument(def_k_CalcHiddenGradient, def_k_chg_activation, NeuronOCL.Activation())
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_CalcHiddenGradient, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!NeuronOCL.Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::calcOutputGradients(CBufferFloat *target, float &error)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(target) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   if(error <= 0)
      error = 1;
   global_work_size[0] = Neurons();
   if(target.GetOpenCL() != OpenCL)
     {
      if(!Gradient.Fill(target))
         ReturnFalse;
      setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_t, getGradientIndex());
     }
   else
     {
      setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_t, target.GetIndex());
     }
   setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_o, getOutputIndex());
   setBuffer(def_k_CalcOutputGradient, def_k_cog_matrix_ig, getGradientIndex());
   setArgument(def_k_CalcOutputGradient, def_k_cog_activation, (int)activation);
   setArgument(def_k_CalcOutputGradient, def_k_cog_error, error);
   ResetLastError();
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_CalcOutputGradient, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::calcAlignmentGradient(const CNeuronBaseOCL *neuron, const bool add = false)
  {
   if(!OpenCL || !neuron || neuron.Activation() != activation)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   int kernel = def_k_CalcAlignmentGradient;
   ResetLastError();
   setBuffer(kernel, def_k_aliggr_matrix_o1, getOutputIndex())
   setBuffer(kernel, def_k_aliggr_matrix_o2, neuron.getOutputIndex())
   setBuffer(kernel, def_k_aliggr_matrix_g1, getGradientIndex())
   setBuffer(kernel, def_k_aliggr_matrix_g2, neuron.getGradientIndex())
   setArgument(kernel, def_k_aliggr_activation, Activation())
   setArgument(kernel, def_k_aliggr_add, (int)add)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!neuron.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2], local_work_size[2];
   global_work_size[0] = Neurons();
   global_work_size[1] = NeuronOCL.Neurons() + 1;
   uint rest = 0;
   float lt = lr;
   switch(NeuronOCL.Optimization())
     {
      case SGD:
         setBuffer(def_k_UpdateWeightsMomentum, def_k_uwm_matrix_w, NeuronOCL.getWeightsIndex())
         setBuffer(def_k_UpdateWeightsMomentum, def_k_uwm_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsMomentum, def_k_uwm_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsMomentum, def_k_uwm_matrix_dw, NeuronOCL.getDeltaWeightsIndex())
         setArgument(def_k_UpdateWeightsMomentum, def_k_uwm_inputs, NeuronOCL.Neurons())
         setArgument(def_k_UpdateWeightsMomentum, def_k_uwm_learning_rates, lr)
         setArgument(def_k_UpdateWeightsMomentum, def_k_uwm_momentum, alpha)
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsMomentum, global_work_offset, global_work_size)
         break;
      case ADAM:
         setBuffer(def_k_UpdateWeightsAdam, def_k_uwa_matrix_w, NeuronOCL.getWeightsIndex())
         setBuffer(def_k_UpdateWeightsAdam, def_k_uwa_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsAdam, def_k_uwa_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsAdam, def_k_uwa_matrix_m, NeuronOCL.getFirstMomentumIndex())
         setBuffer(def_k_UpdateWeightsAdam, def_k_uwa_matrix_v, NeuronOCL.getSecondMomentumIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsAdam, def_k_uwa_inputs, NeuronOCL.Neurons())
         setArgument(def_k_UpdateWeightsAdam, def_k_uwa_l, lt)
         setArgument(def_k_UpdateWeightsAdam, def_k_uwa_b1, b1)
         setArgument(def_k_UpdateWeightsAdam, def_k_uwa_b2, b2)
         ////Comment(com+"\n UpdateWeightsAdam");
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsAdam, global_work_offset, global_work_size)
         t++;
         break;
      case LS:
         setBuffer(def_k_UpdateWeightsLS, def_k_uwls_matrix_w, NeuronOCL.getWeightsIndex())
         setBuffer(def_k_UpdateWeightsLS, def_k_uwls_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsLS, def_k_uwls_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsLS, def_k_uwls_matrix_xg, NeuronOCL.getFirstMomentumIndex())
         setBuffer(def_k_UpdateWeightsLS, def_k_uwls_matrix_xx, NeuronOCL.getSecondMomentumIndex())
         setArgument(def_k_UpdateWeightsLS, def_k_uwls_inputs, NeuronOCL.Neurons())
         setArgument(def_k_UpdateWeightsLS, def_k_uwls_l, lr)
         setArgument(def_k_UpdateWeightsLS, def_k_uwls_update, (int)(t >= iBatch))
         rest = global_work_size[1] % 4;
         global_work_size[1] = (global_work_size[1] - rest) / 4 + (rest > 0 ? 1 : 0);
         ////Comment(com+"\n UpdateWeightsAdam");
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsLS, global_work_offset, global_work_size)
         if(t >= iBatch)
            t = 0;
         else
            t++;
         break;
      case ADAM_MINI:
         setBuffer(def_k_UpdateWeightsAdamMini, def_k_wuam_matrix_w, NeuronOCL.getWeightsIndex())
         setBuffer(def_k_UpdateWeightsAdamMini, def_k_wuam_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsAdamMini, def_k_wuam_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsAdamMini, def_k_wuam_matrix_m, NeuronOCL.getFirstMomentumIndex())
         setBuffer(def_k_UpdateWeightsAdamMini, def_k_wuam_matrix_v, NeuronOCL.getSecondMomentumIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsAdamMini, def_k_wuam_l, lt)
         setArgument(def_k_UpdateWeightsAdamMini, def_k_wuam_b1, b1)
         setArgument(def_k_UpdateWeightsAdamMini, def_k_wuam_b2, b2)
         global_work_size[0] = NeuronOCL.Neurons() + 1;
         global_work_size[1] = Neurons();
         local_work_size[0] = global_work_size[0];
         local_work_size[1] = 1;
         ////Comment(com+"\n UpdateWeightsAdam");
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecuteLoc(def_k_UpdateWeightsAdamMini, global_work_offset, global_work_size, local_work_size)
         t++;
         break;
      default:
         ReturnFalse;
         break;
     }
#ifdef  _DEBUG
   if(!NeuronOCL.getWeights().BufferRead())
      ReturnFalse;
#endif
//---
     {
      uint global_work_offset_n[] = {0};
      uint global_work_size_n[] = {Neurons()};
      setBuffer(def_k_NormalizeWeights, def_k_norm_buffer, NeuronOCL.getWeightsIndex())
      setArgument(def_k_NormalizeWeights, def_k_norm_dimension, int(NeuronOCL.Neurons() + 1))
      kernelExecute(def_k_NormalizeWeights, global_work_offset_n, global_work_size_n)
     }
#ifdef  _DEBUG
   if(!NeuronOCL.getWeights().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::CalcHiddenGradients(CObject *TargetObject, CBufferFloat *SecondInput, CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None)
  {
//---
   if(CheckPointer(TargetObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = TargetObject;
//---
   switch(TargetObject.Type())
     {
      case defNeuronBaseOCL:
         return calcHiddenGradients(temp);
      case defNeuronMultiModels:
         return temp.CalcHiddenGradients(GetPointer(this));
      default:
         return temp.calcInputGradients(GetPointer(this), SecondInput, SecondGradient, SecondActivation);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::CalcHiddenGradientsSecond(CObject *TargetObject, CNeuronBaseOCL *SecondNeuron)
  {
   if(CheckPointer(TargetObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = TargetObject;
   switch(TargetObject.Type())
     {
      case defNeuronBaseOCL:
         return calcHiddenGradients(temp);
      case defNeuronMultiModels:
         return temp.CalcHiddenGradients(GetPointer(this));
      default:
         return temp.calcInputGradientsSecond(GetPointer(this), SecondNeuron);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::UpdateInputWeights(CObject *SourceObject, CBufferFloat *SecondInput = NULL)
  {
//---
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = SourceObject;
   if(!bTrain && !temp.TrainMode())
      return true;
   if(Type() == defNeuronSoftMaxOCL)
      return true;
//---
   return updateInputWeights(temp, SecondInput);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::UpdateInputWeightsSecond(CObject *SourceObject, CNeuronBaseOCL *SecondNeuron)
  {
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      ReturnFalse;
//---
   CNeuronBaseOCL *temp = SourceObject;
   if(!bTrain && !temp.TrainMode())
      return true;
   if(Type() == defNeuronSoftMaxOCL)
      return true;
//---
   return updateInputWeightsSecond(temp, SecondNeuron);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Save(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, Type()) < INT_VALUE)
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)activation, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)optimization, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iBatch, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)t, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)bTrain, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   if(CheckPointer(Output) == POINTER_INVALID || !Output.Save(file_handle))
      ReturnFalse;
   if(CheckPointer(PrevOutput) == POINTER_INVALID || !PrevOutput.Save(file_handle))
      ReturnFalse;
   if(CheckPointer(Gradient) == POINTER_INVALID || !Gradient.Save(file_handle))
      ReturnFalse;
//---
   if(CheckPointer(Weights) == POINTER_INVALID)
     {
      FileWriteInteger(file_handle, 0);
      return true;
     }
   else
      FileWriteInteger(file_handle, 1);
//---
   if(CheckPointer(Weights) == POINTER_INVALID || !Weights.Save(file_handle))
      ReturnFalse;
   if(optimization == SGD)
     {
      if(CheckPointer(DeltaWeights) == POINTER_INVALID || !DeltaWeights.Save(file_handle))
         ReturnFalse;
     }
   else
     {
      if(CheckPointer(FirstMomentum) == POINTER_INVALID || !FirstMomentum.Save(file_handle))
         ReturnFalse;
      if(CheckPointer(SecondMomentum) == POINTER_INVALID || !SecondMomentum.Save(file_handle))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Load(const int file_handle)
  {
   if(file_handle == INVALID_HANDLE)
      ReturnFalse;
//---
   activation = (ENUM_ACTIVATION)FileReadInteger(file_handle, INT_VALUE);
   optimization = (ENUM_OPTIMIZATION)FileReadInteger(file_handle, INT_VALUE);
   iBatch = (uint)FileReadInteger(file_handle, INT_VALUE);
   t = (uint)FileReadInteger(file_handle, INT_VALUE);
   bTrain = (bool)FileReadInteger(file_handle, INT_VALUE);
   if(CheckPointer(Output) == POINTER_INVALID)
     {
      Output = new CBufferFloat();
      if(CheckPointer(Output) == POINTER_INVALID)
         ReturnFalse;
     }
   if(Output.GetIndex() >= 0)
      Output.BufferFree();
   if(!Output.Load(file_handle))
      ReturnFalse;
   if(Output.Total() > 0 && !Output.BufferCreate(OpenCL))
      ReturnFalse;
   prev_output.Init(Output.Total());
//---
   if(CheckPointer(PrevOutput) == POINTER_INVALID)
     {
      PrevOutput = new CBufferFloat();
      if(CheckPointer(PrevOutput) == POINTER_INVALID)
         ReturnFalse;
     }
   if(PrevOutput.GetIndex() >= 0)
      PrevOutput.BufferFree();
   if(!PrevOutput.Load(file_handle))
      ReturnFalse;
   if(PrevOutput.Total() > 0 && !PrevOutput.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(CheckPointer(Gradient) == POINTER_INVALID)
     {
      Gradient = new CBufferFloat();
      if(CheckPointer(Gradient) == POINTER_INVALID)
         ReturnFalse;
     }
   if(Gradient.GetIndex() >= 0)
      Gradient.BufferFree();
   if(!Gradient.Load(file_handle))
      ReturnFalse;
   if(Gradient.Total() > 0 && !Gradient.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(FileReadInteger(file_handle) == 0)
     {
      DeleteObj(Weights);
      DeleteObj(DeltaWeights);
      DeleteObj(FirstMomentum);
      DeleteObj(SecondMomentum);
      return true;
     }
//---
   if(CheckPointer(Weights) == POINTER_INVALID)
     {
      Weights = new CBufferFloat();
      if(CheckPointer(Weights) == POINTER_INVALID)
         ReturnFalse;
     }
   if(Weights.GetIndex() >= 0)
      Weights.BufferFree();
   if(!Weights.Load(file_handle))
      ReturnFalse;
   if(Weights.Total() > 0 && !Weights.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(optimization == SGD)
     {
      if(CheckPointer(DeltaWeights) == POINTER_INVALID)
        {
         DeltaWeights = new CBufferFloat();
         if(CheckPointer(DeltaWeights) == POINTER_INVALID)
            ReturnFalse;
        }
      if(DeltaWeights.GetIndex() >= 0)
         DeltaWeights.BufferFree();
      if(!DeltaWeights.Load(file_handle))
         ReturnFalse;
      if(DeltaWeights.Total() > 0 && !DeltaWeights.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      if(CheckPointer(FirstMomentum) == POINTER_INVALID)
        {
         FirstMomentum = new CBufferFloat();
         if(CheckPointer(FirstMomentum) == POINTER_INVALID)
            ReturnFalse;
        }
      if(FirstMomentum.GetIndex() >= 0)
         FirstMomentum.BufferFree();
      if(!FirstMomentum.Load(file_handle))
         ReturnFalse;
      if(FirstMomentum.Total() > 0 && !FirstMomentum.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(CheckPointer(SecondMomentum) == POINTER_INVALID)
        {
         SecondMomentum = new CBufferFloat();
         if(CheckPointer(SecondMomentum) == POINTER_INVALID)
            ReturnFalse;
        }
      if(SecondMomentum.GetIndex() >= 0)
         SecondMomentum.BufferFree();
      if(!SecondMomentum.Load(file_handle))
         ReturnFalse;
      if(SecondMomentum.Total() > 0 && !SecondMomentum.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronBaseOCL::GetLayerInfo(void)
  {
   CLayerDescription* result = new CLayerDescription();
   if(!result)
      return result;
//---
   result.type = Type();
   result.count = Output.Total();
   result.optimization = optimization;
   result.activation = activation;
   result.batch = (int)(optimization == LS ? iBatch : 1);
   result.layers = 1;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::numOutputs(const uint outputs, ENUM_OPTIMIZATION optimization_type)
  {
   if(outputs > 0)
     {
      if(CheckPointer(Weights) == POINTER_INVALID)
        {
         Weights = new CBufferFloat();
         if(CheckPointer(Weights) == POINTER_INVALID)
            ReturnFalse;
        }
      Weights.BufferFree();
      Weights.Clear();
      int count = (int)((Output.Total() + 1) * outputs);
      if(!Weights.Reserve(count))
         ReturnFalse;
      float k = (float)(1 / sqrt(Output.Total() + 1));
      for(int i = 0; i < count; i++)
        {
         if(!Weights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
            ReturnFalse;
        }
      if(!Weights.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(optimization_type == SGD)
        {
         if(CheckPointer(DeltaWeights) == POINTER_INVALID)
           {
            DeltaWeights = new CBufferFloat();
            if(CheckPointer(DeltaWeights) == POINTER_INVALID)
               ReturnFalse;
           }
         DeltaWeights.BufferFree();
         if(!DeltaWeights.BufferInit(count, 0))
            ReturnFalse;
         if(!DeltaWeights.BufferCreate(OpenCL))
            ReturnFalse;
         DeleteObj(FirstMomentum);
         DeleteObj(SecondMomentum);
        }
      else
        {
         DeleteObj(DeltaWeights);
         //---
         if(CheckPointer(FirstMomentum) == POINTER_INVALID)
           {
            FirstMomentum = new CBufferFloat();
            if(CheckPointer(FirstMomentum) == POINTER_INVALID)
               ReturnFalse;
           }
         FirstMomentum.BufferFree();
         if(!FirstMomentum.BufferInit(count, 0))
            ReturnFalse;
         if(!FirstMomentum.BufferCreate(OpenCL))
            ReturnFalse;
         //---
         if(CheckPointer(SecondMomentum) == POINTER_INVALID)
           {
            SecondMomentum = new CBufferFloat();
            if(CheckPointer(SecondMomentum) == POINTER_INVALID)
               ReturnFalse;
           }
         SecondMomentum.BufferFree();
         if(!SecondMomentum.BufferInit(count, 0))
            ReturnFalse;
         if(!SecondMomentum.BufferCreate(OpenCL))
            ReturnFalse;
        }
     }
   else
     {
      DeleteObj(Weights);
      DeleteObj(DeltaWeights);
      DeleteObj(FirstMomentum);
      DeleteObj(SecondMomentum);
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBaseOCL::SetOpenCL(COpenCLMy *obj)
  {
   if(OpenCL == obj)
      return;
   if(!obj)
      return;
   OpenCL = obj;
   if(!!Output && Output.Total() > 0)
      Output.BufferCreate(OpenCL);             ///< Buffer of Output tensor
   if(!!PrevOutput && PrevOutput.Total() > 0)
      PrevOutput.BufferCreate(OpenCL);         ///< Buffer of previous iteration Output tensor
   if(!!Weights && Weights.Total() > 0)
      Weights.BufferCreate(OpenCL);            ///< Buffer of weights matrix
   if(!!DeltaWeights && DeltaWeights.Total() > 0)
      DeltaWeights.BufferCreate(OpenCL);       ///< Buffer of last delta weights matrix (#SGD)
   if(!!Gradient && Gradient.Total() > 0)
      Gradient.BufferCreate(OpenCL);           ///< Buffer of gradient tensor
   if(!!FirstMomentum && FirstMomentum.Total() > 0)
      FirstMomentum.BufferCreate(OpenCL);      ///< Buffer of first momentum matrix (#ADAM)
   if(!!SecondMomentum && SecondMomentum.Total() > 0)
      SecondMomentum.BufferCreate(OpenCL);     ///< Buffer of second momentum matrix (#ADAM)
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SetGradient(CBufferFloat *buffer, bool delete_prev = true)
  {
   if(Gradient == buffer)
      return true;
//---
   if(!!Gradient)
     {
      if(Output.Total() > buffer.Total())
         ReturnFalse;
      if(delete_prev)
         DeleteObj(Gradient);
     }
   Gradient = buffer;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SetOutput(CBufferFloat *buffer, bool delete_prev = true)
  {
   if(Output == buffer)
      return true;
//---
   if(!!Output)
     {
      if(Output.Total() > buffer.Total())
         ReturnFalse;
      if(delete_prev)
         DeleteObj(Output);
     }
   Output = buffer;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProofOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, int window, int step, int units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count, optimization_type, batch))
      ReturnFalse;
//---
   iWindow = window;
   iStep = step;
   activation = None;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronProofOCL::~CNeuronProofOCL(void)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProofOCL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Output.Total();
   setBuffer(def_k_FeedForwardProof, def_k_ffp_matrix_i, NeuronOCL.getOutputIndex());
   setBuffer(def_k_FeedForwardProof, def_k_ffp_matrix_o, Output.GetIndex());
   setArgument(def_k_FeedForwardProof, def_k_ffp_inputs, NeuronOCL.Neurons());
   setArgument(def_k_FeedForwardProof, def_k_ffp_window, (int)iWindow);
   setArgument(def_k_FeedForwardProof, def_k_ffp_step, (int)iStep);
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_FeedForwardProof, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProofOCL::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = NeuronOCL.Neurons();
   setBuffer(def_k_CalcInputGradientProof, def_k_cigp_matrix_i, NeuronOCL.getOutputIndex());
   setBuffer(def_k_CalcInputGradientProof, def_k_cigp_matrix_o, Output.GetIndex());
   setBuffer(def_k_CalcInputGradientProof, def_k_cigp_matrix_g, Gradient.GetIndex());
   setBuffer(def_k_CalcInputGradientProof, def_k_cigp_matrix_ig, NeuronOCL.getGradientIndex());
   setArgument(def_k_CalcInputGradientProof, def_k_cigp_outputs, Output.Total());
   setArgument(def_k_CalcInputGradientProof, def_k_cigp_window, (int)iWindow);
   setArgument(def_k_CalcInputGradientProof, def_k_cigp_step, (int)iStep);
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_CalcInputGradientProof, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronProofOCL::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
   result.window = (int)iWindow;
   result.step = (int)iStep;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronConvOCL::~CNeuronConvOCL(void)
  {
   DeleteObj(WeightsConv);
   DeleteObj(DeltaWeightsConv);
   DeleteObj(FirstMomentumConv);
   DeleteObj(SecondMomentumConv);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window_in, uint step, uint window_out, uint units_count, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window_in, step, window_out, units_count, 1, optimization_type, batch);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint window_in, uint step, uint window_out, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if((optimization == LS && batch <= 0))
      ReturnFalse;
   if(!CNeuronProofOCL::Init(numOutputs, myIndex, open_cl, window_in, step, units_count * window_out * variables, optimization_type, batch))
      ReturnFalse;
//---
   iWindowOut = window_out;
   iVariables = variables;
//---
   if(CheckPointer(WeightsConv) == POINTER_INVALID)
     {
      WeightsConv = new CBufferFloat();
      if(CheckPointer(WeightsConv) == POINTER_INVALID)
         ReturnFalse;
     }
   int count = (int)((iWindow + 1) * iWindowOut * iVariables);
   if(!WeightsConv.Reserve(count))
      ReturnFalse;
   float k = (float)(1 / sqrt(iWindow + 1));
   for(int i = 0; i < count; i++)
     {
      if(!WeightsConv.Add((GenerateWeight() * 2 * k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!WeightsConv.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(optimization == SGD)
     {
      if(CheckPointer(DeltaWeightsConv) == POINTER_INVALID)
        {
         DeltaWeightsConv = new CBufferFloat();
         if(CheckPointer(DeltaWeightsConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!DeltaWeightsConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!DeltaWeightsConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      if(CheckPointer(FirstMomentumConv) == POINTER_INVALID)
        {
         FirstMomentumConv = new CBufferFloat();
         if(CheckPointer(FirstMomentumConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!FirstMomentumConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!FirstMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(CheckPointer(SecondMomentumConv) == POINTER_INVALID)
        {
         SecondMomentumConv = new CBufferFloat();
         if(CheckPointer(SecondMomentumConv) == POINTER_INVALID)
            ReturnFalse;
        }
      if(!SecondMomentumConv.BufferInit(count, 0.0))
         ReturnFalse;
      if(!SecondMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = { Output.Total() / (iWindowOut * iVariables),
                                iWindowOut, iVariables
                              };
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_w, WeightsConv.GetIndex());
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_i, NeuronOCL.getOutputIndex());
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_o, Output.GetIndex());
   setArgument(def_k_FeedForwardConv, def_k_ffc_inputs, NeuronOCL.Neurons() / iVariables);
   setArgument(def_k_FeedForwardConv, def_k_ffc_step, (int)iStep);
   setArgument(def_k_FeedForwardConv, def_k_ffc_window_in, (int)iWindow);
   setArgument(def_k_FeedForwardConv, def_k_ffс_window_out, (int)iWindowOut);
   setArgument(def_k_FeedForwardConv, def_k_ffc_activation, (int)activation);
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_FeedForwardConv, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//float delta = MathAbs((temp - prev_output)).Sum();
//prev_output = temp;
//string error;
//CLGetInfoString(OpenCL.GetContext(), CL_ERROR_DESCRIPTION, error);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Direct non-owning device-buffer input                            |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::FeedForwardBuffer(CBufferFloat *source)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(source) == POINTER_INVALID ||
      source.GetIndex() < 0 || iWindowOut == 0 || iVariables == 0)
      ReturnFalse;
   const uint units = Output.Total() / (iWindowOut * iVariables);
   const uint expected = (units == 0 ? 0 : ((units - 1) * iStep + iWindow) * iVariables);
   if(units == 0 || source.Total() != (int)expected)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {units, iWindowOut, iVariables};
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_w, WeightsConv.GetIndex());
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_i, source.GetIndex());
   setBuffer(def_k_FeedForwardConv, def_k_ffc_matrix_o, Output.GetIndex());
   setArgument(def_k_FeedForwardConv, def_k_ffc_inputs, source.Total() / (int)iVariables);
   setArgument(def_k_FeedForwardConv, def_k_ffc_step, (int)iStep);
   setArgument(def_k_FeedForwardConv, def_k_ffc_window_in, (int)iWindow);
   setArgument(def_k_FeedForwardConv, def_k_ffс_window_out, (int)iWindowOut);
   setArgument(def_k_FeedForwardConv, def_k_ffc_activation, (int)activation);
   kernelExecute(def_k_FeedForwardConv, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2];
   global_work_size[0] = NeuronOCL.Neurons() / iVariables;
   global_work_size[1] = iVariables;
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_w, WeightsConv.GetIndex());
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_g, Gradient.GetIndex());
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_o, NeuronOCL.getOutputIndex());
   setBuffer(def_k_CalcHiddenGradientConv, def_k_chgc_matrix_ig, NeuronOCL.getGradientIndex());
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_outputs, Neurons() / iVariables);
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_step, (int)iStep);
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_window_in, (int)iWindow);
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_window_out, (int)iWindowOut);
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_activation, (int)NeuronOCL.Activation());
   setArgument(def_k_CalcHiddenGradientConv, def_k_chgc_shift_out, (int)0);
//Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
   kernelExecute(def_k_CalcHiddenGradientConv, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = { 0 };
   uint global_work_size[1] = { WeightsConv.Total() };
   uint global_work_offset_am[3] = { 0, 0, 0 };
   uint global_work_size_am[3] = { iWindow + 1, iWindowOut, iVariables };
   uint local_work_size_am[3] = { global_work_size_am[0], 1, 1 };
   float lt = 0;
   switch(optimization)
     {
      case SGD:
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_w, WeightsConv.GetIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_g, getGradientIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_i, NeuronOCL.getOutputIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_dw, DeltaWeightsConv.GetIndex());
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_inputs, NeuronOCL.Neurons() / iVariables);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_learning_rates, lr);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_momentum, alpha);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_window_in, (int)iWindow);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_window_out, (int)iWindowOut);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_step, (int)iStep);
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsConvMomentum, global_work_offset, global_work_size)
         break;
      case ADAM:
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_m, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_v, SecondMomentumConv.GetIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_inputs, NeuronOCL.Neurons() / iVariables)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_l, lt)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_b1, b1)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_b2, b2)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_window_in, (int)iWindow)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_window_out, (int)iWindowOut)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_step, (int)iStep)
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsConvAdam, global_work_offset, global_work_size)
         t++;
         break;
      case LS:
         global_work_size[0] = iWindow + 1;
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_xg, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_xx, SecondMomentumConv.GetIndex())
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwls_inputs, NeuronOCL.Neurons())
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_l, lr)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_update, (int)(t >= iBatch))
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_window_in, (int)iWindow)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_window_out, (int)iWindowOut)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_step, (int)iStep)
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecute(def_k_UpdateWeightsConvLS, global_work_offset, global_work_size)
         if(t >= iBatch)
            t = 0;
         else
            t++;
         break;
      case ADAM_MINI:
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_i, NeuronOCL.getOutputIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_m, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_v, SecondMomentumConv.GetIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_inputs, NeuronOCL.Neurons() / iVariables)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_l, lt)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_b1, b1)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_b2, b2)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_step, (int)iStep)
         ResetLastError();
         //Comment(com+"\n "+(string)__LINE__+"-"__FUNCTION__);
         kernelExecuteLoc(def_k_UpdateWeightsConvAdamMini, global_work_offset_am, global_work_size_am, local_work_size_am)
         t++;
         break;
      default:
         ReturnFalse;
         break;
     }
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
   return true;
  }
//+------------------------------------------------------------------+
//| Direct non-owning device-buffer input weight update              |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::UpdateInputWeightsBuffer(CBufferFloat *source)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(source) == POINTER_INVALID ||
      source.GetIndex() < 0 || iWindowOut == 0 || iVariables == 0)
      ReturnFalse;
   const uint units = Output.Total() / (iWindowOut * iVariables);
   const uint expected = (units == 0 ? 0 : ((units - 1) * iStep + iWindow) * iVariables);
   if(units == 0 || source.Total() != (int)expected)
      ReturnFalse;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {WeightsConv.Total()};
   uint global_work_offset_am[3] = {0, 0, 0};
   uint global_work_size_am[3] = {iWindow + 1, iWindowOut, iVariables};
   uint local_work_size_am[3] = {global_work_size_am[0], 1, 1};
   const int inputs = source.Total() / (int)iVariables;
   float lt = 0;
   switch(optimization)
     {
      case SGD:
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_w, WeightsConv.GetIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_g, getGradientIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_i, source.GetIndex());
         setBuffer(def_k_UpdateWeightsConvMomentum, def_k_uwcm_matrix_dw, DeltaWeightsConv.GetIndex());
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_inputs, inputs);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_learning_rates, lr);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_momentum, alpha);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_window_in, (int)iWindow);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_window_out, (int)iWindowOut);
         setArgument(def_k_UpdateWeightsConvMomentum, def_k_uwcm_step, (int)iStep);
         ResetLastError();
         kernelExecute(def_k_UpdateWeightsConvMomentum, global_work_offset, global_work_size)
         break;
      case ADAM:
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_i, source.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_m, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdam, def_k_uwca_matrix_v, SecondMomentumConv.GetIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_inputs, inputs)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_l, lt)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_b1, b1)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_b2, b2)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_window_in, (int)iWindow)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_window_out, (int)iWindowOut)
         setArgument(def_k_UpdateWeightsConvAdam, def_k_uwca_step, (int)iStep)
         ResetLastError();
         kernelExecute(def_k_UpdateWeightsConvAdam, global_work_offset, global_work_size)
         t++;
         break;
      case LS:
         global_work_size[0] = iWindow + 1;
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_i, source.GetIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_xg, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvLS, def_k_uwcls_matrix_xx, SecondMomentumConv.GetIndex())
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwls_inputs, inputs)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_l, lr)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_update, (int)(t >= iBatch))
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_window_in, (int)iWindow)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_window_out, (int)iWindowOut)
         setArgument(def_k_UpdateWeightsConvLS, def_k_uwcls_step, (int)iStep)
         ResetLastError();
         kernelExecute(def_k_UpdateWeightsConvLS, global_work_offset, global_work_size)
         if(t >= iBatch)
            t = 0;
         else
            t++;
         break;
      case ADAM_MINI:
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_w, WeightsConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_g, getGradientIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_i, source.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_m, FirstMomentumConv.GetIndex())
         setBuffer(def_k_UpdateWeightsConvAdamMini, def_k_wucam_matrix_v, SecondMomentumConv.GetIndex())
         lt = (float)(lr * MathSqrt(1.0 - MathPow((double)b2, (double)t)) / (1.0 - MathPow((double)b1, (double)t)));
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_inputs, inputs)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_l, lt)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_b1, b1)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_b2, b2)
         setArgument(def_k_UpdateWeightsConvAdamMini, def_k_wucam_step, (int)iStep)
         ResetLastError();
         kernelExecuteLoc(def_k_UpdateWeightsConvAdamMini, global_work_offset_am, global_work_size_am, local_work_size_am)
         t++;
         break;
      default:
         ReturnFalse;
     }
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronConvOCL::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronProofOCL::GetLayerInfo();
   if(!result)
      return result;
   result.window_out = (int)iWindowOut;
   result.count /= (int)iWindowOut;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronConvOCL::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!WeightsConv)
      WeightsConv.BufferCreate(obj);                                         ///< Matrix of weights to previous layer
   if(!!DeltaWeightsConv)
      DeltaWeightsConv.BufferCreate(obj);                                    ///< Matrix of delta weights to previous layer (#SGD)
   if(!!FirstMomentumConv)
      FirstMomentumConv.BufferCreate(obj);                                   ///< Matrix of first momentum to previous layer (#ADAM)
   if(!!SecondMomentumConv)
      SecondMomentumConv.BufferCreate(obj);                                  ///< Matrix of second momentum to previous layer (#ADAM)
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CBufferFloat::BufferToCSV(const string file_name)
  {
   BufferRead();
   int h = FileOpen(file_name, FILE_CSV | FILE_WRITE);
   if(h == INVALID_HANDLE)
      return;
//---
   for(int i = 0; i < m_data_total; i++)
      FileWrite(h, m_data[i]);
   FileFlush(h);
   FileClose(h);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::Load(const int file_handle)
  {
   if(!CNeuronProofOCL::Load(file_handle))
      ReturnFalse;
//---
   if(CheckPointer(WeightsConv) == POINTER_INVALID)
      WeightsConv = new CBufferFloat();
   if(WeightsConv.GetIndex() >= 0)
      WeightsConv.BufferFree();
   if(!WeightsConv.Load(file_handle))
      ReturnFalse;
   if(!WeightsConv.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(optimization == SGD)
     {
      if(CheckPointer(DeltaWeightsConv) == POINTER_INVALID)
         DeltaWeightsConv = new CBufferFloat();
      if(DeltaWeightsConv.GetIndex() >= 0)
         DeltaWeightsConv.BufferFree();
      if(!DeltaWeightsConv.Load(file_handle))
         ReturnFalse;
      if(!DeltaWeightsConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      if(CheckPointer(FirstMomentumConv) == POINTER_INVALID)
         FirstMomentumConv = new CBufferFloat();
      if(FirstMomentumConv.GetIndex() >= 0)
         FirstMomentumConv.BufferFree();
      if(!FirstMomentumConv.Load(file_handle))
         ReturnFalse;
      if(!FirstMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
      //---
      if(CheckPointer(SecondMomentumConv) == POINTER_INVALID)
         SecondMomentumConv = new CBufferFloat();
      if(SecondMomentumConv.GetIndex() >= 0)
         SecondMomentumConv.BufferFree();
      if(!SecondMomentumConv.Load(file_handle))
         ReturnFalse;
      if(!SecondMomentumConv.BufferCreate(OpenCL))
         ReturnFalse;
     }
   iWindowOut = FileReadInteger(file_handle);
   iVariables = FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::Save(const int file_handle)
  {
   if(!CNeuronProofOCL::Save(file_handle))
      ReturnFalse;
//---
   if(CheckPointer(WeightsConv) == POINTER_INVALID || !WeightsConv.Save(file_handle))
      ReturnFalse;
   if(optimization == SGD && (CheckPointer(DeltaWeightsConv) == POINTER_INVALID || !DeltaWeightsConv.Save(file_handle)))
      ReturnFalse;
   if(optimization != SGD && (CheckPointer(FirstMomentumConv) == POINTER_INVALID || !FirstMomentumConv.Save(file_handle)))
      ReturnFalse;
   if(optimization != SGD && (CheckPointer(SecondMomentumConv) == POINTER_INVALID || !SecondMomentumConv.Save(file_handle)))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iWindowOut, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, iVariables, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProofOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iWindow, INT_VALUE) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, iStep, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronProofOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   iWindow = FileReadInteger(file_handle);
   iStep = FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SumAndNormalize(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out, int dimension, bool normilize = true, int shift_in1 = 0, int shift_in2 = 0, int shift_out = 0, float multiplyer = 0.5f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor1) == POINTER_INVALID || CheckPointer(tensor2) == POINTER_INVALID
      || CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor1.GetIndex() < 0)
      ReturnFalse;
   if(tensor2.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   int size = MathMin(MathMin(tensor1.Total() - shift_in1, tensor2.Total() - shift_in2), out.Total() - shift_out);
   if(size <= 0 || size < dimension)
     {
      PrintFormat("Tensor 1 total = %d -> shift = %d", tensor1.Total(), shift_in1);
      PrintFormat("Tensor 2 total = %d -> shift = %d", tensor2.Total(), shift_in2);
      PrintFormat("Output total = %d -> shift = %d", out.Total(), shift_out);
      PrintFormat("size=%d -> dimension=%d", size, dimension);
      ReturnFalse;
     }
   global_work_size[0] = size / dimension;
   const int kernel = def_k_MatrixSum;
   setBuffer(kernel, def_k_sum_matrix1, tensor1.GetIndex())
   setBuffer(kernel, def_k_sum_matrix2, tensor2.GetIndex())
   setBuffer(kernel, def_k_sum_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_sum_dimension, dimension)
   setArgument(kernel, def_k_sum_shift_in1, shift_in1)
   setArgument(kernel, def_k_sum_shift_in2, shift_in2)
   setArgument(kernel, def_k_sum_shift_out, shift_out)
   setArgument(kernel, def_k_sum_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
   if(!normilize)
      return true;
   return Normalize(out, dimension);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Normalize(CBufferFloat *tensor, int dimension)
  {
   if(!OpenCL || !tensor)
      ReturnFalse;
   if(tensor.GetIndex() < 0 || dimension <= 0)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {tensor.Total() / dimension};
   const int kernel = def_k_Normalize;
   setBuffer(kernel, def_k_norm_buffer, tensor.GetIndex())
   setArgument(kernel, def_k_norm_dimension, dimension)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!tensor.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Different(CBufferFloat *tensor1, CBufferFloat *tensor2,
                               CBufferFloat *out, int dimension, int shift_in1 = 0,
                               int shift_in2 = 0, int shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor1) == POINTER_INVALID || CheckPointer(tensor2) == POINTER_INVALID
      || CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor1.GetIndex() < 0)
      ReturnFalse;
   if(tensor2.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   int size = MathMin(
                 MathMin(tensor1.Total() / (shift_in1 + dimension), tensor2.Total() / (shift_in2 + dimension)),
                 out.Total() / (shift_out + dimension));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {size, dimension};
   const int kernel = def_k_MatrixDif;
   setBuffer(kernel, def_k_dif_matrix1, tensor1.GetIndex())
   setBuffer(kernel, def_k_dif_matrix2, tensor2.GetIndex())
   setBuffer(kernel, def_k_dif_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_dif_shift_in1, shift_in1)
   setArgument(kernel, def_k_dif_shift_in2, shift_in2)
   setArgument(kernel, def_k_dif_shift_out, shift_out)
   setArgument(kernel, def_k_dif_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DifferentGrad(CBufferFloat *tensor1, CBufferFloat *tensor2,
                                   CBufferFloat *out, int dimension, int shift_in1 = 0,
                                   int shift_in2 = 0, int shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor1) == POINTER_INVALID || CheckPointer(tensor2) == POINTER_INVALID
      || CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor1.GetIndex() < 0)
      ReturnFalse;
   if(tensor2.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   int size = MathMin(
                 MathMin(tensor1.Total() / (shift_in1 + dimension), tensor2.Total() / (shift_in2 + dimension)),
                 out.Total() / (shift_out + dimension));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {size, dimension};
   const int kernel = def_k_MatrixDifGrad;
   setBuffer(kernel, def_k_dif_matrix1, tensor1.GetIndex())
   setBuffer(kernel, def_k_dif_matrix2, tensor2.GetIndex())
   setBuffer(kernel, def_k_dif_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_dif_shift_in1, shift_in1)
   setArgument(kernel, def_k_dif_shift_in2, shift_in2)
   setArgument(kernel, def_k_dif_shift_out, shift_out)
   setArgument(kernel, def_k_dif_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::IdentSum(CBufferFloat *tensor, CBufferFloat *out,
                              int dimension, int shift_in = 0,
                              int shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor) == POINTER_INVALID ||
      CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   int size = MathMin(tensor.Total() / (shift_in + dimension), out.Total() / (shift_out + dimension));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {size, dimension};
   const int kernel = def_k_IdentMatrixSum;
   setBuffer(kernel, def_k_idsum_matrix_in, tensor.GetIndex())
   setBuffer(kernel, def_k_idsum_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_idsum_shift_in, shift_in)
   setArgument(kernel, def_k_idsum_shift_out, shift_out)
   setArgument(kernel, def_k_idsum_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::IdentDifferent(CBufferFloat* tensor, CBufferFloat* out,
                                    int dimension, int shift_in = 0,
                                    int shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor) == POINTER_INVALID ||
      CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   int size = MathMin(tensor.Total() / (shift_in + dimension), out.Total() / (shift_out + dimension));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {size, dimension};
   const int kernel = def_k_IdentMatrixDif;
   setBuffer(kernel, def_k_iddif_matrix_in, tensor.GetIndex())
   setBuffer(kernel, def_k_iddif_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_iddif_shift_in, shift_in)
   setArgument(kernel, def_k_iddif_shift_out, shift_out)
   setArgument(kernel, def_k_iddif_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::IdentDifferentGrad(CBufferFloat* tensor, CBufferFloat* out,
                                        int dimension, int shift_in = 0,
                                        int shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(tensor) == POINTER_INVALID ||
      CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(tensor.GetIndex() < 0)
      ReturnFalse;
   if(out.GetIndex() < 0)
      ReturnFalse;
//---
   int size = MathMin(tensor.Total() / (shift_in + dimension), out.Total() / (shift_out + dimension));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {size, dimension};
   const int kernel = def_k_IdentMatrixDifGrad;
   setBuffer(kernel, def_k_iddif_matrix_in, tensor.GetIndex())
   setBuffer(kernel, def_k_iddif_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_iddif_shift_in, shift_in)
   setArgument(kernel, def_k_iddif_shift_out, shift_out)
   setArgument(kernel, def_k_iddif_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SumVecMatrix(CBufferFloat *vector_in, CBufferFloat *matrix_in,
                                  CBufferFloat *out, uint dimension, uint variables,
                                  uint shift_in1 = 0, uint shift_in2 = 0,
                                  uint shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(vector_in) == POINTER_INVALID ||
      CheckPointer(matrix_in) == POINTER_INVALID || CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(variables < 1)
      ReturnFalse;
   if(vector_in.GetIndex() < 0 || vector_in.Total() < int((shift_in1 + dimension)*variables))
      ReturnFalse;
   if(matrix_in.GetIndex() < 0 || matrix_in.Total() < int((shift_in2 + dimension)*variables))
      ReturnFalse;
   if(out.GetIndex() < 0 || out.Total() < int((shift_out + dimension)*variables))
      ReturnFalse;
//---
   int size = MathMin(matrix_in.Total() / int((shift_in2 + dimension) * variables),
                      out.Total() / int((shift_out + dimension) * variables));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {size, dimension, variables};
   const int kernel = def_k_SumVecMatrix;
   setBuffer(kernel, def_k_svecmat_vector_in, vector_in.GetIndex())
   setBuffer(kernel, def_k_svecmat_matrix_in, matrix_in.GetIndex())
   setBuffer(kernel, def_k_svecmat_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_svecmat_shift_in1, shift_in1)
   setArgument(kernel, def_k_svecmat_shift_in2, shift_in2)
   setArgument(kernel, def_k_svecmat_shift_out, shift_out)
   setArgument(kernel, def_k_svecmat_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SumVecMatrixGrad(CBufferFloat *vector_in, CBufferFloat *matrix_in,
                                      CBufferFloat *out, uint dimension, uint variables,
                                      uint shift_in1 = 0, uint shift_in2 = 0,
                                      uint shift_out = 0, float multiplyer = 1.0f)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(vector_in) == POINTER_INVALID ||
      CheckPointer(matrix_in) == POINTER_INVALID || CheckPointer(out) == POINTER_INVALID)
      ReturnFalse;
   if(variables < 1)
      ReturnFalse;
   if(vector_in.GetIndex() < 0 || vector_in.Total() < int((shift_in1 + dimension)*variables))
      ReturnFalse;
   if(matrix_in.GetIndex() < 0 || matrix_in.Total() < int((shift_in2 + dimension)*variables))
      ReturnFalse;
   if(out.GetIndex() < 0 || out.Total() < int((shift_out + dimension)*variables))
      ReturnFalse;
//---
   int size = MathMin(matrix_in.Total() / int((shift_in2 + dimension) * variables),
                      out.Total() / int((shift_out + dimension) * variables));
   if(size < 1)
      ReturnFalse;
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {size, dimension, variables};
   const int kernel = def_k_SumVecMatrixGrad;
   setBuffer(kernel, def_k_svecmat_vector_in, vector_in.GetIndex())
   setBuffer(kernel, def_k_svecmat_matrix_in, matrix_in.GetIndex())
   setBuffer(kernel, def_k_svecmat_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_svecmat_shift_in1, shift_in1)
   setArgument(kernel, def_k_svecmat_shift_in2, shift_in2)
   setArgument(kernel, def_k_svecmat_shift_out, shift_out)
   setArgument(kernel, def_k_svecmat_multiplyer, multiplyer)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ConcatVecMatrix(CBufferFloat *vector_in, CBufferFloat *matrix_in,
                                     CBufferFloat *out, uint dimension_v,
                                     uint rows, uint cols, uint variables, bool multvarsecond)
  {
   if(!OpenCL || !vector_in || !matrix_in || !out)
      ReturnFalse;
   if(variables < 1)
      ReturnFalse;
   if(vector_in.GetIndex() < 0 || vector_in.Total() < int(dimension_v * variables))
      ReturnFalse;
   if(matrix_in.GetIndex() < 0 || matrix_in.Total() < int(cols * rows * (multvarsecond ? variables : 1)))
      ReturnFalse;
   if(out.GetIndex() < 0 || out.Total() < int((dimension_v + cols)*rows * variables))
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {rows, MathMax(dimension_v, cols), variables};
   const int kernel = def_k_ConcatVecMatrix;
   setBuffer(kernel, def_k_concvm_vector_in, vector_in.GetIndex())
   setBuffer(kernel, def_k_concvm_matrix_in, matrix_in.GetIndex())
   setBuffer(kernel, def_k_concvm_matrix_out, out.GetIndex())
   setArgument(kernel, def_k_concvm_dimension_v, dimension_v)
   setArgument(kernel, def_k_concvm_colums_m, cols)
   setArgument(kernel, def_k_concvm_multvarsecond, int(multvarsecond))
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ConcatVecMatrixGrad(CBufferFloat *vector_in, CBufferFloat *matrix_in,
      CBufferFloat *out, uint dimension_v,
      uint rows, uint cols, uint variables, bool multvarsecond)
  {
   if(!OpenCL || !vector_in || !matrix_in || !out)
      ReturnFalse;
   if(variables < 1)
      ReturnFalse;
   if(vector_in.GetIndex() < 0 || vector_in.Total() < int(dimension_v * variables))
      ReturnFalse;
   if(matrix_in.GetIndex() < 0 || matrix_in.Total() < int(cols * rows * (multvarsecond ? variables : 1)))
      ReturnFalse;
   if(out.GetIndex() < 0 || out.Total() < int((dimension_v + cols)*rows * variables))
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {(uint)MathMin(MathMax(rows, variables), OpenCL.GetMaxLocalSize(0)),
                               MathMax(dimension_v, cols),
                               MathMax(rows, variables)
                              };
   uint local_work_size[3] = {global_work_size[0], 1, 1};
   const int kernel = def_k_ConcatVecMatrixGrad;
   setBuffer(kernel, def_k_concvm_gr_vector_in_gr, vector_in.GetIndex())
   setBuffer(kernel, def_k_concvm_gr_matrix_in_gr, matrix_in.GetIndex())
   setBuffer(kernel, def_k_concvm_gr_matrix_out_gr, out.GetIndex())
   setArgument(kernel, def_k_concvm_gr_dimension_v, (int)dimension_v)
   setArgument(kernel, def_k_concvm_gr_colums_m, (int)cols)
   setArgument(kernel, def_k_concvm_gr_rows, (int)rows)
   setArgument(kernel, def_k_concvm_gr_variables, (int)variables)
   setArgument(kernel, def_k_concvm_gr_multvarsecond, (int)multvarsecond)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::MatMul(const CBufferFloat* matr1, const CBufferFloat* matr2, CBufferFloat* result, const int rows1, const int cols1, const int cols2, const int variables = 1, const bool mult_var_second = true)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {rows1, cols2, variables};
   setBuffer(def_k_MatMult, def_k_mm_matr1, matr1.GetIndex())
   setBuffer(def_k_MatMult, def_k_mm_matr2, matr2.GetIndex())
   setBuffer(def_k_MatMult, def_k_mm_result, result.GetIndex())
   setArgument(def_k_MatMult, def_k_mm_dimension, (int)cols1)
   setArgument(def_k_MatMult, def_k_mm_multvarsecond, (int)mult_var_second)
//---
   kernelExecute(def_k_MatMult, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!result.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::MatMulGrad(const CBufferFloat* matr1, CBufferFloat* matr1_gr, const CBufferFloat* matr2, CBufferFloat* matr2_gr, const CBufferFloat* result_gr, const int rows1, const int cols1, const int cols2, const int variables = 1, const bool mult_var_second = true)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {rows1, cols2, variables};
   setBuffer(def_k_MatMultGrad, def_k_mmg_matr1, matr1.GetIndex())
   setBuffer(def_k_MatMultGrad, def_k_mmg_matr1_gr, matr1_gr.GetIndex())
   setBuffer(def_k_MatMultGrad, def_k_mmg_matr2, matr2.GetIndex())
   setBuffer(def_k_MatMultGrad, def_k_mmg_matr2_gr, matr2_gr.GetIndex())
   setBuffer(def_k_MatMultGrad, def_k_mmg_result_gr, result_gr.GetIndex())
   setArgument(def_k_MatMultGrad, def_k_mmg_dimension, (int)cols1)
   setArgument(def_k_MatMultGrad, def_k_mmg_multvarsecond, (int)mult_var_second)
//---
   kernelExecute(def_k_MatMultGrad, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!matr1_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DiagMatMul(const CBufferFloat* diag, const CBufferFloat* matr, CBufferFloat* result, const int rows, const int cols, const int variables = 1, const int activat_func = None)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {rows, cols, variables};
   uint local_work_size[3] = {1, cols, 1};
   uint kernel = def_k_DiagMatMult;
   setBuffer(kernel, def_k_diagmm_diag, diag.GetIndex())
   setBuffer(kernel, def_k_diagmm_matr, matr.GetIndex())
   setBuffer(kernel, def_k_diagmm_result, result.GetIndex())
   setArgument(kernel, def_k_diagmm_activation, (int)activat_func)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!result.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DiagMatMulGrad(const CBufferFloat* diag, CBufferFloat* diag_gr, const CBufferFloat* matr, CBufferFloat* matr_gr, const CBufferFloat* result_gr, const int rows, const int cols, const int variables = 1)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {rows, cols, variables};
   uint local_work_size[3] = {1, cols, 1};
   int kernel = def_k_DiagMatMultGrad;
   setBuffer(kernel, def_k_diagmmgr_diag, diag.GetIndex())
   setBuffer(kernel, def_k_diagmmgr_grad_diag, diag_gr.GetIndex())
   setBuffer(kernel, def_k_diagmmgr_matr, matr.GetIndex())
   setBuffer(kernel, def_k_diagmmgr_grad_matr, matr_gr.GetIndex())
   setBuffer(kernel, def_k_diagmmgr_grad_result, result_gr.GetIndex())
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!matr_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseMatMul(const CBufferFloat *sparse_indexs,
                                  const CBufferFloat *sparse_data,
                                  const CBufferFloat *full,
                                  CBufferFloat *result,
                                  const int sparse_rows,
                                  const int sparse_cols,
                                  const int full_rows,
                                  const int full_cols)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {sparse_rows, sparse_cols, full_cols};
   uint local_work_size[3] = {1, global_work_size[1], 1};
   uint kernel = def_k_SparseMatMult;
   setBuffer(kernel, def_k_smm_sparse_index, sparse_indexs.GetIndex())
   setBuffer(kernel, def_k_smm_sparse_data, sparse_data.GetIndex())
   setBuffer(kernel, def_k_smm_full, full.GetIndex())
   setBuffer(kernel, def_k_smm_result, result.GetIndex())
   setArgument(kernel, def_k_smm_full_rows, full_rows)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!result.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseMatMulGrad(const CBufferFloat *sparse_indexs,
                                      const CBufferFloat *sparse_data,
                                      CBufferFloat *sparse_grad,
                                      const CBufferFloat *full,
                                      CBufferFloat *full_gr,
                                      const CBufferFloat *grad,
                                      const int sparse_rows,
                                      const int sparse_cols,
                                      const int full_rows,
                                      const int full_cols)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {(uint)MathMax(sparse_rows, full_rows),
                               (uint)MathMin(MathMax(sparse_cols, full_cols), (int)OpenCL.GetMaxLocalSize(1)),
                               (uint)MathMax(sparse_cols, full_cols)
                              };
   uint local_work_size[3] = {1, global_work_size[1], 1};
   uint kernel = def_k_SparseMatMultGrad;
   setBuffer(kernel, def_k_smm_gr_sparse_index, sparse_indexs.GetIndex())
   setBuffer(kernel, def_k_smm_gr_sparse_data, sparse_data.GetIndex())
   setBuffer(kernel, def_k_smm_gr_sparse_gr, sparse_grad.GetIndex())
   setBuffer(kernel, def_k_smm_gr_full, full.GetIndex())
   setBuffer(kernel, def_k_smm_gr_full_gr, full_gr.GetIndex())
   setBuffer(kernel, def_k_smm_gr_result_gr, grad.GetIndex())
   setArgument(kernel, def_k_smm_gr_sparse_rows, sparse_rows)
   setArgument(kernel, def_k_smm_gr_sparse_cols, sparse_cols)
   setArgument(kernel, def_k_smm_gr_full_rows, full_rows)
   setArgument(kernel, def_k_smm_gr_full_cols, full_cols)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!full_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseConcatenate(const CBufferFloat *sparse_indexs,
                                       const CBufferFloat *sparse_data,
                                       const CBufferFloat *full,
                                       CBufferFloat *result,
                                       const int sparse_rows,
                                       const int sparse_cols,
                                       const int full_rows,
                                       const int full_cols)

  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {sparse_rows, sparse_cols, full_cols};
   uint local_work_size[3] = {1, global_work_size[1], 1};
   uint kernel = def_k_SparseConcatenate;
   setBuffer(kernel, def_k_sparconc_sparse_index, sparse_indexs.GetIndex())
   setBuffer(kernel, def_k_sparconc_sparse_data, sparse_data.GetIndex())
   setBuffer(kernel, def_k_sparconc_full, full.GetIndex())
   setBuffer(kernel, def_k_sparconc_result, result.GetIndex())
   setArgument(kernel, def_k_sparconc_full_rows, full_rows)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!result.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseConcatenateGrad(const CBufferFloat *sparse_indexs,
      const CBufferFloat *sparse_data,
      CBufferFloat *sparse_grad,
      const CBufferFloat *full,
      CBufferFloat *full_gr,
      const CBufferFloat *grad,
      const int sparse_rows,
      const int sparse_cols,
      const int full_rows,
      const int full_cols)
  {
   if(!OpenCL)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {(uint)MathMax(sparse_rows, full_rows),
                               (uint)MathMin(MathMax(sparse_cols, full_cols), (int)OpenCL.GetMaxLocalSize(1)),
                               (uint)MathMax(sparse_cols, full_cols)
                              };
   uint local_work_size[3] = {1, global_work_size[1], 1};
   uint kernel = def_k_SparseConcatenateGrad;
   setBuffer(kernel, def_k_sparconc_gr_sparse_index, sparse_indexs.GetIndex())
   setBuffer(kernel, def_k_sparconc_gr_sparse_data, sparse_data.GetIndex())
   setBuffer(kernel, def_k_sparconc_gr_sparse_gr, sparse_grad.GetIndex())
   setBuffer(kernel, def_k_sparconc_gr_full, full.GetIndex())
   setBuffer(kernel, def_k_sparconc_gr_full_gr, full_gr.GetIndex())
   setBuffer(kernel, def_k_sparconc_gr_result_gr, grad.GetIndex())
   setArgument(kernel, def_k_sparconc_gr_sparse_rows, sparse_rows)
   setArgument(kernel, def_k_sparconc_gr_sparse_cols, sparse_cols)
   setArgument(kernel, def_k_sparconc_gr_full_rows, full_rows)
   setArgument(kernel, def_k_sparconc_gr_full_cols, full_cols)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!full_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ScalarToVector(CBufferFloat *scalar,
                                    CBufferFloat *vector_in,
                                    CBufferFloat *vector_out,
                                    int dimension)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   if(!scalar || !vector_in || !vector_out)
      ReturnFalse;
   int total = scalar.Total() * dimension;
   if(vector_in.Total() < total ||
      vector_out.Total() < total)
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {(uint)scalar.Total(),
                              (uint)dimension
                             };
   uint kernel = def_k_ScalarToVector;
   setBuffer(kernel, def_k_stv_scalar, scalar.GetIndex())
   setBuffer(kernel, def_k_stv_vector_in, vector_in.GetIndex())
   setBuffer(kernel, def_k_stv_vector_out, vector_out.GetIndex())
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!vector_out.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ScalarToVectorGrad(CBufferFloat *scalar,
                                        CBufferFloat *scalar_gr,
                                        CBufferFloat *vector_in,
                                        CBufferFloat *vector_in_gr,
                                        CBufferFloat *vector_out_gr,
                                        int dimension)
  {
   if(!OpenCL)
      ReturnFalse;
//---
   if(!scalar || !vector_in || !vector_out_gr ||
      !scalar_gr || !vector_in_gr)
      ReturnFalse;
   int total = scalar.Total() * dimension;
   if(vector_in.Total() < total ||
      vector_in_gr.Total() < total ||
      vector_out_gr.Total() < total ||
      scalar.Total() > scalar_gr.Total())
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {(uint)scalar.Total(),
                              (uint)MathMin(dimension, (int)OpenCL.GetMaxLocalSize(1))
                             };
   uint local_work_size[] = {1, global_work_size[1]};
//---
   uint kernel = def_k_ScalarToVectorGrad;
   setBuffer(kernel, def_k_stvg_scalar, scalar.GetIndex())
   setBuffer(kernel, def_k_stvg_scalar_gr, scalar_gr.GetIndex())
   setBuffer(kernel, def_k_stvg_vector_in, vector_in.GetIndex())
   setBuffer(kernel, def_k_stvg_vector_in_gr, vector_in_gr.GetIndex())
   setBuffer(kernel, def_k_stvg_vector_out_gr, vector_out_gr.GetIndex())
   setArgument(kernel, def_k_stvg_dimension, dimension)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!vector_in_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CNeuronBaseOCL::GenerateWeight(void)
  {
   xor128;
   float result = (float)rnd_w / UINT_MAX;
   if(result == 0)
      result = GenerateWeight();
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDropoutOCL::CNeuronDropoutOCL(void) :  OutProbability((float)0.1),
   OutNumber(0),
   dInitValue(1.0)
  {
   PrevLayer = NULL;
   DropOutMultiplier = new CBufferFloat();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronDropoutOCL::~CNeuronDropoutOCL(void)
  {
   DeleteObj(DropOutMultiplier);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, float out_prob, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   OutProbability = out_prob;
   OutNumber = (int)(numNeurons * out_prob);
   dInitValue = 1 / (1 - OutProbability);
   if(CheckPointer(DropOutMultiplier) == POINTER_INVALID)
      DropOutMultiplier = new CBufferFloat();
   if(!DropOutMultiplier.BufferInit(numNeurons + 1, dInitValue))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   activation = (ENUM_ACTIVATION)NeuronOCL.Activation();
   PrevLayer = NeuronOCL;
   if(!bTrain)
      return true;
//---
   if(CheckPointer(DropOutMultiplier) == POINTER_INVALID)
      DropOutMultiplier = new CBufferFloat();
   if(!DropOutMultiplier.BufferInit(NeuronOCL.Neurons(), dInitValue))
      ReturnFalse;
   for(int i = 0; i < OutNumber; i++)
     {
      uint p = RND(Neurons());
      float val = DropOutMultiplier.At(p);
      if(val == 0 || val == DBL_MAX)
        {
         i--;
         continue;
        }
      if(!DropOutMultiplier.Update(RND(Neurons()), 0))
         ReturnFalse;
     }
//---
   if(!DropOutMultiplier.BufferCreate(OpenCL))
      ReturnFalse;
   return ElementMult(NeuronOCL.getOutput(), DropOutMultiplier, Output);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ElementMult(const CBufferFloat* input1, const CBufferFloat* input2, CBufferFloat* output)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = (output.Total() + 3) / 4;
   setBuffer(def_k_Dropout, def_k_dout_input, input1.GetIndex())
   setBuffer(def_k_Dropout, def_k_dout_map, input2.GetIndex())
   setBuffer(def_k_Dropout, def_k_dout_out, output.GetIndex())
   setArgument(def_k_Dropout, def_k_dout_dimension, output.Total())
   ResetLastError();
   kernelExecute(def_k_Dropout, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::GateElementMult(const CBufferFloat* input1, const CBufferFloat* input2,
                                     const CBufferFloat* gate, CBufferFloat* output)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {output.Total()};
   uint kernel = def_k_GateElementMul;
   setBuffer(kernel, def_k_gem_inputs1, input1.GetIndex())
   setBuffer(kernel, def_k_gem_inputs2, input2.GetIndex())
   setBuffer(kernel, def_k_gem_gate, gate.GetIndex())
   setBuffer(kernel, def_k_gem_out, output.GetIndex())
   ResetLastError();
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::GateElementMultGrad(const CBufferFloat* input1,
      CBufferFloat* input1_gr,
      const CBufferFloat* input2,
      CBufferFloat* input2_gr,
      const CBufferFloat* gate,
      CBufferFloat* gate_gr,
      const CBufferFloat* output_gr,
      const int activation1,
      const int activation2,
      const int activation_gate)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = output_gr.Total();
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_inputs1, input1.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_inputs1_gr, input1_gr.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_inputs2, input2.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_inputs2_gr, input2_gr.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_gate, gate.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_gate_gr, gate_gr.GetIndex())
   setBuffer(def_k_GateElementMulGrad, def_k_gemg_out, output_gr.GetIndex())
   setArgument(def_k_GateElementMulGrad, def_k_gemg_activ1, int(activation1))
   setArgument(def_k_GateElementMulGrad, def_k_gemg_activ2, int(activation2))
   setArgument(def_k_GateElementMulGrad, def_k_gemg_activ_gr, int(activation_gate))
   ResetLastError();
   kernelExecute(def_k_GateElementMulGrad, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!input1_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   if(!bTrain)
      return true;
//---
   if(CheckPointer(DropOutMultiplier) == POINTER_INVALID)
      ReturnFalse;
//---
   if(!DropOutMultiplier.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return ElementMult(Gradient, DropOutMultiplier, NeuronOCL.getGradient());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteDouble(file_handle, OutProbability) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronDropoutOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   OutProbability = (float)FileReadDouble(file_handle);
   OutNumber = (int)(Neurons() * OutProbability);
   dInitValue = 1 / (1 - OutProbability);
   if(CheckPointer(DropOutMultiplier) == POINTER_INVALID)
      DropOutMultiplier = new CBufferFloat();
   if(!DropOutMultiplier.BufferInit(Neurons() + 1, dInitValue))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronDropoutOCL::GetLayerInfo(void)
  {
   CLayerDescription* result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
//---
   result.probability = OutProbability;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBatchNormOCL::CNeuronBatchNormOCL(void)  :  iBatchSize(1)
  {
   PrevLayer = NULL;
   BatchOptions = NULL;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronBatchNormOCL::~CNeuronBatchNormOCL(void)
  {
   if(CheckPointer(PrevLayer) != POINTER_INVALID)
      PrevLayer = NULL;
   DeleteObj(BatchOptions);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return Init(numOutputs, myIndex, open_cl, numNeurons, batch, optimization_type);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, uint batchSize, ENUM_OPTIMIZATION optimization_type)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batchSize))
      ReturnFalse;
   activation = None;
   iBatchSize = (int)batchSize;
//---
   if(!BatchOptions)
      BatchOptions = new CBufferFloat();
   BatchOptions.BufferFree();
   int count = (int)numNeurons * (optimization_type == SGD ? 7 : 9);
   if(!BatchOptions.BufferInit(count, 0.0f))
      ReturnFalse;
   if(!BatchOptions.BufferCreate(OpenCL))
      ReturnFalse;
   iBatchCount = 1;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   PrevLayer = NeuronOCL;
   if(iBatchSize <= 1)
     {
      activation = (ENUM_ACTIVATION)NeuronOCL.Activation();
      return true;
     }
//---
   if(CheckPointer(BatchOptions) == POINTER_INVALID)
     {
      int count = Neurons() * (optimization == SGD ? 7 : 9);
      BatchOptions = new CBufferFloat();
      if(!BatchOptions.BufferInit(count, 0))
         ReturnFalse;
      if(!BatchOptions.BufferCreate(OpenCL))
         ReturnFalse;
     }
   if(BatchOptions.GetIndex() < 0 && !BatchOptions.BufferCreate(OpenCL))
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
   iBatchCount = MathMin(iBatchCount, iBatchSize);
   setBuffer(def_k_BatchFeedForward, def_k_bff_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_BatchFeedForward, def_k_bff_options, BatchOptions.GetIndex())
   setBuffer(def_k_BatchFeedForward, def_k_bff_output, Output.GetIndex())
   setArgument(def_k_BatchFeedForward, def_k_bff_batch, (int)iBatchCount)
   setArgument(def_k_BatchFeedForward, def_k_bff_optimization, (int)optimization)
   setArgument(def_k_BatchFeedForward, def_k_bff_activation, (int)activation)
   setArgument(def_k_BatchFeedForward, def_k_bff_training, (int)(bTrain ? 1 : 0))
   ResetLastError();
   kernelExecute(def_k_BatchFeedForward, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
   if(bTrain)
      iBatchCount++;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   if(iBatchSize <= 1)
      return (CheckPointer(PrevLayer) != POINTER_INVALID);
//---
   if(CheckPointer(BatchOptions) == POINTER_INVALID || BatchOptions.GetIndex() < 0)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
   setBuffer(def_k_CalcHiddenGradientBatch, def_k_bchg_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(def_k_CalcHiddenGradientBatch, def_k_bchg_options, BatchOptions.GetIndex())
   setBuffer(def_k_CalcHiddenGradientBatch, def_k_bchg_matrix_g, Gradient.GetIndex())
   setBuffer(def_k_CalcHiddenGradientBatch, def_k_bchg_matrix_ig, NeuronOCL.getGradientIndex())
   setArgument(def_k_CalcHiddenGradientBatch, def_k_bchg_activation, NeuronOCL.Activation())
   setArgument(def_k_CalcHiddenGradientBatch, def_k_bchg_batch, (int)iBatchCount)
   setArgument(def_k_CalcHiddenGradientBatch, def_k_bchg_optimization, (int)optimization)
   ResetLastError();
   kernelExecute(def_k_CalcHiddenGradientBatch, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
//---
   if(iBatchSize <= 1)
      return (CheckPointer(PrevLayer) != POINTER_INVALID);
//---
   if(CheckPointer(BatchOptions) == POINTER_INVALID || BatchOptions.GetIndex() < 0)
      ReturnFalse;
   if(iBatchCount < 100 * iBatchSize)
      return true;
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
//---
   if(optimization == SGD)
     {
      setBuffer(def_k_UpdateBatchOptionsMomentum, def_k_buom_options, BatchOptions.GetIndex())
      setBuffer(def_k_UpdateBatchOptionsMomentum, def_k_buom_matrix_g, Gradient.GetIndex())
      setArgument(def_k_UpdateBatchOptionsMomentum, def_k_buom_learning_rates, lr)
      setArgument(def_k_UpdateBatchOptionsMomentum, def_k_buom_momentum, alpha)
      ResetLastError();
      kernelExecute(def_k_UpdateBatchOptionsMomentum, global_work_offset, global_work_size)
     }
   else
     {
      setBuffer(def_k_UpdateBatchOptionsAdam, def_k_buoa_options, BatchOptions.GetIndex())
      setBuffer(def_k_UpdateBatchOptionsAdam, def_k_buoa_matrix_g, Gradient.GetIndex())
      setArgument(def_k_UpdateBatchOptionsAdam, def_k_buoa_l, lr)
      setArgument(def_k_UpdateBatchOptionsAdam, def_k_buoa_b1, b1)
      setArgument(def_k_UpdateBatchOptionsAdam, def_k_buoa_b2, b2)
      ResetLastError();
      kernelExecute(def_k_UpdateBatchOptionsAdam, global_work_offset, global_work_size)
     }
#ifdef  _DEBUG
   if(!BatchOptions.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, iBatchSize, INT_VALUE) <= 0)
      ReturnFalse;
   return BatchOptions.Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   iBatchSize = FileReadInteger(file_handle, INT_VALUE);
   if(CheckPointer(BatchOptions) == POINTER_INVALID)
     {
      BatchOptions = new CBufferFloat();
      if(CheckPointer(BatchOptions) == POINTER_INVALID)
         ReturnFalse;
     }
   if(BatchOptions.GetIndex() >= 0)
      BatchOptions.BufferFree();
   if(!BatchOptions.Load(file_handle))
      ReturnFalse;
   if(!!OpenCL && !BatchOptions.BufferCreate(OpenCL))
      ReturnFalse;
   iBatchCount = iBatchSize;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronBatchNormOCL::GetLayerInfo(void)
  {
   CLayerDescription* result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
//---
   result.batch = (int)iBatchSize;
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBatchNormOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!BatchOptions)
      BatchOptions.BufferCreate(obj);    ///< Container of method parameters
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::GetLayerOutput(uint layer, CBufferFloat *&result)
  {
   if(!layers || layers.Total() <= (int)layer)
      ReturnFalse;
   CLayer *Layer = layers.At(layer);
   if(!Layer)
      ReturnFalse;
//---
   if(!result)
     {
      result = new CBufferFloat();
      if(!result)
         ReturnFalse;
     }
//---
   CNeuronBaseOCL *temp = Layer.At(0);
   if(!temp || temp.getOutputVal(result) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Return the existing device output; caller does not own it.      |
//+------------------------------------------------------------------+
bool CNet::GetLayerOutputDevice(uint layer, CBufferFloat *&result)
  {
   result = NULL;
   if(!layers || layers.Total() <= (int)layer)
      ReturnFalse;
   CLayer *Layer = layers.At(layer);
   if(!Layer || Layer.Total() <= 0)
      ReturnFalse;
   CNeuronBaseOCL *temp = Layer.At(0);
   if(!temp)
      ReturnFalse;
   result = temp.getOutput();
   if(!result || result.GetIndex() == INVALID_HANDLE || result.GetOpenCL() != opencl)
     {
      result = NULL;
      ReturnFalse;
     }
//--- The returned pointer stays owned by the layer/net.
   return true;
  }
//+------------------------------------------------------------------+
//| Return the first neuron object of a layer without a host copy.    |
//+------------------------------------------------------------------+
CNeuronBaseOCL *CNet::Layer(int layer)
  {
   if(!layers || layers.Total() <= 0)
      return(NULL);
   const int total = layers.Total();
   if(layer < 0)
      layer += total;
   if(layer < 0 || layer >= total)
      return(NULL);
   CLayer *LayerObject = layers.At(layer);
   return(LayerObject ? LayerObject[0] : NULL);
  }
//+------------------------------------------------------------------+
//| Feed forward from an existing device-resident layer.             |
//+------------------------------------------------------------------+
bool CNet::FeedForwardLayer(CNeuronBaseOCL *source_layer)
  {
   if(!source_layer || !layers || layers.Total() <= 1)
      ReturnFalse;
   CNeuronBaseOCL *previous = source_layer;
   for(int i = 1; i < layers.Total(); i++)
     {
      CLayer *layer = layers.At(i);
      CNeuronBaseOCL *current = (layer ? layer[0] : NULL);
      if(!current || !current.FeedForward(previous))
         ReturnFalse;
      previous = current;
     }
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::GetLayerOutput(uint layer, vector<float> &result)
  {
   if(!layers || layers.Total() <= (int)layer)
      ReturnFalse;
   CLayer *Layer = layers.At(layer);
   if(!Layer)
      ReturnFalse;
//---
//---
   CNeuronBaseOCL *temp = Layer.At(0);
   if(!temp || temp.getOutputVal(result) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Bind a library-owned immutable RAG snapshot to a RAG-MPI layer.  |
//| This is deliberately typed: consumers cannot reach arbitrary     |
//| internal layers or pass a token buffer from another OpenCL context.|
//+------------------------------------------------------------------+
bool CNet::BindRAGMemory(const uint layer, CNeuronRAGMemory *memory)
  {
   if(memory == NULL || layers == NULL || opencl == NULL || layer >= (uint)layers.Total())
      return false;
   CLayer *target = layers.At(layer);
   if(target == NULL)
      return false;
   CNeuronBaseOCL *base = target.At(0);
   if(base == NULL || base.Type() != defNeuronVLADriverRAGMPI)
      return false;
   CNeuronVLADriverRAGMPI *mpi = base;
   if(memory.GetOpenCL() != opencl || memory.TokenCount() != mpi.RAGTokenCount() ||
      memory.TokenWidth() != mpi.RAGTokenWidth())
      return false;
   if(memory.TokenWidth() == 0 || memory.TokenCount() > uint(INT_MAX) / memory.TokenWidth())
      return false;
   CBufferFloat *tokens = memory.Tokens();
   CBufferFloat *absolute_relevance = memory.AbsoluteRelevance();
   if(tokens == NULL || tokens.GetOpenCL() != opencl ||
      tokens.Total() != int(memory.TokenCount()*memory.TokenWidth()) ||
      absolute_relevance == NULL || absolute_relevance.GetOpenCL() != opencl ||
      absolute_relevance.Total() != 1)
      return false;
   return mpi.SetRAGTokens(tokens, absolute_relevance, memory.TokenWidth());
  }
//+------------------------------------------------------------------+
//| Detach RAG pointers before replacing their library-owned COW set.|
//+------------------------------------------------------------------+
bool CNet::UnbindRAGMemory(const uint layer)
  {
   if(layers == NULL || layer >= (uint)layers.Total())
      return false;
   CLayer *target = layers.At(layer);
   if(target == NULL)
      return false;
   CNeuronBaseOCL *base = target.At(0);
   if(base == NULL || base.Type() != defNeuronVLADriverRAGMPI)
      return false;
   CNeuronVLADriverRAGMPI *mpi = base;
   return mpi.SetRAGTokens(NULL, NULL, mpi.RAGTokenWidth());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNet::SetOpenCL(COpenCLMy * obj)
  {
   if(opencl == obj)
      return;
   COpenCLMy *previous = opencl;
   const bool own_previous = bOwnOpenCL;
   if(!obj || !layers)
      return;
//--- Rebind every buffer while its original context is still alive.  Layers
//--- and neurons are non-owning observers; only CNet may destroy its context.
   for(int i = 0; i < layers.Total(); i++)
     {
      CLayer *layer = layers.At(i);
      if(layer)
         layer.SetOpenCL(obj);
     }
   opencl = obj;
   bOwnOpenCL = false;
   if(own_previous && CheckPointer(previous) != POINTER_INVALID)
     {
      previous.Shutdown();
      delete previous;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSoftMaxOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint size = Output.Total() / iHeads;
   uint global_work_size[2] = { size, iHeads };
   uint local_work_size[2] = { size, 1 };
   setBuffer(def_k_SoftMax_FeedForward, def_k_softmaxff_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_SoftMax_FeedForward, def_k_softmaxff_outputs, getOutputIndex())
   kernelExecuteLoc(def_k_SoftMax_FeedForward, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSoftMaxOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[2] = {0, 0};
   uint size = Output.Total() / iHeads;
   uint global_work_size[2] = { size, iHeads };
   uint local_work_size[2] = { size, 1 };
   setBuffer(def_k_SoftMax_HiddenGradient, def_k_softmaxhg_input_gr, NeuronOCL.getGradientIndex());
   setBuffer(def_k_SoftMax_HiddenGradient, def_k_softmaxhg_output_gr, getGradientIndex());
   setBuffer(def_k_SoftMax_HiddenGradient, def_k_softmaxhg_outputs, getOutputIndex());
   kernelExecuteLoc(def_k_SoftMax_HiddenGradient, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSoftMaxOCL::calcOutputGradients(CBufferFloat* Target, float & error)
  {
   if(!OpenCL || !Target)
      ReturnFalse;
//---
   if(!Output.BufferRead())
      ReturnFalse;
   if(Output.Total() != Target.Total())
      ReturnFalse;
   error = 0;
   for(int i = 0; i < Output.Total(); i++)
     {
      if(Output[i] > 0)
         error += -MathAbs(Target[i]) * MathLog(Output[i]);
      Gradient.Update(i, Target[i]);
     }
   error /= (float)iHeads;
   if(!Gradient.BufferWrite())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Output.Total();
   setBuffer(def_k_SoftMax_OutputGradient, def_k_softmaxog_targets, getGradientIndex());
   setBuffer(def_k_SoftMax_OutputGradient, def_k_softmaxog_output_gr, getGradientIndex());
   setBuffer(def_k_SoftMax_OutputGradient, def_k_softmaxog_outputs, getOutputIndex());
   kernelExecute(def_k_SoftMax_OutputGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLayerDescription* CNeuronSoftMaxOCL::GetLayerInfo(void)
  {
   CLayerDescription* result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
   result.step = (int)iHeads;
   result.count = (int)(Output.Total() / iHeads);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSoftMaxOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iHeads) <= 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSoftMaxOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   iHeads = (uint)FileReadInteger(file_handle);
   if(iHeads <= 0)
      iHeads = 1;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNet::getResults(vector<float> &resultVals)
  {
   CBufferFloat* temp;
   getResults(temp);
   temp.GetData(resultVals);
   DeleteObj(temp);
   return;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNet::getResults(float & resultVals[])
  {
   CBufferFloat* temp;
   getResults(temp);
   if(!temp.GetData(resultVals))
     {
      DeleteObj(temp);
      return;
     }
   DeleteObj(temp);
   return;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::WeightsUpdate(CNet* net, float tau)
  {
   if(!layers || !net || !net.layers || layers.Total() != net.layers.Total())
      ReturnFalse;
   if(tau < 0)
      ReturnFalse;
   if(tau > 1)
      return true;
//---
   for(int l = 0; l < layers.Total(); l++)
     {
      CLayer *layer = layers.At(l);
      if(!layer  || !layer.WeightsUpdate(net.layers.At(l), tau))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLayer::WeightsUpdate(CLayer * source, float tau)
  {
   if(!source || Type() != source.Type() ||
      m_data_total != source.Total())
      ReturnFalse;
//---
   for(int i = 0; i < m_data_total; i++)
     {
      CNeuronBaseOCL *temp = m_data[i];
      if(temp == source.At(i))
         continue;
      //---
      if(!temp || !temp.WeightsUpdate(source.At(i), tau))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CLayer::ClearStates(void)
  {
   for(int i = 0; i < m_data_total; i++)
     {
      CNeuronBaseOCL *temp = m_data[i];
      if(!temp ||
         !temp.Clear())
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(tau != 1.0f && optimization == ADAM)
      return WeightsUpdateAdam(source, tau);
//---
   if(!OpenCL || !source)
      ReturnFalse;
   if(Type() != source.Type())
      ReturnFalse;
   if(!Weights || Weights.Total() == 0)
      return true;
   if(!source.Weights || Weights.Total() != source.Weights.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Weights.Total()};
   ResetLastError();
   setBuffer(def_k_SoftUpdate, def_k_su_target, Weights.GetIndex())
   setBuffer(def_k_SoftUpdate, def_k_su_source, source.getWeightsIndex())
   setArgument(def_k_SoftUpdate, def_k_su_tau, (float)tau)
   kernelExecute(def_k_SoftUpdate, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Weights.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::WeightsUpdateAdam(CNeuronBaseOCL* source, float tau)
  {
   if(GetPointer(this) == source)
      return true;
//---
   if(!OpenCL || !source)
      ReturnFalse;
   if(Type() != source.Type())
      ReturnFalse;
   if(!Weights || Weights.Total() == 0)
      return true;
   if(!source.Weights || Weights.Total() != source.Weights.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Weights.Total()};
   ResetLastError();
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_target, getWeightsIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_source, source.getWeightsIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_m, getFirstMomentumIndex())
   setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_v, getSecondMomentumIndex())
   setArgument(def_k_SoftUpdateAdam, def_k_sua_tau, (float)tau)
   setArgument(def_k_SoftUpdateAdam, def_k_sua_b1, (float)b1)
   setArgument(def_k_SoftUpdateAdam, def_k_sua_b2, (float)b2)
   kernelExecute(def_k_SoftUpdateAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Weights.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SwapBuffers(CBufferFloat *&buffer1, CBufferFloat *&buffer2, bool check_size = true)
  {
   if(!buffer1 || !buffer2)
      ReturnFalse;
   if(check_size)
     {
      if(buffer1.Total() != buffer2.Total())
         ReturnFalse;
     }
//---
   CBufferFloat *temp = buffer1;
   buffer1 = buffer2;
   buffer2 = temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronConvOCL *temp = source;
   if(WeightsConv.Total() != temp.WeightsConv.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {WeightsConv.Total()};
   ResetLastError();
   if(tau != 1.0f && optimization == ADAM)
     {
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_target, WeightsConv.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_source, temp.WeightsConv.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_m, FirstMomentumConv.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_v, SecondMomentumConv.GetIndex())
      setArgument(def_k_SoftUpdateAdam, def_k_sua_tau, (float)tau)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b1, (float)b1)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b2, (float)b2)
      kernelExecute(def_k_SoftUpdateAdam, global_work_offset, global_work_size)
     }
   else
     {
      setBuffer(def_k_SoftUpdate, def_k_su_target, WeightsConv.GetIndex())
      setBuffer(def_k_SoftUpdate, def_k_su_source, temp.WeightsConv.GetIndex())
      setArgument(def_k_SoftUpdate, def_k_su_tau, (float)tau)
      kernelExecute(def_k_SoftUpdate, global_work_offset, global_work_size)
     }
#ifdef  _DEBUG
   if(!WeightsConv.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronBatchNormOCL *temp = source;
   if(BatchOptions.Total() != temp.BatchOptions.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {BatchOptions.Total()};
   ResetLastError();
   setBuffer(def_k_SoftUpdate, def_k_su_target, BatchOptions.GetIndex())
   setBuffer(def_k_SoftUpdate, def_k_su_source, temp.BatchOptions.GetIndex())
   setArgument(def_k_SoftUpdate, def_k_su_tau, (float)tau)
   kernelExecute(def_k_SoftUpdate, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!BatchOptions.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::feedForward(CNet* inputNet, int inputLayer = -1, CNet *secondNet = NULL, int secondLayer = -1)
  {
   if(!inputNet || !opencl)
      ReturnFalse;
   if(inputLayer < 0)
     {
      inputLayer = inputNet.layers.Total() + inputLayer;
      if(inputLayer < 0)
         ReturnFalse;
     }
//---
   CBufferFloat *second = NULL;
   CNeuronBaseOCL *secondNeuron = NULL;
   bool del_second = false;
   if(!!secondNet)
     {
      if(secondLayer < 0)
         secondLayer = secondNet.layers.Total() - 1;
      if(secondNet.GetOpenCL() != opencl)
        {
         secondNet.GetLayerOutput(secondLayer, second);
         if(!!second)
           {
            if(!second.BufferCreate(opencl))
              {
               DeleteObj(second);
               ReturnFalse;
              }
            del_second = true;
           }
        }
      else
        {
         if(secondNet.layers.Total() <= secondLayer)
           {
            PrintFormat("CNet::feedForward second stream: requested_layer=%d available_layers=%d",
                        secondLayer, secondNet.layers.Total());
            ReturnFalse;
           }
         CLayer *layer = secondNet.layers.At(secondLayer);
         CNeuronBaseOCL *neuron = layer.At(0);
         secondNeuron = neuron;
         second = neuron.getOutput();
        }
     }
//---
   if(inputNet.opencl != opencl)
     {
      CBufferFloat *inputs;
      if(!inputNet.GetLayerOutput(inputLayer, inputs))
        {
         if(del_second)
            DeleteObj(second);
         DeleteObj(inputs);
         ReturnFalse;
        }
      bool result = feedForward(inputs, 1, false, second);
      if(del_second)
         DeleteObj(second);
      DeleteObj(inputs);
      return result;
     }
//---
   CLayer *layer = inputNet.layers.At(inputLayer);
   if(!layer)
     {
      if(del_second)
         DeleteObj(second);
      ReturnFalse;
     }
   CNeuronBaseOCL *neuron = layer.At(0);
   layer = layers.At(0);
   if(!layer)
     {
      if(del_second)
         DeleteObj(second);
      ReturnFalse;
     }
   if(layer.At(0) != neuron)
     {
      CNeuronBaseOCL *temp = layer.At(0);
      if(temp.getConnections() <= 0)
        {
         if(!layer.Update(0, neuron))
           {
            if(del_second)
               DeleteObj(second);
            ReturnFalse;
           }
         else
            layer.FreeMode(false);
        }
      else
        {
         if(neuron.getOutputIndex() != temp.getOutputIndex())
            temp.getOutput().BufferSet(neuron.getOutputIndex());
         if(neuron.getGradientIndex() != temp.getGradientIndex())
            temp.getGradient().BufferSet(neuron.getGradientIndex());
        }
     }
//---
   for(int l = 1; l < layers.Total() && !IsStopped(); l++)
     {
      layer = layers.At(l);
      neuron = layer.At(0);
      layer = layers.At(l - 1);
      if(secondNeuron ? !neuron.FeedForwardSecond(layer.At(0), secondNeuron) : !neuron.FeedForward(layer.At(0), second))
        {
         if(del_second)
            DeleteObj(second);
         ReturnFalse;
        }
     }
//---
   if(del_second)
      DeleteObj(second);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::feedForward(CNet* inputNet, int inputLayer = -1, CBufferFloat *SecondInput = NULL)
  {
   if(!inputNet || !opencl)
      ReturnFalse;
   if(inputLayer < 0)
     {
      inputLayer = inputNet.layers.Total() + inputLayer;
      if(inputLayer < 0)
         ReturnFalse;
     }
//---
   if(!!SecondInput)
     {
      if(SecondInput.GetIndex() < 0)
         if(!SecondInput.BufferCreate(opencl))
            ReturnFalse;
     }
//---
   if(inputNet.opencl != opencl)
     {
      CBufferFloat *inputs;
      if(!inputNet.GetLayerOutput(inputLayer, inputs))
         ReturnFalse;
      bool result = feedForward(inputs, 1, false, SecondInput);
      return result;
     }
//---
   CLayer *layer = inputNet.layers.At(inputLayer);
   if(!layer)
      ReturnFalse;
   CNeuronBaseOCL *neuron = layer.At(0);
   layer = layers.At(0);
   if(!layer)
      ReturnFalse;
   if(layer.At(0) != neuron)
     {
      CNeuronBaseOCL *temp = layer.At(0);
      if(neuron.getOutputIndex() != temp.getOutputIndex())
         temp.getOutput().BufferSet(neuron.getOutputIndex());
      if(neuron.getGradientIndex() != temp.getGradientIndex())
         temp.getGradient().BufferSet(neuron.getGradientIndex());
      temp.SetActivationFunction((ENUM_ACTIVATION)neuron.Activation());
     }
//---
   for(int l = 1; l < layers.Total() && !IsStopped(); l++)
     {
      layer = layers.At(l);
      neuron = layer.At(0);
      layer = layers.At(l - 1);
      if(!neuron.FeedForward(layer.At(0), SecondInput))
         ReturnFalse;
      //neuron.getOutput().BufferRead();
     }
//---
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::backProp(CArrayFloat* targetVals, CNet *secondNet = NULL, int secondLayer = -1)
  {
   if(!secondNet || !secondNet.layers)
      return backPropOCL(targetVals);
//---
   if(secondLayer < 0)
      secondLayer = secondNet.layers.Total() - 1;
//---
   if(opencl != secondNet.GetOpenCL())
     {
      CBufferFloat *second;
      if(!secondNet.GetLayerOutput(secondLayer, second))
         ReturnFalse;
      if(!second)
         return backPropOCL(targetVals);
      CLayer *layer = secondNet.layers.At(secondLayer);
      CNeuronBaseOCL *neuron = layer.At(0);
      CBufferFloat gradient;
      neuron.getGradient().BufferRead();
      gradient.AssignArray(neuron.getGradient());
      if(!second.BufferCreate(opencl) || !gradient.BufferCreate(opencl))
        {
         DeleteObj(second);
         ReturnFalse;
        }
      bool result = backPropOCL(targetVals, second, GetPointer(gradient), false, (ENUM_ACTIVATION)neuron.Activation());
      DeleteObj(second);
      if(result)
        {
         second = neuron.getGradient();
         gradient.BufferRead();
         second.AssignArray(GetPointer(gradient));
         result = second.BufferWrite();
        }
      return result;
     }
//---
   CLayer *layer = layers.At(layers.Total() - 1);
   CNeuronBaseOCL *neuron = layer.At(0);
   float error = 0.0;
   if(neuron.Type() != defNeuronSoftMaxOCL
      && neuron.Type() != defNeuronMantisAttentionUnit
     )
     {
      vector<float> target, output;
      if(neuron.getOutputVal(output) < neuron.Neurons())
         ReturnFalse;
      if(output.HasNan() > 0)
        {
         printf("%s - %d -> Output has NaN", __FUNCTION__, __LINE__);
         ReturnFalse;
        }
      if(!target.Resize(targetVals.Total()))
         ReturnFalse;
      for(int n = 0; n < targetVals.Total(); n++)
         target[n] = targetVals.At(n);
      if(target.Size() > output.Size())
        {
         if(!target.Resize(output.Size()))
            ReturnFalse;
        }
      error = output.Loss(target, LOSS_MSE);
     }
   if(!neuron.calcOutputGradients(targetVals, error))
      ReturnFalse;
//---
   if(neuron.TrainMode())
     {
      if(!MathIsValidNumber(error))
         error = getRecentAverageError();
      backPropCount++;
      recentAverageError += (error - recentAverageError) / fmin(recentAverageSmoothingFactor, (float)backPropCount);
     }
//---
   return backPropGradient(secondNet, secondLayer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::backPropGradient(CBufferFloat *SecondInput = NULL, CBufferFloat *SecondGradient = NULL, int LastLayer = -1, bool load_weights = true)
  {
   if(CheckPointer(layers) == POINTER_INVALID || CheckPointer(opencl) == POINTER_INVALID)
      ReturnFalse;
   if(LastLayer < 0)
     {
      LastLayer = layers.Total() + LastLayer;
      if(LastLayer < 0)
         ReturnFalse;
     }
   CLayer *currentLayer = layers.At(LastLayer);
   CNeuronBaseOCL *neuron = NULL;
   if(CheckPointer(currentLayer) == POINTER_INVALID)
      ReturnFalse;
//--- Calc Hidden Gradients
   for(int layerNum = LastLayer - 1; layerNum >= 0 && !IsStopped(); layerNum--)
     {
      CLayer *nextLayer = currentLayer;
      currentLayer = layers.At(layerNum);
      if(CheckPointer(currentLayer) == POINTER_INVALID)
         ReturnFalse;
      neuron = currentLayer.At(0);
      if(!neuron || !nextLayer.At(0))
         ReturnFalse;
      if(!neuron.CalcHiddenGradients(nextLayer.At(0), SecondInput, SecondGradient))
        {
         PrintFormat("CNet backprop CalcHiddenGradients failed layer=%d neuron=%d", layerNum, neuron.Type());
         ReturnFalse;
        }
     }
//---
   CLayer *prevLayer = layers.At(LastLayer);
   for(int layerNum = LastLayer; layerNum > 0 && !IsStopped(); layerNum--)
     {
      currentLayer = prevLayer;
      prevLayer = layers.At(layerNum - 1);
      neuron = currentLayer.At(0);
      if(!neuron || !prevLayer.At(0))
         ReturnFalse;
      if(!neuron.UpdateInputWeights(prevLayer.At(0), SecondInput))
        {
         PrintFormat("CNet backprop UpdateInputWeights failed layer=%d neuron=%d", layerNum, neuron.Type());
         ReturnFalse;
        }
     }
//---
   if(!load_weights)
      return true;
//---
   bool result = false;
   for(int layerNum = 0; (layerNum < layers.Total() && !result && !IsStopped()); layerNum++)
     {
      currentLayer = layers.At(layerNum);
      CNeuronBaseOCL *temp = currentLayer.At(0);
      CNeuronConvOCL *conv = NULL;
      CNeuronConcatenate *conc = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      if(!temp.TrainMode())
        {
         if(layerNum == layers.Total() - 1)
            result = true;
         continue;
        }
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            if(!!conv.GetWeightsConv() && conv.GetWeightsConv().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         case defNeuronConcatenate:
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
              {
               result = true;
               break;
              }
            conc = temp;
            if(!!conc.getConcatWeights() && conc.getConcatWeights().BufferRead())
               result = true;
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            if(!!batch.getBatchOptions() && batch.getBatchOptions().BufferRead())
               result = true;
            else
               if(!!temp.getWeights() && temp.getWeights().BufferRead())
                  result = true;
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            if(!!lstm.getLSTMWeights() && lstm.getLSTMWeights().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         default:
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::backPropGradient(CNet* secondNet, int secondLayer = -1, int LastLayer = -1, bool load_weights = true)
  {
   if(CheckPointer(layers) == POINTER_INVALID || CheckPointer(opencl) == POINTER_INVALID ||
      !secondNet)
      ReturnFalse;
   if(LastLayer < 0)
     {
      LastLayer = layers.Total() + LastLayer;
      if(LastLayer < 0)
         ReturnFalse;
     }
   if(secondLayer < 0)
      secondLayer = secondNet.layers.Total() - 1;
   CLayer *currentLayer = layers.At(LastLayer);
   CNeuronBaseOCL *neuron = NULL;
   if(!currentLayer || !secondNet.layers.At(secondLayer))
      ReturnFalse;
   CNeuronBaseOCL *secondNeuron = ((CArrayObj *)secondNet.layers.At(secondLayer)).At(0);
   if(!secondNeuron)
      ReturnFalse;
//--- Calc Hidden Gradients
   for(int layerNum = LastLayer - 1; layerNum >= 0 && !IsStopped(); layerNum--)
     {
      CLayer *nextLayer = currentLayer;
      currentLayer = layers.At(layerNum);
      if(CheckPointer(currentLayer) == POINTER_INVALID)
         ReturnFalse;
      neuron = currentLayer.At(0);
      if(!neuron || !neuron.CalcHiddenGradientsSecond(nextLayer.At(0), secondNeuron))
         ReturnFalse;
     }
//---
   CLayer *prevLayer = layers.At(LastLayer);
   for(int layerNum = LastLayer; layerNum > 0 && !IsStopped(); layerNum--)
     {
      currentLayer = prevLayer;
      prevLayer = layers.At(layerNum - 1);
      neuron = currentLayer.At(0);
      if(!neuron.UpdateInputWeightsSecond(prevLayer.At(0), secondNeuron))
         ReturnFalse;
     }
//---
   if(!load_weights)
      return true;
//---
   bool result = false;
   for(int layerNum = 0; (layerNum < layers.Total() && !result && !IsStopped()); layerNum++)
     {
      currentLayer = layers.At(layerNum);
      CNeuronBaseOCL *temp = currentLayer.At(0);
      CNeuronConvOCL *conv = NULL;
      CNeuronConcatenate *conc = NULL;
      CNeuronBatchNormOCL *batch = NULL;
      CNeuronLSTMOCL *lstm = NULL;
      if(!temp)
         continue;
      if(!temp.TrainMode())
        {
         if(layerNum == layers.Total() - 1)
            result = true;
         continue;
        }
      switch(temp.Type())
        {
         case defNeuronConvOCL:
            conv = temp;
            if(!!conv.GetWeightsConv() && conv.GetWeightsConv().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         case defNeuronConcatenate:
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
              {
               result = true;
               break;
              }
            conc = temp;
            if(!!conc.getConcatWeights() && conc.getConcatWeights().BufferRead())
               result = true;
            break;
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            batch = temp;
            if(!!batch.getBatchOptions() && batch.getBatchOptions().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         case defNeuronLSTMOCL:
            lstm = temp;
            if(!!lstm.getLSTMWeights() && lstm.getLSTMWeights().BufferRead())
               result = true;
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
         default:
            if(!!temp.getWeights() && temp.getWeights().BufferRead())
               result = true;
            break;
        }
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::AlphasGradient(CNet* PolicyNet)
  {
   if(!PolicyNet || !PolicyNet.layers)
      ReturnFalse;
   int total = PolicyNet.layers.Total();
   if(total <= 0)
      ReturnFalse;
   CLayer *layer = PolicyNet.layers.At(total - 1);
   if(!layer || !layer.At(0))
      ReturnFalse;
   if(layer.At(0).Type() != defNeuronSoftActorCritic)
      return true;
//---
   CNeuronSoftActorCritic *neuron = layer.At(0);
   if(!layers)
      ReturnFalse;
   total = layers.Total();
   if(total <= 0 || !layers.At(total - 1))
      ReturnFalse;
   layer = layers.At(total - 1);
//---
   return neuron.calcAlphaGradients((CNeuronBaseOCL*) layer.At(0));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::GetLogProbs(vectorf & log_probs)
  {
//---
   if(!layers)
      ReturnFalse;
   int total = layers.Total();
   if(total <= 0 || !layers.At(total - 1))
      ReturnFalse;
   CLayer *layer = layers.At(total - 1);
   if(!layer.At(0) || layer.At(0).Type() != defNeuronSoftActorCritic)
      ReturnFalse;
//---
   CNeuronSoftActorCritic *neuron = layer.At(0);
//---
   return neuron.GetAlphaLogProbs(log_probs);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::CalcLogProbs(CBufferFloat* buffer)
  {
   if(!layers)
      ReturnFalse;
   int total = layers.Total();
   if(total <= 0 || !layers.At(total - 1))
      ReturnFalse;
   CLayer *layer = layers.At(total - 1);
   if(!layer.At(0) || layer.At(0).Type() != defNeuronSoftActorCritic)
      ReturnFalse;
//---
   CNeuronSoftActorCritic *neuron = layer.At(0);
//---
   return neuron.CalcLogProbs(buffer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::SetResult(float & result[])
  {
   if(!layers || layers.Total() <= 0)
      ReturnFalse;
//---
   CNeuronBaseOCL *output = ((CLayer*)layers.At(layers.Total() - 1)).At(0);
   if(!output)
      ReturnFalse;
   if(!output.getOutput().AssignArray(result) || !output.getOutput().BufferWrite())
      ReturnFalse;
   if(output.Type() == defNeuronSoftActorCritic && !output.ReCalcLogProbs())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronEmbeddingOCL::CNeuronEmbeddingOCL(void)
  {
   ArrayFree(a_Windows);
//---
   i_WindowsBuffer = INVALID_HANDLE;
   i_STDBuffer = INVALID_HANDLE;
   i_WindowOut = 0;
   i_StackSize = 1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNeuronEmbeddingOCL::~CNeuronEmbeddingOCL(void)
  {
   ArrayFree(a_Windows);
   if(!!OpenCL)
     {
      if(i_WindowsBuffer >= 0)
         OpenCL.BufferFree(i_WindowsBuffer);
      if(i_STDBuffer >= 0)
         OpenCL.BufferFree(i_STDBuffer);
     }
//---
   i_WindowsBuffer = INVALID_HANDLE;
   i_STDBuffer = INVALID_HANDLE;
   i_WindowOut = 0;
   i_StackSize = 1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronEmbeddingOCL::SetOpenCL(COpenCLMy * obj)
  {
   if(OpenCL == obj)
      return;
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!OpenCL)
      return;
//---
   i_WindowsBuffer = OpenCL.AddBuffer(sizeof(int) * a_Windows.Size(), CL_MEM_READ_WRITE);
   if(i_WindowsBuffer >= 0)
      OpenCL.BufferWrite(i_WindowsBuffer, a_Windows, 0, 0, a_Windows.Size());
   i_STDBuffer = OpenCL.AddBuffer(sizeof(float) * a_Windows.Size(), CL_MEM_READ_WRITE);
//---
   WeightsEmbedding.BufferCreate(OpenCL);
   FirstMomentumEmbed.BufferCreate(OpenCL);
   SecondMomentumEmbed.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint stack_size, uint window_out, uint & windows[])
  {
   if(CheckPointer(open_cl) == POINTER_INVALID || window_out <= 0 || windows.Size() <= 0 || stack_size <= 0)
      ReturnFalse;
   if(OpenCL != open_cl)
      DeleteObj(OpenCL);
   uint numNeurons = window_out * windows.Size() * stack_size;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, ADAM, 1))
      ReturnFalse;
//---
   uint weights = 0;
   ArrayCopy(a_Windows, windows);
   i_WindowOut = (int)window_out;
   i_StackSize = (int)stack_size;
   for(uint i = 0; i < windows.Size(); i++)
      weights += (windows[i] + 1) * window_out;
   if(!WeightsEmbedding.Reserve(weights))
      ReturnFalse;
   float k = 1.0f / sqrt((float)weights / (float)window_out);
   for(uint i = 0; i < weights; i++)
      if(!WeightsEmbedding.Add(k * (2 * GenerateWeight() - 1.0f)*WeightsMultiplier))
         ReturnFalse;
   if(!WeightsEmbedding.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!FirstMomentumEmbed.BufferInit(weights, 0))
      ReturnFalse;
   if(!FirstMomentumEmbed.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!SecondMomentumEmbed.BufferInit(weights, 0))
      ReturnFalse;
   if(!SecondMomentumEmbed.BufferCreate(OpenCL))
      ReturnFalse;
//---
   i_WindowsBuffer = OpenCL.AddBuffer(sizeof(int) * a_Windows.Size(), CL_MEM_READ_WRITE);
   if(i_WindowsBuffer < 0 || !OpenCL.BufferWrite(i_WindowsBuffer, a_Windows, 0, 0, a_Windows.Size()))
      ReturnFalse;
   i_STDBuffer = OpenCL.AddBuffer(sizeof(float) * a_Windows.Size(), CL_MEM_READ_WRITE);
   if(i_STDBuffer < 0)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   bool res = true;
   int check_w[];
   ArrayResize(check_w, a_Windows.Size());
   if(!OpenCL.BufferRead(i_WindowsBuffer, check_w, 0, 0, a_Windows.Size()))
      ReturnFalse;
   for(uint i = 0; (i < a_Windows.Size() && res); i++)
      res = check_w[i] == a_Windows[i];
   if(!res)
     {
      if(!OpenCL.BufferWrite(i_WindowsBuffer, a_Windows, 0, 0, a_Windows.Size()))
         ReturnFalse;
     }
//---
   setBuffer(def_k_Embedding, def_k_emb_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_Embedding, def_k_emb_outputs, getOutputIndex())
   setBuffer(def_k_Embedding, def_k_emb_std, i_STDBuffer)
   setBuffer(def_k_Embedding, def_k_emb_weights, WeightsEmbedding.GetIndex())
   setBuffer(def_k_Embedding, def_k_emb_windows, i_WindowsBuffer)
   setArgument(def_k_Embedding, def_k_emb_stack_size, i_StackSize)
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2]   = {i_WindowOut, a_Windows.Size()};
   uint local_work_size[2]    = {i_WindowOut, 1};
   kernelExecuteLoc(def_k_Embedding, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::Clear(void)
  {
   if(!Output.BufferInit(Output.Total(), 0))
      ReturnFalse;
   if(!OpenCL)
      return true;
//---
   return Output.BufferWrite();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   setBuffer(def_k_EmbeddingHiddenGradient, def_k_ehg_inputs_gradient, NeuronOCL.getGradientIndex())
   setBuffer(def_k_EmbeddingHiddenGradient, def_k_ehg_outputs_gradient, getGradientIndex())
   setBuffer(def_k_EmbeddingHiddenGradient, def_k_ehg_std, i_STDBuffer)
   setBuffer(def_k_EmbeddingHiddenGradient, def_k_ehg_weights, WeightsEmbedding.GetIndex())
   setBuffer(def_k_EmbeddingHiddenGradient, def_k_ehg_windows, i_WindowsBuffer)
   setArgument(def_k_EmbeddingHiddenGradient, def_k_ehg_window_out, i_WindowOut)
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1]   = {NeuronOCL.Neurons()};
   kernelExecute(def_k_EmbeddingHiddenGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_gradient, getGradientIndex())
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_std, i_STDBuffer)
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_weights, WeightsEmbedding.GetIndex())
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_matrix_m, FirstMomentumEmbed.GetIndex())
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_matrix_v, SecondMomentumEmbed.GetIndex())
   setBuffer(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_windows, i_WindowsBuffer)
   setArgument(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_window_out, i_WindowOut)
   setArgument(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_learning_rate, lr)
   setArgument(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_b1, b1)
   setArgument(def_k_EmbeddingUpdateWeightsAdam, def_k_euw_b2, b2)
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1]   = {WeightsEmbedding.Total()};
   kernelExecute(def_k_EmbeddingUpdateWeightsAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!WeightsEmbedding.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronEmbeddingOCL *temp = source;
   if(WeightsEmbedding.Total() != temp.WeightsEmbedding.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {WeightsEmbedding.Total()};
   ResetLastError();
   if(tau != 1.0f && optimization == ADAM)
     {
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_target, WeightsEmbedding.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_source, temp.WeightsEmbedding.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_m, FirstMomentumEmbed.GetIndex())
      setBuffer(def_k_SoftUpdateAdam, def_k_sua_matrix_v, SecondMomentumEmbed.GetIndex())
      setArgument(def_k_SoftUpdateAdam, def_k_sua_tau, (float)tau)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b1, (float)b1)
      setArgument(def_k_SoftUpdateAdam, def_k_sua_b2, (float)b2)
      kernelExecute(def_k_SoftUpdateAdam, global_work_offset, global_work_size)
     }
   else
     {
      setBuffer(def_k_SoftUpdate, def_k_su_target, WeightsEmbedding.GetIndex())
      setBuffer(def_k_SoftUpdate, def_k_su_source, temp.WeightsEmbedding.GetIndex())
      setArgument(def_k_SoftUpdate, def_k_su_tau, (float)tau)
      kernelExecute(def_k_SoftUpdate, global_work_offset, global_work_size)
     }
#ifdef  _DEBUG
   if(!WeightsEmbedding.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!WeightsEmbedding.Save(file_handle))
      ReturnFalse;
   if(!FirstMomentumEmbed.Save(file_handle))
      ReturnFalse;
   if(!SecondMomentumEmbed.Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)a_Windows.Size()) != INT_VALUE)
      ReturnFalse;
   if(FileWriteArray(file_handle, a_Windows, 0, (int)a_Windows.Size()) != a_Windows.Size())
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, i_WindowOut) != INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, i_StackSize) != INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronEmbeddingOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!WeightsEmbedding.Load(file_handle))
      ReturnFalse;
   if(!FirstMomentumEmbed.Load(file_handle))
      ReturnFalse;
   if(!SecondMomentumEmbed.Load(file_handle))
      ReturnFalse;
//---
   int size = FileReadInteger(file_handle);
   ArrayResize(a_Windows, size);
   if(FileReadArray(file_handle, a_Windows, 0, size) != size)
      ReturnFalse;
//---
   i_WindowOut = FileReadInteger(file_handle);
   i_StackSize = FileReadInteger(file_handle);
//---
   if(!OpenCL)
      return true;
//---
   if(i_WindowsBuffer != INVALID_HANDLE)
      OpenCL.BufferFree(i_WindowsBuffer);
   i_WindowsBuffer = OpenCL.AddBuffer(sizeof(int) * a_Windows.Size(), CL_MEM_READ_WRITE);
   if(i_WindowsBuffer >= 0)
      if(!OpenCL.BufferWrite(i_WindowsBuffer, a_Windows, 0, 0, size))
         ReturnFalse;
   if(i_STDBuffer != INVALID_HANDLE)
      OpenCL.BufferFree(i_STDBuffer);
   i_STDBuffer = OpenCL.AddBuffer(sizeof(float) * a_Windows.Size(), CL_MEM_READ_WRITE);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::Clear(void)
  {
   if(!opencl)
      return true;
   if(!layers || layers.Total() <= 0)
      return true;
   for(int i = 0; i < layers.Total(); i++)
     {
      CLayer *layer = layers.At(i);
      if(!layer)
         ReturnFalse;
      for(int n = 0; n < layer.Total(); n++)
        {
         CNeuronBaseOCL *neuron = layer.At(n);
         if(!neuron)
           {
            PrintFormat("%s: layer=%d neuron=%d is NULL", __FUNCTION__, i, n);
            ReturnFalse;
           }
         if(!neuron.Clear())
           {
            CBufferFloat *output = neuron.getOutput();
            CBufferFloat *gradient = neuron.getGradient();
            PrintFormat("%s: layer=%d neuron=%d type=%d output=%d index=%d gradient=%d index=%d",
                        __FUNCTION__, i, n, neuron.Type(), (output ? output.Total() : -1),
                        (output ? output.GetIndex() : -1), (gradient ? gradient.Total() : -1),
                        (gradient ? gradient.GetIndex() : -1));
            ReturnFalse;
           }
        }
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::feedForward(CArrayFloat* inputVals, int window = 1, bool tem = true, CNet *secondNet = NULL, int secondLayer = -1)
  {
   if(CheckPointer(layers) == POINTER_INVALID || CheckPointer(inputVals) == POINTER_INVALID || layers.Total() <= 1)
      ReturnFalse;
//---
   CBufferFloat *SecondInput = NULL;
   CNeuronBaseOCL *SecondNeuron = NULL;
   bool del_second = false;
   if(!!secondNet)
     {
      if(secondLayer < 0)
         secondLayer = secondNet.layers.Total() - 1;
      if(secondNet.GetOpenCL() != opencl)
        {
         secondNet.GetLayerOutput(secondLayer, SecondInput);
         if(!!SecondInput)
           {
            if(!SecondInput.BufferCreate(opencl))
              {
               DeleteObj(SecondInput);
               ReturnFalse;
              }
            del_second = true;
           }
        }
      else
        {
         if(secondNet.layers.Total() <= secondLayer)
            ReturnFalse;
         CLayer *layer = secondNet.layers.At(secondLayer);
         CNeuronBaseOCL *neuron = layer.At(0);
         SecondNeuron = neuron;
         SecondInput = neuron.getOutput();
        }
     }
//---
   CLayer *previous = NULL;
   CLayer *current = layers.At(0);
   bool result = true;
   int total = MathMin(current.Total(), inputVals.Total());
   CNeuronBase *neuron = NULL;
   if(CheckPointer(opencl) == POINTER_INVALID)
     {
      for(int i = 0; i < total; i++)
        {
         neuron = current.At(i);
         if(CheckPointer(neuron) == POINTER_INVALID)
           {
            result = false;
            break;
           }
         int pos = i;
         int d = 0;
         if(window > 1)
           {
            d = i % window;
            pos = (i - d) / window;
           }
         neuron.setOutputVal(inputVals.At(i) + (float)(tem ? (d % 2 == 0 ? sin(pos / MathPow(10000.0, (2.0 * d + 1.0) / (window + 1.0))) : cos(pos / MathPow(10000.0, (2.0 * d + 1.0) / (window + 1.0)))) : 0));
         if(tem)
            inputVals.Update(i, neuron.getOutputVal());
        }
     }
   else
     {
      CNeuronBaseOCL *neuron_ocl = current.At(0);
      CBufferFloat *inputs = neuron_ocl.getOutput();
      if(tem)
        {
         int total_data = inputVals.Total();
         for(int d = 0; d < total_data; d++)
           {
            int pos = d;
            int dim = 0;
            if(window > 1)
              {
               dim = d % window;
               pos = (d - dim) / window;
              }
            float value = (float)(pos / MathPow(10000.0, (2.0 * dim + 1.0) / (window + 1.0)));
            value = (float)(inputVals.At(d) + (dim % 2 == 0 ? sin(value) : cos(value)));
            if(!inputVals.Update(d, value))
              {
               result = false;
               break;
              }
           }
        }
      if(!result || !inputs.Fill(inputVals))
        {
         if(del_second)
            DeleteObj(SecondInput);
         ReturnFalse;
        }
     }
//---
   CObject *temp = NULL;
   for(int l = 1; (l < layers.Total() && result && !IsStopped()); l++)
     {
      previous = current;
      current = layers.At(l);
      if(CheckPointer(current) == POINTER_INVALID)
        {
         result = false;
         break;
        }
      //---
      if(CheckPointer(opencl) != POINTER_INVALID)
        {
         CNeuronBaseOCL *current_ocl = current.At(0);
         CNeuronBaseOCL *previous_ocl = previous.At(0);
         if(SecondNeuron ? !current_ocl.FeedForwardSecond(previous.At(0), SecondNeuron) : !current_ocl.FeedForward(previous.At(0), SecondInput))
           {
            PrintFormat("CNet::feedForward: layer=%d type=%d previous_type=%d previous=%d second_type=%d second=%d",
                        l, current_ocl.Type(), previous_ocl.Type(), previous_ocl.Neurons(),
                        (SecondNeuron ? SecondNeuron.Type() : -1), (SecondNeuron ? SecondNeuron.Neurons() : 0));
            result = false;
            break;
           }
         continue;
        }
      //---
      total = current.Total();
      if(current.At(0).Type() == defNeuron)
         total--;
      //---
      for(int n = 0; n < total && !IsStopped(); n++)
        {
         neuron = current.At(n);
         if(CheckPointer(neuron) == POINTER_INVALID)
           {
            result = false;
            break;
           }
         if(previous.At(0).Type() == defNeuron)
           {
            temp = previous;
            if(!neuron.feedForward(temp))
              {
               result = false;
               break;
              }
            continue;
           }
         if(neuron.Type() == defNeuron)
           {
            if(n == 0)
              {
               CLayer *temp_l = new CLayer(total);
               if(CheckPointer(temp_l) == POINTER_INVALID)
                 {
                  result = false;
                  break;
                 }
               CNeuronProof *proof = NULL;
               for(int p = 0; p < previous.Total(); p++)
                 {
                  proof = previous.At(p);
                  if(CheckPointer(proof) == POINTER_INVALID)
                    {
                     result = false;
                     break;
                    }
                  temp_l.AddArray(proof.getOutputLayer());
                 }
               temp = temp_l;
              }
            if(!neuron.feedForward(temp))
              {
               result = false;
               break;
              }
            if(n == total - 1)
              {
               CLayer *temp_l = temp;
               temp_l.FreeMode(false);
               temp_l.Shutdown();
               DeleteObj(temp_l);
              }
            continue;
           }
         temp = previous.At(n);
         if(CheckPointer(temp) == POINTER_INVALID)
           {
            result = false;
            break;
           }
         if(!neuron.feedForward(temp))
           {
            result = false;
            break;
           }
        }
     }
//---
   if(del_second)
      DeleteObj(SecondInput);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPositionEncoder::Init(uint numOutputs, uint myIndex,
                                  COpenCLMy * open_cl, uint count, uint window,
                                  ENUM_OPTIMIZATION optimization_type, uint batch
                                 )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, count * window, optimization_type, batch))
      ReturnFalse;
//---
   matrix<float> pe = matrix<float>::Zeros(count, window);
   vector<float> position = vector<float>::Ones(count);
   position = position.CumSum() - 1;
   for(uint i = 0; i < window; i += 2)
     {
      vector<float> temp = position / MathPow(10000.0f, 2.0f * uint(i / 2) / window);
      pe.Col(MathSin(temp), i);
      if((i + 1) < window)
         pe.Col(MathCos(temp), i + 1);
     }
   if(!PositionEncoder.AssignArray(pe))
      ReturnFalse;
//---
   return PositionEncoder.BufferCreate(open_cl);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPositionEncoder::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!Gradient || Gradient != NeuronOCL.getGradient())
     {
      if(!!Gradient)
         DeleteObj(Gradient);
      Gradient = NeuronOCL.getGradient();
     }
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
   setBuffer(def_k_MatrixSum, def_k_sum_matrix1, NeuronOCL.getOutputIndex())
   setBuffer(def_k_MatrixSum, def_k_sum_matrix2, PositionEncoder.GetIndex())
   setBuffer(def_k_MatrixSum, def_k_sum_matrix_out, Output.GetIndex())
   setArgument(def_k_MatrixSum, def_k_sum_dimension, (int)1)
   setArgument(def_k_MatrixSum, def_k_sum_multiplyer, 1.0f)
   setArgument(def_k_MatrixSum, def_k_sum_shift_in1, 0.0f)
   setArgument(def_k_MatrixSum, def_k_sum_shift_in2, 0.0f)
   setArgument(def_k_MatrixSum, def_k_sum_shift_out, 0.0f)
   kernelExecute(def_k_MatrixSum, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPositionEncoder::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   return PositionEncoder.Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPositionEncoder::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   PositionEncoder.BufferFree();
   if(!PositionEncoder.Load(file_handle))
      ReturnFalse;
   if(!PositionEncoder.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPositionEncoder::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(!!OpenCL)
      PositionEncoder.BufferCreate(OpenCL);
   else
      PositionEncoder.BufferFree();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTransposeOCL::Init(uint numOutputs, uint myIndex,
                               COpenCLMy * open_cl, uint count,
                               uint window,
                               ENUM_OPTIMIZATION optimization_type,
                               uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, count * window,
                            optimization_type, batch))
      ReturnFalse;
//---
   iWindow = window;
   iCount = count;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTransposeOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   SetActivationFunction((ENUM_ACTIVATION)NeuronOCL.Activation());
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iCount, iWindow};
   setBuffer(def_k_Transpose, def_k_tr_matrix_in, NeuronOCL.getOutputIndex())
   setBuffer(def_k_Transpose, def_k_tr_matrix_out, Output.GetIndex())
   kernelExecute(def_k_Transpose, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTransposeOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {iWindow, iCount};
   setBuffer(def_k_Transpose, def_k_tr_matrix_out, NeuronOCL.getGradientIndex())
   setBuffer(def_k_Transpose, def_k_tr_matrix_in, Gradient.GetIndex())
   kernelExecute(def_k_Transpose, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTransposeOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iCount) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iWindow) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTransposeOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   iCount = (uint)FileReadInteger(file_handle);
   iWindow = (uint)FileReadInteger(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::ElementMultGrad(const CBufferFloat* input1, CBufferFloat* input1_gr, const CBufferFloat* input2, CBufferFloat* input2_gr, const CBufferFloat* output, const int activation1, const int activation2)
  {
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
   ResetLastError();
   setBuffer(def_k_CGConv_HiddenGradient, def_k_cgc_matrix_f, input1.GetIndex())
   setBuffer(def_k_CGConv_HiddenGradient, def_k_cgc_matrix_fg, input1_gr.GetIndex())
   setBuffer(def_k_CGConv_HiddenGradient, def_k_cgc_matrix_s, input2.GetIndex())
   setBuffer(def_k_CGConv_HiddenGradient, def_k_cgc_matrix_sg, input2_gr.GetIndex())
   setBuffer(def_k_CGConv_HiddenGradient, def_k_cgc_matrix_g, output.GetIndex())
   setArgument(def_k_CGConv_HiddenGradient, def_k_cgc_activationf, activation1)
   setArgument(def_k_CGConv_HiddenGradient, def_k_cgc_activations, activation2)
   kernelExecute(def_k_CGConv_HiddenGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!input2_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::LoadInsideLayer(int file_handle, CNeuronBaseOCL* neuron)
  {
   if(!neuron)
      ReturnFalse;
   int type = FileReadInteger(file_handle);
   if(type != neuron.Type())
      ReturnFalse;
   neuron.AttachOpenCL(OpenCL);
   if(!neuron.Load(file_handle))
      ReturnFalse;
   neuron.SetOpenCL(OpenCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                         uint window, uint window_out, uint count,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * count, optimization_type, batch))
      ReturnFalse;
//---
   if(!cConvs[0].Init(0, 0, OpenCL, window, window, window_out, count, 1, optimization, iBatch))
      ReturnFalse;
   if(!cNorm[0].Init(0, 1, OpenCL, window_out * count, iBatch, optimization))
      ReturnFalse;
   cNorm[0].SetActivationFunction(GELU);
   if(!cConvs[1].Init(0, 2, OpenCL, window_out, window_out, window_out, count, 1, optimization, iBatch))
      ReturnFalse;
   if(!cNorm[1].Init(0, 3, OpenCL, window_out * count, iBatch, optimization))
      ReturnFalse;
   cNorm[1].SetActivationFunction(None);
//---
   if(!cConvs[2].Init(0, 4, OpenCL, window, window, window_out, count, 1, optimization, iBatch))
      ReturnFalse;
   if(!cNorm[2].Init(0, 5, OpenCL, window_out * count, iBatch, optimization))
      ReturnFalse;
   cNorm[2].SetActivationFunction(None);
//---
   if(!cTemp.Init(0, 6, OpenCL, window * count, optimization, batch))
      ReturnFalse;
//---
   cNorm[1].SetGradient(getGradient());
   cNorm[2].SetGradient(getGradient());
   SetActivationFunction(None);
   iWindowOut = (int)window_out;
//---
   for(int i = 0; i < 3; i++)
      cConvs[i].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//---
   if(!cConvs[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNorm[0].FeedForward(GetPointer(cConvs[0])))
      ReturnFalse;
   if(!cConvs[1].FeedForward(GetPointer(cNorm[0])))
      ReturnFalse;
   if(!cNorm[1].FeedForward(GetPointer(cConvs[1])))
      ReturnFalse;
//---
   if(cTemp.getOutput() != NeuronOCL.getOutput())
      cTemp.SetOutput(NeuronOCL.getOutput(), true);
//---
   if(!cConvs[2].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNorm[2].FeedForward(GetPointer(cConvs[2])))
      ReturnFalse;
//---
   if(!SumAndNormalize(cNorm[1].getOutput(), cNorm[2].getOutput(), Output, iWindowOut, true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::calcInputGradients(CNeuronBaseOCL* prevLayer)
  {
   if(!cConvs[2].CalcHiddenGradients(cNorm[2].AsObject()))
      ReturnFalse;
   if(!cTemp.CalcHiddenGradients(cConvs[2].AsObject()))
      ReturnFalse;
//---
   if(!cConvs[1].CalcHiddenGradients(cNorm[1].AsObject()))
      ReturnFalse;
   if(!cNorm[0].CalcHiddenGradients(cConvs[1].AsObject()))
      ReturnFalse;
   if(!prevLayer.CalcHiddenGradients(cNorm[0].AsObject()))
      ReturnFalse;
//---
   if(!SumAndNormalize(prevLayer.getGradient(), cTemp.getGradient(), prevLayer.getGradient(), iWindowOut, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   for(int i = 2; i >= 0; i--)
     {
      if(!cNorm[i].UpdateInputWeights(cConvs[i].AsObject()))
         ReturnFalse;
     }
   if(!cConvs[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cConvs[1].UpdateInputWeights(cNorm[0].AsObject()))
      ReturnFalse;
   if(!cConvs[2].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CResidualConv *temp = source;
   for(uint i = 0; i < cNorm.Size(); i--)
      if(!cNorm[i].WeightsUpdate(temp.cNorm[i].AsObject(), tau))
         ReturnFalse;
   for(uint i = 0; i < cConvs.Size(); i--)
      if(!cConvs[i].WeightsUpdate(temp.cConvs[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CResidualConv::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(int i = 0; i < 3; i++)
     {
      cConvs[i].SetOpenCL(OpenCL);
      cNorm[i].SetOpenCL(OpenCL);
     }
   cTemp.SetOpenCL(OpenCL);
//---
   if(cNorm[2].getGradient() != getGradient())
     {
      SetGradient(cNorm[2].getGradient());
      cNorm[1].SetGradient(cNorm[2].getGradient());
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CResidualConv::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   for(int i = 0; i < 3; i++)
     {
      cConvs[i].TrainMode(bTrain);
      cNorm[i].TrainMode(bTrain);
     }
   cTemp.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   for(int i = 0; i < 3; i++)
     {
      if(!cConvs[i].Save(file_handle))
         ReturnFalse;
      if(!cNorm[i].Save(file_handle))
         ReturnFalse;
     }
   if(!cTemp.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iWindowOut) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CResidualConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   for(int i = 0; i < 3; i++)
     {
      if(!LoadInsideLayer(file_handle, GetPointer(cConvs[i])))
         ReturnFalse;
      if(!LoadInsideLayer(file_handle, GetPointer(cNorm[i])))
         ReturnFalse;
     }
   if(!LoadInsideLayer(file_handle, GetPointer(cTemp)))
      ReturnFalse;
   iWindowOut = FileReadInteger(file_handle);
//---
   if(cNorm[2].getGradient() != getGradient())
     {
      SetGradient(cNorm[2].getGradient());
      cNorm[1].SetGradient(cNorm[2].getGradient());
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::CopyBufferRaw(CBufferFloat *source, CBufferFloat *destination,
                                   const uint total)
  {
   if(!OpenCL || !source || !destination || total == 0 ||
      source.Total() != (int)total || destination.Total() != (int)total ||
      source.GetIndex() < 0 || destination.GetIndex() < 0 ||
      source.GetOpenCL() != OpenCL || destination.GetOpenCL() != OpenCL)
      ReturnFalse;
   if(source == destination)
      return(true);
   const int items = (int)total;
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {total};
   setBuffer(def_k_CopyBufferRaw, def_k_copy_raw_source, source.GetIndex())
   setBuffer(def_k_CopyBufferRaw, def_k_copy_raw_destination, destination.GetIndex())
   setArgument(def_k_CopyBufferRaw, def_k_copy_raw_total, items)
   kernelExecute(def_k_CopyBufferRaw, global_work_offset, global_work_size)
//---
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Concat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* output, int window1, int window2, int count)
  {
   return Concat(input1, input2, input1, input2, output, window1, window2, 0, 0, count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Concat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* input3, CBufferFloat* output, int window1, int window2, int window3, int count)
  {
   return Concat(input1, input2, input3, input1, output, window1, window2, window3, 0, count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Concat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* input3, CBufferFloat* input4, CBufferFloat* output, int window1, int window2, int window3, int window4, int count)
  {
   return Concat(input1.GetIndex(), input2.GetIndex(), (!input3 ? input1.GetIndex() : input3.GetIndex()), (!input4 ? input1.GetIndex() : input4.GetIndex()), output.GetIndex(), window1, window2, (!input3 ? 0 : window3), (!input3 ? 0 : window3), count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Concat(int input1, int input2, int input3, int input4, int output, int window1, int window2, int window3, int window4, int count)
  {
   uint global_work_offset[2] = {0};
   uint global_work_size[2];
   global_work_size[0] = count;
   global_work_size[1] = MathMax(MathMax(window1, window2), MathMax(window3, window4));
   setBuffer(def_k_ConcatenateMatrix, def_k_conc_input1, input1)
   setArgument(def_k_ConcatenateMatrix, def_k_conc_window1, window1)
   setBuffer(def_k_ConcatenateMatrix, def_k_conc_input2, input2)
   setArgument(def_k_ConcatenateMatrix, def_k_conc_window2, window2)
   setBuffer(def_k_ConcatenateMatrix, def_k_conc_input3, input3)
   setArgument(def_k_ConcatenateMatrix, def_k_conc_window3, window3)
   setBuffer(def_k_ConcatenateMatrix, def_k_conc_input4, input4)
   setArgument(def_k_ConcatenateMatrix, def_k_conc_window4, window4)
   setBuffer(def_k_ConcatenateMatrix, def_k_conc_out, output)
   kernelExecute(def_k_ConcatenateMatrix, global_work_offset, global_work_size)
#ifdef  _DEBUG
   ulong  s = OpenCL.GetBufferSize(output) / sizeof(float);
   vector<float> temp = vector<float>::Zeros(s);
   if(!OpenCL.BufferToVector(output, temp))
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DeConcat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* output, int window1, int window2, int count)
  {
   return DeConcat(input1, input2, input1, input2, output, window1, window2, 0, 0, count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DeConcat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* input3, CBufferFloat* output, int window1, int window2, int window3, int count)
  {
   return DeConcat(input1, input2, input3, input1, output, window1, window2, window3, 0, count);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DeConcat(CBufferFloat* input1, CBufferFloat* input2, CBufferFloat* input3, CBufferFloat* input4, CBufferFloat* output, int window1, int window2, int window3, int window4, int count)
  {
   uint global_work_offset[2] = {0};
   uint global_work_size[2] = {count, MathMax(MathMax(window1, window2), MathMax(window3, window4))};
   setBuffer(def_k_DeconcatenateMatrix, def_k_dconc_output1, input1.GetIndex())
   setArgument(def_k_DeconcatenateMatrix, def_k_dconc_window1, window1)
   setBuffer(def_k_DeconcatenateMatrix, def_k_dconc_output2, input2.GetIndex())
   setArgument(def_k_DeconcatenateMatrix, def_k_dconc_window2, window2)
   setBuffer(def_k_DeconcatenateMatrix, def_k_dconc_output3, (!input3 ? input1.GetIndex() : input3.GetIndex()))
   setArgument(def_k_DeconcatenateMatrix, def_k_dconc_window3, (!input3 ? 0 : window3))
   setBuffer(def_k_DeconcatenateMatrix, def_k_dconc_output4, (!input4 ? input1.GetIndex() : input4.GetIndex()))
   setArgument(def_k_DeconcatenateMatrix, def_k_dconc_window4, (!input4 ? 0 : window4))
   setBuffer(def_k_DeconcatenateMatrix, def_k_dconc_inputs, output.GetIndex())
   kernelExecute(def_k_DeconcatenateMatrix, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRevINDenormOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, int NormLayer, CNeuronBatchNormOCL * normLayer)
  {
   if(NormLayer >= 0)
     {
      if(!normLayer)
         ReturnFalse;
      switch(normLayer.Type())
        {
         case defNeuronBatchNormOCL:
         case defNeuronBatchNormWithNoise:
            break;
         default:
            ReturnFalse;
        }
      if(BatchOptions != normLayer.BatchOptions)
         DeleteObj(BatchOptions);
      if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, numNeurons, normLayer.iBatchSize, normLayer.Optimization()))
         ReturnFalse;
      if(BatchOptions != normLayer.BatchOptions)
        {
         DeleteObj(BatchOptions);
         BatchOptions = normLayer.BatchOptions;
        }
     }
   else
      if(!CNeuronBatchNormOCL::Init(numOutputs, myIndex, open_cl, numNeurons, 0, ADAM))
         ReturnFalse;
   iBatchNormLayer = NormLayer;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRevINDenormOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   PrevLayer = NeuronOCL;
//---
   if(!BatchOptions)
      iBatchSize = 0;
   if(iBatchSize <= 1)
     {
      activation = (ENUM_ACTIVATION)NeuronOCL.Activation();
      return true;
     }
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   ResetLastError();
   setBuffer(def_k_RevInFeedForward, def_k_revffinputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_RevInFeedForward, def_k_revffoptions, BatchOptions.GetIndex())
   setBuffer(def_k_RevInFeedForward, def_k_revffoutput, Output.GetIndex())
   setArgument(def_k_RevInFeedForward, def_k_revffoptions_size, (int)BatchOptions.Total())
   setArgument(def_k_RevInFeedForward, def_k_revffoptimization, (int)optimization)
   kernelExecute(def_k_RevInFeedForward, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRevINDenormOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   if(iBatchSize <= 1)
      return true;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {Neurons()};
   ResetLastError();
   setBuffer(def_k_RevInHiddenGraddient, def_k_revhginputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_RevInHiddenGraddient, def_k_revhginputs_gr, NeuronOCL.getGradientIndex())
   setBuffer(def_k_RevInHiddenGraddient, def_k_revhgoptions, BatchOptions.GetIndex())
   setBuffer(def_k_RevInHiddenGraddient, def_k_revhgoutput_gr, Gradient.GetIndex())
   setArgument(def_k_RevInHiddenGraddient, def_k_revhgoptions_size, (int)BatchOptions.Total())
   setArgument(def_k_RevInHiddenGraddient, def_k_revhgoptimization, (int)optimization)
   setArgument(def_k_RevInHiddenGraddient, def_k_revhgactivation, (int)NeuronOCL.Activation())
   kernelExecute(def_k_RevInHiddenGraddient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRevINDenormOCL::Save(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iBatchNormLayer, INT_VALUE) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronRevINDenormOCL::Load(const int file_handle)
  {
   if(!CNeuronBatchNormOCL::Load(file_handle))
      ReturnFalse;
   iBatchNormLayer = FileReadInteger(file_handle, INT_VALUE);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNet::OpenCLInit(void)
  {
   if(!opencl)
     {
      opencl = new COpenCLMy();
      if(!opencl)
         ReturnFalse;
      bOwnOpenCL = true;
     }
   if(!opencl.Initialize(cl_program, true))
     {
      DeleteObj(opencl);
      bOwnOpenCL = false;
      ReturnFalse;
     }
//--- create kernels
   opencl.SetKernelsCount(313);
   CreateKernel(def_k_FeedForward, "FeedForward")
   CreateKernel(def_k_CalcOutputGradient, "CalcOutputGradient")
   CreateKernel(def_k_CalcHiddenGradient, "CalcHiddenGradient")
   CreateKernel(def_k_UpdateWeightsMomentum, "UpdateWeightsMomentum")
   CreateKernel(def_k_UpdateWeightsAdam, "UpdateWeightsAdam")
   CreateKernel(def_k_UpdateWeightsLS, "UpdateWeightsLS")
   CreateKernel(def_k_AttentionGradients, "AttentionInsideGradients")
   CreateKernel(def_k_AttentionOut, "AttentionOut")
   CreateKernel(def_k_AttentionScore, "AttentionScore")
   CreateKernel(def_k_CalcHiddenGradientConv, "CalcHiddenGradientConv")
   CreateKernel(def_k_CalcInputGradientProof, "CalcInputGradientProof")
   CreateKernel(def_k_FeedForwardConv, "FeedForwardConv")
   CreateKernel(def_k_FeedForwardProof, "FeedForwardProof")
   CreateKernel(def_k_MatrixSum, "SumMatrix")
   CreateKernel(def_k_Matrix5Sum, "Sum5Matrix")
   CreateKernel(def_k_UpdateWeightsConvAdam, "UpdateWeightsConvAdam")
   CreateKernel(def_k_UpdateWeightsConvMomentum, "UpdateWeightsConvMomentum")
   CreateKernel(def_k_UpdateWeightsConvLS, "UpdateWeightsConvLS")
   CreateKernel(def_k_Normalize, "Normalize")
   CreateKernel(def_k_NormalizeWeights, "NormalizeWeights")
   CreateKernel(def_k_ConcatenateMatrix, "ConcatenateBuffers")
   CreateKernel(def_k_DeconcatenateMatrix, "DeconcatenateBuffers")
   CreateKernel(def_k_MHAttentionGradients, "MHAttentionInsideGradients")
   CreateKernel(def_k_MHAttentionScore, "MHAttentionScore")
   CreateKernel(def_k_MHAttentionOut, "MHAttentionOut")
   CreateKernel(def_k_Dropout, "Dropout")
   CreateKernel(def_k_BatchFeedForward, "BatchFeedForward")
   CreateKernel(def_k_CalcHiddenGradientBatch, "CalcHiddenGradientBatch")
   CreateKernel(def_k_UpdateBatchOptionsMomentum, "UpdateBatchOptionsMomentum")
   CreateKernel(def_k_UpdateBatchOptionsAdam, "UpdateBatchOptionsAdam")
   CreateKernel(def_k_VAEFeedForward, "VAE_FeedForward")
   CreateKernel(def_k_VAECalcHiddenGradient, "VAE_CalcHiddenGradient")
   CreateKernel(def_k_LSTM_ConcatenatedGradient, "LSTM_ConcatenatedGradient")
   CreateKernel(def_k_LSTM_HiddenGradient, "LSTM_HiddenGradient")
   CreateKernel(def_k_LSTM_FeedForward, "LSTM_FeedForward")
   CreateKernel(def_k_LSTM_UpdateWeightsAdam, "LSTM_UpdateWeightsAdam")
   CreateKernel(def_k_SoftMax_FeedForward, "SoftMax_FeedForward")
   CreateKernel(def_k_SoftMax_HiddenGradient, "SoftMax_HiddenGradient")
   CreateKernel(def_k_SoftMax_OutputGradient, "SoftMax_OutputGradient")
   CreateKernel(def_k_FQF_Cosine, "FQF_Cosine")
   CreateKernel(def_k_FQF_Output, "FQF_Output")
   CreateKernel(def_k_FQF_OutputGradient, "FQF_OutputGradient")
   CreateKernel(def_k_FQF_QuantileGradient, "FQF_QuantileGradient")
   CreateKernel(def_k_FQF_CosineGradient, "FQF_CosineGradient")
   CreateKernel(def_k_MHSparseAttentionScore, "MHSparseAttentionScore")
   CreateKernel(def_k_MHSparseAttentionOut, "MHSparseAttentionOut")
   CreateKernel(def_k_FFMultiModels, "FeedForwardMultiModels")
   CreateKernel(def_k_HGMultiModels, "CalcHiddenGradientMultiModels")
   CreateKernel(def_k_UWMultiModels, "UpdateWeightsAdamMultiModels")
   CreateKernel(def_k_ConcatFeedForward, "Concat_FeedForward")
   CreateKernel(def_k_ConcatCalcHiddenGradient, "Concat_HiddenGradient")
   CreateKernel(def_k_ConcatUpdWeightsMomentum, "Concat_UpdateWeightsMomentum")
   CreateKernel(def_k_ConcatUpdWeightsAdam, "Concat_UpdateWeightsAdam")
   CreateKernel(def_k_SoftUpdate, "SoftUpdate")
   CreateKernel(def_k_SoftUpdateAdam, "SoftUpdateAdam")
   CreateKernel(def_k_SAC_AlphaLogProbs, "SAC_AlphaLogProbs")
   CreateKernel(def_k_SAC_AlphaGradients, "SAC_AlphaGradients")
   CreateKernel(def_k_SAC_OutputGradient, "SAC_OutputGradient")
   CreateKernel(def_k_SAC_CalcLogProbs, "SAC_CalcLogProbs")
   CreateKernel(def_k_Embedding, "Embedding")
   CreateKernel(def_k_EmbeddingHiddenGradient, "EmbeddingHiddenGradient")
   CreateKernel(def_k_EmbeddingUpdateWeightsAdam, "EmbeddingUpdateWeightsAdam")
   CreateKernel(def_k_Transpose, "Transpose")
   CreateKernel(def_k_TransposeRCD, "TransposeRCD")
   CreateKernel(def_k_MH2AttentionOut, "MH2AttentionOut")
   CreateKernel(def_k_MH2AttentionInsideGradients, "MH2AttentionInsideGradients")
   CreateKernel(def_k_CGConv_HiddenGradient, "CGConv_HiddenGradient")
   CreateKernel(def_k_XCiTFeedForward, "XCiTFeedForward")
   CreateKernel(def_k_XCiTInsideGradients, "XCiTInsideGradients")
   CreateKernel(def_k_DOTFeedForward, "DOTFeedForward")
   CreateKernel(def_k_DOTInsideGradients, "DOTInsideGradients")
   CreateKernel(def_k_RPBUpdateAdam, "RPBUpdateAdam")
   CreateKernel(def_k_GTEFeedForward, "GTEFeedForward")
   CreateKernel(def_k_GTEInsideGradients, "GTEInsideGradients")
   CreateKernel(def_k_FeedForwardNODEF, "FeedForwardNODEF")
   CreateKernel(def_k_FeedForwardNODEInpK, "FeedForwardNODEInpK")
   CreateKernel(def_k_HiddenGradientNODEF, "HiddenGradientNODEF")
   CreateKernel(def_k_HiddenGradientNODEInpK, "HiddenGradientNODEInpK")
   CreateKernel(def_k_NODEF_UpdateWeightsAdam, "NODEF_UpdateWeightsAdam")
   CreateKernel(def_k_FeedForwardContAtt, "FeedForwardContAtt")
   CreateKernel(def_k_TimeDerivative, "TimeDerivative")
   CreateKernel(def_k_HGTimeDerivative, "HiddenGradientTimeDerivative")
   CreateKernel(def_k_HiddenGradientContAtt, "HiddenGradientContAtt")
   CreateKernel(def_k_RevInFeedForward, "RevInFeedForward")
   CreateKernel(def_k_RevInHiddenGraddient, "RevInHiddenGraddient")
   CreateKernel(def_k_DeActivation, "DeActivation")
   CreateKernel(def_k_PatchCreate, "PatchCreate")
   CreateKernel(def_k_PatchHiddenGradient, "PatchHiddenGradient")
   CreateKernel(def_k_PatchUpdateWeightsAdam, "PatchUpdateWeightsAdam")
   CreateKernel(def_k_MatMult, "MatMult")
   CreateKernel(def_k_FFT, "FFT")
   CreateKernel(def_k_ComplexLayer, "ComplexLayer")
   CreateKernel(def_k_ComplexLayerGradient, "ComplexLayerGradient")
   CreateKernel(def_k_GradientMSA, "GradientMSA")
   CreateKernel(def_k_CumulativeGradient, "CumulativeGradient")
   CreateKernel(def_k_FeedForwardComplexConv, "FeedForwardComplexConv")
   CreateKernel(def_k_CalcHiddenGradientComplexConv, "CalcHiddenGradientComplexConv")
   CreateKernel(def_k_UpdateWeightsComplexConvMomentum, "UpdateWeightsComplexConvMomentum")
   CreateKernel(def_k_UpdateWeightsComplexConvAdam, "UpdateWeightsComplexConvAdam")
   CreateKernel(def_k_ComplexMHAttentionGradients, "ComplexMHAttentionGradients")
   CreateKernel(def_k_ComplexMHAttentionScore, "ComplexMHAttentionScore")
   CreateKernel(def_k_ComplexMHAttentionOut, "ComplexMHAttentionOut")
   CreateKernel(def_k_ComplexSoftMax_FeedForward, "ComplexSoftMax_FeedForward")
   CreateKernel(def_k_ComplexSoftMax_HiddenGradient, "ComplexSoftMax_HiddenGradient")
   CreateKernel(def_k_ComplexSoftMax_OutputGradient, "ComplexSoftMax_OutputGradient")
   CreateKernel(def_k_ComplexNormalize, "ComplexNormalize")
   CreateKernel(def_k_ComplexUnNormalize, "ComplexUnNormalize")
   CreateKernel(def_k_ComplexNormalizeGradient, "ComplexNormalizeGradient")
   CreateKernel(def_k_ComplexUnNormalizeGradient, "ComplexUnNormalizeGradient")
   CreateKernel(def_k_MainFreqWeight, "MainFreqWeight")
   CreateKernel(def_k_WeightedSum, "WeightedSum")
   CreateKernel(def_k_WeightedSumGradient, "WeightedSumGradient")
   CreateKernel(def_k_FeedForwardS3, "FeedForwardS3")
   CreateKernel(def_k_InsideGradientS3, "InsideGradientS3")
   CreateKernel(def_k_WeightGradientS3, "WeightGradientS3")
   CreateKernel(def_k_MH2PyrAttentionOut, "MH2PyrAttentionOut")
   CreateKernel(def_k_PLR, "PLR")
   CreateKernel(def_k_PLRGrad, "PLRGradient")
   CreateKernel(def_k_PLRMultiAgents, "PLRMultiAgents")
   CreateKernel(def_k_PLRMultiAgentsGrad, "PLRMultiAgentsGradient")
   CreateKernel(def_k_UpdateWeightsAdamMini, "UpdateWeightsAdamMini")
   CreateKernel(def_k_UpdateWeightsConvAdamMini, "UpdateWeightsConvAdamMini")
   CreateKernel(def_k_CutTrendAndOther, "CutTrendAndOther")
   CreateKernel(def_k_CutTrendAndOtherGradient, "CutTrendAndOtherGradient")
   CreateKernel(def_k_CutOneFromAnother, "CutOneFromAnother")
   CreateKernel(def_k_CutOneFromAnotherGradient, "CutOneFromAnotherGradient")
   CreateKernel(def_k_UniTrajPrepare, "UniTrajPrepare")
   CreateKernel(def_k_UniTrajPrepareGrad, "UniTrajPrepareGrad")
   CreateKernel(def_k_UniTrajBTS, "UniTrajBTS")
   CreateKernel(def_k_HiVTPrepare, "HiVTPrepare")
   CreateKernel(def_k_GateElementMul, "GateElementMul")
   CreateKernel(def_k_GateElementMulGrad, "GateElementMulGrad")
   CreateKernel(def_k_MatMultGrad, "MatMultGrad")
   CreateKernel(def_k_OrthoganalLoss, "OrthoganalLoss")
   CreateKernel(def_k_CalcDistance, "CalcDistance")
   CreateKernel(def_k_FeedForwardLocalMax, "FeedForwardLocalMax")
   CreateKernel(def_k_CalcInputGradientLocalMax, "CalcInputGradientLocalMax")
   CreateKernel(def_k_MHMaskAttentionOut, "MHMaskAttentionOut")
   CreateKernel(def_k_MHMaskAttentionInsideGradients, "MHMaskAttentionInsideGradients")
   CreateKernel(def_k_MHPosBiasAttentionOut, "MHPosBiasAttentionOut")
   CreateKernel(def_k_MHPosBiasAttentionInsideGradients, "MHPosBiasAttentionInsideGradients")
   CreateKernel(def_k_CalcPositionBias, "CalcPositionBias")
   CreateKernel(def_k_DiversityLoss, "DiversityLoss")
   CreateKernel(def_k_MHRelativeAttentionOut, "MHRelativeAttentionOut")
   CreateKernel(def_k_MHRelativeAttentionInsideGradients, "MHRelativeAttentionInsideGradients")
   CreateKernel(def_k_CalcAlignmentGradient, "CalcAlignmentGradient")
   CreateKernel(def_k_FeatureSmoothing, "FeatureSmoothing")
   CreateKernel(def_k_FeatureSmoothingGradient, "FeatureSmoothingGradient")
   CreateKernel(def_k_BatchFeedForwardAddNoise, "BatchFeedForwardAddNoise")
   CreateKernel(def_k_HyperProjection, "HyperProjection")
   CreateKernel(def_k_HyperProjectionGrad, "HyperProjectionGrad")
   CreateKernel(def_k_LogMap, "LogMap")
   CreateKernel(def_k_LogMapGrad, "LogMapGrad")
   CreateKernel(def_k_CalcEpsilonWeights, "CalcEpsilonWeights")
   CreateKernel(def_k_CalcEpsilonWeightsConv, "CalcEpsilonWeightsConv")
   CreateKernel(def_k_CalcHiddenGradientMHConv, "CalcHiddenGradientMHConv")
   CreateKernel(def_k_FeedForwardMHConv, "FeedForwardMHConv")
   CreateKernel(def_k_UpdateWeightsMHConvAdam, "UpdateWeightsMHConvAdam")
   CreateKernel(def_k_MoreLessEqual, "MoreLessEqual")
   CreateKernel(def_k_MultiScaleRelativeAttentionOut, "MultiScaleRelativeAttentionOut")
   CreateKernel(def_k_SSM2D_FeedForward, "SSM2D_FeedForward")
   CreateKernel(def_k_SSM2D_CalcHiddenGradient, "SSM2D_CalcHiddenGradient")
   CreateKernel(def_k_PScan, "PScan")
   CreateKernel(def_k_PScan_CalcHiddenGradient, "PScan_CalcHiddenGradient")
   CreateKernel(def_k_DiagMatMult, "DiagMatMult")
   CreateKernel(def_k_DiagMatMultGrad, "DiagMatMultGrad")
   CreateKernel(def_k_TopKgates, "TopKgates")
   CreateKernel(def_k_TopKgatesGrad, "TopKgatesGrad")
   CreateKernel(def_k_MaskByDistance, "MaskByDistance")
   CreateKernel(def_k_MaskAttention, "MaskAttention")
   CreateKernel(def_k_MaskAttentionGradients, "MaskAttentionGradients")
   CreateKernel(def_k_FeedForwardMultWinConv, "FeedForwardMultWinConv")
   CreateKernel(def_k_CalcHiddenGradientMultWinConv, "CalcHiddenGradientMultWinConv")
   CreateKernel(def_k_UpdateWeightsMultWinConvAdam, "UpdateWeightsMultWinConvAdam")
   CreateKernel(def_k_MaskAttentionComplex, "MaskAttentionComplex")
   CreateKernel(def_k_MaskAttentionGradientsComplex, "MaskAttentionGradientsComplex")
   CreateKernel(def_k_CSLSTM_FeedForward, "CSLSTM_FeedForward")
   CreateKernel(def_k_CSLSTM_CalcHiddenGradient, "CSLSTM_CalcHiddenGradient")
   CreateKernel(def_k_ProbAttentionQeuryImp, "ProbAttentionQeuryImp")
   CreateKernel(def_k_TopKImportanceToIndex, "TopKImportanceToIndex")
   CreateKernel(def_k_QIndexAttention, "QIndexAttention")
   CreateKernel(def_k_QIndexAttentionGradients, "QIndexAttentionGradients")
   CreateKernel(def_k_TSPositonEncoder, "TSPositonEncoder")
   CreateKernel(def_k_FeedForwardMultWinConvWPad, "FeedForwardMultWinConvWPad")
   CreateKernel(def_k_CalcHiddenGradientMultWinConvWPad, "CalcHiddenGradientMultWinConvWPad")
   CreateKernel(def_k_UpdateWeightsMultWinConvAdamWPad, "UpdateWeightsMultWinConvAdamWPad")
   CreateKernel(def_k_ConcatDiff, "ConcatDiff")
   CreateKernel(def_k_FeedForwardMaskMultWinConv, "FeedForwardMaskMultWinConv")
   CreateKernel(def_k_CalcHiddenGradientMaskMultWinConv, "CalcHiddenGradientMaskMultWinConv")
   CreateKernel(def_k_UpdateWeightsMaskMultWinConvAdam, "UpdateWeightsMaskMultWinConvAdam")
   CreateKernel(def_k_MainFreq, "MainFreq")
   CreateKernel(def_k_FeedForwardAdaptConv, "FeedForwardAdaptConv")
   CreateKernel(def_k_CalcHiddenGradientAdaptConv, "CalcHiddenGradientAdaptConv")
   CreateKernel(def_k_UpdateWeightsAdaptConvAdam, "UpdateWeightsAdaptConvAdam")
   CreateKernel(def_k_RoPE, "RoPE")
   CreateKernel(def_k_CalcHiddenGradRoPE, "CalcHiddenGradRoPE")
   CreateKernel(def_k_MatrixDif, "DifMatrix")
   CreateKernel(def_k_MatrixDifGrad, "DifMatrixGrad")
   CreateKernel(def_k_IdentMatrixSum, "IdentitySumMatrix")
   CreateKernel(def_k_IdentMatrixDif, "IdentityDifMatrix")
   CreateKernel(def_k_IdentMatrixDifGrad, "IdentityDifMatrixGrad")
   CreateKernel(def_k_SumVecMatrix, "SumVecMatrix")
   CreateKernel(def_k_SumVecMatrixGrad, "SumVecMatrixGrad")
   CreateKernel(def_k_InterpolationAttention, "InterpolationAttention")
   CreateKernel(def_k_InterpolationAttentionGrad, "InterpolationAttentionGrad")
   CreateKernel(def_k_Activation, "Activation")
   CreateKernel(def_k_PeriodNorm, "PeriodNorm")
   CreateKernel(def_k_PeriodNormGrad, "PeriodNormGrad")
   CreateKernel(def_k_AdaptSpatialNorm, "AdaptSpatialNorm")
   CreateKernel(def_k_AdaptSpatialNormGrad, "AdaptSpatialNormGrad")
   CreateKernel(def_k_AttentNorm, "AttentNorm")
   CreateKernel(def_k_AttentNormGrad, "AttentNormGrad")
   CreateKernel(def_k_ChebStep, "ChebStep")
   CreateKernel(def_k_ChebStepGrad, "ChebStepGrad")
   CreateKernel(def_k_SignificantNeighborsSampling, "SignificantNeighborsSampling")
   CreateKernel(def_k_SparseMHScores, "SparseMHScores")
   CreateKernel(def_k_SparseMHScoresGrad, "SparseMHScoresGrad")
   CreateKernel(def_k_SparseMatMult, "SparseMatMult")
   CreateKernel(def_k_SparseMatMultGrad, "SparseMatMultGrad")
   CreateKernel(def_k_RandomWalk, "RandomWalk")
   CreateKernel(def_k_ConcatByLabel, "ConcatByLabel")
   CreateKernel(def_k_ConcatByLabelGrad, "ConcatByLabelGrad")
   CreateKernel(def_k_GlobalLocalAttention, "GlobalLocalAttention")
   CreateKernel(def_k_GlobalLocalAttentionGrad, "GlobalLocalAttentionGrad")
   CreateKernel(def_k_SparseSoftMax, "SparseSoftMax")
   CreateKernel(def_k_SparseSoftMaxGrad, "SparseSoftMaxGrad")
   CreateKernel(def_k_FloatToSpike, "FloatToSpike")
   CreateKernel(def_k_FloatToSpikeGrad, "FloatToSpikeGrad")
   CreateKernel(def_k_SpikeMHAttention, "SpikeMHAttention")
   CreateKernel(def_k_SpikeMHAttentionGrad, "SpikeMHAttentionGrad")
   CreateKernel(def_k_STFS, "STFS")
   CreateKernel(def_k_STFSGrad, "STFSGrad")
   CreateKernel(def_k_AddToStack, "AddToStack")
   CreateKernel(def_k_AggregationByTime, "AggregationByTime")
   CreateKernel(def_k_AggregationByTimeGrad, "AggregationByTimeGrad")
   CreateKernel(def_k_GRU, "GRU")
   CreateKernel(def_k_GRU_Grad, "GRU_Grad")
   CreateKernel(def_k_ScalarToVector, "ScalarToVector")
   CreateKernel(def_k_ScalarToVectorGrad, "ScalarToVectorGrad")
   CreateKernel(def_k_CalcFlow, "CalcFlow")
   CreateKernel(def_k_DilatedCorrelation, "DilatedCorrelation")
   CreateKernel(def_k_DilatedCorrelationGrad, "DilatedCorrelationGrad")
   CreateKernel(def_k_DilatedDifference, "DilatedDifference")
   CreateKernel(def_k_DilatedDifferenceGrad, "DilatedDifferenceGrad")
   CreateKernel(def_k_PerturbedMatrix, "PerturbedMatrix")
   CreateKernel(def_k_PerturbedMatrixGrad, "PerturbedMatrixGrad")
   CreateKernel(def_k_LinearUpsample, "LinearUpsample")
   CreateKernel(def_k_LinearUpsampleGrad, "LinearUpsampleGrad")
   CreateKernel(def_k_MixExpertsPredict, "MixExpertsPredict")
   CreateKernel(def_k_MixExpertsPredictGrad, "MixExpertsPredictGrad")
   CreateKernel(def_k_MHFAT, "MHFAT")
   CreateKernel(def_k_MHFATGrad, "MHFATGrad")
   CreateKernel(def_k_SparseConcatenate, "SparseConcatenate")
   CreateKernel(def_k_SparseConcatenateGrad, "SparseConcatenateGrad")
   CreateKernel(def_k_MHFlashAttention, "MHFlashAttention")
   CreateKernel(def_k_MHFlashAttentionGrad, "MHFlashAttentionGrad")
   CreateKernel(def_k_MHFlashSTCA, "MHFlashSTCA")
   CreateKernel(def_k_MHFlashSTCAGrad, "MHFlashSTCAGrad")
   CreateKernel(def_k_ConcatVecMatrix, "ConcatVecMatrix")
   CreateKernel(def_k_ConcatVecMatrixGrad, "ConcatVecMatrixGrad")
   CreateKernel(def_k_INFNetBGU, "INFNetBGU")
   CreateKernel(def_k_INFNetBGUGrad, "INFNetBGUGrad")
   CreateKernel(def_k_MHCrossAttvsSim, "MHCrossAttvsSim")
   CreateKernel(def_k_MHCrossAttvsSimGrad, "MHCrossAttvsSimGrad")
   CreateKernel(def_k_MarketStateDensity, "MarketStateDensity")
   CreateKernel(def_k_NLLGrad, "NLLGrad")
   CreateKernel(def_k_SparseSoftmaxHausdorff, "SparseSoftmaxHausdorff")
   CreateKernel(def_k_SparseSoftmaxHausdorffGrad, "SparseSoftmaxHausdorffGrad")
   CreateKernel(def_k_MomADTTMMeanScores, "MomADTTMMeanScores")
   CreateKernel(def_k_MomADTTMMeanScoresGrad, "MomADTTMMeanScoresGrad")
   CreateKernel(def_k_CogDriverAffineStack, "CogDriverAffineStack")
   CreateKernel(def_k_CogDriverAffineStackGrad, "CogDriverAffineStackGrad")
   CreateKernel(def_k_CogDriverRankTCMInsertPending, "CogDriverRankTCMInsertPending")
   CreateKernel(def_k_CogDriverRankTCMBuildKV, "CogDriverRankTCMBuildKV")
   CreateKernel(def_k_CogDriverRankTCMMeanFlashScore, "CogDriverRankTCMMeanFlashScore")
   CreateKernel(def_k_CogDriverRankTCMSelfKV, "CogDriverRankTCMSelfKV")
   CreateKernel(def_k_CogDriverForecastShiftState, "CogDriverForecastShiftState")
   CreateKernel(def_k_CogDriverForecastConfidence, "CogDriverForecastConfidence")
   CreateKernel(def_k_CogDriverForecastDeltaNorm, "CogDriverForecastDeltaNorm")
   CreateKernel(def_k_CogDriverForecastBuildKV, "CogDriverForecastBuildKV")
   CreateKernel(def_k_CogDriverForecastBuildKVGrad, "CogDriverForecastBuildKVGrad")
   CreateKernel(def_k_CogDriverForecastGradientRMS, "CogDriverForecastGradientRMS")
   CreateKernel(def_k_ScenarioForecastForward, "ScenarioForecastForward")
   CreateKernel(def_k_ScenarioResponsibilities, "ScenarioResponsibilities")
   CreateKernel(def_k_ScenarioCodebookEMA, "ScenarioCodebookEMA")
   CreateKernel(def_k_ScenarioGradientNormalize, "ScenarioGradientNormalize")
   CreateKernel(def_k_RAGScore, "RAGScore")
   CreateKernel(def_k_RAGPartial, "RAGPartial")
   CreateKernel(def_k_RAGMerge, "RAGMerge")
   CreateKernel(def_k_RAGGather, "RAGGather")
   CreateKernel(def_k_RAGNearestPair, "RAGNearestPair")
   CreateKernel(def_k_RAGTokenGate, "RAGTokenGate")
   CreateKernel(def_k_RAGTokenGateGrad, "RAGTokenGateGrad")
   CreateKernel(def_k_D2SkillScoreSlots, "D2SkillScoreSlots")
   CreateKernel(def_k_D2SkillUtility, "D2SkillUtility")
   CreateKernel(def_k_D2SkillItemPartial, "D2SkillItemPartial")
   CreateKernel(def_k_D2SkillItemReduce, "D2SkillItemReduce")
   CreateKernel(def_k_D2SkillItemApply, "D2SkillItemApply")
   CreateKernel(def_k_D2SkillGatherCorrection, "D2SkillGatherCorrection")
   CreateKernel(def_k_D2SkillAgeLifecycle, "D2SkillAgeLifecycle")
   CreateKernel(def_k_D2SkillMetricPartial, "D2SkillMetricPartial")
   CreateKernel(def_k_D2SkillMetricReduce, "D2SkillMetricReduce")
   CreateKernel(def_k_D2SkillCandidateLeader, "D2SkillCandidateLeader")
   CreateKernel(def_k_D2SkillVectorUpdate, "D2SkillVectorUpdate")
   CreateKernel(def_k_D2SkillPromoteSlot, "D2SkillPromoteSlot")
   CreateKernel(def_k_D2SkillDiagnosticsPartial, "D2SkillDiagnosticsPartial")
   CreateKernel(def_k_D2SkillDiagnosticsReduce, "D2SkillDiagnosticsReduce")
   CreateKernel(def_k_D2SkillBankVectorPartial, "D2SkillBankVectorPartial")
   CreateKernel(def_k_CopyBufferRaw, "CopyBufferRaw")
   CreateKernel(def_k_D2SkillInfluenceTotal, "D2SkillInfluenceTotal")
   CreateKernel(def_k_D2SkillResetInfluence, "D2SkillResetInfluence")
   CreateKernel(def_k_D2SkillScoreFinalize, "D2SkillScoreFinalize")
   CreateKernel(def_k_D2SkillDiagnosticsNormPartial, "D2SkillDiagnosticsNormPartial")
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   if(!cInputs.Init(numNeurons, 0, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   CBufferFloat *inp = cInputs.getOutput();
   inp.BufferInit(1, 1);
   inp.BufferWrite();
   cInputs.SetActivationFunction(None);
//---
   if(!cPositionEncoder.Init(0, 1, OpenCL, numNeurons, optimization, iBatch))
      ReturnFalse;
   cPositionEncoder.SetActivationFunction(SIGMOID);
   SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::Activation(CBufferFloat *inputs, CBufferFloat *outputs, int activat)
  {
   if(!OpenCL || !inputs || !outputs)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {inputs.Total()};
   ResetLastError();
   setBuffer(def_k_Activation, def_k_act_inputs, inputs.GetIndex())
   setBuffer(def_k_Activation, def_k_act_outputs, outputs.GetIndex())
   setArgument(def_k_Activation, def_k_act_activation, (int)activat)
//---
   kernelExecute(def_k_Activation, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!outputs.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DeActivation(CBufferFloat* inputs, CBufferFloat* inputs_gr, CBufferFloat* output_gr, int activat)
  {
   if(!OpenCL || !inputs || !inputs_gr || !output_gr)
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {inputs.Total()};
   ResetLastError();
   setBuffer(def_k_DeActivation, def_k_deact_inputs, inputs.GetIndex())
   setBuffer(def_k_DeActivation, def_k_deact_inputs_gr, inputs_gr.GetIndex())
   setBuffer(def_k_DeActivation, def_k_deact_output_gr, output_gr.GetIndex())
   setArgument(def_k_DeActivation, def_k_deact_activation, (int)activat)
//---
   kernelExecute(def_k_DeActivation, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!inputs_gr.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::NLLGrad(const CBufferFloat *log_b, CBufferFloat *grad_in,
                             const CBufferFloat *grad_out, CBufferFloat *grad_log_b)
  {
   if(!OpenCL || !log_b || !grad_in || !grad_out || !grad_log_b)
      ReturnFalse;
   if(log_b.GetIndex() < 0 || grad_in.GetIndex() < 0 || grad_out.GetIndex() < 0 || grad_log_b.GetIndex() < 0)
      ReturnFalse;
   if(log_b.Total() != grad_in.Total() || log_b.Total() != grad_out.Total() || log_b.Total() != grad_log_b.Total())
      ReturnFalse;
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1] = {grad_in.Total()};
   const int kernel = def_k_NLLGrad;
   ResetLastError();
   setBuffer(kernel, def_k_nllgr_log_b, log_b.GetIndex())
   setBuffer(kernel, def_k_nllgr_grad_in, grad_in.GetIndex())
   setBuffer(kernel, def_k_nllgr_grad_out, grad_out.GetIndex())
   setBuffer(kernel, def_k_nllgr_grad_log_b, grad_log_b.GetIndex())
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!grad_in.BufferRead() || !grad_log_b.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseSoftmaxHausdorff(const CBufferFloat *plan,
      const CBufferFloat *decis,
      CBufferFloat *output,
      CBufferFloat *top_index,
      const int dimension,
      const int top_k)
  {
   if(!OpenCL || !plan || !decis || !output || !top_index)
      ReturnFalse;
   if(plan.GetIndex() < 0 || decis.GetIndex() < 0 || output.GetIndex() < 0 || top_index.GetIndex() < 0)
      ReturnFalse;
   if(dimension <= 0 || top_k <= 0 || top_k > dimension || dimension > 64)
      ReturnFalse;
   if(dimension > (int)OpenCL.GetMaxLocalSize(0))
      ReturnFalse;
   if(plan.Total() <= 0 || decis.Total() <= 0 || plan.Total() % dimension != 0 || decis.Total() % dimension != 0)
      ReturnFalse;
//---
   const int count_plan = plan.Total() / dimension;
   const int count_decis = decis.Total() / dimension;
   const int pairs = count_plan * count_decis;
   if(output.Total() != pairs || top_index.Total() != pairs * top_k)
      ReturnFalse;
//---
   const int kernel = def_k_SparseSoftmaxHausdorff;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {(uint)dimension, (uint)count_plan, (uint)count_decis};
   uint local_work_size[3] = {(uint)dimension, 1, 1};
   ResetLastError();
   setBuffer(kernel, def_k_ssh_plan, plan.GetIndex())
   setBuffer(kernel, def_k_ssh_decis, decis.GetIndex())
   setBuffer(kernel, def_k_ssh_output, output.GetIndex())
   setBuffer(kernel, def_k_ssh_top_index, top_index.GetIndex())
   setArgument(kernel, def_k_ssh_dimension, dimension)
   setArgument(kernel, def_k_ssh_top_k, top_k)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!output.BufferRead() || !top_index.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::SparseSoftmaxHausdorffGrad(const CBufferFloat *plan,
      CBufferFloat *grad_plan,
      const CBufferFloat *decis,
      CBufferFloat *grad_decis,
      const CBufferFloat *grad_out,
      const CBufferFloat *top_index,
      const int dimension,
      const int top_k)
  {
   if(!OpenCL || !plan || !grad_plan || !decis || !grad_decis || !grad_out || !top_index)
      ReturnFalse;
   if(plan.GetIndex() < 0 || grad_plan.GetIndex() < 0 || decis.GetIndex() < 0 || grad_decis.GetIndex() < 0 ||
      grad_out.GetIndex() < 0 || top_index.GetIndex() < 0)
      ReturnFalse;
   if(dimension <= 0 || top_k <= 0 || top_k > dimension || dimension > 64)
      ReturnFalse;
   if(plan.Total() <= 0 || decis.Total() <= 0 || plan.Total() % dimension != 0 || decis.Total() % dimension != 0)
      ReturnFalse;
   if(grad_plan.Total() != plan.Total() || grad_decis.Total() != decis.Total())
      ReturnFalse;
//---
   const int count_plan = plan.Total() / dimension;
   const int count_decis = decis.Total() / dimension;
   const int pairs = count_plan * count_decis;
   if(grad_out.Total() != pairs || top_index.Total() != pairs * top_k)
      ReturnFalse;
//---
   const int kernel = def_k_SparseSoftmaxHausdorffGrad;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {(uint)dimension, (uint)MathMax(count_plan, count_decis), 2};
   ResetLastError();
   setBuffer(kernel, def_k_sshg_plan, plan.GetIndex())
   setBuffer(kernel, def_k_sshg_grad_plan, grad_plan.GetIndex())
   setBuffer(kernel, def_k_sshg_decis, decis.GetIndex())
   setBuffer(kernel, def_k_sshg_grad_decis, grad_decis.GetIndex())
   setBuffer(kernel, def_k_sshg_grad_out, grad_out.GetIndex())
   setBuffer(kernel, def_k_sshg_top_index, top_index.GetIndex())
   setArgument(kernel, def_k_sshg_dimension, dimension)
   setArgument(kernel, def_k_sshg_count_plan, count_plan)
   setArgument(kernel, def_k_sshg_count_decis, count_decis)
   setArgument(kernel, def_k_sshg_top_k, top_k)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!grad_plan.BufferRead() || !grad_decis.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cPositionEncoder.FeedForward(cInputs.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), cPositionEncoder.getOutput(), Output, 1, false, 0, 0, 0, 1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!DeActivation(cPositionEncoder.getOutput(), cPositionEncoder.getGradient(), Gradient, cPositionEncoder.Activation()))
      ReturnFalse;
   if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), Gradient, NeuronOCL.Activation()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return cPositionEncoder.UpdateInputWeights(cInputs.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!source || source.Type() != Type())
      ReturnFalse;
   CNeuronLearnabledPE *Source = source;
   return cPositionEncoder.WeightsUpdate(GetPointer(Source.cPositionEncoder), tau);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronLearnabledPE::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cPositionEncoder.SetOpenCL(OpenCL);
   cInputs.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cInputs.Save(file_handle))
      ReturnFalse;
   if(!cPositionEncoder.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLearnabledPE::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cInputs.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cPositionEncoder.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                           uint window_in, uint step, uint window_out,
                           uint count, uint variables,
                           ENUM_OPTIMIZATION optimization_type, uint batch
                          )
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * count * variables, optimization_type, batch))
      ReturnFalse;
//---
   iWindowIn = MathMax(window_in, 1);
   iWindowOut = MathMax(window_out, 1);
   iStep = MathMax(step, 1);
   iVariables = MathMax(variables, 1);
   iCount = MathMax(count, 1);
//---
   int total = int((window_in + 1) * window_out * variables);
   if(!cPatchWeights.Reserve(total))
      ReturnFalse;
   float k = float(1 / sqrt(total));
   for(int i = 0; i < total; i++)
     {
      if(!cPatchWeights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
         ReturnFalse;
     }
   if(!cPatchWeights.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!cPatchFirstMomentum.BufferInit(total, 0) ||
      !cPatchFirstMomentum.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cPatchSecondMomentum.BufferInit(total, 0) ||
      !cPatchSecondMomentum.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iCount, iWindowOut, iVariables};
   ResetLastError();
   setBuffer(def_k_PatchCreate, def_k_ptc_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_PatchCreate, def_k_ptc_weights, cPatchWeights.GetIndex())
   setBuffer(def_k_PatchCreate, def_k_ptc_outputs, Output.GetIndex())
   setArgument(def_k_PatchCreate, def_k_ptc_activation, (int)activation)
   setArgument(def_k_PatchCreate, def_k_ptc_inputs_total, (int)NeuronOCL.Neurons())
   setArgument(def_k_PatchCreate, def_k_ptc_window_in, (int)iWindowIn)
   setArgument(def_k_PatchCreate, def_k_ptc_step, (int)iStep)
//---
   kernelExecute(def_k_PatchCreate, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {NeuronOCL.Neurons() / iVariables, iVariables};
   ResetLastError();
   setBuffer(def_k_PatchHiddenGradient, def_k_pthg_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_PatchHiddenGradient, def_k_pthg_inputs_gr, NeuronOCL.getGradientIndex())
   setBuffer(def_k_PatchHiddenGradient, def_k_pthg_weights, cPatchWeights.GetIndex())
   setBuffer(def_k_PatchHiddenGradient, def_k_pthg_outputs_gr, Gradient.GetIndex())
   setArgument(def_k_PatchHiddenGradient, def_k_pthg_activation, (int)NeuronOCL.Activation())
   setArgument(def_k_PatchHiddenGradient, def_k_pthg_outputs_total, (int)iCount)
   setArgument(def_k_PatchHiddenGradient, def_k_pthg_window_in, (int)iWindowIn)
   setArgument(def_k_PatchHiddenGradient, def_k_pthg_step, (int)iStep)
   setArgument(def_k_PatchHiddenGradient, def_k_pthg_window_out, (int)iWindowOut)
//---
   kernelExecute(def_k_PatchHiddenGradient, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Gradient.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL || !OpenCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {iWindowIn + 1, iWindowOut, iVariables};
   ResetLastError();
   setBuffer(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_inputs, NeuronOCL.getOutputIndex())
   setBuffer(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_outputs_gr, getGradientIndex())
   setBuffer(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_weights, cPatchWeights.GetIndex())
   setBuffer(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_weights_m, cPatchFirstMomentum.GetIndex())
   setBuffer(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_weights_v, cPatchSecondMomentum.GetIndex())
   setArgument(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_inputs_total, (int)NeuronOCL.Neurons())
   setArgument(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_l, lr)
   setArgument(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_b1, b1)
   setArgument(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_step, (int)iStep)
   setArgument(def_k_PatchUpdateWeightsAdam, def_k_ptuwa_b2, b2)
//---
   kernelExecute(def_k_PatchUpdateWeightsAdam, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cPatchSecondMomentum.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, (int)iWindowIn) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iWindowOut) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iStep) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iCount) < INT_VALUE)
      ReturnFalse;
   if(FileWriteInteger(file_handle, (int)iVariables) < INT_VALUE)
      ReturnFalse;
//---
   if(!cPatchWeights.Save(file_handle))
      ReturnFalse;
   if(!cPatchFirstMomentum.Save(file_handle))
      ReturnFalse;
   if(!cPatchSecondMomentum.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPatching::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   iWindowIn = (uint)FileReadInteger(file_handle);
   iWindowOut = (uint)FileReadInteger(file_handle);
   iStep = (uint)FileReadInteger(file_handle);
   iCount = (uint)FileReadInteger(file_handle);
   iVariables = (uint)FileReadInteger(file_handle);
//---
   if(!cPatchWeights.Load(file_handle))
      ReturnFalse;
   if(!cPatchFirstMomentum.Load(file_handle))
      ReturnFalse;
   if(!cPatchSecondMomentum.Load(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPatching::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cPatchWeights.BufferCreate(OpenCL);
   cPatchFirstMomentum.BufferCreate(OpenCL);
   cPatchSecondMomentum.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLegendreWavelets::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint step, uint units_count, uint filters, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window, step, filters, units_count, optimization_type, batch))
      ReturnFalse;
//---
   WeightsConv.BufferInit(WeightsConv.Total(), 0);
   for(uint i = 0; i < iWindow; i++)
     {
      uint shift = i;
      float k = float(2.0 * i - 1.0) / iWindow;
      for(uint f = 1; f <= filters; f++)
        {
         float value = 0;
         switch(f)
           {
            case 1:
               value = k;
               break;
            case 2:
               value = (3 * k * k - 1) / 2;
               break;
            default:
               value = ((2 * f - 1) * k * WeightsConv.At(shift - (iWindow + 1)) - (f - 1) * WeightsConv.At(shift - 2 * (iWindow + 1))) / f;
               break;
           }
         if(!WeightsConv.Update(shift, value))
            ReturnFalse;
         shift += iWindow + 1;
        }
     }
   if(!!OpenCL)
      if(!WeightsConv.BufferWrite())
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DiversityLoss(CNeuronBaseOCL* neuron, const int units,
                                   const int dimension, const bool add = false)
  {
   if(!OpenCL || !neuron)
      ReturnFalse;
//---
   uint global_work_offset[] = {0, 0};
   uint global_work_size[] = {units/*Q units*/, (uint)MathMin(units, OpenCL.GetMaxLocalSize(1))};
   uint local_work_size[] = {1, global_work_size[1] };
   int kernel = def_k_DiversityLoss;
//---
   ResetLastError();
   setBuffer(kernel, def_k_dl_data, neuron.getOutputIndex())
   setBuffer(kernel, def_k_dl_grad, neuron.getGradientIndex())
   setArgument(kernel, def_k_dl_dimension, dimension)
   setArgument(kernel, def_k_dl_activation, neuron.Activation())
   setArgument(kernel, def_k_dl_add, (int)add)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!neuron.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBatchNormWithNoise::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!bTrain)
      return CNeuronBatchNormOCL::feedForward(NeuronOCL);
//---
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   PrevLayer = NeuronOCL;
   if(iBatchSize <= 1)
     {
      activation = (ENUM_ACTIVATION)NeuronOCL.Activation();
      return true;
     }
//---
   double random[];
   if(!MathRandomNormal(0, 1, Neurons(), random))
      ReturnFalse;
   if(cNoise.Total() != Neurons() ||
      cNoise.GetOpenCL() != OpenCL)
     {
      cNoise.BufferFree();
      if(!cNoise.AssignArray(random))
         ReturnFalse;
      if(!cNoise.BufferCreate(OpenCL))
         ReturnFalse;
     }
   else
     {
      if(!cNoise.AssignArray(random))
         ReturnFalse;
      if(!cNoise.BufferWrite())
         ReturnFalse;
     }
//---
   iBatchCount = MathMin(iBatchCount, iBatchSize);
   float noise_alpha = float(1.0 - MathRand() / 32767.0 * 0.01);
//---
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Neurons();
   int kernel = def_k_BatchFeedForwardAddNoise;
   ResetLastError();
   setBuffer(kernel, def_k_normwithnoise_inputs, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_normwithnoise_noise, cNoise.GetIndex())
   setBuffer(kernel, def_k_normwithnoise_options, BatchOptions.GetIndex())
   setBuffer(kernel, def_k_normwithnoise_output, Output.GetIndex())
   setArgument(kernel, def_k_normwithnoise_activation, int(activation))
   setArgument(kernel, def_k_normwithnoise_alpha, noise_alpha)
   setArgument(kernel, def_k_normwithnoise_batch, iBatchCount)
   setArgument(kernel, def_k_normwithnoise_optimization, int(optimization))
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
   iBatchCount++;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return Init(numOutputs, myIndex, open_cl, numNeurons, 0.7f, optimization_type, batch);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, float rho, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   fRho = fabs(rho);
   if(fRho == 0 || !Weights)
      return true;
//---
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::calcEpsilonWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[2] = { 0, 0 };
   uint global_work_size[2] = { NeuronOCL.Neurons() + 1, Output.Total() };
   uint local_work_size[2] = { global_work_size[0], 1 };
   const int kernel = def_k_CalcEpsilonWeights;
//---
   setBuffer(kernel, def_k_epsw_matrix_w, NeuronOCL.getWeightsIndex())
   setBuffer(kernel, def_k_epsw_matrix_g, getGradientIndex())
   setBuffer(kernel, def_k_epsw_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_epsw_matrix_epsw, NeuronOCL.getWeightsSAMIndex())
   setArgument(kernel, def_k_epsw_rho, fRho)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!NeuronOCL.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::feedForwardSAM(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[1] = { 0 };
   uint global_work_size[1] = { Output.Total() };
//---
   const int kernel = def_k_FeedForward;
   setBuffer(kernel, def_k_ff_matrix_w, NeuronOCL.getWeightsSAMIndex())
   setBuffer(kernel, def_k_ff_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ff_matrix_o, Output.GetIndex())
   setArgument(kernel, def_k_ff_inputs, NeuronOCL.Neurons())
   setArgument(kernel, def_k_ff_activation, (int)activation)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(NeuronOCL.getWeightsSAMIndex() < 0 ||
      fRho == 0)
      return CNeuronBaseOCL::updateInputWeights(NeuronOCL);
//---
   if(!SumAndNormalize(Gradient, Output, Gradient, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!calcEpsilonWeights(NeuronOCL))
      ReturnFalse;
   if(!feedForwardSAM(NeuronOCL))
      ReturnFalse;
   float error = 1;
   if(!calcOutputGradients(Gradient, error))
      ReturnFalse;
//---
   return CNeuronBaseOCL::updateInputWeights(NeuronOCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::calcHiddenGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(activation == None)
      return CNeuronBaseOCL::calcHiddenGradients(NeuronOCL);
//---
   CBufferFloat *temp = Gradient;
   Gradient = PrevOutput;
   if(!DeActivation(Output, temp, Gradient, activation) ||
      !CNeuronBaseOCL::calcHiddenGradients(NeuronOCL))
      ReturnFalse;
   Gradient = temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronBaseSAMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   if(cWeightsSAM.GetOpenCL() != OpenCL)
      cWeightsSAM.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteFloat(file_handle, fRho) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseSAMOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   fRho = FileReadFloat(file_handle);
   if(fRho == 0 || !Weights)
      return true;
   cWeightsSAM.BufferFree();
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window_in, uint step, uint window_out, uint units_count, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return CNeuronConvSAMOCL::Init(numOutputs, myIndex, open_cl, window_in, step, window_out, units_count, variables, 0.7f, optimization_type, batch);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window_in, uint step, uint window_out, uint units_count, uint variables, float rho, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window_in, step, window_out, units_count, variables, optimization_type, batch))
      ReturnFalse;
   fRho = fabs(rho);
   if(fRho == 0)
      return true;
//---
   cWeightsSAMConv.BufferFree();
   if(!cWeightsSAMConv.BufferInit(WeightsConv.Total(), 0) ||
      !cWeightsSAMConv.BufferCreate(OpenCL))
      ReturnFalse;
//---
   cWeightsSAM.BufferFree();
   if(!Weights)
      return true;
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::calcEpsilonWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      ReturnFalse;
//---
   uint global_work_offset[3] = { 0, 0, 0 };
   uint global_work_size[3] = { iWindow + 1, iWindowOut, iVariables };
   uint local_work_size[3] = { global_work_size[0], 1, 1 };
   const int kernel = def_k_CalcEpsilonWeightsConv;
//---
   setBuffer(kernel, def_k_epswconv_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_epswconv_matrix_epsw, cWeightsSAMConv.GetIndex())
   setBuffer(kernel, def_k_epswconv_matrix_g, getGradientIndex())
   setBuffer(kernel, def_k_epswconv_matrix_w, WeightsConv.GetIndex())
   setArgument(kernel, def_k_epswconv_rho, fRho)
   setArgument(kernel, def_k_epswconv_inputs, int(NeuronOCL.Neurons() / iVariables))
   setArgument(kernel, def_k_epswconv_step, int(iStep))
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cWeightsSAMConv.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::feedForwardSAM(CNeuronBaseOCL* NeuronOCL)
  {
   if(CheckPointer(OpenCL) == POINTER_INVALID || CheckPointer(NeuronOCL) == POINTER_INVALID)
      ReturnFalse;
   uint global_work_offset[3] = {0, 0, 0};
   uint global_work_size[3] = {Output.Total() / (iWindowOut * iVariables),
                               iWindowOut, iVariables
                              };
//---
   const int kernel = def_k_FeedForwardConv;
   setBuffer(kernel, def_k_ffc_matrix_w, cWeightsSAMConv.GetIndex())
   setBuffer(kernel, def_k_ffc_matrix_i, NeuronOCL.getOutputIndex())
   setBuffer(kernel, def_k_ffc_matrix_o, Output.GetIndex())
   setArgument(kernel, def_k_ffc_inputs, NeuronOCL.Neurons() / iVariables)
   setArgument(kernel, def_k_ffc_step, (int)iStep)
   setArgument(kernel, def_k_ffc_window_in, (int)iWindow)
   setArgument(kernel, def_k_ffс_window_out, (int)iWindowOut)
   setArgument(kernel, def_k_ffc_activation, (int)activation)
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!Output.BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(fRho == 0)
      return CNeuronConvOCL::updateInputWeights(NeuronOCL);
//---
   if(!SumAndNormalize(Gradient, Output, Gradient, iWindowOut, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!calcEpsilonWeights(NeuronOCL))
      ReturnFalse;
   if(!feedForwardSAM(NeuronOCL))
      ReturnFalse;
   float error = 1;
   if(!calcOutputGradients(Gradient, error))
      ReturnFalse;
//---
   return CNeuronConvOCL::updateInputWeights(NeuronOCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(activation == None)
      return CNeuronConvOCL::calcInputGradients(NeuronOCL);
//---
   CBufferFloat *temp = Gradient;
   Gradient = PrevOutput;
   if(!DeActivation(Output, temp, Gradient, activation) ||
      !CNeuronConvOCL::calcInputGradients(NeuronOCL))
      ReturnFalse;
   Gradient = temp;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronConvSAMOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   cWeightsSAMConv.BufferCreate(OpenCL);
   if(!!Weights)
      cWeightsSAM.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(FileWriteFloat(file_handle, fRho) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   fRho = FileReadFloat(file_handle);
//---
   cWeightsSAMConv.BufferFree();
   cWeightsSAM.BufferFree();
   cWeightsSAMConv.Clear();
   cWeightsSAM.Clear();
   if(fRho <= 0)
      return true;
//---
   if(!cWeightsSAMConv.BufferInit(WeightsConv.Total(), 0) ||
      !cWeightsSAMConv.BufferCreate(OpenCL))
      ReturnFalse;
//---
   if(!Weights)
      return true;
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CBufferFloat::BufferInitLike(CBufferFloat* master)
  {
   if(!master)
      ReturnFalse;
   if(!!OpenCL && m_myIndex >= 0)
     {
      if(!BufferFree())
         ReturnFalse;
     }
   if(!BufferInit(master.Total(), 0))
      ReturnFalse;
   if(!BufferCreate(master.GetOpenCL()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::InitBufferLike(CBufferFloat *&buffer, CBufferFloat* master)
  {
   if(!buffer)
     {
      buffer = new CBufferFloat();
      if(!buffer)
         ReturnFalse;
     }
//---
   return buffer.BufferInitLike(master);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronConvSAMOCL::ReplaceBuffer(CBufferFloat *&buffer, CBufferFloat* master)
  {
   if(buffer == master)
      return;
   if(!!buffer)
     {
      buffer.BufferFree();
      DeleteObj(buffer);
     }
//---
   buffer = master;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronConvSAMOCL::InitPS(CNeuronConvSAMOCL * master)
  {
   if(!master ||
      master.Type() != Type()
     )
      ReturnFalse;
//---
   alpha = master.alpha;
   iBatch = master.iBatch;
   t = master.t;
   m_myIndex = master.m_myIndex;
   activation = master.activation;
   optimization = master.optimization;
   iWindow = master.iWindow;
   iStep = master.iStep;
   iWindowOut = master.iWindowOut;
   iVariables = master.iVariables;
   bTrain = master.bTrain;
   fRho = master.fRho;
//---
   if(!InitBufferLike(Output, master.Output))
      ReturnFalse;
   if(!!master.getPrevOutput())
      if(!InitBufferLike(PrevOutput, master.getPrevOutput()))
         ReturnFalse;
   if(!InitBufferLike(Gradient, master.Gradient))
      ReturnFalse;
//---
   ReplaceBuffer(Weights, master.Weights);
   ReplaceBuffer(DeltaWeights, master.DeltaWeights);
   ReplaceBuffer(FirstMomentum, master.FirstMomentum);
   ReplaceBuffer(SecondMomentum, master.SecondMomentum);
//---
   ReplaceBuffer(WeightsConv, master.WeightsConv);
   ReplaceBuffer(DeltaWeightsConv, master.DeltaWeightsConv);
   ReplaceBuffer(FirstMomentumConv, master.FirstMomentumConv);
   ReplaceBuffer(SecondMomentumConv, master.SecondMomentumConv);
//---
   if(master.cWeightsSAM.Total() > 0)
     {
      CBufferFloat *buf = GetPointer(cWeightsSAM);
      if(!InitBufferLike(buf, GetPointer(master.cWeightsSAM)))
         ReturnFalse;
     }
   else
     {
      cWeightsSAM.BufferFree();
      cWeightsSAM.Clear();
     }
//---
   if(fRho > 0)
     {
      CBufferFloat *buf = GetPointer(cWeightsSAMConv);
      if(!InitBufferLike(buf, GetPointer(master.cWeightsSAMConv)))
         ReturnFalse;
     }
   else
     {
      cWeightsSAMConv.BufferFree();
      cWeightsSAMConv.Clear();
     }
//---
   SetOpenCL(master.OpenCL);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                          uint window, uint window_out, uint units_count,
                          uint variables, float rho,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvSAMOCL::Init(numOutputs, myIndex, open_cl, window, window, window_out,
                               units_count, variables, rho, optimization_type, batch))
      ReturnFalse;
   if(!acConvolution[0].Init(0, 0, OpenCL, iWindow, iWindow, iWindowOut, units_count,
                             iVariables, fRho, optimization, iBatch))
      ReturnFalse;
   acConvolution[0].SetActivationFunction(GELU);
   if(!acConvolution[1].Init(0, 1, OpenCL, iWindowOut, iWindowOut, iWindow, units_count,
                             iVariables, fRho, optimization, iBatch))
      ReturnFalse;
   acConvolution[1].SetActivationFunction(None);
   if(!cResidual.Init(0, 2, OpenCL, acConvolution[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(acConvolution[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::InitPS(CNeuronPSBlock * master)
  {
   if(!CNeuronConvSAMOCL::InitPS((CNeuronConvSAMOCL*)master))
      ReturnFalse;
   for(int i = 0; i < 2; i++)
      if(!acConvolution[i].InitPS(master.acConvolution[i].AsObject()))
         ReturnFalse;
   if(!cResidual.Init(0, 2, OpenCL, acConvolution[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(acConvolution[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!acConvolution[0].FeedForward(NeuronOCL))
      ReturnFalse;
   if(!acConvolution[1].FeedForward(acConvolution[0].AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(NeuronOCL.getOutput(), acConvolution[1].getOutput(), cResidual.getOutput(), iWindow, true, 0, 0, 0, 1))
      ReturnFalse;
   if(!CNeuronConvSAMOCL::feedForward(cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronConvSAMOCL::calcInputGradients(cResidual.AsObject()))
      ReturnFalse;
   if(!acConvolution[0].CalcHiddenGradients(acConvolution[1].AsObject()))
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(acConvolution[0].AsObject()))
      ReturnFalse;
   if(NeuronOCL.Activation() == None)
     {
      if(!SumAndNormalize(NeuronOCL.getGradient(), cResidual.getGradient(), NeuronOCL.getGradient(), iWindow, false, 0, 0, 0, 1))
         ReturnFalse;
     }
   else
     {
      if(!DeActivation(NeuronOCL.getOutput(), cResidual.getGradient(), cResidual.getPrevOutput(), NeuronOCL.Activation()) ||
         !SumAndNormalize(NeuronOCL.getGradient(), cResidual.getPrevOutput(), NeuronOCL.getGradient(), iWindow, false, 0, 0, 0, 1))
         ReturnFalse;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!CNeuronConvSAMOCL::updateInputWeights(cResidual.AsObject()))
      ReturnFalse;
   if(!acConvolution[1].UpdateInputWeights(acConvolution[0].AsObject()))
      ReturnFalse;
   if(!acConvolution[0].UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::Save(const int file_handle)
  {
   if(!CNeuronConvSAMOCL::Save(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
      if(!acConvolution[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronPSBlock::Load(const int file_handle)
  {
   if(!CNeuronConvSAMOCL::Load(file_handle))
      ReturnFalse;
//---
   for(int i = 0; i < 2; i++)
      if(!LoadInsideLayer(file_handle, acConvolution[i].AsObject()))
         ReturnFalse;
//---
   if(!cResidual.Init(0, 2, OpenCL, acConvolution[1].Neurons(), optimization, iBatch))
      ReturnFalse;
   if(!cResidual.SetGradient(acConvolution[1].getGradient(), true))
      ReturnFalse;
   cResidual.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronPSBlock::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvSAMOCL::SetOpenCL(obj);
   for(int i = 0; i < 2; i++)
      acConvolution[i].SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronLegendreWaveletsHL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint step, uint units_count, uint filters, uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window, step, 2 * filters, units_count, variables, optimization_type, batch))
      ReturnFalse;
//---
   WeightsConv.BufferInit(WeightsConv.Total(), 0);
   const uint shift_hight = (iWindow + 1) * filters;
   for(uint i = 0; i < iWindow; i++)
     {
      uint shift = i;
      float k = float(2.0 * i - 1.0) / iWindow;
      for(uint f = 1; f <= filters; f++)
        {
         float value = 0;
         switch(f)
           {
            case 1:
               value = k;
               break;
            case 2:
               value = (3 * k * k - 1) / 2;
               break;
            default:
               value = ((2 * f - 1) * k * WeightsConv.At(shift - (iWindow + 1)) - (f - 1) * WeightsConv.At(shift - 2 * (iWindow + 1))) / f;
               break;
           }
         for(uint v = 0; v < iVariables; v++)
           {
            uint shift_var = 2 * shift_hight * v;
            if(!WeightsConv.Update(shift + shift_var, value))
               ReturnFalse;
            if(!WeightsConv.Update(shift + shift_var + shift_hight, MathPow(-1.0f, float(i))*value))
               ReturnFalse;
           }
         shift += iWindow + 1;
        }
     }
   if(!!OpenCL)
     {
      if(!WeightsConv.BufferWrite())
         ReturnFalse;
      uint global_work_size[] = {iWindowOut * iVariables};
      uint global_work_offset[] = {0};
      setBuffer(def_k_NormalizeWeights, def_k_norm_buffer, WeightsConv.GetIndex());
      setArgument(def_k_NormalizeWeights, def_k_norm_dimension, (int)iWindow + 1);
      kernelExecute(def_k_NormalizeWeights, global_work_offset, global_work_size)
#ifdef  _DEBUG
      if(!WeightsConv.BufferRead())
         ReturnFalse;
#endif
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                    uint chanels_in, uint chanels_out, uint window,
                                    uint step, uint units_count, uint group_size, uint groups,
                                    ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   int units_out = ((int)units_count - (int)window + (int)step - 1) / (int)step + 1;
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, group_size * groups, group_size * groups, chanels_out, units_out, 1, optimization_type, batch))
      ReturnFalse;
//--- Projection In
   uint index = 0;
   if(!cProjectionIn.Init(0, index, OpenCL, chanels_in, chanels_in, group_size * groups, units_count, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormalizeIn.Init(0, index, OpenCL, cProjectionIn.Neurons(), iBatch, optimization))
      ReturnFalse;
   cNormalizeIn.SetActivationFunction(LReLU);
   index++;
   if(!cTransposeIn.Init(0, index, OpenCL, units_count, groups, group_size, optimization, iBatch))
      ReturnFalse;
   cTransposeIn.SetActivationFunction((ENUM_ACTIVATION)cNormalizeIn.Activation());
//--- Feature Extraction
   index++;
   if(!cFeatureExtraction.Init(0, index, OpenCL, group_size * window, group_size * step, group_size, units_out, groups, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormalizeFeature.Init(0, index, OpenCL, cFeatureExtraction.Neurons(), iBatch, optimization))
      ReturnFalse;
   cNormalizeFeature.SetActivationFunction(LReLU);
//--- Projection Out
   index++;
   if(!cTransposeOut.Init(0, index, OpenCL, groups, units_out, group_size, optimization, iBatch))
      ReturnFalse;
   cTransposeOut.SetActivationFunction((ENUM_ACTIVATION)cNormalizeFeature.Activation());
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Projection In
   if(!cProjectionIn.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cNormalizeIn.FeedForward(cProjectionIn.AsObject()))
      ReturnFalse;
   if(!cTransposeIn.FeedForward(cNormalizeIn.AsObject()))
      ReturnFalse;
//--- Feature Extraction
   if(!cFeatureExtraction.FeedForward(cTransposeIn.AsObject()))
      ReturnFalse;
   if(!cNormalizeFeature.FeedForward(cFeatureExtraction.AsObject()))
      ReturnFalse;
//--- Projection Out
   if(!cTransposeOut.FeedForward(cNormalizeFeature.AsObject()))
      ReturnFalse;
   return CNeuronConvOCL::feedForward(cTransposeOut.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Projection Out
   if(!CNeuronConvOCL::calcInputGradients(cTransposeOut.AsObject()))
      ReturnFalse;
//--- Feature Extraction\
   if(!cNormalizeFeature.CalcHiddenGradients(cTransposeOut.AsObject()))
      ReturnFalse;
   if(!cFeatureExtraction.CalcHiddenGradients(cNormalizeFeature.AsObject()))
      ReturnFalse;
//--- Projection In
   if(!cTransposeIn.CalcHiddenGradients(cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!cNormalizeIn.CalcHiddenGradients(cTransposeIn.AsObject()))
      ReturnFalse;
   if(!cProjectionIn.CalcHiddenGradients(cNormalizeIn.AsObject()))
      ReturnFalse;
//--- Inputs
   if(!NeuronOCL.CalcHiddenGradients(cProjectionIn.AsObject()))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- Projection Out
   if(!CNeuronConvOCL::updateInputWeights(cTransposeOut.AsObject()))
      ReturnFalse;
//--- Feature Extraction
   if(!cNormalizeFeature.UpdateInputWeights(cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!cFeatureExtraction.UpdateInputWeights(cTransposeIn.AsObject()))
      ReturnFalse;
//--- Projection In
   if(!cNormalizeIn.UpdateInputWeights(cProjectionIn.AsObject()))
      ReturnFalse;
   if(!cProjectionIn.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
   if(!cProjectionIn.Save(file_handle))
      ReturnFalse;
   if(!cNormalizeIn.Save(file_handle))
      ReturnFalse;
   if(!cTransposeIn.Save(file_handle))
      ReturnFalse;
   if(!cFeatureExtraction.Save(file_handle))
      ReturnFalse;
   if(!cNormalizeFeature.Save(file_handle))
      ReturnFalse;
   if(!cTransposeOut.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBottleneck::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjectionIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormalizeIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFeatureExtraction.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormalizeFeature.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeOut.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResNeXtBottleneck::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
   cProjectionIn.SetOpenCL(obj);
   cNormalizeIn.SetOpenCL(obj);
   cTransposeIn.SetOpenCL(obj);
   cFeatureExtraction.SetOpenCL(obj);
   cNormalizeFeature.SetOpenCL(obj);
   cTransposeOut.SetOpenCL(obj);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                                  uint chanels_in, uint chanels_out,
                                  uint units_in, uint units_out,
                                  ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, chanels_in, chanels_in, chanels_out,
                            units_out, 1, optimization_type, batch))
      ReturnFalse;
//---
   int index = 0;
   if(!cTransposeIn.Init(0, index, OpenCL, units_in, chanels_in, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cProjectionTime.Init(0, index, OpenCL, units_in, units_in, units_out, chanels_in, 1, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cNormalizeTime.Init(0, index, OpenCL, cProjectionTime.Neurons(), iBatch, optimization))
      ReturnFalse;
   index++;
   if(!cTransposeOut.Init(0, index, OpenCL, chanels_in, units_out, optimization, iBatch))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
//--- Projection Timeline
   if(!cTransposeIn.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cProjectionTime.FeedForward(cTransposeIn.AsObject()))
      ReturnFalse;
   if(!cNormalizeTime.FeedForward(cProjectionTime.AsObject()))
      ReturnFalse;
//--- Projection Chanels
   if(!cTransposeOut.FeedForward(cNormalizeTime.AsObject()))
      ReturnFalse;
   return CNeuronConvOCL::feedForward(cTransposeOut.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//--- Projection Chanels
   if(!CNeuronConvOCL::calcInputGradients(cTransposeOut.AsObject()))
      ReturnFalse;
//--- Projection Timeline
   if(!cNormalizeTime.CalcHiddenGradients(cTransposeOut.AsObject()))
      ReturnFalse;
   if(!cProjectionTime.CalcHiddenGradients(cNormalizeTime.AsObject()))
      ReturnFalse;
   if(!cTransposeIn.CalcHiddenGradients(cProjectionTime.AsObject()))
      ReturnFalse;
//--- Inputs
   return NeuronOCL.CalcHiddenGradients(cTransposeIn.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
//--- Projection Chanels
   if(!CNeuronConvOCL::updateInputWeights(cTransposeOut.AsObject()))
      ReturnFalse;
//--- Projection Timeline
   if(!cNormalizeTime.UpdateInputWeights(cProjectionTime.AsObject()))
      ReturnFalse;
   if(!cProjectionTime.UpdateInputWeights(cTransposeIn.AsObject()))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      ReturnFalse;
//---
   if(!cTransposeIn.Save(file_handle))
      ReturnFalse;
   if(!cProjectionTime.Save(file_handle))
      ReturnFalse;
   if(!cNormalizeTime.Save(file_handle))
      ReturnFalse;
   if(!cTransposeOut.Save(file_handle))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtResidual::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      ReturnFalse;
//---
   if(!LoadInsideLayer(file_handle, cTransposeIn.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjectionTime.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cNormalizeTime.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cTransposeOut.AsObject()))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResNeXtResidual::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronConvOCL::SetOpenCL(obj);
//---
   cTransposeIn.SetOpenCL(OpenCL);
   cProjectionTime.SetOpenCL(OpenCL);
   cNormalizeTime.SetOpenCL(OpenCL);
   cTransposeOut.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                               uint chanels_in, uint chanels_out, uint window,
                               uint step, uint units_count, uint group_size, uint groups,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   int units_out = ((int)units_count - (int)window + (int)step - 1) / (int)step + 1;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_out * chanels_out, optimization_type, batch))
      ReturnFalse;
//---
   iChanelsOut = chanels_out;
//---
   int index = 0;
   if(!cBottleneck.Init(0, index, OpenCL, chanels_in, chanels_out, window, step, units_count,
                        group_size, groups, optimization, iBatch))
      ReturnFalse;
   index++;
   if(!cResidual.Init(0, index, OpenCL, chanels_in, chanels_out, units_count, units_out, optimization, iBatch))
      ReturnFalse;
//---
   if(!cResidual.SetGradient(cBottleneck.getGradient(), true))
      ReturnFalse;
   if(!SetGradient(cBottleneck.getGradient(), true))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cBottleneck.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!cResidual.FeedForward(NeuronOCL))
      ReturnFalse;
   if(!SumAndNormalize(cBottleneck.getOutput(), cResidual.getOutput(), Output, iChanelsOut, true, 0, 0, 0, 1))
      ReturnFalse;
//--- result
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   if(!NeuronOCL.CalcHiddenGradients(cBottleneck.AsObject()))
      ReturnFalse;
//---
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(cBuffer.GetOpenCL() != OpenCL ||
      cBuffer.Total() != temp.Total())
     {
      if(!cBuffer.BufferInitLike(temp))
         ReturnFalse;
     }
   if(!NeuronOCL.SetGradient(GetPointer(cBuffer), false))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(cResidual.AsObject()))
      ReturnFalse;
   if(!SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
      ReturnFalse;
   if(!NeuronOCL.SetGradient(temp, false))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cBottleneck.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
   if(!cResidual.UpdateInputWeights(NeuronOCL))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   if(FileWriteInteger(file_handle, int(iChanelsOut)) < INT_VALUE)
      ReturnFalse;
//---
   if(!cBottleneck.Save(file_handle))
      ReturnFalse;
   if(!cResidual.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronResNeXtBlock::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iChanelsOut = (uint)FileReadInteger(file_handle);
//---
   if(!LoadInsideLayer(file_handle, cBottleneck.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cResidual.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronResNeXtBlock::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
//---
   cBottleneck.SetOpenCL(OpenCL);
   cResidual.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::TopKgates(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint window = cProjection.GetFilters() / 2;
   uint units = cProjection.GetUnits();
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {window, units};
   uint local_work_size[2] = {window, 1};
//---
   uint kernel = def_k_TopKgates;
   setBuffer(kernel, def_k_topK_inputs, cProjection.getOutputIndex())
   setBuffer(kernel, def_k_topK_noises, cbNoise.GetIndex())
   setBuffer(kernel, def_k_topK_gates, cGates.getOutputIndex())
   setArgument(kernel, def_k_topK_k, iK)
//---
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef  _DEBUG
   if(!cGates.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::TopKgatesGradient(void)
  {
   if(!OpenCL)
      ReturnFalse;
   uint window = cProjection.GetFilters() / 2;
   uint units = cProjection.GetUnits();
//---
   uint global_work_offset[2] = {0, 0};
   uint global_work_size[2] = {window, units};
//---
   uint kernel = def_k_TopKgatesGrad;
   setBuffer(kernel, def_k_topKgr_inputs, cProjection.getOutputIndex())
   setBuffer(kernel, def_k_topKgr_grad_inputs, cProjection.getGradientIndex())
   setBuffer(kernel, def_k_topKgr_noises, cbNoise.GetIndex())
   setBuffer(kernel, def_k_topKgr_gates, cGates.getOutputIndex())
   setBuffer(kernel, def_k_topKgr_grad_gates, cGates.getGradientIndex())
//---
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef  _DEBUG
   if(!cGates.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint window, uint units_count, uint gates, uint top_k, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronSoftMaxOCL::Init(numOutputs, myIndex, open_cl, gates * units_count, optimization_type, batch))
      ReturnFalse;
   SetHeads(units_count);
   if(!cProjection.Init(0, 0, OpenCL, window, window, 2 * gates, units_count, 1, optimization, iBatch))
      ReturnFalse;
   cProjection.SetActivationFunction(None);
   if(!cbNoise.BufferInit(Neurons(), 0) ||
      !cbNoise.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGates.Init(0, 1, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cGates.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   if(!cProjection.FeedForward(NeuronOCL))
      ReturnFalse;
//---
   if(bTrain)
     {
      double random[];
      if(!MathRandomNormal(0, 1, Neurons(), random))
         ReturnFalse;
      if(!cbNoise.AssignArray(random))
         ReturnFalse;
      if(!cbNoise.BufferWrite())
         ReturnFalse;
     }
   else
      if(!cbNoise.Fill(0))
         ReturnFalse;
//---
   if(!TopKgates())
      ReturnFalse;
//---
   return CNeuronSoftMaxOCL::feedForward(cGates.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!CNeuronSoftMaxOCL::calcInputGradients(cGates.AsObject()))
      ReturnFalse;
//---
   if(!TopKgatesGradient())
      ReturnFalse;
//---
   return NeuronOCL.CalcHiddenGradients(cProjection.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   return cProjection.UpdateInputWeights(NeuronOCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::Save(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Save(file_handle))
      ReturnFalse;
   if(!cProjection.Save(file_handle))
      ReturnFalse;
   if(FileWriteInteger(file_handle, iK) < INT_VALUE)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronTopKGates::Load(const int file_handle)
  {
   if(!CNeuronSoftMaxOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cProjection.AsObject()))
      ReturnFalse;
   if(FileIsEnding(file_handle))
      ReturnFalse;
   iK = FileReadInteger(file_handle);
//---
   if(!cbNoise.BufferInit(Neurons(), 0) ||
      !cbNoise.BufferCreate(OpenCL))
      ReturnFalse;
   if(!cGates.Init(0, 1, OpenCL, Neurons(), optimization, iBatch))
      ReturnFalse;
   cGates.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronTopKGates::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronSoftMaxOCL::SetOpenCL(obj);
   cProjection.SetOpenCL(OpenCL);
   cGates.SetOpenCL(OpenCL);
   cbNoise.BufferCreate(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl,
                            uint window, uint step, uint window_out, uint units_count,
                            uint variables, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_count * window_out * variables, optimization_type, batch))
      ReturnFalse;
   SetActivationFunction(None);
   for(uint i = 0; i < caProjections.Size(); i++)
     {
      if(!caProjections[i].Init(0, i, OpenCL, window, step, window_out, units_count, variables, optimization, iBatch))
         ReturnFalse;
     }
   caProjections[0].SetActivationFunction(GELU);
   caProjections[1].SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::feedForward(CNeuronBaseOCL* NeuronOCL)
  {
   for(uint i = 0; i < caProjections.Size(); i++)
      if(!caProjections[i].FeedForward(NeuronOCL))
         ReturnFalse;
   if(!ElementMult(caProjections[0].getOutput(), caProjections[1].getOutput(), Output))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::calcInputGradients(CNeuronBaseOCL* NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   if(!ElementMultGrad(caProjections[0].getOutput(), caProjections[0].getGradient(),
                       caProjections[1].getOutput(), caProjections[1].getGradient(),
                       Gradient, caProjections[0].Activation(), caProjections[1].Activation()
                      ))
      ReturnFalse;
//---
   if(!NeuronOCL.CalcHiddenGradients(caProjections[0].AsObject()))
      ReturnFalse;
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(NeuronOCL.getPrevOutput(), false) ||
      !NeuronOCL.CalcHiddenGradients(caProjections[1].AsObject()) ||
      !SumAndNormalize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false)
     )
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::updateInputWeights(CNeuronBaseOCL* NeuronOCL)
  {
   for(uint i = 0; i < caProjections.Size(); i++)
      if(!caProjections[i].UpdateInputWeights(NeuronOCL))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CNeuronSwiGLUOCL *Source = source;
   for(uint i = 0; i < caProjections.Size(); i++)
      if(!caProjections[i].WeightsUpdate(Source.caProjections[i].AsObject(), tau))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caProjections.Size(); i++)
      if(!caProjections[i].Save(file_handle))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronSwiGLUOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   for(uint i = 0; i < caProjections.Size(); i++)
      if(!LoadInsideLayer(file_handle, caProjections[i].AsObject()))
         ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronSwiGLUOCL::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   for(uint i = 0; i < caProjections.Size(); i++)
      caProjections[i].SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::Init(uint numOutputs, uint myIndex, COpenCLMy * open_cl, uint numNeurons, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
   if(!cOne.Init(numNeurons, 0, OpenCL, 1, optimization, iBatch))
      ReturnFalse;
   cOne.SetActivationFunction(None);
   if(!cOne.getOutput().Fill(1))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::FeedForward(void)
  {
   if(!bTrain)
      return true;
//---
   return CNeuronBaseOCL::feedForward(cOne.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::UpdateInputWeights(void)
  {
   return CNeuronBaseOCL::updateInputWeights(cOne.AsObject());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cOne.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cOne.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CParams::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cOne.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::WeightsUpdate(CNeuronBaseOCL* source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CParams *Source = source;
   if(!cOne.WeightsUpdate(Source.cOne.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CParams::Identity(const int rows, const int cols)
  {
   if(rows * cols != Neurons())
      ReturnFalse;
   matrix<float> ident = matrix<float>::Identity(rows, cols);
   if(!ident.Reshape(rows * cols, 1))
      ReturnFalse;
   if(!ident.Resize(rows * cols, 2))
      ReturnFalse;
   if(!ident.Col(vector<float>::Zeros(rows * cols), 1))
      ReturnFalse;
   CBufferFloat *w = cOne.getWeights();
   if(!w || !w.AssignArray(ident))
      ReturnFalse;
   w.BufferFree();
   if(!w.BufferCreate(OpenCL))
      ReturnFalse;
//---
   return FeedForward();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint numNeurons, uint period,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      ReturnFalse;
//---
   cOnes.Clear();
   cOnes.SetOpenCL(OpenCL);
   for(uint i = 0; i < period; i++)
     {
      cCurrent = new CNeuronBaseOCL();
      if(!cCurrent)
         ReturnFalse;
      if(!cCurrent.Init(Neurons(), i, OpenCL, 1, optimization, iBatch) ||
         !cOnes.Add(cCurrent))
        {
         DeleteObj(cCurrent);
         ReturnFalse;
        }
      cCurrent.SetActivationFunction(None);
     }
//---
   if(!SetPosition(0))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::Identity(const int rows, const int cols)
  {
   if(rows * cols != Neurons())
      ReturnFalse;
   matrix<float> ident = matrix<float>::Identity(rows, cols);
   if(!ident.Reshape(rows * cols, 1))
      ReturnFalse;
   if(!ident.Resize(rows * cols, 2))
      ReturnFalse;
   if(!ident.Col(vector<float>::Zeros(rows * cols), 1))
      ReturnFalse;
//---
   CBufferFloat *w = NULL;
   for(int i = 0; i < cOnes.Total(); i++)
     {
      if(!cOnes[i])
         ReturnFalse;
      w = cOnes[i].getWeights();
      if(!w || !w.AssignArray(ident))
         ReturnFalse;
      w.BufferFree();
      if(!w.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return FeedForward();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::Zeros(void)
  {
   matrix<float> zeros = matrix<float>::Zeros(Neurons(), 2);
//---
   CBufferFloat *w = NULL;
   for(int i = 0; i < cOnes.Total(); i++)
     {
      if(!cOnes[i])
         ReturnFalse;
      w = cOnes[i].getWeights();
      if(!w || !w.AssignArray(zeros))
         ReturnFalse;
      w.BufferFree();
      if(!w.BufferCreate(OpenCL))
         ReturnFalse;
     }
//---
   return FeedForward();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::SetPosition(int position)
  {
   cCurrent = cOnes[position];
   if(!cCurrent)
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::FeedForward(void)
  {
   if(!cCurrent)
      if(!SetPosition(0))
         ReturnFalse;
//---
   return CNeuronBaseOCL::feedForward(cCurrent);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::UpdateInputWeights(void)
  {
   return CNeuronBaseOCL::updateInputWeights(cCurrent);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
//---
   CCircleParams* Source = source;
   if(!cOnes.WeightsUpdate(Source.cOnes.AsObject(), tau))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
//---
   return cOnes.Save(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CCircleParams::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
//---
   return cOnes.Load(file_handle);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CCircleParams::SetOpenCL(COpenCLMy * obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cOnes.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBufferFloat* CCircleParams::getWeightsParams(void)
  {
   if(!cCurrent)
      if(!SetPosition(0))
         return NULL;
   return cCurrent.getWeights();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DilatedCorrelation(CNeuronBaseOCL *feature,
                                        CNeuronBaseOCL *correlations,
                                        uint shifts_index,
                                        uint dimension,
                                        uint units)
  {
   if(!OpenCL || !feature || !correlations ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint corrs = correlations.Neurons() / units;
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units,
                               (uint)MathMin(dimension, OpenCL.GetMaxLocalSize(1)),
                               corrs
                             };
   uint local_work_size[] = { 1, global_work_size[1], 1 };
   uint kernel = def_k_DilatedCorrelation;
   setBuffer(kernel, def_k_dcor_correlations, correlations.getOutputIndex())
   setBuffer(kernel, def_k_dcor_feature, feature.getOutputIndex())
   setBuffer(kernel, def_k_dcor_shifts, shifts_index)
   setArgument(kernel, def_k_dcor_dimension, dimension)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!correlations.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DilatedCorrelationGrad(CNeuronBaseOCL *feature,
      CNeuronBaseOCL *correlations,
      uint shifts_index,
      uint dimension,
      uint units)
  {
   if(!OpenCL || !feature || !correlations ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint corrs = correlations.Neurons() / units;
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units,
                               (uint)MathMin(corrs, OpenCL.GetMaxLocalSize(1)),
                               dimension
                             };
   uint local_work_size[] = { 1, global_work_size[1], 1 };
   uint kernel = def_k_DilatedCorrelationGrad;
   setBuffer(kernel, def_k_dcorgr_corr_gr, correlations.getGradientIndex())
   setBuffer(kernel, def_k_dcorgr_feature, feature.getOutputIndex())
   setBuffer(kernel, def_k_dcorgr_feature_gr, feature.getGradientIndex())
   setBuffer(kernel, def_k_dcorgr_shifts, shifts_index)
   setArgument(kernel, def_k_dcorgr_total_corr, corrs)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!feature.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DilatedDifference(CNeuronBaseOCL *feature,
                                       CNeuronBaseOCL *difference,
                                       uint shifts_index,
                                       uint dimension,
                                       uint units)
  {
   if(!OpenCL || !feature || !difference ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint shifts = difference.Neurons() / feature.Neurons();
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units,
                               shifts,
                               dimension
                             };
   uint kernel = def_k_DilatedDifference;
   setBuffer(kernel, def_k_ddiff_differences, difference.getOutputIndex())
   setBuffer(kernel, def_k_ddiff_feature, feature.getOutputIndex())
   setBuffer(kernel, def_k_ddiff_shifts, shifts_index)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!difference.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::DilatedDifferenceGrad(CNeuronBaseOCL *feature,
      CNeuronBaseOCL *difference,
      uint shifts_index,
      uint dimension,
      uint units)
  {
   if(!OpenCL || !feature || !difference ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint shifts = difference.Neurons() / feature.Neurons();
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units,
                               (uint)MathMin(shifts, OpenCL.GetMaxLocalSize(1)),
                               dimension
                             };
   uint local_work_size[] = { 1, global_work_size[1], 1 };
   uint kernel = def_k_DilatedDifferenceGrad;
   setBuffer(kernel, def_k_ddiffgr_differences_gr, difference.getGradientIndex())
   setBuffer(kernel, def_k_ddiffgr_feature, feature.getOutputIndex())
   setBuffer(kernel, def_k_ddiffgr_feature_gr, feature.getGradientIndex())
   setBuffer(kernel, def_k_ddiffgr_shifts, shifts_index)
   setArgument(kernel, def_k_ddiffgr_total_shifts, shifts)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!feature.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CNeuronBaseOCL::CreateShifts(const uint max_displacement, int &count)
  {
   if(!OpenCL || max_displacement < 1)
      return INVALID_HANDLE;
//---
   int shifts[];
   if(ArrayResize(shifts, max_displacement) <= 0)
      return INVALID_HANDLE;
//---
   count = 0;
   for(uint i = 0; i < max_displacement; i++)
     {
      if(i > 2 && bool(i & 1))
         continue;
      shifts[count] = (int)(i + 1);
      count++;
     }
   if(ArrayResize(shifts, count) < (int)count)
      return INVALID_HANDLE;
//---
   return OpenCL.AddBufferFromArray(shifts, 0, count, CL_MEM_READ_ONLY);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::PerturbedMatrix(CNeuronBaseOCL *inputs,
                                     CNeuronBaseOCL *perturb,
                                     CNeuronBaseOCL *outputs,
                                     float perturb_mult,
                                     uint dimension, uint units)
  {
   if(!OpenCL || !inputs || !perturb || !outputs ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint global_work_offset[2] = {0};
   uint global_work_size[] = { dimension, units };
   uint kernel = def_k_PerturbedMatrix;
   setBuffer(kernel, def_k_pm_inputs, inputs.getOutputIndex())
   setBuffer(kernel, def_k_pm_perturb, perturb.getOutputIndex())
   setBuffer(kernel, def_k_pm_output, outputs.getOutputIndex())
   setArgument(kernel, def_k_pm_perturb_mult, perturb_mult)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!outputs.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::PerturbedMatrixGrad(CNeuronBaseOCL *inputs,
      CNeuronBaseOCL *perturb,
      CNeuronBaseOCL *outputs,
      float perturb_mult,
      uint dimension, uint units)
  {
   if(!OpenCL || !inputs || !perturb || !outputs ||
      units <= 0 || dimension <= 0)
      ReturnFalse;
//---
   uint global_work_offset[2] = { 0 };
   uint global_work_size[] = { dimension, units };
   uint kernel = def_k_PerturbedMatrixGrad;
   setBuffer(kernel, def_k_pm_inputs, inputs.getGradientIndex())
   setBuffer(kernel, def_k_pm_perturb, perturb.getGradientIndex())
   setBuffer(kernel, def_k_pm_output, outputs.getGradientIndex())
   setArgument(kernel, def_k_pm_perturb_mult, perturb_mult)
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!inputs.getGradient().BufferRead() ||
      !perturb.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::LinearUpsample(CNeuronBaseOCL *data,
                                    CNeuronBaseOCL *upsample,
                                    uint upsampl_mult,
                                    uint units,
                                    uint variables)
  {
   if(!OpenCL || !data || !upsample ||
      units <= 0 || upsampl_mult <= 0 || variables <= 0)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units, variables, upsampl_mult };
   uint kernel = def_k_LinearUpsample;
   setBuffer(kernel, def_k_lu_data, data.getOutputIndex())
   setBuffer(kernel, def_k_lu_upsample, upsample.getOutputIndex())
   kernelExecute(kernel, global_work_offset, global_work_size)
#ifdef _DEBUG
   if(!upsample.getOutput().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronBaseOCL::LinearUpsampleGrad(CNeuronBaseOCL *data,
                                        CNeuronBaseOCL *upsample,
                                        uint upsampl_mult,
                                        uint units,
                                        uint variables)
  {
   if(!OpenCL || !data || !upsample ||
      units <= 0 || upsampl_mult <= 0 || variables <= 0)
      ReturnFalse;
//---
   uint global_work_offset[3] = {0};
   uint global_work_size[] = { units, variables, upsampl_mult };
   uint local_work_size[] = { 1, 1, (uint)MathMin(upsampl_mult, OpenCL.GetMaxLocalSize(2)) };
   uint kernel = def_k_LinearUpsampleGrad;
   setBuffer(kernel, def_k_lug_data_gr, data.getGradientIndex())
   setBuffer(kernel, def_k_lug_upsample_gr, upsample.getGradientIndex())
   setArgument(kernel, def_k_lug_dimension_htr, (int)upsampl_mult)
   kernelExecuteLoc(kernel, global_work_offset, global_work_size, local_work_size)
#ifdef _DEBUG
   if(!data.getGradient().BufferRead())
      ReturnFalse;
#endif
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint params_to_field, uint fields, uint embed_size,
                             uint candidates, uint topK,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, params_to_field * fields, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cCandidates.Init(0, index, OpenCL, params_to_field * candidates, optimization, iBatch))
      ReturnFalse;
   cCandidates.SetActivationFunction(None);
   index++;
   if(!cFieldEmbeddings.Init(0, index, OpenCL, fields * embed_size, optimization, iBatch))
      ReturnFalse;
   cFieldEmbeddings.SetActivationFunction(SIGMOID);
   if(fields <= embed_size)
      if(!cFieldEmbeddings.Identity(fields, embed_size))
         ReturnFalse;
   index++;
   cScore.Clear();
   cScore.SetOpenCL(OpenCL);
   CNeuronConvOCL* conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, embed_size, embed_size, 4 * candidates, fields, 1, optimization, iBatch) ||
      !cScore.Add(conv))
      DeleteObjAndFalse(conv);
   conv.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 4 * candidates, 4 * candidates, candidates, fields, 1, optimization, iBatch) ||
      !cScore.Add(conv))
      DeleteObjAndFalse(conv);
   conv.SetActivationFunction(SoftPlus);
   index++;
   CNeuronSparseSoftMax* softmax = new CNeuronSparseSoftMax();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, fields, candidates, topK, optimization, iBatch) ||
      !cScore.Add(softmax))
      DeleteObjAndFalse(softmax);
   iFields = fields;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::FeedForward(void)
  {
   if(!bTrain)
      return true;
//---
   if(!cCandidates.FeedForward())
      ReturnFalse;
   if(!cFieldEmbeddings.FeedForward())
      ReturnFalse;
   CNeuronBaseOCL* prev = cFieldEmbeddings.AsObject();
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(prev.Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = prev;
   uint topK = softmax.DimensionOut();
   uint params_to_field = Neurons() / iFields;
   uint candidates = cCandidates.Neurons() / params_to_field;
   if(!SparseMatMul(softmax.GetIndexes(), softmax.getOutput(), cCandidates.getOutput(),
                    Output, iFields, topK, candidates, params_to_field))
      ReturnFalse;
   dActivation(this);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::UpdateInputWeights(void)
  {
//--- Backprop gradients
   CNeuronBaseOCL* next = cScore[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next || next.Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = next;
   uint topK = softmax.DimensionOut();
   uint params_to_field = Neurons() / iFields;
   uint candidates = cCandidates.Neurons() / params_to_field;
   if(!SparseMatMulGrad(softmax.GetIndexes(), softmax.getOutput(), softmax.getGradient(),
                        cCandidates.getOutput(), cCandidates.getGradient(),
                        Gradient, iFields, topK, candidates, params_to_field))
      ReturnFalse;
   Deactivation(cCandidates);
   for(int i = cScore.Total() - 2; i >= 0; i--)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!cFieldEmbeddings.CalcHiddenGradients(next))
      ReturnFalse;
//--- Candidates
   if(!SumAndNormalize(cCandidates.getOutput(), cCandidates.getGradient(), cCandidates.getGradient(),
                       params_to_field, true, 0, 0, 0, 1))
      ReturnFalse;
//--- Update weights
   float error = 1;
   if(!cCandidates.calcOutputGradients(cCandidates.getGradient(), error) ||
      !DiversityLoss(cCandidates.AsObject(), candidates, params_to_field, true) ||
      !cCandidates.UpdateInputWeights())
      ReturnFalse;
   if(!cFieldEmbeddings.UpdateInputWeights())
      ReturnFalse;
   curr = cFieldEmbeddings.AsObject();
   for(int i = 0; i < cScore.Total(); i++)
     {
      next = cScore[i];
      if(!next ||
         !next.UpdateInputWeights(curr))
         ReturnFalse;
      curr = next;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cCandidates.Save(file_handle))
      ReturnFalse;
   if(!cFieldEmbeddings.Save(file_handle))
      ReturnFalse;
   if(!cScore.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCandidates.AsObject()))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cFieldEmbeddings.AsObject()))
      ReturnFalse;
   if(!cScore.Load(file_handle))
      ReturnFalse;
//---
   if(!cScore[-1] || cScore[-1].Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = cScore[-1];
   iFields = softmax.Heads();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CFieldAwareParams::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CFieldAwareParams* Source = source;
   dWeightsUpdate(cCandidates, Source, tau);
   dWeightsUpdate(cFieldEmbeddings, Source, tau);
   dWeightsUpdate(cScore, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFieldAwareParams::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cCandidates.SetOpenCL(OpenCL);
   cFieldEmbeddings.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CFieldAwareParams::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cCandidates.TrainMode(bTrain);
   cFieldEmbeddings.TrainMode(bTrain);
   cScore.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                 uint window_in, uint window_out, uint fields,
                                 uint embed_size, uint candidates, uint topK,
                                 ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * fields, optimization_type, batch))
      ReturnFalse;
   if(!cParams.Init(0, 0, OpenCL, window_in * window_out, fields, embed_size, candidates, topK, optimization, iBatch))
      ReturnFalse;
   cParams.SetActivationFunction(None);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
      if(!cParams.FeedForward())
         ReturnFalse;
//---
   uint fields = cParams.GetFields();
   uint window_out = Neurons() / fields;
   uint window_in = cParams.Neurons() / (fields * window_out);
   if(!NeuronOCL || NeuronOCL.Neurons() < int(window_in * fields))
      ReturnFalse;
   if(!MatMul(NeuronOCL.getOutput(), cParams.getOutput(), Output, 1, window_in, window_out, fields, true))
      ReturnFalse;
   dActivation(this);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
//---
   uint fields = cParams.GetFields();
   uint window_out = Neurons() / fields;
   uint window_in = cParams.Neurons() / (fields * window_out);
   if(NeuronOCL.Neurons() < int(window_in * fields))
      ReturnFalse;
   if(!MatMulGrad(NeuronOCL.getOutput(), NeuronOCL.getGradient(),
                  cParams.getOutput(), cParams.getGradient(),
                  Gradient, 1, window_in, window_out, fields, true))
      ReturnFalse;
   Deactivation(NeuronOCL);
   Deactivation(cParams);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cParams.UpdateInputWeights())
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cParams.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cParams.AsObject()))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldAwareConv::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronFieldAwareConv* Source = source;
   dWeightsUpdate(cParams, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFieldAwareConv::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cParams.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFieldAwareConv::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cParams.TrainMode(bTrain);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                        uint window, uint fields, uint embed_size,
                                        uint candidates, uint topK,
                                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, embed_size * fields, optimization_type, batch))
      ReturnFalse;
//---
   uint index = 0;
   if(!cCandidates.Init(0, index, OpenCL, embed_size * candidates, optimization, iBatch))
      ReturnFalse;
   cCandidates.SetActivationFunction(None);
   index++;
   cScore.Clear();
   cScore.SetOpenCL(OpenCL);
   CNeuronConvOCL* conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, window, window, 2 * candidates, fields, 1, optimization, iBatch) ||
      !cScore.Add(conv))
      DeleteObjAndFalse(conv);
   conv.SetActivationFunction(SoftPlus);
   index++;
   conv = new CNeuronConvOCL();
   if(!conv ||
      !conv.Init(0, index, OpenCL, 2 * candidates, 2 * candidates, candidates, fields, 1, optimization, iBatch) ||
      !cScore.Add(conv))
      DeleteObjAndFalse(conv);
   conv.SetActivationFunction(SoftPlus);
   index++;
   CNeuronSparseSoftMax* softmax = new CNeuronSparseSoftMax();
   if(!softmax ||
      !softmax.Init(0, index, OpenCL, fields, candidates, topK, optimization, iBatch) ||
      !cScore.Add(softmax))
      DeleteObjAndFalse(softmax);
   iFields = fields;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain)
      if(!cCandidates.FeedForward())
         ReturnFalse;
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.FeedForward(prev))
         ReturnFalse;
      prev = curr;
     }
   if(prev.Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = prev;
   uint topK = softmax.DimensionOut();
   uint params_to_field = Neurons() / iFields;
   uint candidates = cCandidates.Neurons() / params_to_field;
   if(!SparseMatMul(softmax.GetIndexes(), softmax.getOutput(), cCandidates.getOutput(),
                    Output, iFields, topK, candidates, params_to_field))
      ReturnFalse;
   dActivation(this);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      ReturnFalse;
   CNeuronBaseOCL* next = cScore[-1];
   CNeuronBaseOCL* curr = NULL;
   if(!next || next.Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = next;
   uint topK = softmax.DimensionOut();
   uint params_to_field = Neurons() / iFields;
   uint candidates = cCandidates.Neurons() / params_to_field;
   if(!SparseMatMulGrad(softmax.GetIndexes(), softmax.getOutput(), softmax.getGradient(),
                        cCandidates.getOutput(), cCandidates.getGradient(),
                        Gradient, iFields, topK, candidates, params_to_field))
      ReturnFalse;
//--- Candidates
   if(!SumAndNormalize(cCandidates.getOutput(), cCandidates.getGradient(), cCandidates.getGradient(),
                       params_to_field, true, 0, 0, 0, 1))
      ReturnFalse;
   float error = 1;
   if(!cCandidates.calcOutputGradients(cCandidates.getGradient(), error) ||
      !DiversityLoss(cCandidates.AsObject(), candidates, params_to_field, true))
      ReturnFalse;
//---
   for(int i = cScore.Total() - 2; i >= 0; i--)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.CalcHiddenGradients(next))
         ReturnFalse;
      next = curr;
     }
   if(!NeuronOCL.CalcHiddenGradients(next))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cCandidates.UpdateInputWeights())
      ReturnFalse;
   CNeuronBaseOCL* prev = NeuronOCL;
   CNeuronBaseOCL* curr = NULL;
   for(int i = 0; i < cScore.Total(); i++)
     {
      curr = cScore[i];
      if(!curr ||
         !curr.UpdateInputWeights(prev))
         ReturnFalse;
      prev = curr;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      ReturnFalse;
   if(!cCandidates.Save(file_handle))
      ReturnFalse;
   if(!cScore.Save(file_handle))
      ReturnFalse;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      ReturnFalse;
   if(!LoadInsideLayer(file_handle, cCandidates.AsObject()))
      ReturnFalse;
   if(!cScore.Load(file_handle))
      ReturnFalse;
//---
   if(!cScore[-1] || cScore[-1].Type() != defNeuronSparseSoftMax)
      ReturnFalse;
   CNeuronSparseSoftMax* softmax = cScore[-1];
   iFields = softmax.Heads();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CNeuronFieldPatternEmbedding::WeightsUpdate(CNeuronBaseOCL *source, float tau)
  {
   if(!CNeuronBaseOCL::WeightsUpdate(source, tau))
      ReturnFalse;
   CNeuronFieldPatternEmbedding* Source = source;
   dWeightsUpdate(cCandidates, Source, tau);
   dWeightsUpdate(cScore, Source, tau);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFieldPatternEmbedding::SetOpenCL(COpenCLMy *obj)
  {
   CNeuronBaseOCL::SetOpenCL(obj);
   cCandidates.SetOpenCL(OpenCL);
   cScore.SetOpenCL(OpenCL);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNeuronFieldPatternEmbedding::TrainMode(bool flag)
  {
   CNeuronBaseOCL::TrainMode(flag);
   cCandidates.TrainMode(bTrain);
   cScore.TrainMode(bTrain);
  }



//+------------------------------------------------------------------+
