//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#ifndef NEURONET_DEFINITIONS_MQH
#define NEURONET_DEFINITIONS_MQH
/// \file
/// \brief NeuroNet.mqh
/// Library for creating Neural network for use in MQL5 experts
/// \author [DNG](https://www.mql5.com/en/users/dng)
/// \copyright Copyright 2019, DNG
//+------------------------------------------------------------------+
///\mainpage NeuronNet
/// Library for creating Neural network for use in MQL5 experts.
/// - \ref const
/// - \ref enums
/// - \ref ObjectTypes
/// - \ref group1
/// - [<b>Class Hierarchy</b>](hierarchy.html)
/// - [<b>Files</b>](files.html)
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, DNG"
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include standart libraries |
//+------------------------------------------------------------------+
#include <Arrays\ArrayFloat.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayObj.mqh>
#include <OpenCL\OpenCL.mqh>
#include <Math\Stat\Normal.mqh>
//+------------------------------------------------------------------+
// Defines
//+------------------------------------------------------------------+
///\defgroup const  Global constants
///@{
#define  lr                3.0e-4f  ///<learning rate
#define  momentum          0.9f     ///<momentum for SGD optimization
#define  WeightsMultiplier 0.01f
//---
/** First momentum multiplier of Adam optimization*/
#define  b1                0.9f
/** Second momentum multiplier of Adam optimization*/
#define  b2                0.999f
/// Pseudo-random sequence generator
#define  xor128            rnd_t=(rnd_x^(rnd_x<<11)); \
                           rnd_x=rnd_y; \
                           rnd_y=rnd_z; \
                           rnd_z=rnd_w; \
                           rnd_w=(rnd_w^(rnd_w>>19))^(rnd_t^(rnd_t>>8))
//---
uint rnd_x = MathRand(), rnd_y = MathRand(), rnd_z = MathRand(), rnd_w = MathRand(), rnd_t = 0;
///@}
#define CreateKernel(id, name)    if(!opencl.KernelCreate(id, name)) { \
                                    PrintFormat("Error of create kernel %s: %d line %d", #name, GetLastError(), __LINE__); \
                                    ReturnFalse; }

#define setBuffer(kernel, id, buffer)  if(!OpenCL.SetArgumentBuffer(kernel, id, buffer)) { \
                                          printf("Error of set parameter kernel %s: %d; line %d", OpenCL.GetKernelName(kernel), GetLastError(), __LINE__); \
                                          ReturnFalse; }
#define setArgument(kernel, id, value) if(!OpenCL.SetArgument(kernel, id, value)) { \
                                          printf("Error of set parameter kernel %s: %d; line %d", OpenCL.GetKernelName(kernel), GetLastError(), __LINE__); \
                                          ReturnFalse; }
#define kernelExecute(kernel,offset,global)  if(!OpenCL.Execute(kernel, global.Size(), offset, global)) { \
                                                string error; \
                                                CLGetInfoString(OpenCL.GetContext(), CL_ERROR_DESCRIPTION, error); \
                                                printf("Error of execution kernel %s %s:\n %s", __FUNCTION__, OpenCL.GetKernelName(kernel), error); \
                                                string  s_global=StringFormat("global = {%d",global[0]); \
                                                for(uint i=1;i<global.Size();i++) s_global=StringFormat("%s, %d",s_global, global[i]); \
                                                printf("%s}",s_global); \
                                                ReturnFalse; }
#define kernelExecuteLoc(kernel,offset,global,local)  if(!OpenCL.Execute(kernel, global.Size(), offset, global, local)) { \
                                                         string error; \
                                                         CLGetInfoString(OpenCL.GetContext(), CL_ERROR_DESCRIPTION, error); \
                                                         printf("Error of execution kernel %s %s:\n %s", __FUNCTION__, OpenCL.GetKernelName(kernel), error); \
                                                         string  s_global=StringFormat("global = {%d",global[0]); \
                                                         for(uint i=1;i<global.Size();i++) s_global=StringFormat("%s, %d",s_global, global[i]); \
                                                         printf("%s}",s_global); \
                                                         string  s_local=StringFormat("local = {%d",local[0]); \
                                                         for(uint i=1;i<local.Size();i++) s_local=StringFormat("%s, %d",s_local, local[i]); \
                                                         printf("%s}",s_local); \
                                                         ReturnFalse; }

#define ReturnFalse  { printf("Error at %s line %d", __FUNCTION__, __LINE__); \
                                                         return false; }
#define DeleteObj(obj)  if(!!obj) delete obj;
#define DeleteObjAndFalse(obj)  { DeleteObj(obj) \
                                  ReturnFalse }
#define Deactivation(obj)     if(obj.Activation()!=None && \
                                 !DeActivation(obj.getOutput(), obj.getGradient(),\
                                               obj.getGradient(),obj.Activation())) ReturnFalse;
#define dActivation(obj)     if(obj.Activation()!=None && \
                                !Activation(obj.getOutput(), obj.getOutput(), obj.Activation())) ReturnFalse;
#define dWeightsUpdate(obj,source,tau)     if(!obj.WeightsUpdate(source.obj.AsObject(),tau)) ReturnFalse;
#define dFileWriteUInt(file_handle, value)  if(FileWriteInteger(file_handle, (int)value,INT_VALUE)<INT_VALUE) ReturnFalse;
#define dFileReadUInt(file_handle, value)   if(FileIsEnding(file_handle)) ReturnFalse; value = (uint)FileReadInteger(file_handle);
#define dFileWriteFloat(file_handle, value)  if(FileWriteFloat(file_handle, (float)value)<sizeof(float)) ReturnFalse;
#define dFileReadUFloat(file_handle, value)   if(FileIsEnding(file_handle)) ReturnFalse; value = FileReadFloat(file_handle);
//--- D2Skill shared MQL5 contract constants.  NeuroNet.cl is a separate
//--- OpenCL program and keeps matching device-side constants in its source.
#define D2SKILL_FORMAT                           8
#define D2SKILL_LAYER_FORMAT                     1
#define D2SKILL_ITEM_FORMAT                      1
#define D2SKILL_DIAGNOSTICS                      24
#define D2SKILL_DISTRIBUTION_BINS                5
#define D2SKILL_METRICS                           10
#define D2SKILL_DIAGNOSTIC_PARTIALS               22
#define D2SKILL_FREE                             0
#define D2SKILL_CANDIDATE                        1
#define D2SKILL_PROTECTED                        2
#define D2SKILL_ACTIVE                           3
#define D2SKILL_INACTIVE                         4
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T1, typename T2>
T1 pow(T2 a, T1 b)
  {
   return (T1)MathPow((T1)a, (T1)b);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
union uaLongChar
  {
   long data[3];
   uchar cdata[24];
  };
//+------------------------------------------------------------------+
///\defgroup ObjectTypes  Defines Object types identified
///Used to identify classes in a library
///@{
//+------------------------------------------------------------------+
///\defgroup arr Arrays
///Used to identify array classes
///\{
#define  defArrayConnects  0x7782   ///<Array of connections \details Identified class #CArrayCon
#define  defLayer          0x7787   ///<Layer of neurons \details Identified class #CLayer
#define  defArrayLayer     0x7788   ///<Array of layers \details Identified class #CArrayLayer
#define  defNet            0x7790   ///<Neuron Net \details Identified class #CNet
///\}
///\defgroup cpu CPU
///Used to identify classes with CPU calculation
///\{
#define  defConnect        0x7781   ///<Connection \details Identified class #CConnection
#define  defNeuronBase     0x7783   ///<Neuron base type \details Identified class #CNeuronBase
#define  defNeuron         0x7784   ///<Full connected neuron \details Identified class #CNeuron
#define  defNeuronConv     0x7785   ///<Convolution neuron \details Identified class #CNeuronConv
#define  defNeuronProof    0x7786   ///<Proof neuron \details Identified class #CNeuronProof
#define  defNeuronLSTM     0x7791   ///<LSTM Neuron \details Identified class #CNeuronLSTM
///\}
///\defgroup gpu GPU
///Used to identify classes with GPU calculation
///\{
#define  defBufferDouble               0x7882   ///<Data Buffer OpenCL \details Identified class #CBufferFloat
#define  defNeuronBaseOCL              0x7883   ///<Neuron Base OpenCL \details Identified class #CNeuronBaseOCL
#define  defNeuronConvOCL              0x7885   ///<Conolution neuron OpenCL \details Identified class #CNeuronConvOCL
#define  defNeuronProofOCL             0x7886   ///<Proof neuron OpenCL \details Identified class #CNeuronProofOCL
#define  defNeuronAttentionOCL         0x7887   ///<Attention neuron OpenCL \details Identified class #CNeuronAttentionOCL
#define  defNeuronMHAttentionOCL       0x7888   ///<Multi-Head Attention neuron OpenCL \details Identified class #CNeuronMHAttentionOCL
#define  defNeuronMLMHAttentionOCL     0x7889   ///<Multilayer multi-headed attention neuron OpenCL \details Identified class #CNeuronMLMHAttentionOCL
#define  defNeuronDropoutOCL           0x7890   ///<Dropout neuron OpenCL \details Identified class #CNeuronDropoutOCL
#define  defNeuronBatchNormOCL         0x7891   ///<Batchnorm neuron OpenCL \details Identified class #CNeuronBatchNormOCL
#define  defNeuronVAEOCL               0x7892   ///<VAE neuron OpenCL \details Identified class #CVAE
#define  defNeuronLSTMOCL              0x7893   ///<LSTM Neuron \details Identified class #CNeuronLSTMOCL
#define  defNeuronSoftMaxOCL           0x7894   ///<SoftMax layer \details Identified class #CNeuronSoftMaxOCL
#define  defNeuronFQF                  0x7895   ///<FQF layer \details Identified class #CNeuronFQF
#define  defNeuronMLMHSparseAttentionOCL        0x7896   ///<Multilayer multi-headed sparse attention neuron OpenCL \details Identified class #CNeuronMLMHAttentionOCL
#define  defNeuronMultiModels          0x7897   ///<Neuron Base MultiModels OpenCL \details Identified class #CNeuronMultiModels
#define  defNeuronConcatenate          0x7898
#define  defNeuronSoftActorCritic      0x7899
#define  defNeuronEmbeddingOCL         0x7900
#define  defNeuronPEOCL                0x7901
#define  defNeuronTransposeOCL         0x7902
#define  defNeuronMH2AttentionOCL      0x7903
#define  defNeuronCGConvOCL            0x7904
#define  defNeuronMFTOCL               0x7905
#define  defNeuronXCiTOCL              0x7906
#define  defNeuronDOTOCL               0x7907
#define  defNeuronFAQOCL               0x7908
#define  defNeuronCrossAttenOCL        0x7909
#define  defNeuronGTE                  0x7910
#define  defResidualConv               0x7911
#define  defCCMREncoder                0x7912
#define  defNeuronCrossXCiTOCL         0x7913
#define  defNeuronCCMROCL              0x7914
#define  defNeuronNODEOCL              0x7915
#define  defNeuronConformerOCL         0x7916
#define  defNeuronRevInDenormOCL       0x7917
#define  defNeuronClientOCL            0x7918
#define  defNeuronLearnabledPE         0x7919
#define  defNeuronUShapeAttention      0x7920
#define  defNeuronPatchingOCL          0x7921
#define  defNeuronTiDEOCL              0x7922
#define  defNeuronLegendreWavelets     0x7923
#define  defNeuronFEDW                 0x7924
#define  defNeuronFITSOCL              0x7925
#define  defNeuronFreDFOCL             0x7926
#define  defNeuronComplexMLMHAttentionOCL       0x7927
#define  defNeuronATFNetOCL            0x7928
#define  defNeuronS3                   0x7929
#define  defNeuronMLMHAttentionMLKV    0x7930
#define  defNeuronMLCrossAttentionMLKV          0x7931
#define  defNeuronCSCMOCL              0x7932
#define  defNeuronSPyrAttentionMLKV    0x7933
#define  defNeuronPLROCL               0x7934
#define  defNeuronTPMEncoder           0x7935
#define  defNeuronTPM                  0x7936
#define  defNeuronSTNNEncoder          0x7937
#define  defNeuronSTNNDecoder          0x7938
#define  defNeuronSparseTSF            0x7939
#define  defNeuronTEMPOOCL             0x7940
#define  defNeuronMVMHAttentionMLKV    0x7941
#define  defNeuronMVCrossMHAttentionMLKV        0x7942
#define  defNeuronInjectTST            0x7943
#define  defNeuronSSMOCL               0x7944
#define  defNeuronMambaOCL             0x7945
#define  defNeuronMambaBlockOCL        0x7946
#define  defNeuronTrajLLMOCL           0x7947
#define  defNeuronUniTrajOCL           0x7948
#define  defNeuronHiVTAAEncoder        0x7949
#define  defNeuronTransposeRCDOCL      0x7950
#define  defNeuronHiVTOCL              0x7951
#define  defNeuronPointNetOCL          0x7952
#define  defNeuronPointNet2OCL         0x7953
#define  defNeuronPointNet2LocalOCL    0x7954
#define  defNeuronPointFormer          0x7955
#define  defNeuronSceneSpecific        0x7956
#define  defNeuronMLMHSceneConditionAttention   0x7957
#define  defNeuronHyperDet             0x7958
#define  defNeuronSEFormer             0x7959
#define  defNeuronSPFormer             0x7960
#define  defNeuronMAFT                 0x7961
#define  defNeuronGRES                 0x7962
#define  defNeuronGEGWA                0x7963
#define  defNeuronLPC                  0x7964
#define  defNeuronOCM                  0x7965
#define  defNeuronRefMask              0x7966
#define  defNeuronRelativeSelfAttention         0x7967
#define  defNeuronRMAT                 0x7968
#define  defNeuronMHAttentionPooling            0x7969
#define  defNeuronMotifs               0x7970
#define  defNeuronMultiScaleAttention           0x7971
#define  defNeuronMolformer            0x7972
#define  defNeuronRelativeCrossAttention        0x7973
#define  defNeuronMotifEncoder         0x7974
#define  defNeuronPropertyAwareAttention        0x7975
#define  defNeuronAMCT                 0x7976
#define  defNeuronNAFS                 0x7977
#define  defNeuronBatchNormWithNoise            0x7978
#define  defNeuronDiffusion            0x7979
#define  defNeuronHyperProjection      0x7980
#define  defNeuronHyperboloids         0x7981
#define  defNeuronHypDiff              0x7982
#define  defNeuronBaseSAMOCL           0x7983
#define  defNeuronConvSAMOCL           0x7984
#define  defNeuronPSBlock              0x7985
#define  defNeuronPSformer             0x7986
#define  defNeuronMarketObserver       0x7987
#define  defNeuronRLAgent              0x7988
#define  defNeuronControlAgent         0x7989
#define  defNeuronMASA                 0x7990
#define  defNeuronPLRMultiAgentsOCL    0x7991
#define  defNeuronCrossSectionalAnalysis  0x7992
#define  defNeuronTemporalAnalysis     0x7993
#define  defNeuronTransposeVRCOCL      0x7994
#define  defNeuronPortfolioGenerator   0x7995
#define  defNeuronMASAAT               0x7996
#define  defNeuronMHConvOCL            0x7997
#define  defNeuronMHFeedForward        0x7998
#define  defNeuronDMHAttention         0x7999
#define  defNeuronCrossDMHAttention    0x8000
#define  defNeuronLegendreWaveletsHL   0x8001
#define  defNeuronDecouplingFlow       0x8002
#define  defNeuronDilatedCasualConv    0x8003
#define  defNeuronMultitaskStockformer 0x8004
#define  defNeuronMemory               0x8005
#define  defNeuronFinMem               0x8006
#define  defNeuronLowLevelReflection   0x8007
#define  defNeuronHighLevelReflection  0x8008
#define  defNeuronFinAgent             0x8009
#define  defNeuronMemoryDistil         0x8010
#define  defNeuronFinConAgent          0x8011
#define  defNeuronFinConManager        0x8012
#define  defNeuronMacroHFT             0x8013
#define  defNeuronMacroHFTHyperAgent   0x8014
#define  defNeuronMacroHFTvsRiskManager   0x8015
#define  defNeuronMultiScaleRelativeSelfAttention     0x8016
#define  defNeuronMultiScaleRelativeCrossAttention    0x8017
#define  defNeuronRecursiveAttention   0x8018
#define  defNeuronLinerAttention       0x8019
#define  defNeuronHidformerTSAgent     0x8020
#define  defNeuronHidformerFreqAgent   0x8021
#define  defNeuronHidformer            0x8022
#define  defNeuronResNeXtBottleneck    0x8023
#define  defNeuronResNeXtResidual      0x8024
#define  defNeuronResNeXtBlock         0x8025
#define  defNeuron2DSSMOCL             0x8026
#define  defNeuronChimera              0x8027
#define  defNeuronMoT                  0x8028
#define  defNeuronChimeraPlus          0x8029
#define  defNeuronHypridDecoder        0x8030
#define  defNeuronAttraos              0x8031
#define  defNeuronTopKGates            0x8032
#define  defNeuronMoE                  0x8033
#define  defNeuronChanelMask           0x8034
#define  defNeuronDUET                 0x8035
#define  defNeuronMultiWindowsConvOCL  0x8036
#define  defNeuronAdaBN                0x8037
#define  defNeuronComplexConvOCL       0x8038
#define  defNeuronComplexMVMHMaskAttention   0x8039
#define  defNeuronCATCH                0x8040
#define  defNeuronSkillsEncoder        0x8041
#define  defNeuronHiSSDLowLevelControler     0x8042
#define  defNeuronCGLSTMOCL            0x8043
#define  defNeuronMHProbAttention      0x8044
#define  defNeuronTSPositionEncoder    0x8045
#define  defMamba4CastEmbeding         0x8046
#define  defNeuronMultiWindowsConvWPadOCL  0x8047
#define  defNeuronConcatDiff           0x8048
#define  defNeuronMantisPatching       0x8049
#define  defNeuronMantisAttentionUnit  0x8050
#define  defNeuronTimeFoundPatching    0x8051
#define  defNeuronTimeFoundTransformerUnit   0x8052
#define  defNeuronSwiGLUOCL            0x8053
#define  defNeuronMaskMultiWinConv     0x8054
#define  defNeuronTimeMoESparseExperts 0x8055
#define  defNeuronTimeMoEAttention     0x8056
#define  defNeuronAdaptConv            0x8057
#define  defNeuronRoPE                 0x8058
#define  defParams                     0x8059
#define  defNeuronK2VAEEncoder         0x8060
#define  defNeuronInterpolationAttention     0x8061
#define  defNeuronAGCN                 0x8062
#define  defNeuronGinARCell            0x8063
#define  defNeuronGinAR                0x8064
#define  defNeuronPeriodNorm           0x8065
#define  defNeuronAdaptSpatialNorm     0x8066
#define  defNeuronSCNNEncoder          0x8067
#define  defNeuronSCNN                 0x8068
#define  defNeuronAttentNorm           0x8069
#define  defNeuronSAttentNorm          0x8070
#define  defNeuronPolynomialRegression 0x8071
#define  defNeuronSSCNNEncoder         0x8072
#define  defNeuronSSCNN                0x8073
#define  defCircleParams               0x8074
#define  defNeuronTQMHA                0x8075
#define  defChebPolinom                0x8076
#define  defNeuronHimNetGrapConv       0x8077
#define  defNeuronHimNetGCRU           0x8078
#define  defNeuronHimNetTempEncoder    0x8079
#define  defNeuronHimNetSpatEncoder    0x8080
#define  defNeuronHimNetEncoder        0x8081
#define  defNeuronHimNetDecoder        0x8082
#define  defNeuronSNSMHAttention       0x8083
#define  defNeuronFastGConv            0x8084
#define  defNeuronSAGDFN               0x8085
#define  defNeuronSpatialEmbedding     0x8086
#define  defNeuronTempEmbedding        0x8087
#define  defNeuronGlobalLocalAttention 0x8088
#define  defNeuronExtralonger          0x8089
#define  defNeuronGraphons             0x8090
#define  defNeuronGraphAttention       0x8091
#define  defNeuronSparseSoftMax        0x8092
#define  defNeuronSparseGraphAttention 0x8093
#define  defNeuronGlobLocGraphAtt      0x8094
#define  defNeuronExtralongerGraph     0x8095
#define  defNeuronSpikeConv            0x8096
#define  defNeuronGateLineAttention    0x8097
#define  defNeuronSpikeSparseAttention 0x8098
#define  defNeuronMoEConv              0x8099
#define  defNeuronSpikingBrain         0x809A
#define  defNeuronSpikeActivation      0x809B
#define  defNeuronSpikeResNeXtBottleneck  0x809C
#define  defNeuronSpikeResNeXtResidual 0x809D
#define  defNeuronSpikeResNeXtBlock    0x809E
#define  defNeuronSSAM                 0x809F
#define  defNeuronSSAMResNeXtBlock     0x80A0
#define  defNeuronSTFS                 0x80A1
#define  defNeuronSpikeConvGRU         0x80A2
#define  defNeuronAddToStack           0x80A3
#define  defNeuronStackCorrelation     0x80A4
#define  defNeuronSTEFlowNetEncoder    0x80A5
#define  defNeuronSTEFlowNetResidualBlock 0x80A6
#define  defNeuronSTEFlowNetDecoder    0x80A7
#define  defNeuronSTEFlowNet           0x80A8
#define  defNeuronMultiScaleStackCorrelation 0x80A9
#define  defNeuronSpikeConvGRU2D       0x80AA
#define  defNeuronRAFT                 0x80AB
#define  defNeuronMultScalStackCorrelBySegments 0x80AC
#define  defNeuronTMAMFE               0x80AD
#define  defNeuronTMAMPA               0x80AE
#define  defNeuronTMA                  0x80AF
#define  defNeuronSpikeSTConvGRU       0x80B0
#define  defNeuronFERENet              0x80B1
#define  defNeuronFGDModule            0x80B2
#define  defNeuronEVMGRFlowNet         0x80B3
#define  defNeuronSpikeConvBlock       0x80B4
#define  defNeuronMSRes                0x80B5
#define  defNeuronSpikeQKAttention     0x80B6
#define  defNeuronSDSA                 0x80B7
#define  defNeuronSDformerUBlock       0x80B8
#define  defNeuronSDformerFlow         0x80B9
#define  defNeuronSpikePatchStak       0x80BA
#define  defNeuronSpike2DSSMOCL        0x80BB
#define  defNeuronSpikeMamba2D         0x80BC
#define  defNeuronSpikeSTSSM           0x80BD
#define  defNeuronESTMEncoder          0x80BE
#define  defNeuronCreateFlow           0x80BF
#define  defNeuronBiDirectCorrelation  0x80C0
#define  defNeuronSATMA                0x80C1
#define  defNeuronBAT                  0x80C2
#define  defNeuronSpikeSCM             0x80C3
#define  defNeuronSpikeFAM             0x80C4
#define  defNeuronSpikeMDC             0x80C5
#define  defNeuronMDS                  0x80C6
#define  defNeuronSpikeADM             0x80C7
#define  defNeuronBATvADM              0x80C8
#define  defNeuronCDCSelfCorrector     0x80C9
#define  defNeuronSpikeCDC             0x80CA
#define  defNeuronSpikeEEMFLow         0x80CB
#define  defNeuronSpikeDepthWiseConv   0x80CC
#define  defNeuronMultiScaleExcitation 0x80CD
#define  defNeuronMultiScaleDifference 0x80CE
#define  defNeuronCorrelationEncoder   0x80CF
#define  defNeuronEDCFlow              0x80D0
#define  defNeuronCreateICEFlow        0x80D1
#define  defNeuronSpikeMFE             0x80D2
#define  defNeuronSpikeMHCrossAttention   0x80D3
#define  defNeuronSpikeMixFusion       0x80D4
#define  defNeuronSTFlow               0x80D5
#define  defNeuronPSSE                 0x80D6
#define  defNeuronSpikeDepthWiseResidual  0x80D7
#define  defNeuronSpikeSuperKernelBlock   0x80D8
#define  defNeuronSpikeGMA             0x80D9
#define  defNeuronETROF                0x80DA
#define  defNeuronPSSEFlow             0x80DB
#define  defNeuronSpikeUSR             0x80DC
#define  defNeuronInitialFlow          0x80DD
#define  defNeuronSpikeSMR             0x80DE
#define  defNeuronEVAFlow              0x80DF
#define  defNeuronResFlowHTR           0x80E0
#define  defNeuronResFlow              0x80E1
#define  defNeuronConfGateTailAwareMoE 0x80E2
#define  defNeuronResFlowLattice       0x80E3
#define  defFieldAwareParams           0x80E4
#define  defNeuronFieldAwareConv       0x80E5
#define  defNeuronMHFAT                0x80E6
#define  defNeuronMHCrossFAT           0x80E7
#define  defNeuronFieldPatternEmbedding  0x80E8
#define  defNeuronAutoToken            0x80E9
#define  defNeuronMultiMixAttention    0x80EA
#define  defNeuronSparseMoEConv        0x80EB
#define  defNeuronMTmixAttBlock        0x80EC
#define  defCrossMHFlashAttention       0x80ED
#define  defNeuronMixFeedForward       0x80EE
#define  defNeuronOneTrans             0x80EF
#define  defNeuronMHTHCrossAttention   0x80F0
#define  defNeuronSTCA                 0x80F1
#define  defNeuronWeightGenerator      0x80F2
#define  defNeuronPCGR                 0x80F3
#define  defNeuronADS                  0x80F4
#define  defNeuronPerTokenFFN          0x80F5
#define  defNeuronScenariosToken       0x80F6
#define  defNeuronUnifiedTokenizer     0x80F7
#define  defNeuronDomainAwareAttention 0x80F8
#define  defNeuronFeatureSelfIteration 0x80F9
#define  defNeuronDomainFused          0x80FA
#define  defNeuronMDL                  0x80FB
#define  defNeuronUniMixerTokenizer    0x80FC
#define  defNeuronPerTokenSwiGLU       0x80FD
#define  defNeuronUniMixer             0x80FE
#define  defNeuronSiameseNorm          0x80FF
#define  defNeuronUniMixerBlock        0x8100
#define  defNeuronINFNetScenarios      0x8101
#define  defNeuronINFNetHUBSequence    0x8102
#define  defNeuronINFNetBlock          0x8103
#define  defNeuronSequenceTokenizer    0x8104
#define  defNeuronDifferenceTokenizer  0x8105
#define  defNeuronReGENTADBackbone     0x8106
#define  defNeuronReGENTADHead         0x8107
#define  defNeuronReGENTAD             0x8108
#define  defNeuronDictionaryCrossAtt   0x8109
#define  defNeuronGDformerBackbone     0x810A
#define  defNeuronGDformer             0x810B
#define  defNeuronUncAD_MSD            0x810C
#define  defNeuronUncAD_MUE            0x810D
#define  defNeuronUncAD_UGP            0x810E
#define  defNeuronUncAD_UGPL           0x810F
#define  defNeuronUncAD                0x8110
#define  defNeuronMomADTTM             0x8111
#define  defNeuronMomADMPI             0x8112
#define  defNeuronMomAD                0x8113
#define  defNeuronCogDriverData        0x8114
#define  defNeuronCogDriverRankTCM     0x8115
#define  defNeuronCogDriverForecastHead 0x8116
#define  defNeuronCogDriverForecastToken 0x8117
#define  defNeuronScenarioForecast      0x8118
#define  defNeuronScenarioCrossAttention 0x8119
#define  defNeuronRAGMemory               0x811A
#define  defNeuronVLADriverRAGMPI          0x811B
#define  defNeuronD2SkillItem             0x811C
#define  defNeuronD2SkillBank             0x811D
#define  defNeuronD2Skill                 0x811E
///\}
///@}
//+------------------------------------------------------------------+
///\defgroup group1 Defines for OpenCL kernels identified
/// Used as indexes when calling functions OpenCL.
///@{
///\defgroup neuron_base Neuron Base
/// Describes the process for the Neuron Base.
///\details Detailed description on [the link.](https://www.mql5.com/ru/articles/8435#para4)
///@{
///\defgroup neuron_base_ff Feed Forward proccess kernel
/// Describes the forward path process for the Neuron Base.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para41">the link.</A>
///@{
#define  def_k_FeedForward       0  ///< Index of #FeedForward kernel
#define  def_k_ff_matrix_w       0  ///< Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
#define  def_k_ff_matrix_i       1  ///< Inputs tesor
#define  def_k_ff_matrix_o       2  ///< Output tensor
#define  def_k_ff_inputs         3  ///< Number of inputs
#define  def_k_ff_activation     4  ///< Activation type (#ENUM_ACTIVATION)
///@}
///\defgroup neuron_base_gr Gradients Calculation kernels
/// Describes the process of gradients calculation for the Neuron Base.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
///@{
#define  def_k_CalcOutputGradient 1  ///< Index of Output gradients calculation kernel (#CalcOutputGradient)
#define  def_k_cog_matrix_t       0  ///< Target tensor
#define  def_k_cog_matrix_o       1  ///< Output tensor
#define  def_k_cog_matrix_ig      2  ///< Tensor of gradients at previous layer
#define  def_k_cog_activation     3  ///< Activation type (#ENUM_ACTIVATION)
#define  def_k_cog_error          4  ///< Error
//---
#define  def_k_CalcHiddenGradient 2  ///< Index of Hidden gradients calculation kernel (#CalcHiddenGradient)
#define  def_k_chg_matrix_w       0  ///< Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define  def_k_chg_matrix_g       1  ///< Tensor of gradients at current layer
#define  def_k_chg_matrix_o       2  ///< Output tensor
#define  def_k_chg_matrix_ig      3  ///< Tensor of gradients at previous layer
#define  def_k_chg_outputs        4  ///< Number of outputs
#define  def_k_chg_activation     5  ///< Activation type (#ENUM_ACTIVATION)
///@}
///\defgroup neuron_base_opt Updating Weights Calculation kernel
/// Describes the process of optimization weights for the Neuron Base.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para43">the link.</A>
/// For Adam optimization look <A HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
///@{
#define  def_k_UpdateWeightsMomentum      3   ///< Index SGD optomization Update weights kernel (#UpdateWeightsMomentum)
#define  def_k_uwm_matrix_w        0  ///< SGD Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define  def_k_uwm_matrix_g        1  ///< SGD Tensor of gradients at current layer
#define  def_k_uwm_matrix_i        2  ///< SGD Inputs tesor
#define  def_k_uwm_matrix_dw       3  ///< SGD Matrix of delta weights in last correction
#define  def_k_uwm_inputs          4  ///< SGD Number of inputs
#define  def_k_uwm_learning_rates  5  ///< SGD Learning rates
#define  def_k_uwm_momentum        6  ///< SGD Momentum multiplier
//---
#define  def_k_UpdateWeightsAdam   4  ///< Index Adam optomization Update weights kernel (#UpdateWeightsAdam)
#define  def_k_uwa_matrix_w        0  ///< Adam Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define  def_k_uwa_matrix_g        1  ///< Adam Tensor of gradients at current layer
#define  def_k_uwa_matrix_i        2  ///< Adam Inputs tesor
#define  def_k_uwa_matrix_m        3  ///< Adam Matrix of first momentum
#define  def_k_uwa_matrix_v        4  ///< Adam Matrix of seconfd momentum
#define  def_k_uwa_inputs          5  ///< Adam Number of inputs
#define  def_k_uwa_l               6  ///< Adam Learning rates
#define  def_k_uwa_b1              7  ///< Adam First momentum multiplier
#define  def_k_uwa_b2              8  ///< Adam Second momentum multiplier
//---
#define  def_k_UpdateWeightsLS    28  ///< Index Least Squares optomization Update weights kernel (#UpdateWeightsLS)
#define  def_k_uwls_matrix_w       0  ///< Least Squares Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define  def_k_uwls_matrix_g       1  ///< Least Squares Tensor of gradients at current layer
#define  def_k_uwls_matrix_i       2  ///< Least Squares Inputs tesor
#define  def_k_uwls_matrix_xg      3  ///< Least Squares Matrix of summ x*g
#define  def_k_uwls_matrix_xx      4  ///< Least Squares Matrix of summ x*x
#define  def_k_uwls_inputs         5  ///< Least Squares Number of inputs
#define  def_k_uwls_l              6  ///< Least Squares Learning rates
#define  def_k_uwls_update         7  ///< Least Squares Update flag
///@}
///@}
//---
///\defgroup neuron_proof Pooling layer's neuron
/// Describes the process for the Neuron of pooling layer.
///@{
///\defgroup neuron_proof_ff Pooling layer's neuron Feed Forward
/// Describes the feed forward process for the Neuron of pooling layer.
///@{
#define  def_k_FeedForwardProof    5  ///< Index of the kernel of the Pooling neuron for Feed forward process (#FeedForwardProof)
#define  def_k_ffp_matrix_i        0  ///< Inputs tesor
#define  def_k_ffp_matrix_o        1  ///< Output tensor
#define  def_k_ffp_inputs          2  ///< Number of inputs
#define  def_k_ffp_window          3  ///< Size of input window
#define  def_k_ffp_step            4  ///< Step size
///@}
//---
///\defgroup neuron_proof_gr Pooling layer's neuron Gradients Calculation kernels
/// Describes the gradient calculation process for the Neuron of pooling layer.
///@{
#define  def_k_CalcInputGradientProof 6  ///< Index of the kernel of the Pooling neuron to transfer gradient to previous layer (#CalcInputGradientProof)
#define  def_k_cigp_matrix_i      0  ///< Inputs tesor
#define  def_k_cigp_matrix_g      1  ///< Tensor of gradients at current layer
#define  def_k_cigp_matrix_o      2  ///< Output tensor
#define  def_k_cigp_matrix_ig     3  ///< Tensor of gradients at previous layer
#define  def_k_cigp_outputs       4  ///< Number of outputs
#define  def_k_cigp_window        5  ///< Size of input window
#define  def_k_cigp_step          6  ///< Step size
///@}
///@}
//---
///\defgroup neuron_conv Convolution layer's neuron
/// Describes the process for the Neuron of convolution layer.
///@{
///\defgroup neuron_conv_ff Convolution layer's neuron Feed Forward
/// Describes the feed forward process for the Neuron of convolution layer.
///@{
#define  def_k_FeedForwardConv    7  ///< Index of the kernel of the convolution neuron for Feed forward process (#FeedForwardConv)
#define  def_k_ffc_matrix_w       0  ///< Weights matrix (m+1)*n, where m - input window and n - output window
#define  def_k_ffc_matrix_i       1  ///< Inputs tesor
#define  def_k_ffc_matrix_o       2  ///< Output tensor
#define  def_k_ffc_inputs         3  ///< Number of inputs
#define  def_k_ffc_step           4  ///< Step size
#define  def_k_ffc_window_in      5  ///< Size of input window
#define  def_k_ffс_window_out     6  ///< Size of output window
#define  def_k_ffc_activation     7  ///< Activation type (#ENUM_ACTIVATION)
///@}
//---
///\defgroup neuron_conv_gr Convolution layer's neuron Gradients Calculation kernels
/// Describes the gradient calculation process for the Neuron of convolution layer.
///@{
#define  def_k_CalcHiddenGradientConv 8  ///< Index of the kernel of the convolution neuron to transfer gradient to previous layer (#CalcHiddenGradientConv)
#define  def_k_chgc_matrix_w      0  ///< Weights matrix (m+1)*n, where m - input window and n - output window
#define  def_k_chgc_matrix_g      1  ///< Tensor of gradients at current layer
#define  def_k_chgc_matrix_o      2  ///< Output tensor
#define  def_k_chgc_matrix_ig     3  ///< Tensor of gradients at previous layer
#define  def_k_chgc_outputs       4  ///< Number of outputs
#define  def_k_chgc_step          5  ///< Step size
#define  def_k_chgc_window_in     6  ///< Size of input window
#define  def_k_chgc_window_out    7  ///< Size of output window
#define  def_k_chgc_activation    8  ///< Activation type (#ENUM_ACTIVATION)
#define  def_k_chgc_shift_out     9  ///< Activation type (#ENUM_ACTIVATION)
///@}
//---
///\defgroup neuron_conv_opt Convolution layer's neuron Update weights kernels
/// Describes the optimization process for the Neuron of convolution layer.
///@{
#define  def_k_UpdateWeightsConvMomentum      9  ///< Index of the kernel of the convolution neuron to update weights SGD (#UpdateWeightsConvMomentum)
#define  def_k_uwcm_matrix_w       0  ///< Weights matrix (m+1)*n, where m - input window and n - output window
#define  def_k_uwcm_matrix_g       1  ///< Tensor of gradients at current layer
#define  def_k_uwcm_matrix_i       2  ///< Inputs tesor
#define  def_k_uwcm_matrix_dw      3  ///< Matrix of delta weights in last correction
#define  def_k_uwcm_inputs         4  ///< Number of inputs
#define  def_k_uwcm_learning_rates 5  ///< Learning rates
#define  def_k_uwcm_momentum       6  ///< Momentum multiplier
#define  def_k_uwcm_window_in      7  ///< Size of input window
#define  def_k_uwcm_window_out     8  ///< Size of output window
#define  def_k_uwcm_step           9  ///< Step size
//---
#define  def_k_UpdateWeightsConvAdam   10  ///< Index of the kernel of the convolution neuron to update weights Adam (#UpdateWeightsConvAdam)
#define  def_k_uwca_matrix_w      0  ///< Weights matrix (m+1)*n, where m - input window and n - output window
#define  def_k_uwca_matrix_g      1  ///< Tensor of gradients at current layer
#define  def_k_uwca_matrix_i      2  ///< Inputs tesor
#define  def_k_uwca_matrix_m      3  ///< Matrix of first momentum
#define  def_k_uwca_matrix_v      4  ///< Matrix of seconfd momentum
#define  def_k_uwca_inputs        5  ///< Number of inputs
#define  def_k_uwca_l             6  ///< Learning rates
#define  def_k_uwca_b1            7  ///< First momentum multiplier
#define  def_k_uwca_b2            8  ///< Second momentum multiplier
#define  def_k_uwca_window_in     9  ///< Size of input window
#define  def_k_uwca_window_out    10  ///< Size of output window
#define  def_k_uwca_step          11  ///< Step size
//---
#define  def_k_UpdateWeightsConvLS   29  ///< Index of the kernel of the convolution neuron to update weights Least Squares(#UpdateWeightsConvLS)
#define  def_k_uwcls_matrix_w     0  ///< Weights matrix (m+1)*n, where m - input window and n - output window
#define  def_k_uwcls_matrix_g     1  ///< Tensor of gradients at current layer
#define  def_k_uwcls_matrix_i     2  ///< Inputs tesor
#define  def_k_uwcls_matrix_xg    3  ///< Matrix of first momentum
#define  def_k_uwcls_matrix_xx    4  ///< Matrix of seconfd momentum
#define  def_k_uwcls_inputs       5  ///< Number of inputs
#define  def_k_uwcls_l            6  ///< Learning rates
#define  def_k_uwcls_update       7  ///< Update flag
#define  def_k_uwcls_window_in    8  ///< Size of input window
#define  def_k_uwcls_window_out   9  ///< Size of output window
#define  def_k_uwcls_step         10  ///< Step size
///@}
///@}
//---
///\defgroup neuron_atten Attention layer's neuron
/// Describes the process for the Neuron of attention layer.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para4">the link.</A>
///@{
///\defgroup neuron_atten_ff Attention layer's neuron Feed Forward
/// Describes the feed forward process for the Neuron of attention layer.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
///@{
#define  def_k_AttentionScore     11 ///< Index of the kernel of the attention neuron to calculate score matrix (#AttentionScore)
#define  def_k_as_querys          0  ///< Matrix of Querys
#define  def_k_as_keys            1  ///< Matriz of Keys
#define  def_k_as_score           2  ///< Matrix of Scores
#define  def_k_as_dimension       3  ///< Dimension of Key
#define  def_k_as_mask            4  ///< 1 - calc only previous units, 0 - calc all
//---
#define  def_k_AttentionOut       12 ///< Index of the Attention Neuron Output calculation kernel (#AttentionOut)
#define  def_k_aout_scores        0  ///< Matrix of Scores
#define  def_k_aout_values        1  ///< Matrix of Values
#define  def_k_aout_inputs        2  ///< Inputs tesor
#define  def_k_aout_out           3  ///< Output tesor
//---
#define  def_k_MatrixSum          13 ///< Index of the kernel for calculation Sum of 2 matrix with multiplyer (#SumMatrix)
#define  def_k_sum_matrix1        0  ///< First matrix
#define  def_k_sum_matrix2        1  ///< Second matrix
#define  def_k_sum_matrix_out     2  ///< Output matrix
#define  def_k_sum_dimension      3  ///< Dimension of matrix
#define  def_k_sum_multiplyer     4  ///< Multiplyer for output
#define  def_k_sum_shift_in1      5
#define  def_k_sum_shift_in2      6
#define  def_k_sum_shift_out      7
//---
#define  def_k_Matrix5Sum          19 ///< Index of the kernel for calculation Sum of 2 matrix with multiplyer (#SumMatrix)
#define  def_k_sum5_matrix1        0  ///< First matrix
#define  def_k_sum5_matrix2        1  ///< Second matrix
#define  def_k_sum5_matrix3        2  ///< Third matrix
#define  def_k_sum5_matrix4        3  ///< Fourth matrix
#define  def_k_sum5_matrix5        4  ///< Fifth matrix
#define  def_k_sum5_matrix_out     5  ///< Output matrix
#define  def_k_sum5_dimension      6  ///< Dimension of matrix
#define  def_k_sum5_multiplyer     7  ///< Multiplyer for output
//---
#define  def_k_MHAttentionScore    20 ///< Index of the kernel of the multi-heads attention neuron to calculate score matrix (#MHAttentionScore)
#define  def_k_mhas_qkv            0  ///< Matrix of Queries, Keys, Values
#define  def_k_mhas_score          1  ///< Matrix of Scores
#define  def_k_mhas_dimension      2  ///< Dimension of Key
#define  def_k_mhas_mask           3  ///< 1 - calc only previous units, 0 - calc all
//---
#define  def_k_MHAttentionOut      21 ///< Index of the kernel of the multi-heads attention neuron to calculate multi-heads out matrix (#MHAttentionOut)
#define  def_k_mhao_score          0  ///< Matrix of Scores
#define  def_k_mhao_qkv            1  ///< Matrix of Queries, Keys, Values
#define  def_k_mhao_out            2  ///< Matrix of Outputs
#define  def_k_mhao_dimension      3  ///< Dimension of Key
//---
#define  def_k_ConcatenateMatrix  17 ///< Index of the Multi Head Attention Neuron Concatenate Output kernel (#ConcatenateBuffers)
#define  def_k_conc_input1        0  ///< Matrix of Buffer 1
#define  def_k_conc_window1       1  ///< Window of Buffer 1
#define  def_k_conc_input2        2  ///< Matrix of Buffer 2
#define  def_k_conc_window2       3  ///< Window of Buffer 2
#define  def_k_conc_input3        4  ///< Matrix of Buffer 3
#define  def_k_conc_window3       5  ///< Window of Buffer 3
#define  def_k_conc_input4        6  ///< Matrix of Buffer 4
#define  def_k_conc_window4       7  ///< Window of Buffer 4
#define  def_k_conc_out           8  ///< Output tesor
///@}
//---
///\defgroup neuron_atten_gr Attention layer's neuron Gradients Calculation
/// Describes the gradients calculation process for the Neuron of attention layer.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para44">the link.</A>
///@{
#define  def_k_AttentionGradients 14    ///< Index of the kernel for gradients calculation process (#AttentionInsideGradients)
#define  def_k_ag_querys          0     ///< Matrix of Querys
#define  def_k_ag_querys_g        1     ///< Matrix of Querys' Gradients
#define  def_k_ag_keys            2     ///< Matrix of Keys
#define  def_k_ag_keys_g          3     ///< Matrix of Keys' Gradients
#define  def_k_ag_values          4     ///< Matrix of Values
#define  def_k_ag_values_g        5     ///< Matrix of Values' Gradients
#define  def_k_ag_scores          6     ///< Matrix of Scores
#define  def_k_ag_gradient        7     ///< Matrix of Gradients from previous iteration
//---
#define  def_k_DeconcatenateMatrix 18 ///< Index of the Multi Head Attention Neuron Deconcatenate Output kernel (#DeconcatenateBuffers)
#define  def_k_dconc_output1       0  ///< Matrix of Buffer 1
#define  def_k_dconc_window1       1  ///< Window of Buffer 1
#define  def_k_dconc_output2       2  ///< Matrix of Buffer 2
#define  def_k_dconc_window2       3  ///< Window of Buffer 2
#define  def_k_dconc_output3       4  ///< Matrix of Buffer 3
#define  def_k_dconc_window3       5  ///< Window of Buffer 3
#define  def_k_dconc_output4       6  ///< Matrix of Buffer 4
#define  def_k_dconc_window4       7  ///< Window of Buffer 4
#define  def_k_dconc_inputs        8  ///< Input tesor
//---
#define def_k_MHAttentionGradients  22    ///< Index of the kernel for gradients calculation process (#AttentionInsideGradients)
#define def_k_mhag_qkv              0     ///< Matrix of Queries, Keys, Values
#define def_k_mhag_qkv_g            1     ///< Matrix of Gradients to Queries, Keys, Values
#define def_k_mhag_score            2     ///< Matrix of Scores
#define def_k_mhag_gradient         3     ///< Matrix of Gradients from previous iteration
//---
#define def_k_Dropout               23    ///< Index of the kernel for Dropout process (#Dropout)
#define def_k_dout_input            0     ///< Inputs Tensor
#define def_k_dout_map              1     ///< Map Tensor
#define def_k_dout_out              2     ///< Out Tensor
#define def_k_dout_dimension        3     ///< Dimension of Inputs
///@}
///@}
//---
///\defgroup neuron_norm Kernels of matrix normalization process
/// Describes the process of matrix normalization.
///\details Detailed description on <A HREF="https://arxiv.org/abs/1607.06450">the link.</A>
///@{
#define def_k_Normalize          15    ///< Index of the kernel for matrix normalization (#Normalize)
#define def_k_norm_buffer        0     ///< In/Out Matrix
#define def_k_norm_dimension     1     ///< Dimension of matrix
//---
#define def_k_NormalizeWeights   16    ///< Index of the kernel for weights matrix normalization (#NormalizeWeights)
//---
#define def_k_BatchFeedForward         24 ///< Index of the kernel for Batch Normalization Feed Forward process (#CNeuronBathcNormOCL)
#define def_k_bff_inputs               0  ///< Inputs data tenzor
#define def_k_bff_options              1  ///< Tenzor of variables
#define def_k_bff_output               2  ///< Tenzor of output data
#define def_k_bff_batch                3  ///< Batch size
#define def_k_bff_optimization         4  ///< Optimization type
#define def_k_bff_activation           5  ///< Activation type
#define def_k_bff_training             6  ///< Training mode (stats are immutable in inference)
//---
#define def_k_CalcHiddenGradientBatch   25 ///< Index of the Kernel of the Batch neuron to transfer gradient to previous layer (#CNeuronBathcNormOCL)
#define def_k_bchg_options             0  ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
#define def_k_bchg_matrix_g            1  ///<[in] Tensor of gradients at current layer
#define def_k_bchg_matrix_i            2  ///<[in] Tensor of previous layer output
#define def_k_bchg_matrix_ig           3  ///<[out] Tensor of gradients at previous layer
#define def_k_bchg_activation          4  ///< Activation type (#ENUM_ACTIVATION)
#define def_k_bchg_batch               5  ///< Batch size
#define def_k_bchg_optimization        6  ///< Optimization type
//---
#define def_k_UpdateBatchOptionsMomentum  26 ///< Index of the kernel for Describe the process of SGD optimization options for the Batch normalization Neuron (#CNeuronBatchNormOCL).
#define def_k_buom_options                0  ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
#define def_k_buom_matrix_g               1  ///<[in] Tensor of gradients at current layer
#define def_k_buom_learning_rates         2  ///< Learning rates
#define def_k_buom_momentum               3  ///< Momentum multiplier
//---
#define def_k_UpdateBatchOptionsAdam      27 ///< Index of the kernel for Describe the process of Adam optimization options for the Batch normalization Neuron (#CNeuronBatchNormOCL).
#define def_k_buoa_options                0  ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
#define def_k_buoa_matrix_g               1  ///<[in] Tensor of gradients at current layer
#define def_k_buoa_l                      2  ///< Learning rates
#define def_k_buoa_b1                     3  ///< First momentum multiplier
#define def_k_buoa_b2                     4  ///< Second momentum multiplier
///@}
///\defgroup VAE neuron Kernels of Variant Aoutoencodre
/// Describes the process of Variant Aoutoencodre.
///@{
#define def_k_VAEFeedForward             30
#define def_k_vaeff_inputs                0
#define def_k_vaeff_random                1
#define def_k_vaeff_outputd               2
//---
#define def_k_VAECalcHiddenGradient      31
#define def_k_vaehg_input                 0
#define def_k_vaehg_inp_grad              1
#define def_k_vaehg_random                2
#define def_k_vaehg_gradient              3
#define def_k_vaehg_kld_mult              4
///@}
///\defgroup LSTM neuron Kernels of RNN unit
/// Describes the process of RNN.
///@{
#define def_k_LSTM_FeedForward            32
#define def_k_lstmff_inputs               0
#define def_k_lstmff_inputs_size          1
#define def_k_lstmff_weights              2
#define def_k_lstmff_concatenated         3
#define def_k_lstmff_memory               4
#define def_k_lstmff_outputs              5
//---
#define def_k_LSTM_ConcatenatedGradient   33
#define def_k_lstmcg_gradient             0
#define def_k_lstmcg_concatenated_gradient 1
#define def_k_lstmcg_memory               2
#define def_k_lstmcg_concatenated         3
//---
#define def_k_LSTM_HiddenGradient         34
#define def_k_lstmhg_concatenated_gradient 0
#define def_k_lstmhg_inputs_gradient      1
#define def_k_lstmhg_weights_gradient     2
#define def_k_lstmhg_hidden_state         3
#define def_k_lstmhg_inputs               4
#define def_k_lstmhg_weeights             5
#define def_k_lstmhg_output               6
#define def_k_lstmhg_hidden_size          7
#define def_k_lstmhg_inputs_size          8
//---
#define def_k_LSTM_UpdateWeightsAdam      35
#define def_k_lstmuw_weights              0
#define def_k_lstmuw_weights_gradient     1
#define def_k_lstmuw_matrix_m             2
#define def_k_lstmuw_matrix_v             3
#define def_k_lstmuw_l                    4
#define def_k_lstmuw_b1                   5
#define def_k_lstmuw_b2                   6
///@}
///\defgroup SoftMax activation Kernels
///@{
#define def_k_SoftMax_FeedForward         36
#define def_k_softmaxff_inputs            0
#define def_k_softmaxff_outputs           1
//---
#define def_k_SoftMax_HiddenGradient      37
#define def_k_softmaxhg_outputs           0
#define def_k_softmaxhg_output_gr         1
#define def_k_softmaxhg_input_gr          2
//---
#define def_k_SoftMax_OutputGradient      38
#define def_k_softmaxog_outputs           0
#define def_k_softmaxog_targets           1
#define def_k_softmaxog_output_gr         2
///@}
///\defgroup SoftMax activation Kernels
///@{
#define def_k_FQF_Cosine                  39
#define def_k_fqf_cosine_softmax          0
#define def_k_fqf_cosine_outputs          1
//---
#define def_k_FQF_Output                  40
#define def_k_fqfout_quantiles            0
#define def_k_fqfout_delta_taus           1
#define def_k_fqfout_output               2
#define def_k_fqfout_total                3
//---
#define def_k_FQF_OutputGradient          41
#define def_k_fqfoutgr_quantiles          0
#define def_k_fqfoutgr_taus               1
#define def_k_fqfoutgr_output_gr          2
#define def_k_fqfoutgr_quantiles_gr       3
#define def_k_fqfoutgr_taus_gr            4
//---
#define def_k_FQF_QuantileGradient        42
#define def_k_fqfqgr_state_enbeding       0
#define def_k_fqfqgr_taus_embedding       1
#define def_k_fqfqgr_quantiles_gr         2
#define def_k_fqfqgr_state_gr             3
#define def_k_fqfqgr_taus_gr              4
//---
#define def_k_FQF_CosineGradient          43
#define def_k_fqfcosgr_softmax            0
#define def_k_fqfcosgr_output_gr          1
#define def_k_fqfcosgr_softmax_gr         2
//---
//---
#define def_k_MHSparseAttentionScore    44 ///< Index of the kernel of the multi-heads sparse attention neuron to calculate score matrix (#MHSparseAttentionScore)
#define def_k_mhas_sparse                3  ///< less than 1.0 сoefficient of sparse
//---
#define def_k_MHSparseAttentionOut      45 ///< Index of the kernel of the multi-heads sparse attention neuron to calculate multi-heads out matrix (#MHSparseAttentionOut)
//---
#define def_k_FFMultiModels             46 ///< Index of the kernel of the multi-models neuron to calculate feed forward
#define def_k_HGMultiModels             47 ///< Index of the kernel of the multi-models neuron to calculate hiden gradient
#define def_k_chg_model                 6  ///< Number of model to calculate
#define def_k_UWMultiModels             48 ///< Index of the kernel of the multi-models neuron to update weights
#define def_k_uwa_model                 9  ///< Number of model to update
///@}
///@}
#define def_k_ConcatFeedForward        49 ///< Index of #FeedForward kernel
#define def_k_cff_matrix_w             0  ///< Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
#define def_k_cff_matrix_i1            1  ///< Inputs tesor
#define def_k_cff_matrix_i2            2  ///< Inputs tesor
#define def_k_cff_matrix_o             3  ///< Output tensor
#define def_k_cff_inputs1              4  ///< Number of inputs
#define def_k_cff_inputs2              5  ///< Number of inputs
#define def_k_cff_activation           6  ///< Activation type (#ENUM_ACTIVATION)
//---
#define def_k_ConcatCalcHiddenGradient 50 ///< Index of Hidden gradients calculation kernel (#CalcHiddenGradient)
#define def_k_cchg_matrix_w            0  ///< Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define def_k_cchg_matrix_g            1  ///< Tensor of gradients at current layer
#define def_k_cchg_matrix_o1           2  ///< Output tensor
#define def_k_cchg_matrix_o2           3  ///< Output tensor
#define def_k_cchg_matrix_ig1          4  ///< Tensor of gradients at previous layer
#define def_k_cchg_matrix_ig2          5  ///< Tensor of gradients at previous layer
#define def_k_cchg_outputs             6  ///< Number of outputs
#define def_k_cchg_inputs1             7  ///< Number of inputs1
#define def_k_cchg_inputs2             8  ///< Number of inputs2
#define def_k_cchg_activation1         9  ///< Activation type (#ENUM_ACTIVATION)
#define def_k_cchg_activation2         10  ///< Activation type (#ENUM_ACTIVATION)
//---
#define def_k_ConcatUpdWeightsMomentum 51 ///< Index SGD optomization Update weights kernel (#UpdateWeightsMomentum)
#define def_k_cuwm_matrix_w            0  ///< SGD Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define def_k_cuwm_matrix_g            1  ///< SGD Tensor of gradients at current layer
#define def_k_cuwm_matrix_i1           2  ///< SGD Inputs tesor
#define def_k_cuwm_matrix_i2           3  ///< SGD Inputs tesor
#define def_k_cuwm_matrix_dw           4  ///< SGD Matrix of delta weights in last correction
#define def_k_cuwm_inputs1             5  ///< SGD Number of inputs
#define def_k_cuwm_inputs2             6  ///< SGD Number of inputs
#define def_k_cuwm_learning_rates      7  ///< SGD Learning rates
#define def_k_cuwm_momentum            8  ///< SGD Momentum multiplier
//---
#define def_k_ConcatUpdWeightsAdam     52 ///< Index Adam optomization Update weights kernel (#UpdateWeightsAdam)
#define def_k_cuwa_matrix_w            0  ///< Adam Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
#define def_k_cuwa_matrix_g            1  ///< Adam Tensor of gradients at current layer
#define def_k_cuwa_matrix_i1           2  ///< Adam Inputs tesor
#define def_k_cuwa_matrix_i2           3  ///< Adam Inputs tesor
#define def_k_cuwa_matrix_m            4  ///< Adam Matrix of first momentum
#define def_k_cuwa_matrix_v            5  ///< Adam Matrix of seconfd momentum
#define def_k_cuwa_inputs1             6  ///< Adam Number of inputs
#define def_k_cuwa_inputs2             7  ///< Adam Number of inputs
#define def_k_cuwa_l                   8  ///< Adam Learning rates
#define def_k_cuwa_b1                  9  ///< Adam First momentum multiplier
#define def_k_cuwa_b2                  10 ///< Adam Second momentum multiplier
//---
#define def_k_SoftUpdate               53
#define def_k_su_target                0
#define def_k_su_source                1
#define def_k_su_tau                   2
//---
#define def_k_SoftUpdateAdam           54
#define def_k_sua_target               0
#define def_k_sua_source               1
#define def_k_sua_matrix_m             2
#define def_k_sua_matrix_v             3
#define def_k_sua_tau                  4
#define def_k_sua_b1                   5
#define def_k_sua_b2                   6
//---
#define def_k_SAC_AlphaLogProbs        55
#define def_k_sac_alp_outputs          0
#define def_k_sac_alp_quantiles        1
#define def_k_sac_alp_probs            2
#define def_k_sac_alp_alphas           3
#define def_k_sac_alp_log_probs        4
#define def_k_sac_alp_random           5
#define def_k_sac_alp_count_quants     6
#define def_k_sac_alp_activation       7
//---
#define def_k_SAC_AlphaGradients       56
#define def_k_sac_alg_outputs          0
#define def_k_sac_alg_gradient         1
#define def_k_sac_alg_log_probs        2
#define def_k_sac_alg_alphas_grad      3
#define def_k_sac_alg_activation       4
//---
#define def_k_SAC_OutputGradient       57
#define def_k_sacoutgr_quantiles       0
#define def_k_sacoutgr_taus            1
#define def_k_sacoutgr_output_gr       2
#define def_k_sacoutgr_quantiles_gr    3
#define def_k_sacoutgr_taus_gr         4
#define def_k_sacoutgr_outputs         5
#define def_k_sacoutgr_count_quants    6
#define def_k_sacoutgr_activation      7
//---
#define def_k_SAC_CalcLogProbs         58
#define def_k_sacclp_outputs           0
#define def_k_sacclp_quantiles         1
#define def_k_sacclp_probs             2
#define def_k_sacclp_alphas            3
#define def_k_sacclp_log_probs         4
#define def_k_sacclp_count_quants      5
#define def_k_sacclp_activation        6
//---
#define def_k_Embedding                59
#define def_k_emb_inputs               0
#define def_k_emb_outputs              1
#define def_k_emb_weights              2
#define def_k_emb_windows              3
#define def_k_emb_std                  4
#define def_k_emb_stack_size           5
//---
#define def_k_EmbeddingHiddenGradient  60
#define def_k_ehg_inputs_gradient      0
#define def_k_ehg_outputs_gradient     1
#define def_k_ehg_weights              2
#define def_k_ehg_windows              3
#define def_k_ehg_std                  4
#define def_k_ehg_window_out           5
//---
#define def_k_EmbeddingUpdateWeightsAdam  61
#define def_k_euw_weights              0
#define def_k_euw_gradient             1
#define def_k_euw_inputs               2
#define def_k_euw_matrix_m             3
#define def_k_euw_matrix_v             4
#define def_k_euw_windows              5
#define def_k_euw_std                  6
#define def_k_euw_window_out           7
#define def_k_euw_learning_rate        8
#define def_k_euw_b1                   9
#define def_k_euw_b2                   10
//---
#define def_k_Transpose                62
#define def_k_tr_matrix_in             0
#define def_k_tr_matrix_out            1
//---
#define  def_k_MH2AttentionOut         63
#define  def_k_mh2ao_q                 0  ///< Matrix of Queries
#define  def_k_mh2ao_kv                1  ///< Matrix of Keys, Values
#define  def_k_mh2ao_score             2  ///< Matrix of Scores
#define  def_k_mh2ao_out               3  ///< Matrix of Outputs
#define  def_k_mh2ao_dimension         4  ///< Dimension of Key
#define  def_k_mh2ao_heads_kv          5  ///< Heads of Key
#define  def_k_mh2ao_mask              6  ///< mask 1 - calc only previous units, 0 - calc all
//---
#define  def_k_MH2AttentionInsideGradients   64
#define  def_k_mh2aig_q                0  ///< Matrix of Queries
#define  def_k_mh2aig_qg               1  ///< Matrix of Queries gradient
#define  def_k_mh2aig_kv               2  ///< Matrix of Keys, Values
#define  def_k_mh2aig_kvg              3  ///< Matrix of Keys, Values gradient
#define  def_k_mh2aig_score            4  ///< Matrix of Scores
#define  def_k_mh2aig_outg             5  ///< Matrix of Outputs gradient
#define  def_k_mh2aig_kunits           6  ///< Size of Key
#define  def_k_mh2aig_heads_kv         7  ///< Heads of Key
//---
#define  def_k_CGConv_HiddenGradient   65
#define  def_k_cgc_matrix_g            0  ///<[in] Tensor of gradients at current layer
#define  def_k_cgc_matrix_f            1  ///<[in] Previous layer Output tensor
#define  def_k_cgc_matrix_s            2  ///<[in] Previous layer Output tensor
#define  def_k_cgc_matrix_fg           3  ///<[out] Tensor of gradients at previous layer
#define  def_k_cgc_matrix_sg           4  ///<[out] Tensor of gradients at previous layer
#define  def_k_cgc_activationf         5  ///< Activation type (#ENUM_ACTIVATION)
#define  def_k_cgc_activations         6  ///< Activation type (#ENUM_ACTIVATION)
//---
#define  def_k_XCiTFeedForward         66
#define  def_k_XCiTff_qkv              0
#define  def_k_XCiTff_score            1
#define  def_k_XCiTff_out              2
//---
#define def_k_XCiTInsideGradients      67
#define def_k_XCiTig_qkv               0
#define def_k_XCiTig_qkv_g             1
#define def_k_XCiTig_scores            2
#define def_k_XCiTig_gradient          3
//---
#define def_k_DOTFeedForward           68
#define def_k_dot_qkv                  0
#define def_k_dot_score                1
#define def_k_dot_rpb                  2
#define def_k_dot_out                  3
//---
#define def_k_DOTInsideGradients       69
#define def_k_dotg_qkv                 0
#define def_k_dotg_qkv_g               1
#define def_k_dotg_scores              2
#define def_k_dotg_rpb                 3
#define def_k_dotg_rpb_g               4
#define def_k_dotg_gradient            5
//---
#define def_k_RPBUpdateAdam            70
#define def_k_rpbw_rpb                 0
#define def_k_rpbw_gradient            1
#define def_k_rpbw_matrix_m            2
#define def_k_rpbw_matrix_v            3
#define def_k_rpbw_b1                  4
#define def_k_rpbw_b2                  5
//---
#define def_k_GTEFeedForward           71
#define def_k_gteff_qkv                0
#define def_k_gteff_score              1
#define def_k_gteff_out                2
#define def_k_gteff_dimension          3
//---
#define def_k_GTEInsideGradients       72
#define def_k_gteig_qkv                0
#define def_k_gteig_qkv_g              1
#define def_k_gteig_scores             2
#define def_k_gteig_gradient           3
//---
#define def_k_FeedForwardNODEInpK     73
#define def_k_ffdopriInp_matrix_i      0
#define def_k_ffdopriInp_matrix_k1     1
#define def_k_ffdopriInp_matrix_k2     2
#define def_k_ffdopriInp_matrix_k3     3
#define def_k_ffdopriInp_matrix_k4     4
#define def_k_ffdopriInp_matrix_k5     5
#define def_k_ffdopriInp_matrix_k6     6
#define def_k_ffdopriInp_matrix_beta   7
#define def_k_ffdopriInp_matrix_o      8
//---
#define def_k_FeedForwardNODEF        74
#define def_k_ffdoprif_matrix_w        0
#define def_k_ffdoprif_matrix_i        1
#define def_k_ffdoprif_matrix_o        2
#define def_k_ffdoprif_dimension       3
#define def_k_ffdoprif_step            4
#define def_k_ffdoprif_activation      5
//---
#define def_k_HiddenGradientNODEInpK  75
//---
#define def_k_HiddenGradientNODEF     76
#define def_k_hddoprif_matrix_w        0
#define def_k_hddoprif_matrix_g        1
#define def_k_hddoprif_matrix_i        2
#define def_k_hddoprif_matrix_ig       3
#define def_k_hddoprif_dimension_out   4
#define def_k_hddoprif_activation      5
//---
#define def_k_NODEF_UpdateWeightsAdam 77
#define def_k_uwdoprif_matrix_w        0
#define def_k_uwdoprif_matrix_gk1      1
#define def_k_uwdoprif_matrix_gk2      2
#define def_k_uwdoprif_matrix_gk3      3
#define def_k_uwdoprif_matrix_gk4      4
#define def_k_uwdoprif_matrix_gk5      5
#define def_k_uwdoprif_matrix_gk6      6
#define def_k_uwdoprif_matrix_ik1      7
#define def_k_uwdoprif_matrix_ik2      8
#define def_k_uwdoprif_matrix_ik3      9
#define def_k_uwdoprif_matrix_ik4      10
#define def_k_uwdoprif_matrix_ik5      11
#define def_k_uwdoprif_matrix_ik6      12
#define def_k_uwdoprif_matrix_m        13
#define def_k_uwdoprif_matrix_v        14
#define def_k_uwdoprif_alpha           15
#define def_k_uwdoprif_lenth           16
#define def_k_uwdoprif_l               17
#define def_k_uwdoprif_b1              18
#define def_k_uwdoprif_b2              19
//---
#define def_k_TimeDerivative           78
#define def_k_tdqkv                    0
#define def_k_tddqkv                   1
#define def_k_tddimension              2
//---
#define def_k_HGTimeDerivative         79
//---
#define def_k_FeedForwardContAtt      80
#define def_k_caqkv                    0
#define def_k_cadqkv                   1
#define def_k_cascore                  2
#define def_k_caout                    3
#define def_k_cadimension              4
#define def_k_caheads                  5
//---
#define def_k_HiddenGradientContAtt    81
#define def_k_hgcaqkv                  0
#define def_k_hgcaqkv_g                1
#define def_k_hgcadqkv                 2
#define def_k_hgcadqkv_g               3
#define def_k_hgcascore                4
#define def_k_hgcaout_g                5
#define def_k_hgcadimension            6
//---
#define def_k_RevInFeedForward         82
#define def_k_revffinputs              0
#define def_k_revffoptions             1
#define def_k_revffoutput              2
#define def_k_revffoptions_size        3
#define def_k_revffoptimization        4
//---
#define def_k_RevInHiddenGraddient     83
#define def_k_revhginputs              0
#define def_k_revhginputs_gr           1
#define def_k_revhgoptions             2
#define def_k_revhgoutput_gr           3
#define def_k_revhgoptions_size        4
#define def_k_revhgoptimization        5
#define def_k_revhgactivation          6
//---
#define def_k_DeActivation             84
#define def_k_deact_inputs             0
#define def_k_deact_inputs_gr          1
#define def_k_deact_output_gr          2
#define def_k_deact_activation         3
//---
#define def_k_PatchCreate              85
#define def_k_ptc_inputs               0
#define def_k_ptc_weights              1
#define def_k_ptc_outputs              2
#define def_k_ptc_inputs_total         3
#define def_k_ptc_window_in            4
#define def_k_ptc_step                 5
#define def_k_ptc_activation           6
//---
#define def_k_PatchHiddenGradient      86
#define def_k_pthg_inputs              0
#define def_k_pthg_inputs_gr           1
#define def_k_pthg_weights             2
#define def_k_pthg_outputs_gr          3
#define def_k_pthg_window_in           4
#define def_k_pthg_step                5
#define def_k_pthg_window_out          6
#define def_k_pthg_outputs_total       7
#define def_k_pthg_activation          8
//---
#define def_k_PatchUpdateWeightsAdam   87
#define def_k_ptuwa_weights            0
#define def_k_ptuwa_outputs_gr         1
#define def_k_ptuwa_inputs             2
#define def_k_ptuwa_weights_m          3
#define def_k_ptuwa_weights_v          4
#define def_k_ptuwa_inputs_total       5
#define def_k_ptuwa_l                  6
#define def_k_ptuwa_b1                 7
#define def_k_ptuwa_b2                 8
#define def_k_ptuwa_step               9
//---
#define def_k_MatMult                  88
#define def_k_mm_matr1                 0
#define def_k_mm_matr2                 1
#define def_k_mm_result                2
#define def_k_mm_dimension             3
#define def_k_mm_multvarsecond         4
//---
#define def_k_FFT                      89
#define def_k_fft_inputs_re            0
#define def_k_fft_inputs_im            1
#define def_k_fft_outputs_re           2
#define def_k_fft_outputs_im           3
#define def_k_fft_input_window         4
#define def_k_fft_input_complex        5
#define def_k_fft_output_window        6
#define def_k_fft_reverse              7
//---
#define def_k_ComplexLayer             90
#define def_k_cl_inputs_re             0
#define def_k_cl_inputs_im             1
#define def_k_cl_outputs_re            2
#define def_k_cl_outputs_im            3
//---
#define def_k_ComplexLayerGradient     91
//---
#define def_k_GradientMSA              92
#define def_k_gmsa_target              0
#define def_k_gmsa_forecast            1
#define def_k_gmsa_gradient            2
//---
#define def_k_CumulativeGradient       93
#define def_k_cg_gradient1             0
#define def_k_cg_gradient2             1
#define def_k_cg_gradient_out          2
#define def_k_cg_alpha                 3
//---
#define def_k_FeedForwardComplexConv            94
#define def_k_CalcHiddenGradientComplexConv     95
#define def_k_UpdateWeightsComplexConvMomentum  96
#define def_k_UpdateWeightsComplexConvAdam      97
#define def_k_ComplexMHAttentionGradients       98
#define def_k_ComplexMHAttentionScore           99
#define def_k_ComplexMHAttentionOut             100
#define def_k_ComplexSoftMax_FeedForward        101
#define def_k_ComplexSoftMax_HiddenGradient     102
#define def_k_ComplexSoftMax_OutputGradient     103
//---
#define def_k_ComplexNormalize                  104
#define def_k_cn_inputs                         0
#define def_k_cn_outputs                        1
#define def_k_cn_means                          2
#define def_k_cn_vars                           3
#define def_k_cn_dimension                      4
//---
#define def_k_ComplexUnNormalize                105
//---
#define def_k_ComplexNormalizeGradient          106
#define def_k_cng_inputs_gr                     0
#define def_k_cng_outputs_gr                    1
#define def_k_cng_vars                          2
#define def_k_cng_dimension                     3
//---
#define def_k_ComplexUnNormalizeGradient        107
//---
#define def_k_MainFreqWeight                    108
#define def_k_mfw_freq                          0
#define def_k_mfw_weight                        1
#define def_k_mfw_dimension                     2
//---
#define def_k_WeightedSum                       109
#define def_k_ws_inputs1                        0
#define def_k_ws_inputs2                        1
#define def_k_ws_outputs                        2
#define def_k_ws_weight                         3
#define def_k_ws_dimension                      4
//---
#define def_k_WeightedSumGradient               110
//---
#define def_k_FeedForwardS3                     111
#define def_k_s3_inputs                         0
#define def_k_s3_probability                    1
#define def_k_s3_weights                        2
#define def_k_s3_outputs                        3
#define def_k_s3_positions                      4
#define def_k_s3_window                         5
#define def_k_s3_total                          6
//---
#define def_k_InsideGradientS3                  112
#define def_k_s3g_inputs                        0
#define def_k_s3g_inputs_gr                     1
#define def_k_s3g_probability                   2
#define def_k_s3g_probability_gr                3
#define def_k_s3g_weights                       4
#define def_k_s3g_outputs_gr                    5
#define def_k_s3g_positions                     6
#define def_k_s3g_window                        7
#define def_k_s3g_total                         8
//---
#define def_k_WeightGradientS3                  113
#define def_k_s3w_inputs                        0
#define def_k_s3w_positions                     1
#define def_k_s3w_outputs_gr                    2
#define def_k_s3w_weights_gr                    3
#define def_k_s3w_window                        4
#define def_k_s3w_total                         5
//---
#define def_k_MH2PyrAttentionOut                114
#define def_k_pam_q                             0
#define def_k_pam_kv                            1
#define def_k_pam_score                         2
#define def_k_pam_out                           3
#define def_k_pam_dimension                     4
#define def_k_pam_heads_kv                      5
#define def_k_pam_window                        6
//---
#define def_k_PLR                               115
#define def_k_plr_inputs                        0
#define def_k_plr_outputs                       1
#define def_k_plt_isttp                         2
#define def_k_plr_transpose                     3
#define def_k_plr_step                          4
//---
#define def_k_PLRGrad                           116
#define def_k_plrg_inputs_gr                    0
#define def_k_plrg_outputs                      1
#define def_k_plrg_outputs_gr                   2
#define def_k_plrg_transpose                    3
//---
#define def_k_UpdateWeightsAdamMini             117
#define def_k_wuam_matrix_w                     0
#define def_k_wuam_matrix_g                     1
#define def_k_wuam_matrix_i                     2
#define def_k_wuam_matrix_m                     3
#define def_k_wuam_matrix_v                     4
#define def_k_wuam_l                            5
#define def_k_wuam_b1                           6
#define def_k_wuam_b2                           7
//---
#define def_k_UpdateWeightsConvAdamMini         118
#define def_k_wucam_matrix_w                    0
#define def_k_wucam_matrix_g                    1
#define def_k_wucam_matrix_i                    2
#define def_k_wucam_matrix_m                    3
#define def_k_wucam_matrix_v                    4
#define def_k_wucam_inputs                      5
#define def_k_wucam_l                           6
#define def_k_wucam_b1                          7
#define def_k_wucam_b2                          8
#define def_k_wucam_step                        9
//---
#define def_k_CutTrendAndOther                  119
#define def_k_ct_inputs                         0
#define def_k_ct_plr                            1
#define def_k_ct_trend                          2
#define def_k_ct_other                          3
//---
#define def_k_CutTrendAndOtherGradient          120
#define def_k_ctg_inputs_gr                     0
#define def_k_ctg_plr                           1
#define def_k_ctg_plr_gr                        2
#define def_k_ctg_trend_gr                      3
#define def_k_ctg_other_gr                      4
//---
#define def_k_CutOneFromAnother                 121
#define def_k_ctofa_inputs                      0
#define def_k_ctofa_cut                         1
#define def_k_ctofa_other                       2
//---
#define def_k_CutOneFromAnotherGradient         122
//---
#define def_k_UniTrajPrepare                    123
#define def_k_utp_history                       0
#define def_k_utp_h_mask                        1
#define def_k_utp_future                        2
#define def_k_utp_f_mask                        3
#define def_k_utp_output                        4
#define def_k_utp_h_total                       5
#define def_k_utp_f_total                       6
//---
#define def_k_UniTrajPrepareGrad                124
#define def_k_utpg_history_gr                   0
#define def_k_utpg_future_gr                    1
#define def_k_utpg_output                       2
#define def_k_utpg_output_gr                    3
#define def_k_utpg_h_total                      4
#define def_k_utpg_f_total                      5
//---
#define def_k_UniTrajBTS                        125
#define def_k_utbts_concat_inp                  0
#define def_k_utbts_d_forw                      1
#define def_k_utbts_d_bakw                      2
#define def_k_utbts_total                       3
//---
#define def_k_GateElementMul                    126
#define def_k_gem_inputs1                       0
#define def_k_gem_inputs2                       1
#define def_k_gem_gate                          2
#define def_k_gem_out                           3
//---
#define def_k_GateElementMulGrad                127
#define def_k_gemg_inputs1                      0
#define def_k_gemg_inputs1_gr                   1
#define def_k_gemg_inputs2                      2
#define def_k_gemg_inputs2_gr                   3
#define def_k_gemg_gate                         4
#define def_k_gemg_gate_gr                      5
#define def_k_gemg_out                          6
#define def_k_gemg_activ1                       7
#define def_k_gemg_activ2                       8
#define def_k_gemg_activ_gr                     9
//---
#define def_k_HiVTPrepare                       128
#define def_k_hivtp_data                        0
#define def_k_hivtp_output                      1
//---
#define def_k_TransposeRCD                      129
//---
#define def_k_MatMultGrad                       130
#define def_k_mmg_matr1                         0
#define def_k_mmg_matr1_gr                      1
#define def_k_mmg_matr2                         2
#define def_k_mmg_matr2_gr                      3
#define def_k_mmg_result_gr                     4
#define def_k_mmg_dimension                     5
#define def_k_mmg_multvarsecond                 6
//---
#define def_k_OrthoganalLoss                    131
#define def_k_ol_data                           0
#define def_k_ol_grad                           1
#define def_k_ol_add                            2
//---
#define def_k_CalcDistance                      132
#define def_k_cd_data                           0
#define def_k_cd_distance                       1
#define def_k_cd_dimension                      2
//---
#define def_k_FeedForwardLocalMax               133
#define def_k_fflm_matrix_i                     0
#define def_k_fflm_distance                     1
#define def_k_fflm_matrix_o                     2
#define def_k_fflm_radius                       3
//---
#define def_k_CalcInputGradientLocalMax         134
#define def_k_cglm_matrix_i                     0
#define def_k_cglm_matrix_ig                    1
#define def_k_cglm_distance                     2
#define def_k_cglm_matrix_o                     3
#define def_k_cglm_matrix_g                     4
#define def_k_cglm_radius                       5
//---
#define def_k_MHMaskAttentionOut                135
#define def_k_mask_at_q                         0
#define def_k_mask_at_kv                        1
#define def_k_mask_at_score                     2
#define def_k_mask_at_mask                      3
#define def_k_mask_at_out                       4
#define def_k_mask_at_dimension                 5
#define def_k_mask_at_heads_kv                  6
#define def_k_mask_at_mask_level                7
//---
#define def_k_MHMaskAttentionInsideGradients    136
#define def_k_mask_atg_q                        0
#define def_k_mask_atg_q_g                      1
#define def_k_mask_atg_kv                       2
#define def_k_mask_atg_kv_g                     3
#define def_k_mask_atg_mask                     4
#define def_k_mask_atg_mask_g                   5
#define def_k_mask_atg_scores                   6
#define def_k_mask_atg_gradient                 7
#define def_k_mask_atg_kunits                   8
#define def_k_mask_atg_heads_kv                 9
#define def_k_mask_atg_mask_level               10
//---
#define def_k_CalcPositionBias                  137
#define def_k_cpb_data1                         0
#define def_k_cpb_data2                         1
#define def_k_cpb_result                        2
#define def_k_cpb_dimension                     3
//---
#define def_k_MHPosBiasAttentionOut             138
#define def_k_pbao_q                            0
#define def_k_pbao_k                            1
#define def_k_pbao_v                            2
#define def_k_pbao_score                        3
#define def_k_pbao_pos_bias                     4
#define def_k_pbao_out                          5
#define def_k_pbao_dimension                    6
#define def_k_pbao_heads_kv                     7
#define def_k_pbao_use_pos_bias                 8
//---
#define def_k_MHPosBiasAttentionInsideGradients 139
#define def_k_pbaog_q                           0
#define def_k_pbaog_q_g                         1
#define def_k_pbaog_k                           2
#define def_k_pbaog_k_g                         3
#define def_k_pbaog_v                           4
#define def_k_pbaog_v_g                         5
#define def_k_pbaog_scores                      6
#define def_k_pbaog_gradient                    7
#define def_k_pbaog_kunits                      8
#define def_k_pbaog_heads_kv                    9
//---
#define def_k_DiversityLoss                     140
#define def_k_dl_data                           0
#define def_k_dl_grad                           1
#define def_k_dl_dimension                      2
#define def_k_dl_activation                     3
#define def_k_dl_add                            4
//---
#define def_k_MHRelativeAttentionOut            141
#define def_k_rat_q                             0
#define def_k_rat_k                             1
#define def_k_rat_v                             2
#define def_k_rat_bk                            3
#define def_k_rat_bv                            4
#define def_k_rat_gc                            5
#define def_k_rat_gp                            6
#define def_k_rat_score                         7
#define def_k_rat_out                           8
#define def_k_rat_dimension                     9
//---
#define def_k_MHRelativeAttentionInsideGradients   142
#define def_k_ratg_q                            0
#define def_k_ratg_q_g                          1
#define def_k_ratg_k                            2
#define def_k_ratg_k_g                          3
#define def_k_ratg_v                            4
#define def_k_ratg_v_g                          5
#define def_k_ratg_bk                           6
#define def_k_ratg_bk_g                         7
#define def_k_ratg_bv                           8
#define def_k_ratg_bv_g                         9
#define def_k_ratg_gc                           10
#define def_k_ratg_gc_g                         11
#define def_k_ratg_gp                           12
#define def_k_ratg_gp_g                         13
#define def_k_ratg_scores                       14
#define def_k_ratg_gradient                     15
#define def_k_ratg_kunits                       16
//---
#define def_k_CalcAlignmentGradient             143
#define def_k_aliggr_matrix_o1                  0
#define def_k_aliggr_matrix_o2                  1
#define def_k_aliggr_matrix_g1                  2
#define def_k_aliggr_matrix_g2                  3
#define def_k_aliggr_activation                 4
#define def_k_aliggr_add                        5
//---
#define def_k_FeatureSmoothing                  144
#define def_k_fs_feature                        0
#define def_k_fs_outputs                        1
#define def_k_fs_smoothing                      2
//---
#define def_k_FeatureSmoothingGradient          145
//---
#define def_k_BatchFeedForwardAddNoise          146
#define def_k_normwithnoise_inputs              0
#define def_k_normwithnoise_options             1
#define def_k_normwithnoise_noise               2
#define def_k_normwithnoise_output              3
#define def_k_normwithnoise_batch               4
#define def_k_normwithnoise_optimization        5
#define def_k_normwithnoise_activation          6
#define def_k_normwithnoise_alpha               7
//---
#define def_k_HyperProjection                   147
#define def_k_lp_inputs                         0
#define def_k_lp_outputs                        1
//---
#define def_k_HyperProjectionGrad               148
#define def_k_lpg_inputs                        0
#define def_k_lpg_inputs_gr                     1
#define def_k_lpg_outputs_gr                    2
//---
#define def_k_LogMap                            149
#define def_k_logmap_features                   0
#define def_k_logmap_centroids                  1
#define def_k_logmap_curvatures                 2
#define def_k_logmap_outputs                    3
#define def_k_logmap_product                    4
#define def_k_logmap_distance                   5
#define def_k_logmap_norma                      6
//---
#define def_k_LogMapGrad                        150
#define def_k_logmapgr_features                 0
#define def_k_logmapgr_features_gr              1
#define def_k_logmapgr_centroids                2
#define def_k_logmapgr_centroids_gr             3
#define def_k_logmapgr_curvatures               4
#define def_k_logmapgr_curvatures_gr            5
#define def_k_logmapgr_outputs                  6
#define def_k_logmapgr_outputs_gr               7
#define def_k_logmapgr_product                  8
#define def_k_logmapgr_distance                 9
#define def_k_logmapgr_norma                    10
//---
#define def_k_CalcEpsilonWeights                151
#define def_k_epsw_matrix_w                     0
#define def_k_epsw_matrix_g                     1
#define def_k_epsw_matrix_i                     2
#define def_k_epsw_matrix_epsw                  3
#define def_k_epsw_rho                          4
//---
#define def_k_CalcEpsilonWeightsConv            152
#define def_k_epswconv_matrix_w                 0
#define def_k_epswconv_matrix_g                 1
#define def_k_epswconv_matrix_i                 2
#define def_k_epswconv_matrix_epsw              3
#define def_k_epswconv_inputs                   4
#define def_k_epswconv_rho                      5
#define def_k_epswconv_step                     6
//---
#define def_k_PLRMultiAgents                    153
//---
#define def_k_PLRMultiAgentsGrad                154
#define def_k_plrg_agents                       4
//---
#define  def_k_FeedForwardMHConv                155
//---
#define  def_k_CalcHiddenGradientMHConv         156
#define  def_k_chgc_heads                       10
//---
#define  def_k_UpdateWeightsMHConvAdam          157
#define  def_k_uwca_heads                       12
//---
#define def_k_MoreLessEqual                     158
#define def_k_mle_inputs                        0
#define def_k_mle_outputs                       1
//---
#define  def_k_MultiScaleRelativeAttentionOut   159
//---
#define def_k_SSM2D_FeedForward                 160
#define def_k_ssm2d_ah                          0
#define def_k_ssm2d_b_time                      1
#define def_k_ssm2d_b_var                       2
#define def_k_ssm2d_px_time                     3
#define def_k_ssm2d_px_var                      4
#define def_k_ssm2d_c_time                      5
#define def_k_ssm2d_c_var                       6
#define def_k_ssm2d_delta_time                  7
#define def_k_ssm2d_delta_var                   8
#define def_k_ssm2d_hidden                      9
#define def_k_ssm2d_y                           10
//---
#define def_k_SSM2D_CalcHiddenGradient          161
#define def_k_ssm2dhg_ah                        0
#define def_k_ssm2dhg_grad_ah                   1
#define def_k_ssm2dhg_b_time                    2
#define def_k_ssm2dhg_grad_b_time               3
#define def_k_ssm2dhg_b_var                     4
#define def_k_ssm2dhg_grad_b_var                5
#define def_k_ssm2dhg_px_time                   6
#define def_k_ssm2dhg_grad_px_time              7
#define def_k_ssm2dhg_px_var                    8
#define def_k_ssm2dhg_grad_px_var               9
#define def_k_ssm2dhg_c_time                    10
#define def_k_ssm2dhg_grad_c_time               11
#define def_k_ssm2dhg_c_var                     12
#define def_k_ssm2dhg_grad_c_var                13
#define def_k_ssm2dhg_delta_time                14
#define def_k_ssm2dhg_grad_delta_time           15
#define def_k_ssm2dhg_delta_var                 16
#define def_k_ssm2dhg_grad_delta_var            17
#define def_k_ssm2dhg_hidden                    18
#define def_k_ssm2dhg_grad_y                    19
//---
#define def_k_PScan                             162
#define def_k_psc_A                             0
#define def_k_psc_X                             1
#define def_k_psc_H                             2
#define def_k_psc_X_out                         3
//---
#define def_k_PScan_CalcHiddenGradient          163
#define def_k_pscgr_A                           0
#define def_k_pscgr_X                           1
#define def_k_pscgr_H                           2
#define def_k_pscgr_grad_X_out                  3
#define def_k_pscgr_grad_A                      4
#define def_k_pscgr_grad_X                      5
#define def_k_pscgr_grad_H                      6
//---
#define def_k_DiagMatMult                       164
#define def_k_diagmm_diag                       0
#define def_k_diagmm_matr                       1
#define def_k_diagmm_result                     2
#define def_k_diagmm_activation                 3
//---
#define def_k_DiagMatMultGrad                   165
#define def_k_diagmmgr_diag                     0
#define def_k_diagmmgr_grad_diag                1
#define def_k_diagmmgr_matr                     2
#define def_k_diagmmgr_grad_matr                3
#define def_k_diagmmgr_grad_result              4
//---
#define def_k_TopKgates                         166
#define def_k_topK_inputs                       0
#define def_k_topK_noises                       1
#define def_k_topK_gates                        2
#define def_k_topK_k                            3
//---
#define def_k_TopKgatesGrad                     167
#define def_k_topKgr_inputs                     0
#define def_k_topKgr_grad_inputs                1
#define def_k_topKgr_noises                     2
#define def_k_topKgr_gates                      3
#define def_k_topKgr_grad_gates                 4
//---
#define def_k_MaskByDistance                    168
#define def_k_maskDist_buf_real                 0
#define def_k_maskDist_buf_imag                 1
#define def_k_maskDist_mask                     2
#define def_k_maskDist_dimension                3
//---
#define def_k_MaskAttention                     169
#define def_k_maskatt_q                         0
#define def_k_maskatt_kv                        1
#define def_k_maskatt_scores                    2
#define def_k_maskatt_masks                     3
#define def_k_maskatt_out                       4
#define def_k_maskatt_dimension                 5
#define def_k_maskatt_heads_kv                  6
//---
#define def_k_MaskAttentionGradients            170
#define def_k_maskattgr_q                       0
#define def_k_maskattgr_q_g                     1
#define def_k_maskattgr_kv                      2
#define def_k_maskattgr_kv_g                    3
#define def_k_maskattgr_scores                  4
#define def_k_maskattgr_gradient                5
#define def_k_maskattgr_kunits                  6
#define def_k_maskattgr_heads_kv                7
//---
#define def_k_FeedForwardMultWinConv            171
#define def_k_ffmwc_matrix_w                    0
#define def_k_ffmwc_matrix_i                    1
#define def_k_ffmwc_matrix_o                    2
#define def_k_ffmwc_windows_in                  3
#define def_k_ffmwc_inputs                      4
#define def_k_ffmwc_windows_total               5
#define def_k_ffmwc_window_out                  6
#define def_k_ffmwc_activation                  7
//---
#define def_k_CalcHiddenGradientMultWinConv     172
#define def_k_chgmwc_matrix_w                   0
#define def_k_chgmwc_matrix_i                   1
#define def_k_chgmwc_matrix_ig                  2
#define def_k_chgmwc_matrix_og                  3
#define def_k_chgmwc_windows_in                 4
#define def_k_chgmwc_outputs                    5
#define def_k_chgmwc_windows_total              6
#define def_k_chgmwc_window_out                 7
#define def_k_chgmwc_activation                 8
//---
#define def_k_UpdateWeightsMultWinConvAdam      173
#define def_k_uwmwc_matrix_w                    0
#define def_k_uwmwc_matrix_og                   1
#define def_k_uwmwc_matrix_i                    2
#define def_k_uwmwc_matrix_m                    3
#define def_k_uwmwc_matrix_v                    4
#define def_k_uwmwc_windows_in                  5
#define def_k_uwmwc_windows_total               6
#define def_k_uwmwc_window_out                  7
#define def_k_uwmwc_inputs                      8
#define def_k_uwmwc_outputs                     9
#define def_k_uwmwc_l                           10
#define def_k_uwmwc_b1                          11
#define def_k_uwmwc_b2                          12
//---
#define def_k_MaskAttentionComplex              174
#define def_k_maskattcom_q                      0
#define def_k_maskattcom_kv                     1
#define def_k_maskattcom_scores                 2
#define def_k_maskattcom_masks                  3
#define def_k_maskattcom_out                    4
#define def_k_maskattcom_dimension              5
#define def_k_maskattcom_heads_kv               6
//---
#define def_k_MaskAttentionGradientsComplex     175
#define def_k_maskattcomgr_q                    0
#define def_k_maskattcomgr_q_g                  1
#define def_k_maskattcomgr_kv                   2
#define def_k_maskattcomgr_kv_g                 3
#define def_k_maskattcomgr_scores               4
#define def_k_maskattcomgr_mask                 5
#define def_k_maskattcomgr_mask_g               6
#define def_k_maskattcomgr_gradient             7
#define def_k_maskattcomgr_kunits               8
#define def_k_maskattcomgr_heads_kv             9
//---
#define def_k_CSLSTM_FeedForward                176
#define def_k_cslstmff_concatenated             0
#define def_k_cslstmff_memory                   1
#define def_k_cslstmff_output                   2
//---
#define def_k_CSLSTM_CalcHiddenGradient         177
#define def_k_cslstmhg_concatenated             0
#define def_k_cslstmhg_concatenated_grad        1
#define def_k_cslstmhg_memory                   2
#define def_k_cslstmhg_output_grad              3
//---
#define def_k_ProbAttentionQeuryImp             178
#define def_k_probat_querys                     0
#define def_k_probat_keys_values                1
#define def_k_probat_index_keys                 2
#define def_k_probat_querys_imp                 3
#define def_k_probat_dimension                  4
//---
#define def_k_TopKImportanceToIndex              179
#define def_k_imptoind_importance                0
#define def_k_imptoind_indexes                  1
#define def_k_imptoind_tok_k                    2
//---
#define def_k_QIndexAttention                   180
#define def_k_indatt_q                          0
#define def_k_indatt_kv                         1
#define def_k_indatt_scores                     2
#define def_k_indatt_indexes                    3
#define def_k_indatt_out                        4
#define def_k_indatt_dimension                  5
#define def_k_indatt_heads_kv                   6
//---
#define def_k_QIndexAttentionGradients          181
#define def_k_indattgr_q                        0
#define def_k_indattgr_q_g                      1
#define def_k_indattgr_kv                       2
#define def_k_indattgr_kv_g                     3
#define def_k_indattgr_indexes                  4
#define def_k_indattgr_scores                   5
#define def_k_indattgr_gradient                 6
#define def_k_indattgr_kunits                   7
#define def_k_indattgr_heads_kv                 8
//---
#define def_k_TSPositonEncoder                  182
#define def_k_tspe_data                         0
#define def_k_tspe_time                         1
#define def_k_tspe_output                       2
#define def_k_tspe_period                       3
//---
#define def_k_FeedForwardMultWinConvWPad        183
#define def_k_ffmwconvwpad_matrix_w             0
#define def_k_ffmwconvwpad_matrix_i             1
#define def_k_ffmwconvwpad_matrix_o             2
#define def_k_ffmwconvwpad_windows_in           3
#define def_k_ffmwconvwpad_inputs               4
#define def_k_ffmwconvwpad_step                 5
#define def_k_ffmwconvwpad_window_out           6
#define def_k_ffmwconvwpad_activation           7
//---
#define def_k_CalcHiddenGradientMultWinConvWPad 184
#define def_k_hgmwconvwpad_matrix_w             0
#define def_k_hgmwconvwpad_matrix_i             1
#define def_k_hgmwconvwpad_matrix_ig            2
#define def_k_hgmwconvwpad_matrix_og            3
#define def_k_hgmwconvwpad_windows_in           4
#define def_k_hgmwconvwpad_outputs              5
#define def_k_hgmwconvwpad_step                 6
#define def_k_hgmwconvwpad_window_out           7
#define def_k_hgmwconvwpad_filters              8
#define def_k_hgmwconvwpad_activation           9
//---
#define def_k_UpdateWeightsMultWinConvAdamWPad  185
#define def_k_uwmwconvwpad_matrix_w             0
#define def_k_uwmwconvwpad_matrix_og            1
#define def_k_uwmwconvwpad_matrix_i             2
#define def_k_uwmwconvwpad_matrix_m             3
#define def_k_uwmwconvwpad_matrix_v             4
#define def_k_uwmwconvwpad_windows_in           5
#define def_k_uwmwconvwpad_windows_total        6
#define def_k_uwmwconvwpad_window_out           7
#define def_k_uwmwconvwpad_inputs               8
#define def_k_uwmwconvwpad_step                 9
#define def_k_uwmwconvwpad_outputs              10
#define def_k_uwmwconvwpad_l                    11
#define def_k_uwmwconvwpad_b1                   12
#define def_k_uwmwconvwpad_b2                   13
//---
#define def_k_ConcatDiff                        186
#define def_k_concdiff_data                     0
#define def_k_concdiff_output                   1
#define def_k_concdiff_step                     2
//---
#define def_k_FeedForwardMaskMultWinConv        187
#define def_k_ffmmwc_matrix_w                   0
#define def_k_ffmmwc_matrix_i                   1
#define def_k_ffmmwc_masks                      2
#define def_k_ffmmwc_matrix_o                   3
#define def_k_ffmmwc_inputs                     4
#define def_k_ffmmwc_window_in                  5
#define def_k_ffmmwc_windows_total              6
#define def_k_ffmmwc_activation                 7
//---
#define def_k_CalcHiddenGradientMaskMultWinConv 188
#define def_k_chgmmwc_matrix_w                  0
#define def_k_chgmmwc_matrix_i                  1
#define def_k_chgmmwc_matrix_ig                 2
#define def_k_chgmmwc_matrix_og                 3
#define def_k_chgmmwc_masks                     4
#define def_k_chgmmwc_masks_g                   5
#define def_k_chgmmwc_outputs                   6
#define def_k_chgmmwc_window_in                 7
#define def_k_chgmmwc_window_out                8
#define def_k_chgmmwc_activation                9
//---
#define def_k_UpdateWeightsMaskMultWinConvAdam  189
#define def_k_uwmmwc_matrix_w                   0
#define def_k_uwmmwc_matrix_og                  1
#define def_k_uwmmwc_matrix_i                   2
#define def_k_uwmmwc_masks                      3
#define def_k_uwmmwc_matrix_m                   4
#define def_k_uwmmwc_matrix_v                   5
#define def_k_uwmmwc_windows_total              6
#define def_k_uwmmwc_inputs                     7
#define def_k_uwmmwc_outputs                    8
#define def_k_uwmmwc_l                          9
#define def_k_uwmmwc_b1                         10
#define def_k_uwmmwc_b2                         11
//---
#define def_k_MainFreq                          190
#define def_k_mf_freq_r                         0
#define def_k_mf_freq_im                        1
#define def_k_mf_main_freq                      2
#define def_k_mf_dimension                      3
//---
#define def_k_FeedForwardAdaptConv              191
#define def_k_ffac_matrix_w                     0
#define def_k_ffac_matrix_i                     1
#define def_k_ffac_matrix_o                     2
#define def_k_ffac_main_freq                    3
#define def_k_ffac_inputs                       4
#define def_k_ffac_window_in                    5
#define def_k_ffac_activation                   6
//---
#define def_k_CalcHiddenGradientAdaptConv       192
#define def_k_chgac_matrix_w                    0
#define def_k_chgac_matrix_i                    1
#define def_k_chgac_matrix_ig                   2
#define def_k_chgac_matrix_og                   3
#define def_k_chgac_main_freq                   4
#define def_k_chgac_outputs                     5
#define def_k_chgac_window_in                   6
#define def_k_chgac_window_out                  7
#define def_k_chgac_activation                  8
//---
#define def_k_UpdateWeightsAdaptConvAdam        193
#define def_k_uwac_matrix_w                     0
#define def_k_uwac_matrix_og                    1
#define def_k_uwac_matrix_i                     2
#define def_k_uwac_matrix_m                     3
#define def_k_uwac_matrix_v                     4
#define def_k_uwac_main_freq                    5
#define def_k_uwac_inputs                       6
#define def_k_uwac_outputs                      7
#define def_k_uwac_l                            8
#define def_k_uwac_b1                           9
#define def_k_uwac_b2                           10
//---
#define def_k_RoPE                              194
#define def_k_rope_inputs                       0
#define def_k_rope_position_emb                 1
#define def_k_rope_outputs                      2
//---
#define def_k_CalcHiddenGradRoPE                195
//---
#define  def_k_MatrixDif                        196
#define  def_k_dif_matrix1                      0
#define  def_k_dif_matrix2                      1
#define  def_k_dif_matrix_out                   2
#define  def_k_dif_multiplyer                   3
#define  def_k_dif_shift_in1                    4
#define  def_k_dif_shift_in2                    5
#define  def_k_dif_shift_out                    6
//---
#define  def_k_MatrixDifGrad                    197
//---
#define  def_k_IdentMatrixDif                   198
#define  def_k_iddif_matrix_in                  0
#define  def_k_iddif_matrix_out                 1
#define  def_k_iddif_multiplyer                 2
#define  def_k_iddif_shift_in                   3
#define  def_k_iddif_shift_out                  4
//---
#define  def_k_IdentMatrixDifGrad               199
//---
#define  def_k_SumVecMatrix                     200
#define  def_k_svecmat_vector_in                0
#define  def_k_svecmat_matrix_in                1
#define  def_k_svecmat_matrix_out               2
#define  def_k_svecmat_multiplyer               3
#define  def_k_svecmat_shift_in1                4
#define  def_k_svecmat_shift_in2                5
#define  def_k_svecmat_shift_out                6
//---
#define  def_k_SumVecMatrixGrad                 201
//---
#define  def_k_InterpolationAttention           202
#define  def_k_ia_matrix_in                     0
#define  def_k_ia_W                             1
#define  def_k_ia_A                             2
#define  def_k_ia_GL                            3
#define  def_k_ia_Adj                           4
#define  def_k_ia_H                             5
#define  def_k_ia_Atten                         6
#define  def_k_ia_matrix_out                    7
#define  def_k_ia_dimension                     8
//---
#define  def_k_InterpolationAttentionGrad       203
#define  def_k_iag_matrix_in                    0
#define  def_k_iag_matrix_in_gr                 1
#define  def_k_iag_W                            2
#define  def_k_iag_W_gr                         3
#define  def_k_iag_A                            4
#define  def_k_iag_A_gr                         5
#define  def_k_iag_GL                           6
#define  def_k_iag_GL_gr                        7
#define  def_k_iag_Adj                          8
#define  def_k_iag_H                            9
#define  def_k_iag_H_gr                         10
#define  def_k_iag_Atten                        11
#define  def_k_iag_matrix_out_gr                12
#define  def_k_iag_dimension                    13
//---
#define  def_k_IdentMatrixSum                   204
#define  def_k_idsum_matrix_in                  0
#define  def_k_idsum_matrix_out                 1
#define  def_k_idsum_multiplyer                 2
#define  def_k_idsum_shift_in                   3
#define  def_k_idsum_shift_out                  4
//---
#define  def_k_Activation                       205
#define  def_k_act_inputs                       0
#define  def_k_act_outputs                      1
#define  def_k_act_activation                   2
//---
#define  def_k_PeriodNorm                       206
#define  def_k_pn_inputs                        0
#define  def_k_pn_mean_stdevs                   1
#define  def_k_pn_outputs                       2
#define  def_k_pn_total_inputs                  3
//---
#define  def_k_PeriodNormGrad                   207
#define  def_k_png_inputs                       0
#define  def_k_png_inputs_gr                    1
#define  def_k_png_mean_stdevs                  2
#define  def_k_png_mean_stdevs_gr               3
#define  def_k_png_outputs                      4
#define  def_k_png_outputs_gr                   5
#define  def_k_png_total_inputs                 6
//---
#define  def_k_AdaptSpatialNorm                 208
#define  def_k_asn_inputs                       0
#define  def_k_asn_attention                    1
#define  def_k_asn_mean_stdevs                  2
#define  def_k_asn_outputs                      3
//---
#define  def_k_AdaptSpatialNormGrad             209
#define  def_k_asng_inputs                      0
#define  def_k_asng_inputs_gr                   1
#define  def_k_asng_attention                   2
#define  def_k_asng_attention_gr                3
#define  def_k_asng_mean_stdevs                 4
#define  def_k_asng_mean_stdevs_gr              5
#define  def_k_asng_outputs_gr                  6
#define  def_k_asng_total_inputs                7
//---
#define  def_k_AttentNorm                       210
#define  def_k_atn_inputs                       0
#define  def_k_atn_attention                    1
#define  def_k_atn_means                        2
#define  def_k_atn_stdevs                       3
#define  def_k_atn_outputs                      4
#define  def_k_atn_total_inputs                 5
#define  def_k_atn_segment_size                 6
//---
#define  def_k_AttentNormGrad                   211
#define  def_k_atng_inputs                      0
#define  def_k_atng_inputs_gr                   1
#define  def_k_atng_attention                   2
#define  def_k_atng_attention_gr                3
#define  def_k_atng_means                       4
#define  def_k_atng_stdevs                      5
#define  def_k_atng_means_gr                    6
#define  def_k_atng_outputs_gr                  7
#define  def_k_atng_total_inputs                8
#define  def_k_atng_segment_size                9
//---
#define  def_k_ChebStep                         212
#define  def_k_cheb_support                     0
#define  def_k_cheb_outputs                     1
#define  def_k_cheb_step                        2
//---
#define  def_k_ChebStepGrad                     213
#define  def_k_chebgr_support                   0
#define  def_k_chebgr_support_g                 1
#define  def_k_chebgr_outputs                   2
#define  def_k_chebgr_outputs_g                 3
#define  def_k_chebgr_step                      4
//---
#define  def_k_SignificantNeighborsSampling     214
#define  def_k_sns_data                         0
#define  def_k_sns_candidates                   1
#define  def_k_sns_random_cands                 2
#define  def_k_sns_neighbors                    3
#define  def_k_sns_dimension                    4
//---
#define  def_k_SparseMHScores                   215
#define  def_k_smhs_data                        0
#define  def_k_smhs_indexes                     1
#define  def_k_smhs_scores                      2
#define  def_k_smhs_sparse                      3
//---
#define  def_k_SparseMHScoresGrad               216
#define  def_k_smhs_gr_data_gr                  0
#define  def_k_smhs_gr_indexes                  1
#define  def_k_smhs_gr_scores                   2
#define  def_k_smhs_gr_scores_gr                3
//---
#define  def_k_SparseMatMult                    217
#define  def_k_smm_sparse_index                 0
#define  def_k_smm_sparse_data                  1
#define  def_k_smm_full                         2
#define  def_k_smm_result                       3
#define  def_k_smm_full_rows                    4
//---
#define  def_k_SparseMatMultGrad                218
#define  def_k_smm_gr_sparse_index              0
#define  def_k_smm_gr_sparse_data               1
#define  def_k_smm_gr_sparse_gr                 2
#define  def_k_smm_gr_full                      3
#define  def_k_smm_gr_full_gr                   4
#define  def_k_smm_gr_result_gr                 5
#define  def_k_smm_gr_sparse_rows               6
#define  def_k_smm_gr_sparse_cols               7
#define  def_k_smm_gr_full_rows                 8
#define  def_k_smm_gr_full_cols                 9
//---
#define  def_k_RandomWalk                       219
#define  def_k_rw_data                          0
#define  def_k_rw_inv_diag                      1
#define  def_k_rw_norm                          2
#define  def_k_rw_total_cols                    3
//---
#define def_k_ConcatByLabel                     220
#define def_k_cbl_data                          0
#define def_k_cbl_label                         1
#define def_k_cbl_embedding1                    2
#define def_k_cbl_embedding2                    3
#define def_k_cbl_output                        4
#define def_k_cbl_dimension_data                5
#define def_k_cbl_dimension_emb1                6
#define def_k_cbl_dimension_emb2                7
#define def_k_cbl_frame1                        8
#define def_k_cbl_frame2                        9
#define def_k_cbl_period1                       10
#define def_k_cbl_period2                       11
//---
#define def_k_ConcatByLabelGrad                 221
#define def_k_cbl_gr_units                      12
//---
#define def_k_GlobalLocalAttention              222
#define def_k_glatt_q                           0
#define def_k_glatt_kv                          1
#define def_k_glatt_scores                      2
#define def_k_glatt_mask                        3
#define def_k_glatt_label                       4
#define def_k_glatt_out                         5
#define def_k_glatt_dimension                   6
#define def_k_glatt_total_kv                    7
#define def_k_glatt_total_mask                  8
//---
#define def_k_GlobalLocalAttentionGrad          223
#define def_k_glatt_gr_q                        0
#define def_k_glatt_gr_q_gr                     1
#define def_k_glatt_gr_kv                       2
#define def_k_glatt_gr_kv_gr                    3
#define def_k_glatt_gr_scores                   4
#define def_k_glatt_gr_mask                     5
#define def_k_glatt_gr_mask_gr                  6
#define def_k_glatt_gr_label                    7
#define def_k_glatt_gr_out_gr                   8
#define def_k_glatt_gr_dimension                9
#define def_k_glatt_gr_total_q                  10
#define def_k_glatt_gr_total_kv                 11
#define def_k_glatt_gr_total_mask               12
//---
#define def_k_SparseSoftMax                     224
#define def_k_ssoftmax_data                     0
#define def_k_ssoftmax_outputs                  1
#define def_k_ssoftmax_indexes                  2
#define def_k_ssoftmax_out_dimension            3
//---
#define def_k_SparseSoftMaxGrad                 225
#define def_k_ssoftmax_gr_data_gr               0
#define def_k_ssoftmax_gr_outputs               1
#define def_k_ssoftmax_gr_outputs_gr            2
#define def_k_ssoftmax_gr_indexes               3
#define def_k_ssoftmax_gr_out_dimension         4
//---
#define def_k_FloatToSpike                      226
#define def_k_fts_values                        0
#define def_k_fts_levels                        1
#define def_k_fts_outputs                       2
//---
#define def_k_FloatToSpikeGrad                  227
#define def_k_ftsg_values                       0
#define def_k_ftsg_values_gr                    1
#define def_k_ftsg_levels_gr                    2
#define def_k_ftsg_gradients                    3
//---
#define def_k_SpikeMHAttention                  228
#define def_k_smhat_qkv                         0
#define def_k_smhat_diag_bias                   1
#define def_k_smhat_scores                      2
#define def_k_smhat_out                         3
#define def_k_smhat_dimension                   4
#define def_k_smhat_mask_future                 5
//---
#define def_k_SpikeMHAttentionGrad              229
#define def_k_smhat_gr_qkv                      0
#define def_k_smhat_gr_qkv_gr                   1
#define def_k_smhat_gr_diag_bias                2
#define def_k_smhat_gr_diag_bias_gr             3
#define def_k_smhat_gr_scores                   4
#define def_k_smhat_gr_gradients                5
#define def_k_smhat_gr_dimension                6
#define def_k_smhat_gr_mask_future              7
//---
#define def_k_STFS                              230
#define def_k_stfs_inputs                       0
#define def_k_stfs_mask_time                    1
#define def_k_stfs_mask_spatial                 2
#define def_k_stfs_outputs                      3
//---
#define def_k_STFSGrad                          231
//---
#define def_k_AddToStack                        232
#define def_k_ats_inputs                        0
#define def_k_ats_stack                         1
#define def_k_ats_stack_size                    2
//---
#define def_k_AggregationByTime                 233
#define def_k_agt_inputs                        0
#define def_k_agt_stack                         1
#define def_k_agt_outputs                       2
#define def_k_agt_stack_size                    3
#define def_k_agt_levels                        4
//---
#define def_k_AggregationByTimeGrad             234
#define def_k_agtg_inputs_gr                    0
#define def_k_agtg_outputs_gr                   1
#define def_k_agtg_levels                       2
//---
#define def_k_GRU                               235
#define def_k_gru_XH                            0
#define def_k_gru_prev_state                    1
#define def_k_gru_outputs                       2
//---
#define def_k_GRU_Grad                          236
#define def_k_gru_gr_XH                         0
#define def_k_gru_gr_XH_gr                      1
#define def_k_gru_gr_prev_state                 2
#define def_k_gru_gr_outputs_gr                 3
//---
#define def_k_ScalarToVector                    237
#define def_k_stv_scalar                        0
#define def_k_stv_vector_in                     1
#define def_k_stv_vector_out                    2
//---
#define def_k_ScalarToVectorGrad                238
#define def_k_stvg_scalar                       0
#define def_k_stvg_scalar_gr                    1
#define def_k_stvg_vector_in                    2
#define def_k_stvg_vector_in_gr                 3
#define def_k_stvg_vector_out_gr                4
#define def_k_stvg_dimension                    5
//---
#define def_k_CalcFlow                          239
#define def_k_cf_value                          0
#define def_k_cf_prev_value                     1
#define def_k_cf_flow                           2
//---
#define def_k_DilatedCorrelation                240
#define def_k_dcor_feature                      0
#define def_k_dcor_shifts                       1
#define def_k_dcor_correlations                 2
#define def_k_dcor_dimension                    3
//---
#define def_k_DilatedCorrelationGrad            241
#define def_k_dcorgr_feature                    0
#define def_k_dcorgr_feature_gr                 1
#define def_k_dcorgr_shifts                     2
#define def_k_dcorgr_corr_gr                    3
#define def_k_dcorgr_total_corr                 4
//---
#define def_k_DilatedDifference                 242
#define def_k_ddiff_feature                     0
#define def_k_ddiff_shifts                      1
#define def_k_ddiff_differences                 2
//---
#define def_k_DilatedDifferenceGrad             243
#define def_k_ddiffgr_feature                   0
#define def_k_ddiffgr_feature_gr                1
#define def_k_ddiffgr_shifts                    2
#define def_k_ddiffgr_differences_gr            3
#define def_k_ddiffgr_total_shifts              4
//---
#define def_k_PerturbedMatrix                   244
#define def_k_pm_inputs                         0
#define def_k_pm_perturb                        1
#define def_k_pm_output                         2
#define def_k_pm_perturb_mult                   3
//---
#define def_k_PerturbedMatrixGrad               245
//---
#define def_k_LinearUpsample                    246
#define def_k_lu_data                           0
#define def_k_lu_upsample                       1
//---
#define def_k_LinearUpsampleGrad                247
#define def_k_lug_data_gr                       0
#define def_k_lug_upsample_gr                   1
#define def_k_lug_dimension_htr                 2
//---
#define def_k_MixExpertsPredict                 248
#define def_k_moepred_experts                   0
#define def_k_moepred_outputs                   1
//---
#define def_k_MixExpertsPredictGrad             249
#define def_k_moepredgr_experts                 0
#define def_k_moepredgr_experts_gr              1
#define def_k_moepredgr_outputs_gr              2
//---
#define def_k_MHFAT                             250
#define def_k_mhfat_q                           0
#define def_k_mhfat_kv                          1
#define def_k_mhfat_scale                       2
#define def_k_mhfat_scores                      3
#define def_k_mhfat_out                         4
#define def_k_mhfat_dimension                   5
#define def_k_mhfat_mask_future                 6
//---
#define def_k_MHFATGrad                         251
#define def_k_mhfat_gr_q                        0
#define def_k_mhfat_gr_q_gr                     1
#define def_k_mhfat_gr_kv                       2
#define def_k_mhfat_gr_kv_gr                    3
#define def_k_mhfat_gr_scale                    4
#define def_k_mhfat_gr_scale_gr                 5
#define def_k_mhfat_gr_scores                   6
#define def_k_mhfat_gr_gradients                7
#define def_k_mhfat_gr_dimension                8
#define def_k_mhfat_gr_total_k                  9
#define def_k_mhfat_gr_mask_future              10
//---
#define def_k_SparseConcatenate                 252
#define def_k_sparconc_sparse_index             0
#define def_k_sparconc_sparse_data              1
#define def_k_sparconc_full                     2
#define def_k_sparconc_result                   3
#define def_k_sparconc_full_rows                4
//---
#define def_k_SparseConcatenateGrad             253
#define def_k_sparconc_gr_sparse_index          0
#define def_k_sparconc_gr_sparse_data           1
#define def_k_sparconc_gr_sparse_gr             2
#define def_k_sparconc_gr_full                  3
#define def_k_sparconc_gr_full_gr               4
#define def_k_sparconc_gr_result_gr             5
#define def_k_sparconc_gr_sparse_rows           6
#define def_k_sparconc_gr_sparse_cols           7
#define def_k_sparconc_gr_full_rows             8
#define def_k_sparconc_gr_full_cols             9
//---
#define def_k_MHFlashAttention                  254
#define def_k_mhflat_q                          0
#define def_k_mhflat_kv                         1
#define def_k_mhflat_logsumexp                  2
#define def_k_mhflat_out                        3
#define def_k_mhflat_dimension                  4
#define def_k_mhflat_total_kv                   5
#define def_k_mhflat_mask_future                6
//---
#define def_k_MHFlashAttentionGrad              255
#define def_k_mhflatg_q                         0
#define def_k_mhflatg_q_gr                      1
#define def_k_mhflatg_kv                        2
#define def_k_mhflatg_kv_gr                     3
#define def_k_mhflatg_logsumexp                 4
#define def_k_mhflatg_out                       5
#define def_k_mhflatg_out_gr                    6
#define def_k_mhflatg_dimension                 7
#define def_k_mhflatg_total_q                   8
#define def_k_mhflatg_total_kv                  9
#define def_k_mhflatg_mask_future               10
//---
#define def_k_MHFlashSTCA                       256
#define def_k_mhstca_query                      0
#define def_k_mhstca_X                          1
#define def_k_mhstca_logsumexp                  2
#define def_k_mhstca_output                     3
#define def_k_mhstca_dimension                  4
#define def_k_mhstca_total_X                    5
#define def_k_mhstca_mask_future                6
//---
#define def_k_MHFlashSTCAGrad                   257
#define def_k_mhstca_gr_query                   0
#define def_k_mhstca_gr_query_gr                1
#define def_k_mhstca_gr_X                       2
#define def_k_mhstca_gr_X_gr                    3
#define def_k_mhstca_gr_logsumexp               4
#define def_k_mhstca_gr_output                  5
#define def_k_mhstca_gr_output_gr               6
#define def_k_mhstca_gr_dimension               7
#define def_k_mhstca_gr_total_q                 8
#define def_k_mhstca_gr_total_X                 9
#define def_k_mhstca_gr_mask_future             10
//---
#define def_k_ConcatVecMatrix                   258
#define def_k_concvm_vector_in                  0
#define def_k_concvm_matrix_in                  1
#define def_k_concvm_matrix_out                 2
#define def_k_concvm_dimension_v                3
#define def_k_concvm_colums_m                   4
#define def_k_concvm_multvarsecond              5
//---
#define def_k_ConcatVecMatrixGrad               259
#define def_k_concvm_gr_vector_in_gr            0
#define def_k_concvm_gr_matrix_in_gr            1
#define def_k_concvm_gr_matrix_out_gr           2
#define def_k_concvm_gr_dimension_v             3
#define def_k_concvm_gr_colums_m                4
#define def_k_concvm_gr_rows                    5
#define def_k_concvm_gr_variables               6
#define def_k_concvm_gr_multvarsecond           7
//---
#define def_k_INFNetBGU                         260
#define def_k_infnbgu_scal_and_shift            0
#define def_k_infnbgu_sequence                  1
#define def_k_infnbgu_context                   2
#define def_k_infnbgu_scenarios                 3
#define def_k_infnbgu_sequence_out              4
#define def_k_infnbgu_context_out               5
#define def_k_infnbgu_scenarios_out             6
#define def_k_infnbgu_sequence_units            7
#define def_k_infnbgu_sequence_vars             8
#define def_k_infnbgu_context_units             9
#define def_k_infnbgu_scenarios_units           10
//---
#define def_k_INFNetBGUGrad                     261
#define def_k_infnbgugr_scal_and_shift          0
#define def_k_infnbgugr_scal_and_shift_grad     1
#define def_k_infnbgugr_sequence                2
#define def_k_infnbgugr_context                 3
#define def_k_infnbgugr_scenarios               4
#define def_k_infnbgugr_sequence_grad           5
#define def_k_infnbgugr_context_grad            6
#define def_k_infnbgugr_scenarios_grad          7
#define def_k_infnbgugr_sequence_out_grad       8
#define def_k_infnbgugr_context_out_grad        9
#define def_k_infnbgugr_scenarios_out_grad      10
#define def_k_infnbgugr_sequence_units          11
#define def_k_infnbgugr_sequence_vars           12
#define def_k_infnbgugr_context_units           13
#define def_k_infnbgugr_scenarios_units         14
//---
#define def_k_MHCrossAttvsSim                   262
#define def_k_casim_query                       0
#define def_k_casim_key                         1
#define def_k_casim_value                       2
#define def_k_casim_prototype                   3
#define def_k_casim_levels                      4
#define def_k_casim_logsumexp                   5
#define def_k_casim_output_at                   6
#define def_k_casim_similarity                  7
#define def_k_casim_dimension                   8
#define def_k_casim_total_X                     9
//---
#define def_k_MHCrossAttvsSimGrad               263
#define def_k_casimgr_query                     0
#define def_k_casimgr_key                       1
#define def_k_casimgr_value                     2
#define def_k_casimgr_prototype                 3
#define def_k_casimgr_levels                    4
#define def_k_casimgr_logsumexp                 5
#define def_k_casimgr_output_at                 6
#define def_k_casimgr_similarity                7
#define def_k_casimgr_grad_output_at            8
#define def_k_casimgr_grad_similarity           9
#define def_k_casimgr_grad_query                10
#define def_k_casimgr_grad_key                  11
#define def_k_casimgr_grad_value                12
#define def_k_casimgr_grad_prototype            13
#define def_k_casimgr_grad_levels               14
#define def_k_casimgr_dimension                 15
#define def_k_casimgr_total_X                   16
//---
#define def_k_MarketStateDensity                264
#define def_k_msd_sequence                      0
#define def_k_msd_distribution                  1
#define def_k_msd_sequence_size                 2
#define def_k_msd_quantiles                     3
#define def_k_msd_quant_step                    4
//---
#define def_k_NLLGrad                           265
#define def_k_nllgr_log_b                       0
#define def_k_nllgr_grad_in                     1
#define def_k_nllgr_grad_out                    2
#define def_k_nllgr_grad_log_b                  3
//---
#define def_k_SparseSoftmaxHausdorff            266
#define def_k_ssh_plan                          0
#define def_k_ssh_decis                         1
#define def_k_ssh_output                        2
#define def_k_ssh_top_index                     3
#define def_k_ssh_dimension                     4
#define def_k_ssh_top_k                         5
//---
#define def_k_SparseSoftmaxHausdorffGrad        267
#define def_k_sshg_plan                         0
#define def_k_sshg_grad_plan                    1
#define def_k_sshg_decis                        2
#define def_k_sshg_grad_decis                   3
#define def_k_sshg_grad_out                     4
#define def_k_sshg_top_index                    5
#define def_k_sshg_dimension                    6
#define def_k_sshg_count_plan                   7
#define def_k_sshg_count_decis                  8
#define def_k_sshg_top_k                        9
//---
#define def_k_MomADTTMMeanScores                268
#define def_k_madttm_ms_distance                0
#define def_k_madttm_ms_output                  1
#define def_k_madttm_ms_decisions               2
//---
#define def_k_MomADTTMMeanScoresGrad            269
#define def_k_madttm_msg_scores_gr              0
#define def_k_madttm_msg_distance_gr            1
#define def_k_madttm_msg_decisions              2
//---
#define def_k_CogDriverAffineStack              270
#define def_k_cdas_stack                        0
#define def_k_cdas_snapshot                     1
#define def_k_cdas_affine                       2
#define def_k_cdas_stack_size                   3
#define def_k_cdas_quant_dimension              4
//---
#define def_k_CogDriverAffineStackGrad          271
#define def_k_cdasg_snapshot                    0
#define def_k_cdasg_stack_gr                    1
#define def_k_cdasg_affine_gr                   2
#define def_k_cdasg_stack_size                  3
#define def_k_cdasg_quant_dimension             4
//---
#define def_k_CogDriverRankTCMInsertPending     272
#define def_k_cdrtcm_insert_memory              0
#define def_k_cdrtcm_insert_age                 1
#define def_k_cdrtcm_insert_score               2
#define def_k_cdrtcm_insert_valid               3
#define def_k_cdrtcm_insert_pending_query       4
#define def_k_cdrtcm_insert_memory_len          5
#define def_k_cdrtcm_insert_query_dim           6
#define def_k_cdrtcm_insert_variables           7
//---
#define def_k_CogDriverRankTCMBuildKV           273
#define def_k_cdrtcm_kv_memory                  0
#define def_k_cdrtcm_kv_age                     1
#define def_k_cdrtcm_kv_valid                   2
#define def_k_cdrtcm_kv_out                     3
#define def_k_cdrtcm_kv_memory_count            4
#define def_k_cdrtcm_kv_variables               5
#define def_k_cdrtcm_kv_heads                   6
#define def_k_cdrtcm_kv_head_dim                7
//---
#define def_k_CogDriverRankTCMMeanFlashScore    274
#define def_k_cdrtcm_score_query                0
#define def_k_cdrtcm_score_kv                   1
#define def_k_cdrtcm_score_valid                2
#define def_k_cdrtcm_score_out                  3
#define def_k_cdrtcm_score_logsumexp            4
#define def_k_cdrtcm_score_memory_count         5
#define def_k_cdrtcm_score_variables            6
#define def_k_cdrtcm_score_heads                7
#define def_k_cdrtcm_score_head_dim             8
//---
#define def_k_CogDriverRankTCMSelfKV            275
#define def_k_cdrtcm_selfkv_source              0
#define def_k_cdrtcm_selfkv_out                 1
#define def_k_cdrtcm_selfkv_query_dim           2
#define def_k_cdrtcm_selfkv_variables           3
#define def_k_cdrtcm_selfkv_heads               4
#define def_k_cdrtcm_selfkv_head_dim            5
//---
#define def_k_CogDriverForecastShiftState       276
#define def_k_cdfh_shift_plan                   0
#define def_k_cdfh_shift_confidence             1
#define def_k_cdfh_shift_shifted_plan           2
#define def_k_cdfh_shift_shifted_confidence     3
#define def_k_cdfh_shift_delta_mask             4
#define def_k_cdfh_shift_query_dim              5
#define def_k_cdfh_shift_horizon                6
//---
#define def_k_CogDriverForecastConfidence       277
#define def_k_cdfh_conf_expected_error          0
#define def_k_cdfh_conf_confidence              1
#define def_k_cdfh_conf_total                   2
//---
#define def_k_CogDriverForecastDeltaNorm        278
#define def_k_cdfh_delta_plan                   0
#define def_k_cdfh_delta_shifted_plan           1
#define def_k_cdfh_delta_plan_delta             2
#define def_k_cdfh_delta_plan_delta_norm        3
#define def_k_cdfh_delta_query_dim              4
#define def_k_cdfh_delta_horizon                5
//---
#define def_k_CogDriverForecastBuildKV          279
#define def_k_cdfh_kv_context                   0
#define def_k_cdfh_kv_out                       1
#define def_k_cdfh_kv_dimension                 2
#define def_k_cdfh_kv_horizon                   3
#define def_k_cdfh_kv_variables                 4
#define def_k_cdfh_kv_heads                     5
#define def_k_cdfh_kv_head_dim                  6
//---
#define def_k_CogDriverForecastBuildKVGrad      280
#define def_k_cdfh_kvg_kv_gr                    0
#define def_k_cdfh_kvg_context_gr               1
#define def_k_cdfh_kvg_dimension                2
#define def_k_cdfh_kvg_horizon                  3
#define def_k_cdfh_kvg_variables                4
#define def_k_cdfh_kvg_heads                    5
#define def_k_cdfh_kvg_head_dim                 6
//---
#define def_k_CogDriverForecastGradientRMS      281
#define def_k_cdfh_grms_gradient                0
#define def_k_cdfh_grms_error_target            1
#define def_k_cdfh_grms_dimension               2
#define def_k_cdfh_grms_horizon                 3
//---
#define def_k_ScenarioForecastForward           282
#define def_k_sff_input                           0
#define def_k_sff_prototypes                      1
#define def_k_sff_generator_input                 2
#define def_k_sff_router_input                    3
#define def_k_sff_confidence_input                4
#define def_k_sff_generator_output                5
#define def_k_sff_router_output                   6
#define def_k_sff_confidence_output               7
#define def_k_sff_z                               8
#define def_k_sff_u                               9
#define def_k_sff_pi                             10
#define def_k_sff_scenarios                      11
#define def_k_sff_horizon                        12
#define def_k_sff_dimension                      13
#define def_k_sff_phase                          14
#define def_k_sff_log_pi                         15
#define def_k_sff_log_u                          16
#define def_k_sff_variables                      17
#define def_k_sff_forward_validity               18
#define def_k_sff_generator_weights              19
#define def_k_sff_router_weights                 20
#define def_k_sff_confidence_weights             21
#define def_k_sff_generator_weights_count        22
#define def_k_sff_router_weights_count           23
#define def_k_sff_confidence_weights_count       24
//---
#define def_k_ScenarioResponsibilities          283
#define def_k_sr_target                           0
#define def_k_sr_z                                1
#define def_k_sr_u                                2
#define def_k_sr_pi                               3
#define def_k_sr_responsibilities                 4
#define def_k_sr_generator_gradient               5
#define def_k_sr_router_gradient                  6
#define def_k_sr_confidence_gradient              7
#define def_k_sr_scenarios                        8
#define def_k_sr_topk                             9
#define def_k_sr_horizon                         10
#define def_k_sr_dimension                       11
#define def_k_sr_scores                          12
#define def_k_sr_sparse                          13
#define def_k_sr_indexes                         14
#define def_k_sr_phase                           15
#define def_k_sr_variables                       16
#define def_k_sr_prototypes                      17
#define def_k_sr_batch_control                   18
#define def_k_sr_huber_delta                     19
#define def_k_sr_target_source                   20
#define def_k_sr_forward_validity                21
#define def_k_sr_diagnostic_partials             22
#define def_k_sr_epoch_diagnostics               23
#define def_k_sr_observation_by_scenario         24
#define def_k_sr_diagnostic_scratch              25
#define def_k_sr_usage                           26
#define def_k_sr_inactive                        27
#define def_k_sr_recovery_events                 28
#define def_k_sr_age_zero_events                 29
//---
#define def_k_ScenarioCodebookEMA               284
#define def_k_sce_target                          0
#define def_k_sce_prototypes                      1
#define def_k_sce_ema_sums                        2
#define def_k_sce_ema_counts                      3
#define def_k_sce_usage                           4
#define def_k_sce_inactive                        5
#define def_k_sce_batch_control                   6
#define def_k_sce_scenarios                       7
#define def_k_sce_dimension                       8
#define def_k_sce_decay                           9
#define def_k_sce_inactivity_age                 10
#define def_k_sce_seed                            11
#define def_k_sce_recovery_age                    12
#define def_k_sce_recovery                        13
#define def_k_sce_phase                           14
#define def_k_sce_horizon                         15
#define def_k_sce_variables                       16
#define def_k_sce_responsibilities                17
#define def_k_sce_recovery_events                 18
#define def_k_sce_age_zero_events                 19
//---
#define def_k_ScenarioGradientNormalize         285
#define def_k_sgn_generator_gradient              0
#define def_k_sgn_generator_input_gradient        1
#define def_k_sgn_router_input_gradient           2
#define def_k_sgn_confidence_input_gradient       3
#define def_k_sgn_input_gradient                  4
#define def_k_sgn_responsibilities                5
#define def_k_sgn_scenarios                       6
#define def_k_sgn_horizon                         7
//--- Generic RAG retrieval pipeline. All dimensions are runtime arguments.
#define RAG_TOPK_LOCAL_WIDTH                      64
#define RAG_TOPK_MAX                              (RAG_TOPK_LOCAL_WIDTH / 2)
// ScenarioID is transported in float2.x by the GPU TopK pipeline.
#define RAG_FLOAT_INDEX_LIMIT                     16777216
#define def_k_RAGScore                           287
#define def_k_rags_embedding                       0
#define def_k_rags_centroids                       1
#define def_k_rags_used                            2
#define def_k_rags_scores                          3
#define def_k_rags_scenario_count                  4
#define def_k_rags_embedding_dimension             5
#define def_k_RAGPartial                         288
#define def_k_ragp_scores                          0
#define def_k_ragp_partial                         1
#define def_k_ragp_scenario_count                  2
#define def_k_ragp_block_size                      3
#define def_k_ragp_top_k                           4
#define def_k_RAGMerge                           289
#define def_k_ragm_source                          0
#define def_k_ragm_destination                     1
#define def_k_ragm_candidate_count                 2
#define def_k_ragm_block_size                      3
#define def_k_ragm_top_k                           4
#define def_k_RAGGather                          290
#define def_k_ragg_top                             0
#define def_k_ragg_actions                         1
#define def_k_ragg_tokens                          2
#define def_k_ragg_scenario_count                  3
#define def_k_ragg_action_count                    4
#define def_k_ragg_action_dimension                5
#define def_k_ragg_token_width                     6
#define def_k_ragg_top_k                           7
#define def_k_ragg_action_used                     8
#define def_k_ragg_action_mean_reward              9
#define def_k_ragg_action_hit_count               10
#define def_k_ragg_absolute_relevance             11
#define def_k_RAGNearestPair                      291
#define def_k_ragn_centroids                       0
#define def_k_ragn_used                            1
#define def_k_ragn_result                          2
#define def_k_ragn_candidate_count                 3
#define def_k_ragn_dimension                       4
#define def_k_ragn_batch_count                     5
#define def_k_RAGTokenGate                        292
#define def_k_ragtg_tokens                         0
#define def_k_ragtg_gated_tokens                   1
#define def_k_ragtg_absolute_relevance             2
#define def_k_ragtg_threshold_weights              3
#define def_k_ragtg_branch_activity                4
#define def_k_ragtg_token_count                    5
#define def_k_ragtg_token_width                    6
#define def_k_ragtg_scale                          7
#define def_k_RAGTokenGateGrad                    293
#define def_k_ragtgg_tokens                        0
#define def_k_ragtgg_gated_gradient                1
#define def_k_ragtgg_absolute_relevance            2
#define def_k_ragtgg_threshold_weights             3
#define def_k_ragtgg_threshold_gradient            4
#define def_k_ragtgg_branch_activity_gradient      5
#define def_k_ragtgg_token_count                   6
#define def_k_ragtgg_token_width                   7
#define def_k_ragtgg_scale                         8
#define def_k_sgn_dimension                       8
#define def_k_sgn_phase                           9
#define def_k_sgn_confidence_gradient            10
#define def_k_sgn_variables                      11
#define def_k_sgn_batch_control                  12
//--- D2Skill GPU retrieval, update and hindsight utility kernels.
#define def_k_D2SkillScoreSlots                  294
#define def_k_d2ss_query                           0
#define def_k_d2ss_keys                            1
#define def_k_d2ss_partials                        2
#define def_k_d2ss_slots                           3
#define def_k_d2ss_dimension                       4
#define def_k_D2SkillUtility                      296
#define def_k_d2ut_influence                       0
#define def_k_d2ut_influence_total                 1
#define def_k_d2ut_utility                         2
#define def_k_d2ut_used                            3
#define def_k_d2ut_state                           4
#define def_k_d2ut_delta                           5
#define def_k_d2ut_beta                            6
#define def_k_d2ut_slots                           7
#define def_k_d2ut_diagnostics                     8
#define def_k_d2ut_applied                         9
#define def_k_D2SkillItemPartial                 286
#define def_k_d2ip_source                          0
#define def_k_d2ip_partials                        1
#define def_k_d2ip_dimension                       2
#define def_k_d2ip_output_offset                   3
#define def_k_D2SkillItemReduce                  297
#define def_k_d2ir_partials                        0
#define def_k_d2ir_reduced                         1
#define def_k_d2ir_input_count                     2
#define def_k_d2ir_input_offset                    3
#define def_k_d2ir_output_offset                   4
#define def_k_d2ir_final_output                    5
#define def_k_D2SkillItemApply                   298
#define def_k_d2ia_gradient                        0
#define def_k_d2ia_correction                      1
#define def_k_d2ia_direction                       2
#define def_k_d2ia_scale                           3
#define def_k_d2ia_observations                    4
#define def_k_d2ia_mass                            5
#define def_k_d2ia_reduced_norm                    6
#define def_k_d2ia_dimension                       7
#define def_k_d2ia_representation                  8
#define def_k_d2ia_beta_correction                 9
#define def_k_d2ia_beta_direction                 10
#define def_k_d2ia_beta_magnitude                 11
#define def_k_d2ia_phase                          12
#define def_k_D2SkillGatherCorrection            299
#define def_k_d2gc_top                             0
#define def_k_d2gc_corrections                     1
#define def_k_d2gc_directions                      2
#define def_k_d2gc_scales                          3
#define def_k_d2gc_selected_correction             4
#define def_k_d2gc_selected_slot                   5
#define def_k_d2gc_selected_score                  6
#define def_k_d2gc_uses                            7
#define def_k_d2gc_diagnostics                     8
#define def_k_d2gc_slots                           9
#define def_k_d2gc_dimension                      10
#define def_k_d2gc_representation                 11
#define def_k_d2gc_alpha                          12
#define def_k_d2gc_retrieved                      13
#define def_k_d2gc_mark_retrieved                 14
//--- D2Skill staged bank update.  Scratch is device-only and is never saved.
#define def_k_D2SkillAgeLifecycle                295
#define def_k_d2al_used                            0
#define def_k_d2al_state                           1
#define def_k_d2al_age                             2
#define def_k_d2al_protection                      3
#define def_k_d2al_utility                         4
#define def_k_d2al_free_scores                     5
#define def_k_d2al_eviction_scores                 6
#define def_k_d2al_slots                           7
#define def_k_d2al_inactivity_age                  8
#define def_k_d2al_retrieved                       9
#define def_k_d2al_candidate_scores               10
#define def_k_D2SkillMetricPartial               300
#define def_k_d2mp_query                           0
#define def_k_d2mp_gradient                        1
#define def_k_d2mp_selected_slot                   2
#define def_k_d2mp_keys                            3
#define def_k_d2mp_directions                      4
#define def_k_d2mp_used                            5
#define def_k_d2mp_candidate_top                   6
#define def_k_d2mp_candidate_key                   7
#define def_k_d2mp_candidate_direction             8
#define def_k_d2mp_partials                        9
#define def_k_d2mp_slots                          10
#define def_k_d2mp_dimension                      11
#define def_k_D2SkillMetricReduce                301
#define def_k_d2mr_partials                        0
#define def_k_d2mr_metrics                         1
#define def_k_d2mr_batch_count                     2
#define def_k_d2mr_input_count                     3
#define def_k_d2mr_fields                          4
#define def_k_d2mr_input_offset                    5
#define def_k_d2mr_output_offset                   6
#define def_k_d2mr_max_field_a                     7
#define def_k_d2mr_max_field_b                     8
#define def_k_d2mr_final_output                    9
#define def_k_D2SkillCandidateLeader             302
#define def_k_d2cl_selected_slot                   0
#define def_k_d2cl_used                            1
#define def_k_d2cl_scales                          2
#define def_k_d2cl_observations                    3
#define def_k_d2cl_mass                            4
#define def_k_d2cl_age                             5
#define def_k_d2cl_candidate_streak                6
#define def_k_d2cl_candidate_valid                 7
#define def_k_d2cl_candidate_scale                 8
#define def_k_d2cl_metrics                         9
#define def_k_d2cl_free_top                       10
#define def_k_d2cl_eviction_top                   11
#define def_k_d2cl_decision                       12
#define def_k_d2cl_diagnostics                    13
#define def_k_d2cl_slots                          14
#define def_k_d2cl_representation                 15
#define def_k_d2cl_beta_magnitude                 16
#define def_k_d2cl_max_correction                 17
#define def_k_d2cl_min_confirmations              18
#define def_k_d2cl_similarity_threshold           19
#define def_k_d2cl_direction_threshold            20
#define def_k_d2cl_candidate_top                  21
#define def_k_d2cl_candidate_protection           22
#define def_k_D2SkillVectorUpdate                303
#define def_k_d2vu_query                           0
#define def_k_d2vu_gradient                        1
#define def_k_d2vu_keys                            2
#define def_k_d2vu_corrections                     3
#define def_k_d2vu_directions                      4
#define def_k_d2vu_scales                          5
#define def_k_d2vu_candidate_key                   6
#define def_k_d2vu_candidate_direction             7
#define def_k_d2vu_candidate_scale                 8
#define def_k_d2vu_decision                        9
#define def_k_d2vu_reduced_norm                   10
#define def_k_d2vu_dimension                      11
#define def_k_d2vu_representation                 12
#define def_k_d2vu_beta_correction                13
#define def_k_d2vu_beta_direction                 14
#define def_k_d2vu_beta_key                       15
#define def_k_d2vu_max_correction                 16
#define def_k_d2vu_phase                          17
#define def_k_D2SkillPromoteSlot                 304
#define def_k_d2ps_scales                          0
#define def_k_d2ps_utility                         1
#define def_k_d2ps_observations                    2
#define def_k_d2ps_uses                            3
#define def_k_d2ps_mass                            4
#define def_k_d2ps_used                            5
#define def_k_d2ps_state                           6
#define def_k_d2ps_age                             7
#define def_k_d2ps_protection                      8
#define def_k_d2ps_candidate_streak                9
#define def_k_d2ps_candidate_valid                10
#define def_k_d2ps_candidate_scale                11
#define def_k_d2ps_decision                       12
#define def_k_d2ps_diagnostics                    13
#define def_k_d2ps_slots                          14
#define def_k_d2ps_protection_age                 15
#define def_k_D2SkillDiagnosticsPartial          305
#define def_k_d2dp_norms                           0
#define def_k_d2dp_utility                         1
#define def_k_d2dp_uses                            2
#define def_k_d2dp_used                            3
#define def_k_d2dp_state                           4
#define def_k_d2dp_age                             5
#define def_k_d2dp_partials                        6
#define def_k_d2dp_slots                           7
#define def_k_D2SkillDiagnosticsReduce           306
#define def_k_d2dr_totals                          0
#define def_k_d2dr_diagnostics                     1
#define def_k_d2dr_utility_distribution            2
#define def_k_d2dr_usage_distribution              3
#define def_k_d2dr_slots                           4
#define def_k_D2SkillBankVectorPartial            307
#define def_k_d2bvp_corrections                     0
#define def_k_d2bvp_directions                      1
#define def_k_d2bvp_decision                        2
#define def_k_d2bvp_partials                        3
#define def_k_d2bvp_dimension                       4
#define def_k_d2bvp_representation                  5
#define def_k_d2bvp_output_offset                   6
#define def_k_CopyBufferRaw                       308
#define def_k_copy_raw_source                       0
#define def_k_copy_raw_destination                  1
#define def_k_copy_raw_total                        2
//--- D2Skill terminal hindsight kernels.
#define def_k_D2SkillInfluenceTotal               309
#define def_k_d2it_influence                        0
#define def_k_d2it_used                             1
#define def_k_d2it_state                            2
#define def_k_d2it_partials                         3
#define def_k_d2it_slots                            4
#define def_k_D2SkillResetInfluence                310
#define def_k_d2ri_influence                        0
#define def_k_d2ri_slots                            1
#define def_k_D2SkillScoreFinalize                311
#define def_k_d2sf_metrics                          0
#define def_k_d2sf_utility                          1
#define def_k_d2sf_used                             2
#define def_k_d2sf_state                            3
#define def_k_d2sf_scores                           4
#define def_k_d2sf_slots                            5
#define def_k_d2sf_similarity_threshold             6
#define def_k_d2sf_utility_weight                   7
#define def_k_d2sf_min_utility                      8
#define def_k_d2sf_utility_policy                   9
#define def_k_D2SkillDiagnosticsNormPartial       312
#define def_k_d2dnp_corrections                     0
#define def_k_d2dnp_partials                        1
#define def_k_d2dnp_slots                           2
#define def_k_d2dnp_dimension                       3
//--- Gradient-to-correction influence is reduced entirely on the device.
#define def_k_D2SkillInfluencePartial             313
#define def_k_d2sip_gradient                        0
#define def_k_d2sip_correction                      1
#define def_k_d2sip_partials                        2
#define def_k_d2sip_dimension                       3
#define def_k_D2SkillInfluenceAccumulate           314
#define def_k_d2sia_selected_slot                   0
#define def_k_d2sia_reduced_influence               1
#define def_k_d2sia_influence                       2
#define def_k_d2sia_slots                           3
#define def_k_D2SkillDirectionEMA                  315
#define def_k_d2de_gradient                          0
#define def_k_d2de_directions                        1
#define def_k_d2de_selected_slot                     2
#define def_k_d2de_used                              3
#define def_k_d2de_state                             4
#define def_k_d2de_reduced_norm                      5
#define def_k_d2de_slots                             6
#define def_k_d2de_dimension                         7
#define def_k_d2de_beta_direction                    8
#define def_k_d2de_partials                          9
#define def_k_d2de_phase                            10
#define def_k_D2SkillNormalizeDirection             316
#define def_k_d2nd_directions                         0
#define def_k_d2nd_selected_slot                      1
#define def_k_d2nd_used                               2
#define def_k_d2nd_state                              3
#define def_k_d2nd_partials                           4
#define def_k_d2nd_reduced_norm                       5
#define def_k_d2nd_slots                              6
#define def_k_d2nd_dimension                          7
#define def_k_d2nd_phase                              8
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
//| Enum of activation formula used
//+------------------------------------------------------------------+
//--- OpenCL program resource.
#resource "NeuroNet.cl" as string cl_program
enum ENUM_ACTIVATION
  {
   None = -1,  ///< Without activation formula
   TANH = 0,     ///< Use \f$tanh(x)\f$ for activation neuron
   SIGMOID = 1,  ///< Use \f$\frac{1}{1+e^x}\f$ for activation neuron
   LReLU = 2,    ///< For activation neuron use LReLU \f[\left\{ \begin{array} a x>=0, \ x \\x<0, \ 0.01*x \end{array} \right.\f]
   SoftPlus = 3,
   GELU = 4,
   MinusSoftPlus = 5,
   ELU = 6,
  };
//+------------------------------------------------------------------+
/// Enum of optimization method used
//+------------------------------------------------------------------+
enum ENUM_OPTIMIZATION
  {
   SGD = 0,      ///< Stochastic gradient descent
   ADAM = 1,     ///< Adam
   LS = 2,       ///< Least Squares
   ADAM_MINI = 3 ///< Adam-mini
  };
///@}
//+------------------------------------------------------------------+
///\class CConnection
///\brief Class of connection to anothe neuron
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/7447#para51">the link.</A>
//+------------------------------------------------------------------+
#endif // NEURONET_DEFINITIONS_MQH

#ifndef NEURONET_BUILDING_FACADE
#include "NeuroNet.mqh"
#endif // NEURONET_BUILDING_FACADE



//+------------------------------------------------------------------+
