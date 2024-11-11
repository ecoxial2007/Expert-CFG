<div align="center">
  
# Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models

</div>

## 💡Overview

**Expert-Controlled Classifier-Free Guidance** is a training-free expert-in-the-loop framework designed to align MedVLM with clinical expertise. It integrates token-level uncertainty estimation, a BioMedCLIP-based medical multimodal Retrieval-Augmented Generation (RAG), and interactive expert revisions and highlight-based guidance.

## 🔨Setup

### 🔨Pre-trained weights

#### Medical Visual Language Model:
Our fine-tuning Phi3V-Med and Phi3.5V-Med links (Remove to avoid violating double-blind):

+ Phi-3V-Med: [Huggingface]
+ Phi-3.5V-Med: [Huggingface]

#### Medical Image & Test Encoder for RAG(optional):

Download BiomedCLIP and place it in `./src/backbone/BiomedCLIP`.

BiomedCLIP links:
+ [Huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
+ [Baiduyun]

**Note**: Directly downloading weights from Huggingface might encounter network issues. To facilitate modifications, we have converted the original `.bin` file to PyTorch's `.pth`. We recommend using the Baiduyun version.


### 📑Data Preparation
Our data mainly comes from publicly available, free online Pathology Education Informational Resource ([PEIR](https://peir.path.uab.edu/library/index.php?/category/2)) Digital Library. 
We test our model on:
+ [VQA-RAD](https://osf.io/89kps/)
+ [SLAKE](https://www.med-vqa.com/slake/)
+ [PathVQA](https://github.com/UCSD-AI4H/PathVQA)

Medical Alignment and Instruction Tuning:
+ [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)
+ [Llave-Med](https://github.com/microsoft/LLaVA-Med)


### Prepare BiomedCLIP Pre-extracted Image Feature
Note: We recommend using our pre-extracted BioMedCLIP features. The original images can also be found in the links below:


| Dataset  | Pre-extracted Features  & Original Images |
|----------|------------------------------------------|
| PEIR     | Remove |
| PathVQA  | Remove |
| Slake    | Remove |
| RADVQA   | Remove |


### Demo
Comming soon!!!

### Evaluation
Comming soon!!!

## 📝Acknowledgements
We also reference the excellent repos of [Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook), [HuatuoVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision), [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), in addition to other specific repos to the baseline and dataset we examined (see paper).

## 📝Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```

```
