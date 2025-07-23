<div align="center">
  
# Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models

</div>

## 💡Overview

**Expert-Controlled Classifier-Free Guidance** is a training-free expert-in-the-loop framework designed to align MedVLM with clinical expertise. It integrates token-level uncertainty estimation, a BioMedCLIP-based medical multimodal Retrieval-Augmented Generation (RAG), and interactive expert revisions and highlight-based guidance.

## 🔨Setup
### 🔨Installation
```
conda create -n expert_cfg python=3.10 -y
conda activate expert_cfg
pip install -r requirements.txt
```

### 🔨Pre-trained weights

#### Baseline Model:
Download them to the current directory separately and merge them with `Phi-3-vision-128k-instruct` and `Phi-3.5-vision-instruct` respectively.
+ Phi-3V: [Huggingface](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
+ Phi-3.5V: [Huggingface](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)


#### Medical LoRA:
Our fine-tuning Phi3V-Med and Phi3.5V-Med LoRA links:

+ Phi-3V-Med: [Huggingface](https://huggingface.co/ecoxial2007/Phi-3V-Med)
+ Phi-3.5V-Med: [Huggingface](https://huggingface.co/ecoxial2007/Phi-3.5V-Med)
Download them to the `./lora_weights` folder

#### Demo
```
torchrun --nproc_per_node=1 demo.py \
    --bf16 \
    --use_lora \
    --input_json 'examples/input_queries.json' \
    --img_root 'examples/images' \
    --save_path 'examples/results.json' \
    --output_dir './lora_weights/logs_phi35_pubmed_instruct' 
```

#### Medical Image & Test Encoder for RAG(optional):

Download BiomedCLIP and place it in `./src/backbone/BiomedCLIP`.

BiomedCLIP links:
+ [Huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

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
| PEIR     | [Baiduyun, Rename zzz2zip](https://pan.baidu.com/s/1sJp_3UzjIIvOiuyMB417GQ?pwd=6666)|
| PEIR BioMedCLIP features & keyword & GPT3.5 rewrite caption | [Baiduyun](https://pan.baidu.com/s/1pqHhrxLL-ZdgEat0wNwLmQ?pwd=6666)|
| PathVQA  | [Baiduyun](https://pan.baidu.com/s/1b1SuDSbsNM1rVGzbx8utvg?pwd=6666)|
| Slake    | [Baiduyun](https://pan.baidu.com/s/1mfAoi9_HZkrk7OuyQIn4-w?pwd=6666)|
| RADVQA   | [Baiduyun](https://pan.baidu.com/s/1gBjAjq2L-iIMf0j05QsJ-w?pwd=6666)|






## 📝Acknowledgements
We also reference the excellent repos of [Phi-3CookBook](https://github.com/microsoft/Phi-3CookBook), [HuatuoVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision), [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), in addition to other specific repos to the baseline and dataset we examined (see paper).

## 📝Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@misc{liang2025uncertaintydriven,
    title={Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models},
    author={Xiao Liang and Di Wang and Zhicheng Jiao and Ronghan Li and Pengfei Yang and Quan Wang and Tat-Seng Chua},
    year={2025},
    eprint={2507.09209},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
