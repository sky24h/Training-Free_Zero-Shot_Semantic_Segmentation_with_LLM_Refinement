# Training-Free_Zero-Shot_Semantic_Segmentation_with_LLM_Refinement

This repository contains official implementation of the paper "Training-Free Zero-Shot Semantic Segmentation with LLM Refinement" (BMVC 2024).

Project Page: https://sky24h.github.io/websites/bmvc2024_training-free-semseg-with-LLM/

Huggingface Demo: https://huggingface.co/spaces/sky24h/Training-Free_Zero-Shot_Semantic_Segmentation_with_LLM_Refinement

## Dependencies
Python >= 3.9 (Recommend == 3.11.8)

pip install -r requirements.txt

## Usage
### 1. Download Pretrained Model
All pre-trained models will be downloaded automatically when you run the code.
However, you may need authorization to download the [Llama3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model from Huggingface.

You can use the following command to login to Huggingface, or you can download the model manually to your local machine and modify the "utils/llms_utils.py" file to load the model from the local directory.
```bash
huggingface-cli login
```

### 2. Inference on Single Image
```bash
python inference_single.py --config ./configs/DRAM.yaml --input_path ./sources/DRAM_eg.jpg
python inference_single.py --config ./configs/Cityscapes.yaml --input_path ./sources/Cityscapes_eg.jpg
```

### 3. Inference on dataset
See the configuration files in the "configs" directory for more details on the dataset and model settings.

```bash
CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/VOC2012.yaml --reset --draw_bbox --debug
CUDA_VISIBLE_DEVICES=0 python inference_dataset.py --config ./configs/COCO-81.yaml --reset --draw_bbox --debug
```

| Flag | Description |
|------|-------------|
| --reset | Removes the previous results |
| --draw_bbox | Visualizes the bounding box of the detected objects |
| --debug | Runs only the first 5% of the dataset |
| --use_lower_vram | if you have a low GPU memory, you can use this flag to reduce the memory requirement of the model |


| Model Variant | GPU Memory Requirement |
|---------------|-----------------------|
| LLama-3-8B w/o use_lower_vram | 30GB |
| LLama-3-8B w/ use_lower_vram | 24GB |
| OpenAI API w/o use_lower_vram | 16GB |
| OpenAI API w/ use_lower_vram | 12GB |

### Citation
If you find this work useful, please consider citing the following paper:
```
@inproceedings{Huang2024SemSegLLM,
  author = {Huang, Yuantian and Iizuka, Satoshi and Fukui, Kazuhiro},
  booktitle = {The British Machine Vision Conference (BMVC) 2024},
  title = {Training-Free Zero-Shot Semantic Segmentation with LLM Refinement},
  year = {2024},
}
```