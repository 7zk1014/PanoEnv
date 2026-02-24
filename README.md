# PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-blue)](https://arxiv.org/abs/xxxx.xxxxx)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PanoEnv-yellow)](https://huggingface.co/datasets/7zkk/PanoEnv)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<p align="center">
  <img src="fig/teaser.png" width="100%">
</p>

## 🔥 Highlights

- **PanoEnv-QA**: A large-scale VQA benchmark with 14,827 questions across 5 categories for 360° panoramic spatial reasoning
- **3D-Aware RL Framework**: GRPO-based post-training with geometry-grounded rewards
- **Two-Stage Curriculum**: Structured → Mixed training for stable optimization
- **State-of-the-Art Results**: 52.93% accuracy, +132% improvement on open-ended questions

## 📊 PanoEnv-QA Dataset

**🤗 Download**: [https://huggingface.co/datasets/7zkk/PanoEnv](https://huggingface.co/datasets/7zkk/PanoEnv)

| Category | # Questions | Percentage |
|----------|-------------|------------|
| Attribute Comparison | 2,975 | 20.1% |
| Distance Estimation | 2,975 | 20.1% |
| Relative Spatial Positioning | 2,975 | 20.1% |
| Environment Identification | 2,965 | 20.0% |
| View Source Identification | 2,937 | 19.8% |
| **Total** | **14,827** | **100%** |

## 📁 Project Structure

```
panoenv_code/
├── README.md
├── requirements.txt
├── dataset/                          # Dataset files
│   ├── qa_generation.py              # QA generation pipeline
│   ├── train_relabeled.jsonl         # Training data
│   ├── val_relabeled.jsonl           # Validation data
│   └── test_benchmark_relabeled.jsonl # Test benchmark
├── code/panoenv/
│   ├── run_grpo_tartanair_cloud.sh   # Main training script
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── evaluate_acc_results.py   # Accuracy evaluation
│   │   ├── qwenscore.py              # Qwen-Score evaluation
│   │   └── Prometheusscore.py        # Prometheus-Score evaluation
│   ├── src/
│   │   ├── eval/                     # Inference scripts
│   │   │   ├── test_tartanair_vqa.py # Main evaluation script
│   │   │   └── ...
│   │   └── open-r1-multimodal/       # Core training framework
│   │       └── src/open_r1/
│   │           ├── grpo_rec.py       # GRPO training entry
│   │           ├── trainer/          # GRPO trainer implementation
│   │           └── vlm_modules/      # VLM module implementations
│   └── run_scripts/                  # Additional run scripts
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/panoenv.git
cd panoenv

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
cd code/panoenv/src/open-r1-multimodal
pip install -e ".[dev]"
```

### Data Preparation

**Option 1: Download from Hugging Face (Recommended)**
```bash
# Using huggingface-cli
huggingface-cli download 7zkk/PanoEnv --local-dir ./data

# Or using Python
from datasets import load_dataset
dataset = load_dataset("7zkk/PanoEnv")
```

**Option 2: Build from scratch**
1. Download the TartanAir dataset from [TartanAir](https://theairlab.org/tartanair-dataset/)
2. Process the data to generate ERP panoramas
3. Place the processed images in `./tartanair_output/`
4. Run `python dataset/qa_generation.py` to generate QA pairs

### Training

```bash
# Run GRPO training
cd code/panoenv
bash run_grpo_tartanair_cloud.sh
```

### Evaluation

```bash
# Run inference
cd code/panoenv/src/eval
torchrun --nproc_per_node=4 test_tartanair_vqa.py

# Evaluate accuracy
cd code/panoenv/evaluation
python evaluate_acc_results.py --input <inference_results.jsonl>
```

## 📈 Results

### Main Results on PanoEnv-QA

| Model | Total Acc. | T/F | MCQ | OE | Q-Score | P-Score |
|-------|------------|-----|-----|-----|---------|---------|
| Qwen2.5-VL-7B (Base) | 49.34% | 65.19% | 57.24% | 6.39% | 5.60 | 5.48 |
| Qwen2.5-VL-32B | 42.70% | 62.47% | 44.96% | 8.36% | 5.02 | 4.92 |
| **GRPO-Balanced (Ours)** | **52.93%** | **68.78%** | **58.90%** | **14.83%** | **6.24** | **5.95** |

## 🔧 Configuration

Key hyperparameters in `run_grpo_tartanair_cloud.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_GENERATIONS` | 4 | GRPO group size K |
| `LEARNING_RATE` | 5e-6 | Learning rate |
| `REWARD_WEIGHTS` | 0.90, 0.10 | Accuracy vs Format reward weights |
| `LORA_R` | 16 | LoRA rank |
| `BETA` | 0.01 | KL penalty coefficient |

## 📖 Citation

```bibtex
@inproceedings{
  title={PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning},
  author={Zekai,Lin  and Zheng, Xu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 🙏 Acknowledgements

- [TartanAir](https://theairlab.org/tartanair-dataset/) for the synthetic 3D environment data
- [TRL](https://github.com/huggingface/trl) for the GRPO implementation
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) for the base VLM
- [Hugging Face](https://huggingface.co/) for hosting the dataset

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
