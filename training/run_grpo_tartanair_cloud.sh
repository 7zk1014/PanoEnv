#!/bin/bash

# ========================================
# CUDA Environment Configuration
# ========================================
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================================
# TartanAir GRPO Training Script
# Hardware: 4×B200 GPU (192GB VRAM each)
# ========================================

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# ========================================
# Data Configuration
# ========================================
TRAIN_DATA="/workspace/data/train_relabeled.jsonl"
TEST_DATA="/workspace/data/test_benchmark_relabeled.jsonl"
IMAGE_ROOT="/workspace/tartanair_output"

MODEL_PATH="/workspace/models/Qwen2.5-VL-7B-Instruct"

# ========================================
# Experiment Configuration
# ========================================
export EXP_NAME="Qwen2.5-VL-7B-TartanAir-GRPO_alldata_resume"
TASK_TYPE="omni_vqa"
IS_REWARD_CUSTOMIZED=True

# ========================================
# GPU Configuration
# ========================================
NUM_GPUS=4

# ========================================
# Batch Size Configuration
# Formula: effective_batch = per_device_batch × grad_accum × num_gpus
# Target: 32-64 for stable GRPO training
# ========================================

PER_DEVICE_BATCH_SIZE=4

GRADIENT_ACCUMULATION=2

EFFECTIVE_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))
echo "Configuration:"
echo "  PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "  GRADIENT_ACCUMULATION: ${GRADIENT_ACCUMULATION}"
echo "  NUM_GPUS: ${NUM_GPUS}"
echo "  Effective Batch Size: ${EFFECTIVE_BATCH}"
echo "  Target: 32-64 (GRPO paper recommendation)"

# ========================================
# GRPO Parameters
# ========================================
NUM_GENERATIONS=4

MAX_COMPLETION_LENGTH=256

# ========================================
# Training Hyperparameters
# ========================================
NUM_EPOCHS=2

LEARNING_RATE=5e-6

# ========================================
# Reward Function Configuration
# ========================================
REWARD_FUNCS="accuracy format"

REWARD_WEIGHTS="0.90,0.10"

BETA=0.01

# ========================================
# LoRA Configuration
# ========================================
USE_PEFT="true"

LORA_R=16

LORA_ALPHA=32

LORA_DROPOUT=0.05

FREEZE_VISION_MODULES="true"

# ========================================
# Optimizer Configuration
# ========================================
WARMUP_RATIO=0.05

MAX_GRAD_NORM=10.0

# ========================================
# Model Optimization
# ========================================
ATTN_IMPL="flash_attention_2"

TORCH_DTYPE="bfloat16"

GRADIENT_CHECKPOINTING="false"

# ========================================
# DeepSpeed Configuration
# ========================================
DEEPSPEED_CONFIG="${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json"

# ========================================
# Image Processing
# ========================================
MAX_PIXELS=3276800

MIN_PIXELS=3136

# ========================================
# Checkpoint and Logging
# ========================================
SAVE_STEPS=100

SAVE_TOTAL_LIMIT=50

LOGGING_STEPS=1

# ========================================
# Debug Configuration
# ========================================
export DEBUG_MODE="true"

# ========================================
# Output Directory
# ========================================
CHECKPOINT_DIR="/workspace/VLM-R1/checkpoints/rl/${EXP_NAME}"
LOG_DIR="/workspace/VLM-R1/runs/${EXP_NAME}/log"

RESUME_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint-1700"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

export LOG_PATH="${LOG_DIR}/train.$(date +%Y%m%d_%H%M%S).log"

echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Log directory: ${LOG_DIR}"

if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    RESUME_FLAG="${RESUME_CHECKPOINT}"
else
    echo "Training from scratch"
    RESUME_FLAG="False"
fi

# ========================================
# Environment Variables
# ========================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# ========================================
# Configuration Summary
# ========================================
echo "========================================"
echo "Experiment: ${EXP_NAME}"
echo "========================================"
echo "Data:"
echo "  Train: ${TRAIN_DATA}"
echo "  Test: ${TEST_DATA}"
echo "  Images: ${IMAGE_ROOT}"
echo "  Model: ${MODEL_PATH}"
echo ""
echo "GPU:"
echo "  Number: ${NUM_GPUS}"
echo "  Batch/Device: ${PER_DEVICE_BATCH_SIZE}"
echo "  Grad Accum: ${GRADIENT_ACCUMULATION}"
echo "  Effective Batch: ${EFFECTIVE_BATCH}"
echo ""
echo "GRPO:"
echo "  Generations: ${NUM_GENERATIONS}"
echo "  Max Length: ${MAX_COMPLETION_LENGTH}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Beta: ${BETA}"
echo ""
echo "Reward:"
echo "  Functions: ${REWARD_FUNCS}"
echo "  Weights: ${REWARD_WEIGHTS}"
echo ""
echo "Training:"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Total Steps: ~$((2395 * NUM_EPOCHS / EFFECTIVE_BATCH))"
echo "  Save Steps: ${SAVE_STEPS}"
echo "  Eval: Disabled"
echo ""
echo "Optimization:"
echo "  Attention: ${ATTN_IMPL}"
echo "  Dtype: ${TORCH_DTYPE}"
echo "  Grad Checkpoint: ${GRADIENT_CHECKPOINTING}"
echo "  DeepSpeed: $(basename ${DEEPSPEED_CONFIG})"
echo "========================================"

# ========================================
# Start Training
# ========================================
echo "Starting training..."
cd ${REPO_HOME}/src/open-r1-multimodal

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${CHECKPOINT_DIR} \
    --resume_from_checkpoint ${RESUME_FLAG} \
    --model_name_or_path ${MODEL_PATH} \
    --data_file_paths "${TRAIN_DATA}" \
    --image_folders "${IMAGE_ROOT}" \
    --is_reward_customized_from_vlm_module ${IS_REWARD_CUSTOMIZED} \
    --task_type ${TASK_TYPE} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --eval_strategy no \
    --logging_steps ${LOGGING_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --bf16 \
    --attn_implementation ${ATTN_IMPL} \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --num_generations ${NUM_GENERATIONS} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --reward_funcs ${REWARD_FUNCS} \
    --custom_reward_weights ${REWARD_WEIGHTS} \
    --beta ${BETA} \
    --report_to none \
    --dataset_name tartanair_vqa \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --use_peft true \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    2>&1 | tee ${LOG_PATH}

# ========================================
# Training Complete
# ========================================
echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Model: ${CHECKPOINT_DIR}"
echo "Log: ${LOG_PATH}"
echo "========================================"
