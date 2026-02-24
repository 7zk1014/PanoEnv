PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
data_paths="./data/your_dataset.jsonl"
image_folders="./data/your_images"
model_path="Qwen/Qwen2.5-VL-7B-Instruct"
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-VQA-lora"
TASK_TYPE="omni_vqa"

cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"



export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME}_600steps \
    --resume_from_checkpoint ${CHECKPOINT_PATH} \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module False \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 50 \
    --num_generations 6 \
    --max_completion_length 2048 \
    --reward_funcs format_tagged reasoning_sim answer_sim \
    --custom_reward_weights 0.1,0.45,0.45 \
    --beta 0.04 \
    --report_to none \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true

echo "Training completed for ${EXP_NAME}"