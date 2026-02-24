"""
TartanAir VQA Evaluation Script
Evaluates GRPO-trained models on TartanAir dataset
"""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import torch.distributed as dist
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "open-r1-multimodal/src"))
from open_r1.vlm_modules.qwen_module import Qwen2VLModule

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

main_rank = 0

RUN_NAME = "Qwen2.5-VL-7B-TartanAir-GRPO"
CHECKPOINT_STEP = 300

MODEL_PATH = f"../../checkpoints/rl/{RUN_NAME}/checkpoint-{CHECKPOINT_STEP}"
OUTPUT_PATH = f"./logs/tartanair_results_{RUN_NAME}_step{CHECKPOINT_STEP}.json"

BSZ = 4
DATA_PATH = "../../data/test_simple_only.jsonl"
IMAGE_ROOT = "../../tartanair_output"

if rank == main_rank:
    print("="*50)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Data: {DATA_PATH}")
    print(f"Image Root: {IMAGE_ROOT}")
    print(f"Batch Size: {BSZ}")
    print("="*50)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
qwen_module = Qwen2VLModule()

def extract_answer(content):
    """Extract answer from model output (case-insensitive)"""
    answer_pattern = r'<[Aa]nswer>(.*?)</[Aa]nswer>'
    match = re.search(answer_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def evaluate_answer_with_reward(model_output, gt_answer, accu_reward_method):
    """Evaluate using the same accuracy_reward_vqa method as training"""
    completion = [[{"content": model_output}]]
    
    try:
        reward = qwen_module.accuracy_reward_vqa(
            completion,
            accu_reward_method=[accu_reward_method],
            expected_answer=[gt_answer]
        )[0]
    except Exception as e:
        print(f"Warning: Error in accuracy_reward_vqa: {e}")
        reward = 0.0
    
    return reward >= 0.99, reward

def check_format(content):
    """Check if output format is valid (consistent with training)"""
    completion = [[{"content": content}]]
    try:
        format_score = qwen_module.format_reward_vqa(completion)[0]
        return format_score >= 0.99
    except:
        return False

if rank == main_rank:
    print("Loading test data...")

with open(DATA_PATH, 'r') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

if rank == main_rank:
    print(f"Total samples: {len(data)}")

per_rank_data = len(data) // world_size
start_idx = rank * per_rank_data
end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
rank_data = data[start_idx:end_idx]

if rank == main_rank:
    print(f"Samples per process: {per_rank_data}")

messages = []
for x in rank_data:
    conv_key = 'value' if 'value' in x['conversations'][0] else 'content'
    question = x['conversations'][0][conv_key].replace('<image>', '')
    question_type = x.get('question_type', 'open_ended')
    
    prompt_template = qwen_module.get_question_template(
        task_type='omni_vqa',
        question_type=question_type
    )
    prompt = prompt_template.format(Question=question)
    
    image_path = os.path.join(IMAGE_ROOT, x['image'])
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]
    }]
    messages.append(message)

rank_outputs = []

for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank, desc=f"Rank {rank}"):
    batch_messages = messages[i:i + BSZ]
    
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            use_cache=True, 
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    rank_outputs.extend(batch_output_text)

if rank == main_rank:
    print(f"Rank {rank} processing complete: {len(rank_outputs)} samples")

all_outputs = [None] * len(data)
rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

gathered_results = [None] * world_size
dist.all_gather_object(gathered_results, rank_results)

if rank == main_rank:
    print("Gathering results from all processes...")
    for results in gathered_results:
        for idx, output in results:
            all_outputs[idx] = output
    
    final_output = []
    correct_count = 0
    format_valid_count = 0
    total_count = len(data)
    total_reward = 0.0
    
    stats = {
        'multiple_choice': {'total': 0, 'correct': 0, 'format_valid': 0, 'total_reward': 0.0},
        'true_false': {'total': 0, 'correct': 0, 'format_valid': 0, 'total_reward': 0.0},
        'open_ended': {'total': 0, 'correct': 0, 'format_valid': 0, 'total_reward': 0.0}
    }
    
    for input_example, model_output in zip(data, all_outputs):
        conv_key = 'value' if 'value' in input_example['conversations'][0] else 'content'
        question = input_example['conversations'][0][conv_key]
        ground_truth = input_example['expected_answer']
        question_type = input_example.get('question_type', 'open_ended')
        accu_reward_method = input_example.get('accu_reward_method', 'default')
        
        model_answer = extract_answer(model_output)
        format_valid = check_format(model_output)
        is_correct, reward = evaluate_answer_with_reward(model_output, ground_truth, accu_reward_method)
        
        if is_correct:
            correct_count += 1
            stats[question_type]['correct'] += 1
        if format_valid:
            format_valid_count += 1
            stats[question_type]['format_valid'] += 1
        
        stats[question_type]['total'] += 1
        stats[question_type]['total_reward'] += reward
        total_reward += reward
        
        result = {
            'image': input_example['image'],
            'question': question,
            'question_type': question_type,
            'accu_reward_method': accu_reward_method,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'extracted_answer': model_answer,
            'correct': int(is_correct),
            'format_valid': int(format_valid),
            'reward': float(reward)
        }
        final_output.append(result)
    
    overall_accuracy = correct_count / total_count * 100
    format_accuracy = format_valid_count / total_count * 100
    avg_reward = total_reward / total_count
    
    mcq_accuracy = stats['multiple_choice']['correct'] / stats['multiple_choice']['total'] * 100 if stats['multiple_choice']['total'] > 0 else 0
    mcq_format = stats['multiple_choice']['format_valid'] / stats['multiple_choice']['total'] * 100 if stats['multiple_choice']['total'] > 0 else 0
    mcq_reward = stats['multiple_choice']['total_reward'] / stats['multiple_choice']['total'] if stats['multiple_choice']['total'] > 0 else 0
    
    yes_no_accuracy = stats['true_false']['correct'] / stats['true_false']['total'] * 100 if stats['true_false']['total'] > 0 else 0
    yes_no_format = stats['true_false']['format_valid'] / stats['true_false']['total'] * 100 if stats['true_false']['total'] > 0 else 0
    yes_no_reward = stats['true_false']['total_reward'] / stats['true_false']['total'] if stats['true_false']['total'] > 0 else 0
    
    open_accuracy = stats['open_ended']['correct'] / stats['open_ended']['total'] * 100 if stats['open_ended']['total'] > 0 else 0
    open_format = stats['open_ended']['format_valid'] / stats['open_ended']['total'] * 100 if stats['open_ended']['total'] > 0 else 0
    open_reward = stats['open_ended']['total_reward'] / stats['open_ended']['total'] if stats['open_ended']['total'] > 0 else 0
    
    print("\n" + "="*80)
    print("Evaluation Results (using accuracy_reward_vqa same as training):")
    print("="*80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"Format Accuracy: {format_accuracy:.2f}% ({format_valid_count}/{total_count})")
    print(f"Average Reward: {avg_reward:.4f}")
    print("\n" + "-"*80)
    print(f"{'Question Type':<15} {'Total':<8} {'Accuracy':<20} {'Format Acc':<20} {'Avg Reward':<12}")
    print("-"*80)
    print(f"{'Multiple Choice':<15} {stats['multiple_choice']['total']:<8} {mcq_accuracy:>6.2f}% ({stats['multiple_choice']['correct']:<4}) {mcq_format:>6.2f}% ({stats['multiple_choice']['format_valid']:<4}) {mcq_reward:>10.4f}")
    print(f"{'True/False':<15} {stats['true_false']['total']:<8} {yes_no_accuracy:>6.2f}% ({stats['true_false']['correct']:<4}) {yes_no_format:>6.2f}% ({stats['true_false']['format_valid']:<4}) {yes_no_reward:>10.4f}")
    print(f"{'Open-Ended':<15} {stats['open_ended']['total']:<8} {open_accuracy:>6.2f}% ({stats['open_ended']['correct']:<4}) {open_format:>6.2f}% ({stats['open_ended']['format_valid']:<4}) {open_reward:>10.4f}")
    print("="*80)
    print("Note: Evaluation method identical to training (threshold: reward >= 0.99)")
    print("="*80)
    
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump({
            'overall_accuracy': overall_accuracy,
            'format_accuracy': format_accuracy,
            'avg_reward': avg_reward,
            'mcq_accuracy': mcq_accuracy,
            'yes_no_accuracy': yes_no_accuracy,
            'open_ended_accuracy': open_accuracy,
            'stats': stats,
            'results': final_output,
            'evaluation_method': 'accuracy_reward_vqa (same as training)',
            'threshold': 0.99
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")

dist.barrier()
