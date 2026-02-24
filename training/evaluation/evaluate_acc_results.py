#!/usr/bin/env python3
"""
Evaluation script: Calculate accuracy using the same accuracy_reward_vqa method as training
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src/open-r1-multimodal/src"))

from open_r1.vlm_modules.qwen_module import Qwen2VLModule


def has_valid_format(text: str) -> bool:
    """Check if text has valid format (contains Reasoning and Answer tags)"""
    has_reasoning = bool(re.search(r'<[Rr]easoning>.*?</[Rr]easoning>', text, re.DOTALL))
    has_answer = bool(re.search(r'<[Aa]nswer>.*?</[Aa]nswer>', text, re.DOTALL))
    
    return has_reasoning and has_answer


def extract_answer(text: str) -> str:
    """Extract answer section from generated text"""
    answer_match = re.search(r'<[Aa]nswer>(.*?)</[Aa]nswer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text.strip()


def extract_reasoning(text: str) -> str:
    """Extract reasoning section from generated text"""
    reasoning_match = re.search(r'<[Rr]easoning>(.*?)</[Rr]easoning>', text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return ""


def evaluate_sample(sample: Dict, qwen_module: Qwen2VLModule) -> Dict:
    """
    Evaluate single sample using the same accuracy_reward_vqa function as training
    """
    generated = sample['generated_answer']
    expected = sample.get('expected_answer', '')
    method = sample.get('accu_reward_method', 'default')
    
    format_valid = has_valid_format(generated)
    
    completion = [[{"content": generated}]]
    
    try:
        reward = qwen_module.accuracy_reward_vqa(
            completion,
            accu_reward_method=[method],
            expected_answer=[expected]
        )[0]
    except Exception as e:
        print(f"Warning: Error evaluating sample: {e}")
        reward = 0.0
    
    is_correct = (reward >= 0.99)
    
    return {
        'format_valid': format_valid,
        'accuracy': is_correct,
        'reward': reward,
        'generated_answer': extract_answer(generated),
        'has_reasoning': bool(extract_reasoning(generated)),
        'accu_reward_method': method
    }


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute overall metrics using the same statistical approach as training analysis"""
    metrics = {
        'total': len(results),
        'format_valid': 0,
        'accurate': 0,
        'has_reasoning': 0,
        'total_reward': 0.0,
        'by_question_type': defaultdict(lambda: {
            'total': 0, 'accurate': 0, 'format_valid': 0, 'total_reward': 0.0
        }),
        'by_method': defaultdict(lambda: {
            'total': 0, 'accurate': 0, 'format_valid': 0, 'total_reward': 0.0
        })
    }
    
    for result in results:
        eval_result = result['eval']
        question_type = result.get('question_type', 'unknown')
        method = result.get('accu_reward_method', 'default')
        reward = eval_result.get('reward', 0.0)
        
        if eval_result['format_valid']:
            metrics['format_valid'] += 1
        if eval_result['accuracy']:
            metrics['accurate'] += 1
        if eval_result['has_reasoning']:
            metrics['has_reasoning'] += 1
        metrics['total_reward'] += reward
        
        metrics['by_question_type'][question_type]['total'] += 1
        metrics['by_question_type'][question_type]['total_reward'] += reward
        if eval_result['accuracy']:
            metrics['by_question_type'][question_type]['accurate'] += 1
        if eval_result['format_valid']:
            metrics['by_question_type'][question_type]['format_valid'] += 1
        
        metrics['by_method'][method]['total'] += 1
        metrics['by_method'][method]['total_reward'] += reward
        if eval_result['accuracy']:
            metrics['by_method'][method]['accurate'] += 1
        if eval_result['format_valid']:
            metrics['by_method'][method]['format_valid'] += 1
    
    return metrics


def print_metrics(metrics: Dict):
    """Print metrics in the same format as previous analysis"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Using accuracy_reward_vqa)")
    print("=" * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  Total Samples: {metrics['total']}")
    print(f"  Format Valid: {metrics['format_valid']}/{metrics['total']} ({metrics['format_valid']/metrics['total']*100:.2f}%)")
    print(f"  Accuracy: {metrics['accurate']}/{metrics['total']} ({metrics['accurate']/metrics['total']*100:.2f}%)")
    print(f"  Has Reasoning: {metrics['has_reasoning']}/{metrics['total']} ({metrics['has_reasoning']/metrics['total']*100:.2f}%)")
    print(f"  Average Reward: {metrics['total_reward']/metrics['total']:.4f}")
    
    print(f"\n{'='*80}")
    print("By Question Type:")
    print(f"{'='*80}")
    print(f"{'Type':<20} {'Total':<8} {'Accuracy':<20} {'Format':<20} {'Avg Reward':<12}")
    print(f"{'-'*20} {'-'*8} {'-'*20} {'-'*20} {'-'*12}")
    for qtype, stats in sorted(metrics['by_question_type'].items()):
        acc_pct = stats['accurate'] / stats['total'] * 100 if stats['total'] > 0 else 0
        fmt_pct = stats['format_valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_reward = stats['total_reward'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{qtype:<20} {stats['total']:<8} "
              f"{acc_pct:>6.2f}% ({stats['accurate']:<4}) "
              f"{fmt_pct:>6.2f}% ({stats['format_valid']:<4}) "
              f"{avg_reward:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"By Evaluation Method (accu_reward_method):")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Total':<8} {'Accuracy':<20} {'Format':<20} {'Avg Reward':<12}")
    print(f"{'-'*20} {'-'*8} {'-'*20} {'-'*20} {'-'*12}")
    for method, stats in sorted(metrics['by_method'].items()):
        acc_pct = stats['accurate'] / stats['total'] * 100 if stats['total'] > 0 else 0
        fmt_pct = stats['format_valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_reward = stats['total_reward'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{method:<20} {stats['total']:<8} "
              f"{acc_pct:>6.2f}% ({stats['accurate']:<4}) "
              f"{fmt_pct:>6.2f}% ({stats['format_valid']:<4}) "
              f"{avg_reward:>10.4f}")
    
    # Open-ended question breakdown statistics
    print(f"\n{'='*80}")
    print("Open-Ended Question Breakdown:")
    print(f"{'='*80}")
    open_ended_methods = {k: v for k, v in metrics['by_method'].items() 
                          if k in ['counting', 'distance', 'spatial', 'default']}
    if open_ended_methods:
        print(f"{'Method':<20} {'Total':<8} {'Accuracy':<20} {'Avg Reward':<12}")
        print(f"{'-'*20} {'-'*8} {'-'*20} {'-'*12}")
        for method, stats in sorted(open_ended_methods.items()):
            acc_pct = stats['accurate'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_reward = stats['total_reward'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{method:<20} {stats['total']:<8} "
                  f"{acc_pct:>6.2f}% ({stats['accurate']:<4}) "
                  f"{avg_reward:>10.4f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results using accuracy_reward_vqa")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with inference results")
    parser.add_argument("--output", type=str, help="Output JSONL file with evaluation results (optional)")
    parser.add_argument("--show_errors", action="store_true", help="Show samples with incorrect predictions")
    parser.add_argument("--max_errors", type=int, default=10, help="Max number of errors to show")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.input}")
    
    results = []
    with open(args.input, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            results.append(item)
    
    print(f"Loaded {len(results)} samples")
    
    print("Initializing Qwen2VLModule for evaluation...")
    qwen_module = Qwen2VLModule()
    
    print("Evaluating samples using accuracy_reward_vqa()...")
    for i, result in enumerate(results):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(results)} samples...")
        result['eval'] = evaluate_sample(result, qwen_module)
    
    print(f"Evaluation complete!")
    
    metrics = compute_metrics(results)
    
    print_metrics(metrics)
    
    if args.show_errors:
        print("\n" + "=" * 80)
        print(f"ERROR SAMPLES (showing first {args.max_errors} incorrect predictions)")
        print("=" * 80)
        
        error_count = 0
        for i, result in enumerate(results):
            if not result['eval']['accuracy'] and error_count < args.max_errors:
                error_count += 1
                print(f"\n--- Sample {result['idx']} ---")
                print(f"Question Type: {result.get('question_type', 'unknown')}")
                print(f"Evaluation Method: {result.get('accu_reward_method', 'default')}")
                print(f"Question: {result['question'][:200]}...")
                print(f"Expected: {result.get('expected_answer', 'N/A')}")
                print(f"Generated Answer: {result['eval']['generated_answer'][:200]}...")
                print(f"Reward: {result['eval']['reward']:.4f}")
                print(f"Format Valid: {result['eval']['format_valid']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\nEvaluation results saved to {args.output}")
    
    print("\n" + "=" * 80)
    print("Evaluation Method: Same as training - using accuracy_reward_vqa()")
    print("Threshold: reward >= 0.99 considered correct")
    print("=" * 80)


if __name__ == "__main__":
    main()
