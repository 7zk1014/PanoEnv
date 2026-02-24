"""
Multi-GPU parallel evaluator using Qwen2.5-7B for VLM results scoring.
Supports JSONL format with 4-card parallel processing and large batch sizes.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import re
from collections import defaultdict
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
import numpy as np

class MultiGPUEvaluator:
    """Multi-GPU parallel evaluator"""

    def __init__(
        self,
        local_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        num_gpus: int = 4,
        batch_size_per_gpu: int = 32,
        save_reasoning: bool = False,
    ):
        self.local_model_path = local_model_path
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = batch_size_per_gpu
        self.save_reasoning = save_reasoning
        self.total_batch_size = num_gpus * batch_size_per_gpu

        print(f"🚀 Multi-GPU configuration:")
        print(f"   - GPUs: {num_gpus}")
        print(f"   - Batch per GPU: {batch_size_per_gpu}")
        print(f"   - Total batch: {self.total_batch_size}")

    def evaluate_dataset(
        self,
        results_file: str,
        output_file: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Dict:
        """Evaluate entire dataset with multi-GPU parallelism"""
        print(f"\n📂 Loading data: {results_file}")
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        if max_samples:
            results = results[:max_samples]
            print(f"⚠️ Test mode: evaluating first {max_samples} samples only")

        print(f"📊 Total samples: {len(results)}")

        print(f"\n🔍 Extracting answers...")
        for item in results:
            item['model_answer'] = self._extract_answer(item.get('generated_text', ''))

        print(f"\n🎯 Preparing scoring tasks...")
        tasks = []
        for idx, item in enumerate(results):
            tasks.append({
                'idx': idx,
                'question': item['question'],
                'model_answer': item['model_answer'],
                'expected_answer': item['expected_answer'],
                'question_type': item['question_type'],
            })

        print(f"✅ Prepared: {len(tasks)} tasks")

        tasks_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, task in enumerate(tasks):
            gpu_id = i % self.num_gpus
            tasks_per_gpu[gpu_id].append(task)

        print(f"\n🔀 Task distribution:")
        for gpu_id in range(self.num_gpus):
            print(f"   GPU {gpu_id}: {len(tasks_per_gpu[gpu_id])} tasks")

        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()

        print(f"\n🚀 Starting {self.num_gpus} GPU processes...")
        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self._worker,
                args=(gpu_id, tasks_per_gpu[gpu_id], result_queue)
            )
            p.start()
            processes.append(p)

        print(f"\n⏳ Waiting for scoring completion...")
        scored_results = {}
        with tqdm(total=len(tasks), desc="Collecting results") as pbar:
            for _ in range(len(tasks)):
                result = result_queue.get()
                scored_results[result['idx']] = result
                pbar.update(1)

        for p in processes:
            p.join()

        print(f"✅ All GPU processes completed")

        print(f"\n📝 Merging results...")
        for idx, item in enumerate(results):
            if idx in scored_results:
                item['eval_score'] = scored_results[idx]['score']
                if self.save_reasoning:
                    item['eval_reasoning'] = scored_results[idx].get('reasoning', '')

        if output_file is None:
            output_file = results_file.replace('.jsonl', '_evaluated.jsonl')

        print(f"\n💾 Saving results: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        scores = [item.get("eval_score", 0) for item in results]
        valid_scores = [s for s in scores if s >= 0]
        stats = self._compute_stats(results, valid_scores)

        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self._print_stats(stats)

        return stats

    def _extract_answer(self, generated_text: str) -> str:
        """Extract answer from <Answer> tags in generated text"""
        match = re.search(r'<Answer>(.*?)</Answer>', generated_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return generated_text.strip()

    def _worker(self, gpu_id: int, tasks: List[Dict], result_queue: Queue):
        """Single GPU worker process"""
        try:
            print(f"   GPU {gpu_id}: Initializing model...")

            torch.cuda.set_device(gpu_id)

            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{gpu_id}"
            )
            model.eval()

            print(f"   GPU {gpu_id}: Model loaded, starting scoring...")

            for i in range(0, len(tasks), self.batch_size_per_gpu):
                batch_tasks = tasks[i:i + self.batch_size_per_gpu]

                prompts = []
                for task in batch_tasks:
                    prompt = self._format_scoring_prompt(
                        tokenizer,
                        task['question'],
                        task['model_answer'],
                        task['expected_answer'],
                        task['question_type']
                    )
                    prompts.append(prompt)

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(f"cuda:{gpu_id}")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2000,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                for j, output in enumerate(outputs):
                    input_len = inputs.input_ids[j].shape[0]
                    response = tokenizer.decode(
                        output[input_len:],
                        skip_special_tokens=True
                    ).strip()

                    score = self._parse_score(response)
                    task_result = {
                        'idx': batch_tasks[j]['idx'],
                        'score': score,
                        'reasoning': response if self.save_reasoning else ''
                    }
                    result_queue.put(task_result)

            print(f"   GPU {gpu_id}: Completed all tasks")

        except Exception as e:
            print(f"   GPU {gpu_id} error: {e}")
            import traceback
            traceback.print_exc()

    def _format_scoring_prompt(
        self,
        tokenizer,
        question: str,
        model_answer: str,
        expected_answer: str,
        question_type: str
    ) -> str:
        """Format scoring prompt"""
        system_prompt = """You are an evaluator. Compare answers and give a score 0-10.

Rules:
- YES/NO: yes/true/1 = YES, no/false/0 = NO.
  Same meaning -> 10, different -> 0.
- Multiple choice: output a single option only.
  Ignore articles/case/punctuation. Match -> 10, else -> 0.
- Numeric (distance, e.g. "About X meters"):
  extract numbers. If no valid number -> 0.
    * <=10% relative error -> 10
    * <=20% -> 5-9
    * >20% -> 0
- Numeric (counting, e.g. "3 different views"):
  use only the integer.
    * Exact match -> 10
    * Otherwise -> 0
- Spatial: compare direction words (left/right,
  front/behind, above/below).
  If any axis is opposite (e.g. left vs right),
  score -> 0. Otherwise, let N = axes used in
  reference, C = correctly matched axes;
  score ≈ 10 * C / N.
- Open-ended: judge semantic match.
    * All key info correct -> 10
    * Most info correct -> 8-9
    * Partially correct -> 6-7
    * Mostly wrong -> 0-5

IMPORTANT: End response with "Score: X"
where X is 0-10."""

        user_prompt = f"""Q: {question}
Type: {question_type}
Reference: {expected_answer}
Answer: {model_answer}

Give score 0-10. Must end with "Score: X":"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_score(self, response: str) -> int:
        """Parse score from response"""
        match = re.search(r'Score:\s*(\d+)', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(10, score))

        match = re.search(r'(\d+)\s*/\s*10', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(10, score))

        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.isdigit():
                score = int(line)
                if 0 <= score <= 10:
                    return score

        numbers = re.findall(r'\b(\d+)\b', response)
        for num in reversed(numbers):
            score = int(num)
            if 0 <= score <= 10:
                return score

        return 0

    def _compute_stats(self, results: List[Dict], scores: List[int]) -> Dict:
        """Compute statistics"""
        stats = {
            "total_samples": len(results),
            "evaluated_samples": len(scores),
            "failed_evaluations": len(results) - len(scores),
            "average_score": float(np.mean(scores)) if scores else 0,
            "median_score": float(np.median(scores)) if scores else 0,
            "score_std": float(np.std(scores)) if scores else 0,
            "max_score": int(max(scores)) if scores else 0,
            "min_score": int(min(scores)) if scores else 0,
            "score_distribution": {
                "perfect_10": sum(1 for s in scores if s == 10),
                "excellent_8-9": sum(1 for s in scores if 8 <= s < 10),
                "good_6-7": sum(1 for s in scores if 6 <= s < 8),
                "fair_4-5": sum(1 for s in scores if 4 <= s < 6),
                "poor_2-3": sum(1 for s in scores if 2 <= s < 4),
                "fail_0-1": sum(1 for s in scores if 0 <= s < 2),
            },
            "by_question_type": {}
        }

        type_scores = defaultdict(list)
        for item in results:
            if item.get("eval_score", -1) >= 0:
                qtype = item.get("question_type", "unknown")
                type_scores[qtype].append(item["eval_score"])

        for qtype, scores_list in type_scores.items():
            stats["by_question_type"][qtype] = {
                "count": len(scores_list),
                "average_score": float(np.mean(scores_list))
            }

        return stats

    def _print_stats(self, stats: Dict):
        """Print statistics"""
        print("\n" + "="*70)
        print("📊 Evaluation Statistics")
        print("="*70)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Evaluated: {stats['evaluated_samples']}")
        print(f"Failed: {stats['failed_evaluations']}")
        print(f"\n⭐ Average score: {stats['average_score']:.2f}/10")
        print(f"📊 Median: {stats['median_score']:.2f}")
        print(f"📈 Std dev: {stats['score_std']:.2f}")

        print(f"\n📈 Score distribution:")
        dist = stats['score_distribution']
        total = sum(dist.values())
        for label, count in sorted(dist.items()):
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:15s}: {count:4d} ({pct:5.1f}%) {bar}")

        if stats['by_question_type']:
            print(f"\n📋 By question type:")
            for qtype, qstats in stats['by_question_type'].items():
                print(f"  {qtype:20s}: {qstats['average_score']:.2f}/10 (n={qstats['count']})")

        print("="*70)


def main():
    """Main function"""
    print("="*70)
    print("🎯 Qwen2.5-7B VLM Results Evaluator - Multi-GPU Version")
    print("="*70)
    print()

    if not torch.cuda.is_available():
        print("❌ No GPU detected")
        return

    num_gpus = torch.cuda.device_count()
    print(f"✅ Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({total_memory:.1f} GB)")

    print()

    print("="*70)
    print("📋 Configuration")
    print("="*70)

    results_file = input("Enter JSONL file path: ").strip()
    if not results_file:
        print("❌ Must provide file path")
        return

    if not os.path.exists(results_file):
        print(f"❌ File not found: {results_file}")
        return

    test_mode = input("Test mode (evaluate first 100 only)? [y/N]: ").strip().lower()
    max_samples = 100 if test_mode == 'y' else None

    print()
    print("💡 Batch size recommendations:")
    print("   - Conservative: 32 (per GPU)")
    print("   - Recommended: 64 (per GPU)")
    print("   - Aggressive: 128 (per GPU)")
    batch_input = input("Batch size per GPU [default: 64]: ").strip()
    batch_size_per_gpu = int(batch_input) if batch_input else 64

    gpu_input = input(f"Number of GPUs to use [default: {num_gpus}]: ").strip()
    use_gpus = int(gpu_input) if gpu_input else num_gpus

    output_file = results_file.replace('.jsonl', '_evaluated_qwen7b.jsonl')

    print()
    print(f"📂 Input: {results_file}")
    print(f"💾 Output: {output_file}")
    print(f"🧪 Mode: {'Test (100 samples)' if max_samples else 'Full'}")
    print(f"🚀 Config: {use_gpus} GPU x {batch_size_per_gpu} batch = {use_gpus * batch_size_per_gpu} total batch")
    print()

    confirm = input("Confirm to start? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled")
        return

    print()

    try:
        start_time = time.time()

        evaluator = MultiGPUEvaluator(
            local_model_path="Qwen/Qwen2.5-7B-Instruct",
            num_gpus=use_gpus,
            batch_size_per_gpu=batch_size_per_gpu,
            save_reasoning=False
        )

        stats = evaluator.evaluate_dataset(
            results_file=results_file,
            output_file=output_file,
            max_samples=max_samples
        )

        elapsed_time = time.time() - start_time

        print()
        print("="*70)
        print("✅ Evaluation completed!")
        print("="*70)
        print(f"📄 Results: {output_file}")
        print(f"📊 Stats: {output_file.replace('.jsonl', '_stats.json')}")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"🚀 Average speed: {stats['total_samples']/elapsed_time:.1f} samples/sec")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
