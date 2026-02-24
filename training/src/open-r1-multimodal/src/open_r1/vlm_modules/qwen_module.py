from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        # Support both HF model names and local paths
        model_id_lower = model_id.lower()
        if "qwen2-vl" in model_id_lower or "qwen2vl" in model_id_lower:
            model_cls = Qwen2VLForConditionalGeneration
        elif "qwen2.5-vl" in model_id_lower or "qwen25vl" in model_id_lower:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            # Default to Qwen2.5-VL if path contains "qwen" (for local paths)
            if "qwen" in model_id_lower:
                model_cls = Qwen2_5_VLForConditionalGeneration
            else:
                raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # Process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str, question_type: str = None):
        """
        Return question template based on task type and question type
        
        Args:
            task_type: Task type (rec, ic, omni_vqa, vqa, etc.)
            question_type: Question type (optional, for VQA task subtypes)
        """
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "omni_vqa" | "vqa":
                base_prompt = """You are an expert in analyzing 360° panoramic (ERP) images (2560x1280).  

Analyze the image carefully and focus on the specific objects mentioned in bounding boxes.

{Question}

Provide your reasoning based on the panoramic scene within <Reasoning> tags, 
then give your final answer within <Answer> tags."""
                
                if question_type == "true_false":
                    return f"""{base_prompt}

This is a YES/NO question.
Your <Answer> must be EXACTLY "Yes" or "No" (case-sensitive, no extra words).

Example: <Reasoning>...</Reasoning><Answer>Yes</Answer>"""
                
                elif question_type == "multiple_choice":
                    return f"""{base_prompt}

This is a MULTIPLE CHOICE question.
Your <Answer> must be EXACTLY one of the provided options.

Example: <Reasoning>...</Reasoning><Answer>AbandonedCable</Answer>"""
                
                elif question_type == "open_ended":
                    return f"""{base_prompt}

This is an OPEN-ENDED question.
Your <Answer> should be concise and direct (under 20 words).

Example: <Reasoning>...</Reasoning><Answer>The building is behind and to the right of and below the truck</Answer>"""
                
                else:
                    return f"""{base_prompt}

Example format: <Reasoning>...</Reasoning><Answer>your answer</Answer>
Your <Reasoning> should contain detailed analysis.
Your <Answer> should be concise and direct."""
            
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def format_reward_vqa(completions, **kwargs):
        """Check if VQA output contains <Reasoning> and <Answer> tags"""
        import re
        import os
        from datetime import datetime
        
        pattern = r"<Reasoning>.*?</Reasoning>\s*<Answer>.*?</Answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        
        rewards = [1.0 if match else 0.0 for match in matches]
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug_format_vqa.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            for i, (content, reward) in enumerate(zip(completion_contents, rewards)):
                if reward < 1.0:
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Format Check Failed -------------\n")
                        f.write(f"Content: {content[:500]}...\n")
                        f.write(f"Has <Reasoning>: {'<Reasoning>' in content}\n")
                        f.write(f"Has </Reasoning>: {'</Reasoning>' in content}\n")
                        f.write(f"Has <Answer>: {'<Answer>' in content}\n")
                        f.write(f"Has </Answer>: {'</Answer>' in content}\n\n")
        
        return rewards

    @staticmethod
    def normalize_entity(text):
        """Normalize entity: remove articles, spaces, punctuation, convert to lowercase"""
        import re
        # Remove articles
        text = re.sub(r'\b(the|a|an)\b', '', text, flags=re.IGNORECASE)
        # Remove spaces and punctuation
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        # Convert to lowercase
        return text.lower()
    
    @staticmethod
    def extract_choice_letter(text):
        """Extract choice letter (A/B/C/D/E) from text"""
        import re
        
        # Pattern 1: Option format at line start "A. xxx" or "A) xxx" or "A: xxx"
        match = re.match(r'^([A-E])[.):]\s*', text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        match = re.search(r'(?:answer|option)\s*:?\s*is\s*([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1).upper()
        
        return None
    
    @staticmethod
    def extract_first_subject(text):
        """Extract first subject from sentence for MCQ answer extraction"""
        import re
        
        # Tokenize (simple split by space)
        words = text.split()
        
        # Strategy 1: Short text returned directly (<=4 words considered as direct option)
        if len(words) <= 4:
            return text.strip()
        
                
        # Strategy 2: Long text extract subject (>4 words considered as sentence)
        # Find first noun phrase: content from start to before first verb/preposition
        # Common verbs: is, are, was, were, has, have, can, will, etc.
        # Common prepositions: in, on, at, to, from, with, by, for, of, near, behind, above, below, etc.
        # Key: also recognize prepositions like with/of so "blue vehicle with stripes" truncates at "with"
        
        verb_prep_pattern = r'\b(is|are|was|were|has|have|had|can|could|will|would|should|may|might|must|do|does|did|in|on|at|to|from|with|by|for|of|near|behind|above|below|under|over|between|among|through|during|before|after|since|until)\b'
        
        # Find position of first verb/preposition
        match = re.search(verb_prep_pattern, text, re.IGNORECASE)
        if match:
            # Extract content before verb/preposition as subject
            subject = text[:match.start()].strip()
            # If subject is empty (sentence starts with verb), return original text
            return subject if subject else text.strip()
        
        # If no verb/preposition found, return original text
        return text.strip()
    
    @staticmethod
    def extract_number_with_unit(text):
        """Extract numeric value with unit and convert to meters"""
        import re
        import re
        text_lower = text.lower()
        
        # Mapping from English numbers to Arabic numerals
        english_numbers = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        pattern = r'(\d+\.?\d*)\s*(meter|metre|meters|metres|m|kilometer|kilometre|kilometers|kilometres|km|centimeter|centimetre|centimeters|centimetres|cm|millimeter|millimetre|millimeters|millimetres|mm|feet|foot|ft|inch|inches|in)'
        match = re.search(pattern, text_lower)
        
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2)
                
                unit_conversions = {
                    'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0, 'm': 1.0,
                    'kilometer': 1000.0, 'kilometre': 1000.0, 'kilometers': 1000.0, 'kilometres': 1000.0, 'km': 1000.0,
                    'centimeter': 0.01, 'centimetre': 0.01, 'centimeters': 0.01, 'centimetres': 0.01, 'cm': 0.01,
                    'millimeter': 0.001, 'millimetre': 0.001, 'millimeters': 0.001, 'millimetres': 0.001, 'mm': 0.001,
                    'feet': 0.3048, 'foot': 0.3048, 'ft': 0.3048,
                    'inch': 0.0254, 'inches': 0.0254, 'in': 0.0254,
                }
                
                multiplier = unit_conversions.get(unit, 1.0)
                return value * multiplier
            except:
                pass
        
        num_match = re.search(r'(\d+\.?\d*)', text_lower)
        if num_match:
            try:
                return float(num_match.group(1))
            except:
                pass
        
        for word, num in english_numbers.items():
            if re.search(r'\b' + word + r'\b', text_lower):
                return float(num)
        
        return None
    
    @staticmethod
    def extract_directions(text):
        """Extract 3D spatial direction words with synonym normalization"""
        import re
        
        # Synonym mapping: all synonyms are mapped to standard words
        synonym_map = {
            # front variants
            'front': 'front', 'forward': 'front', 'ahead': 'front', 'in front of': 'front',
            # back variants
            'back': 'back', 'behind': 'back', 'rear': 'back', 'at the back': 'back',
            # left variants
            'left': 'left', 'leftward': 'left', 'to the left': 'left', 'on the left': 'left',
            # right variants
            'right': 'right', 'rightward': 'right', 'to the right': 'right', 'on the right': 'right',
            # up variants
            'above': 'up', 'up': 'up', 'upper': 'up', 'over': 'up', 'top': 'up', 'higher': 'up',
            # down variants
            'below': 'down', 'down': 'down', 'under': 'down', 'lower': 'down', 'beneath': 'down', 'bottom': 'down'
        }
        
        found = {'front_back': None, 'left_right': None, 'up_down': None}
        text_lower = text.lower()
        
        # Sort by length descending (match longer phrases first, e.g., "in front of")
        sorted_keywords = sorted(synonym_map.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Use word boundary matching
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                normalized = synonym_map[keyword]
                
                # Fill corresponding dimension based on normalized result
                if normalized in ['front', 'back']:
                    if found['front_back'] is None:  # Only record the first occurrence
                        found['front_back'] = normalized
                elif normalized in ['left', 'right']:
                    if found['left_right'] is None:
                        found['left_right'] = normalized
                elif normalized in ['up', 'down']:
                    if found['up_down'] is None:
                        found['up_down'] = normalized
        
        return found

    @staticmethod
    def accuracy_reward_vqa(completions, **kwargs):
        """VQA accuracy reward function with multiple matching strategies"""
        import re
        import os
        from datetime import datetime
        
        try:
            from Levenshtein import ratio
        except ImportError:
            print("⚠️ Warning: python-Levenshtein not installed, using exact matching only")
            ratio = lambda a, b: 1.0 if a == b else 0.0
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        reward_methods = kwargs.get('accu_reward_method', ['default'] * len(contents))
        if not isinstance(reward_methods, list):
            reward_methods = [reward_methods] * len(contents)
        
        expected_answers = kwargs.get('expected_answer', [''] * len(contents))
        if not isinstance(expected_answers, list):
            expected_answers = [expected_answers] * len(contents)
        
        answer_pattern = r'<Answer>(.*?)</Answer>'
        
        for idx, (content, sol, method) in enumerate(zip(contents, expected_answers, reward_methods)):
            content_match = re.search(answer_pattern, content, re.DOTALL | re.IGNORECASE)
            if not content_match:
                rewards.append(0.0)
                continue
            
            student_answer = content_match.group(1).strip()
            
            sol_match = re.search(answer_pattern, str(sol), re.DOTALL | re.IGNORECASE)
            ground_truth = sol_match.group(1).strip() if sol_match else str(sol).strip()
            
            reward = 0.0
            
            if method == 'yes_no':
                student_norm = student_answer.lower().strip()
                truth_norm = ground_truth.lower().strip()
                reward = 1.0 if student_norm == truth_norm else 0.0
            
            elif method == 'mcq':
                student_subject = Qwen2VLModule.extract_first_subject(student_answer)
                truth_subject = Qwen2VLModule.extract_first_subject(ground_truth)
                
                student_norm = Qwen2VLModule.normalize_entity(student_subject)
                truth_norm = Qwen2VLModule.normalize_entity(truth_subject)
                
                reward = 1.0 if student_norm == truth_norm else 0.0
            
            elif method == 'distance':
                student_num = Qwen2VLModule.extract_number_with_unit(student_answer)
                truth_num = Qwen2VLModule.extract_number_with_unit(ground_truth)
                
                if student_num is not None and truth_num is not None and truth_num > 0:
                    error_rate = abs(student_num - truth_num) / truth_num
                    
                    if error_rate <= 0.10:
                        reward = 1.0
                    elif error_rate <= 0.20:
                        reward = 0.5
                    else:
                        reward = 0.0
                else:
                    student_norm = student_answer.lower().strip()
                    truth_norm = ground_truth.lower().strip()
                    similarity = ratio(student_norm, truth_norm)
                    reward = 1.0 if similarity > 0.75 else 0.0
            
            elif method == 'spatial':
                student_dirs = Qwen2VLModule.extract_directions(student_answer)
                truth_dirs = Qwen2VLModule.extract_directions(ground_truth)
                
                dimension_scores = []
                for dimension in ['front_back', 'left_right', 'up_down']:
                    truth_dim = truth_dirs[dimension]
                    student_dim = student_dirs[dimension]
                    
                    if truth_dim:
                        if student_dim == truth_dim:
                            dimension_scores.append(1.0)
                        else:
                            dimension_scores.append(0.0)
                
                if dimension_scores:
                    reward = sum(dimension_scores) / len(dimension_scores)
                else:
                    reward = 0.0
            
            elif method == 'counting':
                student_num = Qwen2VLModule.extract_number_with_unit(student_answer)
                truth_num = Qwen2VLModule.extract_number_with_unit(ground_truth)
                
                if student_num is not None and truth_num is not None:
                    reward = 1.0 if abs(student_num - truth_num) < 0.01 else 0.0
                else:
                    reward = 0.0
            
            else:
                student_norm = re.sub(r'\b(the|a|an)\b', '', student_answer, flags=re.IGNORECASE)
                truth_norm = re.sub(r'\b(the|a|an)\b', '', ground_truth, flags=re.IGNORECASE)
                student_norm = ' '.join(student_norm.split()).lower().strip()
                truth_norm = ' '.join(truth_norm.split()).lower().strip()
                
                if student_norm == truth_norm:
                    reward = 1.0
                else:
                    similarity = ratio(student_norm, truth_norm)
                    reward = 1.0 if similarity > 0.75 else 0.0
            
            rewards.append(reward)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "debug_accuracy_vqa.log")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} [{method}] Accuracy: {reward:.2f} -------------\n")
                    f.write(f"Question: {kwargs.get('problem', ['N/A'])[0] if 'problem' in kwargs else 'N/A'}\n")
                    f.write(f"Student Answer: {student_answer}\n")
                    f.write(f"Ground Truth: {ground_truth}\n")
                    f.write(f"Reward: {reward}\n\n")
        
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "omni_vqa" | "vqa":
                    return Qwen2VLModule.accuracy_reward_vqa
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "omni_vqa" | "vqa":
                    return Qwen2VLModule.format_reward_vqa
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func in ["format_tagged", "reasoning_sim", "answer_sim"]:
            return None
        else:
            raise ValueError(f"Unsupported reward function: {func}")
