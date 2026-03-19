# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import random
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
import random
from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import torch

def extract_number_answer(output_str):
    '''# Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    # Compare the extracted answers
    if student_answer == ground_truth:
        reward = 1.0'''
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, output_str)

    if match:
        return match.group(1).strip()
    return None
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        #print(content)
        #print('\n')
        #print(f"Solution: {sol}\n")
        reward = 0.0
        # Try symbolic verification first

        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        # if ground_truth == content:
        #     reward = 1.0
        if ground_truth.lower() == content.lower():
            reward = 1.0

    # for content, sol in zip(contents, solution):
    #     reward = 0.0
        
    #     # 提取ground truth中的分类层级列表
    #     sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    #     ground_truth_str = sol_match.group(1).strip()
    #     taxonomic_ranks = [rank.strip().lower() for rank in re.split(r' ', ground_truth_str) if rank.strip()]

    #     # 计算当前层级数量
    #     k = len(taxonomic_ranks)
        
    #     # 使用等比数列计算权重
    #     # 公比 r > 1，使得越细的层级权重越大
    #     r = 2.0  # 可以调整公比大小
        
    #     # 计算等比数列权重：a, a*r, a*r^2, ..., a*r^(k-1)
    #     # 其中 a 是首项，通过归一化条件确定
    #     weights = [r ** i for i in range(k)]  # 先计算未归一化的权重
        
    #     # 计算权重总和
    #     sum_weights = sum(weights)
        
    #     # 归一化系数：目标总奖励（1.0） / 权重总和
    #     normalize_factor = 1.0 / sum_weights
        
    #     # 计算每个层级的奖励并累加
    #     for idx, rank in enumerate(taxonomic_ranks):
    #         if rank in content.strip().lower():
    #             # 使用等比数列权重：越细的层级权重越大
    #             reward += weights[idx] * normalize_factor

    #     # 确保奖励在 [0, 1] 范围内
    #     reward = min(reward, 1.0)

        '''try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer.lower() == ground_truth.lower() or ground_truth.lower() in student_answer.lower():
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail'''
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
                #print(content)
                #print(sol)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    #print(script_args, training_args, model_args)
    #exit()
    # Get reward functions
    #reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    reward_funcs = [reward_funcs_registry['accuracy'] ]
    # Load the dataset
    #dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dataset = load_dataset('parquet', data_files=f"{script_args.dataset_name}/*.parquet", name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    #QUESTION_TEMPLATE = "{Question}\n Please output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    QUESTION_TEMPLATE = "{Question}\n Please directly output the answer."
    def make_conversation_image(example):
        #print(example['solution'])
        #print(example["problem"])
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example['problem']) },
                    ],
                },
            ],
        }



    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")



    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
