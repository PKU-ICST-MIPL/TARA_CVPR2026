#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

# Qwen3-VL-2B-Instruct (No-Thinking RFT + TARA)

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
--model_path ../CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara \
--output_file eval_results1/iNat21_plant_Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara.json

python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
--model_path ../CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara \
--output_file eval_results1/iNat21_animal_Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara.json

# Qwen2.5-VL-3B-Instruct (No-Thinking RFT + TARA)

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
--model_path ../CLS-RL/checkpoints/Qwen2.5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara \
--output_file eval_results1/iNat21_plant_Qwen2.5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara.json

python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
--model_path ../CLS-RL/checkpoints/Qwen2.5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara \
--output_file eval_results1/iNat21_animal_Qwen2.5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-tara.json