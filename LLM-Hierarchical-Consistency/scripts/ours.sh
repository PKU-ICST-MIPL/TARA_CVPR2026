
export CUDA_VISIBLE_DEVICES=0

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
--model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen2-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken \
--output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen2-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen2_5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen2_5-VL-3B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken-2814 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-final-firstgentoken-2814.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken-1414 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-final-firstgentoken-1414.json



# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-final-firstgentoken.json

# python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final-firstgentoken \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_animal_Qwen3-VL-2B-Instruct-text-visual-align-final-firstgentoken.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-final.json

# python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-final \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_animal_Qwen3-VL-2B-Instruct-text-visual-align-final.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-v2-onlylabel \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/cub_Qwen3-VL-2B-Instruct-text-visual-align-v2-onlylabel.json



# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-visual-align \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-visual-align.json

# python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/CUB200_with_similar_choice_new.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-v2 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/cub_Qwen3-VL-2B-Instruct-text-visual-align-v2.json

# python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice_new_sample10.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-v2 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/imagenet_animal_Qwen3-VL-2B-Instruct-text-visual-align-v2.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-v2_5 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-v2_5.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-1shot-fewshot-ep1-generation4-text-visual-align-v2 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plant_Qwen3-VL-2B-Instruct-text-visual-align-v2.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep1-generation4-text-visual-align-textvisuallastlayer/checkpoint-1347 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plantae_Qwen3-VL-2B-Instruct-text-visual-align-textvisuallastlayer.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align-textlastlayer/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plantae_Qwen3-VL-2B-Instruct-text-visual-align-textlastlayer.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-visual-align/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plantae_Qwen3-VL-2B-Instruct_visual_align.json

# python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/CUB200_with_similar_choice_new.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep1-generation4-text-visual-align-textvisuallastlayer/checkpoint-1347 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/cub_Qwen3-VL-2B-Instruct-text-visual-align-textvisuallastlayer.json

# python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice_new_sample10.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep1-generation4-text-visual-align-textvisuallastlayer/checkpoint-1347 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/imagenet_animal_Qwen3-VL-2B-Instruct-text-visual-align-textvisuallastlayer.json

# python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/CUB200_with_similar_choice_new.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/cub_Qwen3-VL-2B-Instruct_text_visual_align.json

# python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
# --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice_new_sample10.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/imagenet_animal_Qwen3-VL-2B-Instruct_text_visual_align.json

# python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_plantae_Qwen3-VL-2B-Instruct_text_visual_align.json

# python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-2694 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_animal_Qwen3-VL-2B-Instruct_text_visual_align.json

# # Ablation study: checkpoint-2000
# python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-2000 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_animal_Qwen3-VL-2B-Instruct_text_visual_align_checkpoint-2000.json

# # Ablation study: checkpoint-1000
# python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice_new_sample1.jsonl \
# --model_path /data1/hhlx/code/CLS-RL/checkpoints/Qwen3-VL-2B-Instruct-no-thinking-iNat21-animal-4shot-fewshot-ep2-generation4-text-visual-align/checkpoint-1000 \
# --output_file /data1/hhlx/code/LLM-Hierarchical-Consistency/eval_results/iNat21_animal_Qwen3-VL-2B-Instruct_text_visual_align_checkpoint-1000.json
