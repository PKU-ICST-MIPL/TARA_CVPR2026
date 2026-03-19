#!/bin/bash
cd src/cls-rl
DATASET="iNat21"  
MAX_PROMPT_LENGTH=1024  
NUM_TRAIN_EPOCHS=1  

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="debug_log_qwen3vl-2b-instruct-no-thinking_${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}.txt"
OUTPUT_DIR="../../checkpoints/Qwen3-VL-2B-Instruct-no-thinking-${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}/"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONPATH=./src/open_r1/trainer:$PYTHONPATH

PYTHONIOENCODING=utf-8 python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12351" \
    src/open_r1/grpo_direct.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset_name "../../data/${DATASET}-1shot-fewshots" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --run_name "Qwen3-VL-2B-Instruct-${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}" \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance

