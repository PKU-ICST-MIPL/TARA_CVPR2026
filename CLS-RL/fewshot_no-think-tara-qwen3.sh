#!/bin/bash
cd src/cls-rl
DATASET="iNat21"  
MAX_PROMPT_LENGTH=1024  
NUM_TRAIN_EPOCHS=1 

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="debug_log_2b_instruct-no-thinking_${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}-tara.txt"
OUTPUT_DIR="../../checkpoints/Qwen3-VL-2B-Instruct-no-thinking-${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}-tara/"

export CUDA_VISIBLE_DEVICES=0,1 #2,3,4,5,6,7,8,9

export PYTHONPATH=./src/open_r1/trainer:$PYTHONPATH

#/data1/hhlx/code/CLS-RL/src/cls-rl/src/open_r1/trainer:$PYTHONPATH

PYTHONIOENCODING=utf-8 python -m torch.distributed.run --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12353" \
    src/open_r1/grpo_direct_tara.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/ \
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
    --run_name "Qwen3-VL-2B-Instruct-${DATASET}-1shot-fewshot-ep${NUM_TRAIN_EPOCHS}-tara" \
    --save_steps 200 \
    --save_only_model true \
    --num_generations 4