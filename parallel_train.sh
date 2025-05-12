#!/bin/bash

# Define the GPU allocation strategy
GPUS=(0 1)  # Use two GPUs
SEEDS=(11 12 13 14 15 25 26 27 28 29)  # 10 different seeds
TASKS_PER_GPU=3  # Default: 5 tasks per GPU, can be modified as needed

# Ensure the logs directory exists
mkdir -p /4T/chenli/data/log/logs

# Base command with common arguments (using the same arguments as your original script)
BASE_CMD="--logdir /4T/chenli/data/log \
  --env-name sat-v0 \
  --nums-variable 50 \
  --train-problems-paths /4T/chenli/data/flat75-180/train \
  --eval-problems-paths /4T/chenli/data/flat100-239/validation \
  --lr 0.00002 \
  --bsize 64 \
  --buffer-size 20000 \
  --eps-init 1.0 \
  --eps-final 0.01 \
  --gamma 0.99 \
  --eps-decay-steps 30000 \
  --batch-updates 50000 \
  --history-len 1 \
  --init-exploration-steps 5000 \
  --step-freq 4 \
  --target-update-freq 10 \
  --loss mse  \
  --opt adam \
  --save-freq 500 \
  --grad_clip 1 \
  --grad_clip_norm_type 2 \
  --eval-freq 1000 \
  --eval-time-limit 3600  \
  --core-steps 4 \
  --expert-exploration-prob 0.0 \
  --priority_alpha 0.5 \
  --priority_beta 0.5 \
  --e2v-aggregator sum  \
  --n_hidden 1 \
  --hidden_size 64 \
  --decoder_v_out_size 32 \
  --decoder_e_out_size 1 \
  --decoder_g_out_size 1 \
  --encoder_v_out_size 32 \
  --encoder_e_out_size 32 \
  --encoder_g_out_size 32 \
  --core_v_out_size 64 \
  --core_e_out_size 64 \
  --core_g_out_size 32 \
  --activation relu \
  --penalty_size 0.1 \
  --train_time_max_decisions_allowed 500 \
  --test_time_max_decisions_allowed 500 \
  --no_max_cap_fill_buffer \
  --lr_scheduler_gamma 1 \
  --lr_scheduler_frequency 3000 \
  --independent_block_layers 0"

# Launch tasks with dynamic allocation
for i in "${!SEEDS[@]}"; do
    # Determine the GPU index based on TASKS_PER_GPU
    GPU_IDX=$((i / TASKS_PER_GPU))
    
    # Skip if we've run out of GPUs
    if [ $GPU_IDX -ge ${#GPUS[@]} ]; then
        echo "Warning: Not enough GPUs for all seeds. Skipping seed ${SEEDS[$i]}."
        continue
    fi
    
    SEED=${SEEDS[$i]}

    # Run the training process with the current seed
    CUDA_VISIBLE_DEVICES=${GPUS[$GPU_IDX]} \
    nohup python3 dqn.py $BASE_CMD --seed $SEED > /4T/chenli/data/log/logs/seed${SEED}.log 2>&1 &
    
    echo "Started task with seed ${SEED} on GPU ${GPUS[$GPU_IDX]}"
done

# Calculate how many tasks are assigned to each GPU for the summary
declare -A TASKS_COUNT
for i in "${!SEEDS[@]}"; do
    GPU_IDX=$((i / TASKS_PER_GPU))
    if [ $GPU_IDX -lt ${#GPUS[@]} ]; then
        GPU=${GPUS[$GPU_IDX]}
        TASKS_COUNT[$GPU]=$((${TASKS_COUNT[$GPU]:-0} + 1))
    fi
done

# Print summary
echo "Launched tasks with $TASKS_PER_GPU tasks per GPU:"
for GPU in "${GPUS[@]}"; do
    COUNT=${TASKS_COUNT[$GPU]:-0}
    echo "  GPU $GPU: $COUNT tasks"
done