#!/bin/bash

# Define the GPU allocation strategy
GPUS=(0 1)  # Use two GPUs
SEEDS=(11 12 13 14 15 25 26 27 28 29)  # 8 different seeds

# Ensure the logs directory exists
mkdir -p /4T/chenli/data/log/logs

# Base command with common arguments (using the same arguments as your original script)
BASE_CMD="--logdir /4T/chenli/data/log \
  --env-name sat-v0 \
  --nums-variable 50 \
  --train-problems-paths /4T/chenli/data/uf50-218/train \
  --eval-problems-paths /4T/chenli/data/uf50-218/validation \
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

# Launch tasks (each GPU runs 4 tasks)
for i in "${!SEEDS[@]}"; do
    # Determine the GPU index (the first four seeds to GPU0, the rest to GPU1)
    GPU_IDX=$((i / 5))
    SEED=${SEEDS[$i]}

    # Run the training process with the current seed; the dqn.py script will create its own timestamped log subdirectory.
    CUDA_VISIBLE_DEVICES=${GPUS[$GPU_IDX]} \
    nohup python3 dqn.py $BASE_CMD --seed $SEED > /4T/chenli/data/log/logs/seed${SEED}.log 2>&1 &
done

echo "Launched ${#SEEDS[@]} tasks. GPU0 runs seeds ${SEEDS[0]} - ${SEEDS[3]}, GPU1 runs seeds ${SEEDS[4]} - ${SEEDS[7]}."