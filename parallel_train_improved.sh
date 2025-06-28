#!/bin/bash

# Improved parallel training script with better monitoring and error handling

# Define the GPU allocation strategy
GPUS=(0 1)  # Use two GPUs
SEEDS=(11 12 13 14) # Train 4 models (2 per GPU) with different seeds

# Check if GPUs are available
for gpu in "${GPUS[@]}"; do
    if ! nvidia-smi -i $gpu &>/dev/null; then
        echo "Error: GPU $gpu is not available!"
        exit 1
    fi
done

# Ensure the logs directory exists
mkdir -p /4T/chenli/data/log/logs

# Base command with common arguments
BASE_CMD="--logdir /4T/chenli/data/log \
  --env-name sat-v0 \
  --nums-variable 50 \
  --train-problems-paths /4T/chenli/data/uf50-218/train \
  --eval-problems-paths /4T/chenli/data/uf50-218/validation \
  --use_sat_message_passing \
  --mp_heads 8 \
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
  --hidden_size 128 \
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

# Array to store process IDs
PIDS=()

echo "Starting parallel training with ${#SEEDS[@]} processes on ${#GPUS[@]} GPUs..."
echo "GPU allocation: 2 processes per GPU"

# Launch tasks (each GPU runs 2 tasks)
for i in "${!SEEDS[@]}"; do
    # Determine the GPU index (2 tasks per GPU)
    GPU_IDX=$((i / 2))
    SEED=${SEEDS[$i]}
    
    # Create log file path
    LOG_FILE="/4T/chenli/data/log/logs/seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting training with seed $SEED on GPU ${GPUS[$GPU_IDX]}..."
    echo "Log file: $LOG_FILE"
    
    # Run the training process with the current seed
    CUDA_VISIBLE_DEVICES=${GPUS[$GPU_IDX]} \
    nohup python3 dqn.py $BASE_CMD --seed $SEED > "$LOG_FILE" 2>&1 &
    
    # Store the process ID
    PIDS+=($!)
    
    # Small delay to avoid resource conflicts
    sleep 2
done

echo ""
echo "All ${#SEEDS[@]} training processes started successfully!"
echo "Process IDs: ${PIDS[@]}"
echo "GPU0 runs seeds ${SEEDS[0]} and ${SEEDS[1]}"
echo "GPU1 runs seeds ${SEEDS[2]} and ${SEEDS[3]}"
echo ""

# Function to check if all processes are still running
check_processes() {
    local running=0
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            ((running++))
        fi
    done
    echo $running
}

# Monitor processes
echo "Monitoring training processes..."
while true; do
    running=$(check_processes)
    if [ $running -eq 0 ]; then
        echo "All training processes have completed."
        break
    fi
    
    echo "$(date): $running out of ${#PIDS[@]} processes are still running..."
    
    # Show GPU usage
    echo "Current GPU usage:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
    
    sleep 300  # Check every 5 minutes
done

# Check exit status of all processes
echo "Checking exit status of training processes..."
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    seed=${SEEDS[$i]}
    
    # Wait for specific process and get exit code
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Training with seed $seed completed successfully"
    else
        echo "✗ Training with seed $seed failed with exit code $exit_code"
    fi
done

echo ""
echo "All training tasks have finished. Check log files in /4T/chenli/data/log/logs/ for details."
