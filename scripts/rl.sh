#!/bin/bash
# ============================================================================
# Reinforcement Learning (GRPO) Training Script
# ============================================================================
# This script runs RL training for video temporal grounding using GRPO.
#
# Usage:
#   ./scripts/rl.sh [options]
#
# Options:
#   --config         Path to training config (default: configs/training/rl_config.yaml)
#   --data_config    Path to data config (default: configs/data/video_config.yaml)
#   --sft_checkpoint Path to SFT checkpoint (default: ./outputs/sft/checkpoint-best)
#   --deepspeed      Path to DeepSpeed config (optional)
#   --output_dir     Output directory (default: ./outputs/rl)
#   --num_gpus       Number of GPUs to use (default: 1)
#
# Example:
#   ./scripts/rl.sh --sft_checkpoint ./outputs/sft/checkpoint-1000 --num_gpus 4
# ============================================================================

set -e

# Default values
CONFIG="configs/training/rl_config.yaml"
DATA_CONFIG="configs/data/video_config.yaml"
SFT_CHECKPOINT="./outputs/sft/checkpoint-best"
DEEPSPEED=""
OUTPUT_DIR="./outputs/rl"
NUM_GPUS=1
MASTER_PORT=29501

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data_config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --sft_checkpoint)
            SFT_CHECKPOINT="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check SFT checkpoint exists
if [ ! -d "${SFT_CHECKPOINT}" ]; then
    echo "Warning: SFT checkpoint not found at ${SFT_CHECKPOINT}"
    echo "Please run SFT training first or specify a valid checkpoint path."
fi

# Log configuration
echo "============================================"
echo "RL (GRPO) Training Configuration"
echo "============================================"
echo "Training Config: ${CONFIG}"
echo "Data Config: ${DATA_CONFIG}"
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "DeepSpeed Config: ${DEEPSPEED:-None}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "============================================"

# Build command
if [ ${NUM_GPUS} -gt 1 ]; then
    # Multi-GPU training with torchrun
    CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT}"
else
    CMD="python"
fi

# Add training script and arguments
TRAIN_SCRIPT="train_rl.py"

CMD="${CMD} ${TRAIN_SCRIPT} \
    --config ${CONFIG} \
    --data_config ${DATA_CONFIG} \
    --model_path ${SFT_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR}"

# Add DeepSpeed if specified
if [ -n "${DEEPSPEED}" ]; then
    CMD="${CMD} --deepspeed ${DEEPSPEED}"
fi

# Run training
echo ""
echo "Running command:"
echo "${CMD}"
echo ""

${CMD}

echo ""
echo "============================================"
echo "RL Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "============================================"
