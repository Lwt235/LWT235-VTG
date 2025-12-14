#!/bin/bash
# ============================================================================
# Supervised Fine-Tuning (SFT) Training Script
# ============================================================================
# This script runs SFT training for video temporal grounding.
#
# Usage:
#   ./scripts/sft.sh [options]
#
# Options:
#   --config         Path to training config (default: configs/training/sft_config.yaml)
#   --data_config    Path to data config (default: configs/data/video_config.yaml)
#   --deepspeed      Path to DeepSpeed config (optional)
#   --output_dir     Output directory (default: ./outputs/sft)
#   --num_gpus       Number of GPUs to use (default: 1)
#
# Example:
#   ./scripts/sft.sh --num_gpus 2 --gpus 0,1 --deepspeed configs/deepspeed/ds_zero2.json
# ============================================================================

set -e

# Default values
CONFIG="configs/training/sft_config.yaml"
DATA_CONFIG="configs/data/video_config.yaml"
DEEPSPEED=""
OUTPUT_DIR="./outputs/sft"
NUM_GPUS=1
GPUS=""
MASTER_PORT=29500

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
        --gpus)
            GPUS="$2"
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

export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="8.6"

# Set visible GPUs if specified
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES=${GPUS}
    echo "Set CUDA_VISIBLE_DEVICES=${GPUS}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "============================================"
echo "SFT Training Configuration"
echo "============================================"
echo "Training Config: ${CONFIG}"
echo "Data Config: ${DATA_CONFIG}"
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
TRAIN_SCRIPT="train_sft.py"

CMD="${CMD} ${TRAIN_SCRIPT} \
    --config ${CONFIG} \
    --data_config ${DATA_CONFIG} \
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
echo "Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "============================================"
