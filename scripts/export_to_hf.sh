#!/bin/bash
# ============================================================================
# Export Model to Hugging Face Format
# ============================================================================
# This script exports trained models to Hugging Face format for deployment.
#
# Usage:
#   ./scripts/export_to_hf.sh [options]
#
# Options:
#   --checkpoint     Path to checkpoint directory (required)
#   --output_dir     Output directory for exported model (required)
#   --merge_lora     Whether to merge LoRA weights (default: true)
#   --push_to_hub    Whether to push to Hugging Face Hub (default: false)
#   --hub_model_id   Model ID for Hugging Face Hub (required if push_to_hub)
#
# Example:
#   ./scripts/export_to_hf.sh --checkpoint ./outputs/rl/checkpoint-best \
#                              --output_dir ./exports/vtg-model \
#                              --merge_lora true
# ============================================================================

set -e

# Default values
CHECKPOINT=""
OUTPUT_DIR=""
MERGE_LORA="true"
PUSH_TO_HUB="false"
HUB_MODEL_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --merge_lora)
            MERGE_LORA="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB="$2"
            shift 2
            ;;
        --hub_model_id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${CHECKPOINT}" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: --output_dir is required"
    exit 1
fi

if [ "${PUSH_TO_HUB}" = "true" ] && [ -z "${HUB_MODEL_ID}" ]; then
    echo "Error: --hub_model_id is required when --push_to_hub is true"
    exit 1
fi

# Check checkpoint exists
if [ ! -d "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "============================================"
echo "Model Export Configuration"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Merge LoRA: ${MERGE_LORA}"
echo "Push to Hub: ${PUSH_TO_HUB}"
if [ "${PUSH_TO_HUB}" = "true" ]; then
    echo "Hub Model ID: ${HUB_MODEL_ID}"
fi
echo "============================================"

# Run export script
python -c "
import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

checkpoint_path = '${CHECKPOINT}'
output_dir = '${OUTPUT_DIR}'
merge_lora = '${MERGE_LORA}'.lower() == 'true'
push_to_hub = '${PUSH_TO_HUB}'.lower() == 'true'
hub_model_id = '${HUB_MODEL_ID}' if '${HUB_MODEL_ID}' else None

print(f'Loading model from {checkpoint_path}...')

# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Merge LoRA if requested and applicable
if merge_lora and isinstance(model, PeftModel):
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('LoRA weights merged successfully')

# Save to output directory
print(f'Saving model to {output_dir}...')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print('Model exported successfully!')

# Push to Hub if requested
if push_to_hub and hub_model_id:
    print(f'Pushing model to Hugging Face Hub: {hub_model_id}...')
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    processor.push_to_hub(hub_model_id)
    print(f'Model pushed to Hub: https://huggingface.co/{hub_model_id}')
"

echo ""
echo "============================================"
echo "Export complete!"
echo "Model saved to: ${OUTPUT_DIR}"
if [ "${PUSH_TO_HUB}" = "true" ]; then
    echo "Model pushed to: https://huggingface.co/${HUB_MODEL_ID}"
fi
echo "============================================"
