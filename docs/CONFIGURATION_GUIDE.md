# Configuration Guide

This guide explains how to configure the LWT235-VTG framework for your specific environment.

## Table of Contents

1. [Data Configuration](#data-configuration)
2. [Training Configuration](#training-configuration)
3. [Environment Setup](#environment-setup)
4. [Common Configuration Examples](#common-configuration-examples)

---

## Data Configuration

### Data Format

The framework uses JSONL (JSON Lines) format for annotation files. Each line is a valid JSON object with the following structure:

```json
{
  "video": "./data/videos/timerft_data/_0yiT0hhCCM_00:06:44:200_00:07:09:100.mp4",
  "duration": 24.900000000000034,
  "timestamp": [11.3, 15.0],
  "sentence": "the paper bill has an image of a woman holding up a paper",
  "qid": "my|internvid-vtime|_0yiT0hhCCM|the paper bill has an image of a woman holding up a paper",
  "video_start": 404.2,
  "video_end": 429.1,
  "difficulty": 24.650233177881407,
  "pred": [0.0, 15.01]
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Path to video file (relative to `video_dir` or absolute) |
| `duration` | float | Video duration in seconds |
| `timestamp` | [float, float] | Start and end time of the target segment |
| `sentence` | string | Text query describing the video segment |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `video_start` | float | Absolute start time for video trimming |
| `video_end` | float | Absolute end time for video trimming |
| `qid` | string | Unique query identifier |
| `difficulty` | float | Sample difficulty score |
| `pred` | [float, float] | Model predictions for evaluation |

> **Note**: You can add any additional metadata fields directly to the JSON object. They will be automatically collected in the `metadata` field during data loading.

### Modifying Data Configuration

Edit `configs/data/video_config.yaml`:

```yaml
dataset:
  name: "video_temporal_grounding"
  data_root: "./data"                              # Root directory for data
  annotation_file: "./data/annotations/train.jsonl" # Path to training annotations
  video_dir: "./data/videos"                       # Base directory for video files

video:
  max_frames: 32                    # Maximum frames to sample per video
  sampling_strategy: "uniform"      # Frame sampling: uniform, random, keyframe
  resolution:
    height: 384                     # Input resolution height
    width: 384                      # Input resolution width
  fps: 0                            # Frame rate (0 = use original)

temporal:
  num_bins: 100                     # Number of temporal bins for discretization
  use_relative_timestamps: true     # Use normalized timestamps (0-1)
  use_temporal_tokens: false        # Use temporal tokens (<0>~<999>)

dataloader:
  batch_size: 4                     # Batch size per GPU
  num_workers: 4                    # Data loading workers
  prefetch_factor: 2                # Prefetch batches
  pin_memory: true                  # Pin memory for faster GPU transfer

validation:
  annotation_file: "./data/annotations/val.jsonl"  # Validation annotations
  batch_size: 8                     # Validation batch size
```

---

## Training Configuration

### SFT Configuration

Edit `configs/training/sft_config.yaml`:

```yaml
model:
  name_or_path: "Qwen/Qwen3-VL-4B-Instruct"  # Base model
  torch_dtype: "bfloat16"                     # Model precision
  attn_implementation: "flash_attention_2"    # Attention implementation

lora:
  enabled: true                               # Enable LoRA
  r: 64                                       # LoRA rank
  lora_alpha: 128                             # LoRA alpha
  target_modules:                             # Modules to apply LoRA
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"

training:
  learning_rate: 1e-4                         # Base learning rate
  per_device_train_batch_size: 2              # Batch size per GPU
  gradient_accumulation_steps: 8              # Gradient accumulation
  num_train_epochs: 3                         # Training epochs
  warmup_ratio: 0.03                          # Warmup ratio
  weight_decay: 0.01                          # Weight decay
  output_dir: "./outputs/sft"                 # Output directory
```

### RL Configuration

Edit `configs/training/rl_config.yaml`:

```yaml
rl:
  algorithm: "grpo"                           # RL algorithm: grpo, r1
  num_generations: 4                          # Generations per prompt
  temperature: 0.7                            # Generation temperature
  kl_coef: 0.05                               # KL divergence coefficient
  clip_range: 0.2                             # PPO clip range

rewards:
  - name: "temporal_iou"
    weight: 1.0
  - name: "segment_overlap"
    weight: 0.5
    mode: "recall"
```

---

## Environment Setup

### Step 1: Create Python Environment

```bash
# Create conda environment
conda create -n vtg_env python=3.11 -y
conda activate vtg_env

# Install PyTorch with CUDA support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r env/requirements.txt
```

### Step 2: Configure Paths

Update paths in configuration files based on your environment:

1. **Data paths** in `configs/data/video_config.yaml`:
   ```yaml
   dataset:
     data_root: "/path/to/your/data"
     annotation_file: "/path/to/your/annotations/train.jsonl"
     video_dir: "/path/to/your/videos"
   ```

2. **Model paths** in training configs:
   ```yaml
   model:
     name_or_path: "/path/to/model"  # or Hugging Face model ID
   ```

3. **Output paths**:
   ```yaml
   training:
     output_dir: "/path/to/outputs"
   ```

### Step 3: Configure Hardware Settings

Adjust based on your GPU memory:

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB | `batch_size: 1`, `gradient_accumulation: 16`, Use Qwen3-VL-2B |
| 16GB | `batch_size: 2`, `gradient_accumulation: 8`, Use Qwen3-VL-4B |
| 24GB | `batch_size: 4`, `gradient_accumulation: 4`, Use Qwen3-VL-4B |
| 80GB (A100) | `batch_size: 8`, `gradient_accumulation: 2` |

For multi-GPU training with limited memory, use DeepSpeed ZeRO-3:

```bash
./scripts/sft.sh --deepspeed configs/deepspeed/ds_zero3_offload.json
```

---

## Common Configuration Examples

### Example 1: Custom Dataset with Different Paths

```yaml
# configs/data/my_dataset.yaml
dataset:
  name: "my_temporal_grounding"
  data_root: "/home/user/my_data"
  annotation_file: "/home/user/my_data/train.jsonl"
  video_dir: "/home/user/my_data/videos"

validation:
  annotation_file: "/home/user/my_data/val.jsonl"
```

Run training with custom config:
```bash
python train_sft.py --data_config configs/data/my_dataset.yaml
```

### Example 2: Enable Temporal Tokens

```yaml
temporal:
  use_temporal_tokens: true         # Enable temporal tokens
  embedding_init_strategy: "sinusoidal"  # Token initialization
```

### Example 3: Memory-Constrained Environment

```yaml
# Reduce memory usage
video:
  max_frames: 16                    # Reduce frames
  resolution:
    height: 256
    width: 256

dataloader:
  batch_size: 1
  num_workers: 2

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### Example 4: Fast Prototyping

```yaml
# Quick training for testing
training:
  num_train_epochs: 1
  max_steps: 100
  save_steps: 50
  eval_steps: 25
  logging_steps: 10
```

### Example 5: Duration-Based Dynamic Batching

The framework supports duration-based batch sampling to stabilize GPU memory usage when videos have varying lengths. This is especially useful for datasets with diverse video durations.

```yaml
# configs/data/video_config.yaml
duration_batching:
  enabled: true                        # Enable duration-based batching
  target_batch_duration: 1200.0        # Target total duration (seconds) per batch
  max_batch_size: null                 # Optional: Maximum videos per batch
  min_batch_size: 1                    # Minimum videos per batch
  drop_last: false                     # Whether to drop incomplete batches
```

**How it works:**
1. Videos are sorted by duration to group similar-length videos together
2. Batches are formed using greedy bin-packing until target duration is reached
3. This ensures consistent GPU memory usage across batches
4. In multi-GPU training, batches are automatically distributed across GPUs

**Benefits:**
- More stable GPU memory usage with varying video lengths
- Better GPU utilization compared to fixed batch sizes
- Automatic load balancing in distributed training

**Multi-GPU Support:**

Duration-based batching automatically supports distributed training:

```bash
# Single GPU training
python train_sft.py --config configs/training/sft_config.yaml \
                    --data_config configs/data/video_config.yaml

# Multi-GPU training (automatically detected)
torchrun --nproc_per_node=4 train_sft.py \
    --config configs/training/sft_config.yaml \
    --data_config configs/data/video_config.yaml

# Or with DeepSpeed
deepspeed --num_gpus=4 train_sft.py \
    --config configs/training/sft_config.yaml \
    --data_config configs/data/video_config.yaml \
    --deepspeed configs/deepspeed/ds_zero2.json
```

**Batch Logging:**

When duration batching is enabled, the framework automatically logs batch statistics:
- Average, minimum, and maximum batch sizes per logging interval
- Training configuration (batch size, accumulation steps, world size)
- Duration batching parameters at training start

Example log output:
```
DurationBasedBatchSampler initialized: samples=1000, target_duration=1200.0s, avg_sample_duration=45.2s, expected_batch_size=26.5
Duration range: min=10.5s, max=180.3s
Distributed training: 4 GPUs, rank 0
Step 10: batch_size avg=24.3, min=20, max=28
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `batch_size` and increase `gradient_accumulation_steps`
   - Reduce `max_frames` in video config
   - Use DeepSpeed ZeRO-3 with CPU offload

2. **Video Loading Errors**:
   - Verify video paths are correct (relative to `video_dir`)
   - Check video file integrity
   - Ensure required video codecs are installed

3. **Slow Training**:
   - Increase `num_workers` in dataloader
   - Enable `pin_memory: true`
   - Use `flash_attention_2` if available

4. **Data Validation Errors**:
   - Run validation script: `python -m utils.data_validation --file your_data.jsonl`
   - Ensure all required fields are present
   - Check timestamp values are within duration

For more help, please open an issue on GitHub.
