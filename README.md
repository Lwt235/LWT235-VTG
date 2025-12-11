# Qwen3-VL-GRPO-VTG

A robust and extensible deep learning framework for **Video Temporal Grounding (VTG)** using Qwen3-VL as the base Large Vision-Language Model (LVLM).

## Overview

This framework provides a complete pipeline for training video temporal localization models, including:

- **Supervised Fine-Tuning (SFT)**: Align LVLM video-text representations with temporal grounding tasks
- **Reinforcement Learning (GRPO/R1)**: Optimize temporal boundary quality using reward functions

### Key Features

- 🎬 Video temporal grounding: Given a text query, predict start and end times of relevant video segments
- 🔧 LoRA-based fine-tuning with support for lm_head and embed_tokens
- 📊 Multiple reward functions: Temporal IoU, Segment Overlap, Step Consistency
- 🚀 DeepSpeed integration for efficient multi-GPU training
- 📦 Easy export to Hugging Face format

## Installation

### Requirements

- Python 3.11+
- CUDA 12.4 compatible GPU
- At least 16GB GPU memory for Qwen3-VL-4B (8GB for Qwen3-VL-2B)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Lwt235/Qwen3-VL-GRPO-VTG.git
cd Qwen3-VL-GRPO-VTG

# Run the installation script
chmod +x env/install.sh
./env/install.sh

# Or install manually
conda create -n vtg_env python=3.11 -y
conda activate vtg_env
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r env/requirements.txt
```

## Project Structure

```
Qwen3-VL-GRPO-VTG/
├── env/                          # Environment setup
│   ├── install.sh               # Installation script
│   └── requirements.txt         # Python dependencies
├── configs/                      # Configuration files
│   ├── data/
│   │   └── video_config.yaml    # Data and video processing config
│   ├── training/
│   │   ├── sft_config.yaml      # SFT training config
│   │   └── rl_config.yaml       # RL training config
│   └── deepspeed/
│       ├── ds_zero2.json        # DeepSpeed ZeRO-2 config
│       └── ds_zero3_offload.json # DeepSpeed ZeRO-3 with offload
├── utils/                        # Utility modules
│   ├── data_validation.py       # Data consistency validation
│   ├── logging_utils.py         # Logging utilities
│   └── common.py                # Common utilities
├── datasets/                     # Dataset modules
│   ├── video_dataset.py         # Video dataset classes
│   └── collate_fns.py           # Collate functions
├── trainers/                     # Training modules
│   ├── sft_trainer.py           # SFT trainer
│   └── rl_trainer.py            # RL/GRPO trainer
├── rewards/                      # Reward functions
│   ├── temporal_iou.py          # Temporal IoU reward
│   ├── segment_overlap.py       # Segment overlap reward
│   ├── step_consistency.py      # Step consistency reward
│   └── reward_registry.py       # Reward combination
├── inference/                    # Inference modules
│   └── video_infer.py           # Video inference with visualization
├── scripts/                      # Training scripts
│   ├── sft.sh                   # SFT training script
│   ├── rl.sh                    # RL training script
│   └── export_to_hf.sh          # Model export script
├── data/                         # Data directory
│   ├── videos/                  # Video files
│   └── annotations/             # Annotation files
├── ckpts/                        # Model checkpoints
├── outputs/                      # Training outputs
│   ├── sft/
│   ├── rl/
│   ├── logs/
│   └── metrics/
├── train_sft.py                  # SFT training entry point
└── train_rl.py                   # RL training entry point
```

## Data Format

### Annotation File Format (JSONL)

Each line in the annotation file should be a JSON object with the following fields:

```json
{
  "video": "./data/videos/example.mp4",
  "duration": 24.9,
  "timestamp": [11.3, 15.0],
  "sentence": "the paper bill has an image of a woman holding up a paper",
  "video_start": null,
  "video_end": null,
  "kwargs": {"difficulty": "easy", "qid": "sample_001"}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Path to video file |
| `duration` | float | Video duration in seconds |
| `timestamp` | [float, float] | Start and end time of the target segment |
| `sentence` | string | Text query describing the video segment |
| `video_start` | float or null | Optional: trim video from this time |
| `video_end` | float or null | Optional: trim video to this time |
| `kwargs` | dict | Additional metadata (difficulty, qid, etc.) |

## Training

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
# Using the script
./scripts/sft.sh --num_gpus 4 --deepspeed configs/deepspeed/ds_zero2.json

# Or directly with Python
python train_sft.py \
    --config configs/training/sft_config.yaml \
    --data_config configs/data/video_config.yaml \
    --output_dir ./outputs/sft
```

### Stage 2: Reinforcement Learning (GRPO)

```bash
# Using the script
./scripts/rl.sh --sft_checkpoint ./outputs/sft/checkpoint-best --num_gpus 4

# Or directly with Python
python train_rl.py \
    --config configs/training/rl_config.yaml \
    --data_config configs/data/video_config.yaml \
    --model_path ./outputs/sft/checkpoint-best \
    --output_dir ./outputs/rl
```

### Key Training Parameters

#### SFT Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Base learning rate | 1e-4 |
| `mm_projector_lr` | Multimodal projector learning rate | 1e-5 |
| `vision_tower_lr` | Vision encoder learning rate | 0.0 (frozen) |
| `head_lr` | Task head learning rate | 1e-4 |
| `per_device_train_batch_size` | Batch size per GPU | 2 |
| `gradient_accumulation_steps` | Gradient accumulation | 8 |
| `num_train_epochs` | Training epochs | 3 |

#### RL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_generations` | Generations per prompt | 4 |
| `temperature` | Generation temperature | 0.7 |
| `kl_coef` | KL divergence coefficient | 0.05 |
| `clip_range` | PPO clip range | 0.2 |

## Reward Functions

### Temporal IoU

Measures the overlap between predicted and ground truth segments:

```python
from rewards import TemporalIoU

reward_fn = TemporalIoU(scale=1.0)
rewards = reward_fn.compute_from_text(
    predictions=["0.25 to 0.75"],
    ground_truths=[(0.3, 0.7)],
)
```

### Segment Overlap

Measures coverage of ground truth segment:

```python
from rewards import SegmentOverlap

reward_fn = SegmentOverlap(mode="recall")
```

### Composite Reward

Combine multiple rewards with weights:

```python
from rewards import CompositeReward

reward = CompositeReward({
    "temporal_iou": {"weight": 1.0},
    "segment_overlap": {"weight": 0.5, "mode": "recall"},
})
```

## Inference

### Python API

```python
from inference import VideoTemporalInference

# Initialize
engine = VideoTemporalInference(
    model_path="./outputs/rl/checkpoint-final",
)

# Predict
result = engine.predict(
    video_path="./data/videos/example.mp4",
    query="the person opens the door",
)

print(f"Segment: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
```

### Command Line

```bash
python inference/video_infer.py \
    --model_path ./outputs/rl/checkpoint-final \
    --video ./data/videos/example.mp4 \
    --query "the person opens the door" \
    --output ./visualization.png
```

### Visualization

```python
from inference import visualize_temporal_grounding

visualize_temporal_grounding(
    video_path="./data/videos/example.mp4",
    predictions=[result],
    ground_truths=[(10.0, 15.0)],
    output_path="./visualization.png",
)
```

## Model Export

Export trained models to Hugging Face format:

```bash
./scripts/export_to_hf.sh \
    --checkpoint ./outputs/rl/checkpoint-final \
    --output_dir ./exports/vtg-model \
    --merge_lora true \
    --push_to_hub false
```

## Configuration

### Model Configuration

```yaml
model:
  name_or_path: "Qwen/Qwen3-VL-4B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

lora:
  enabled: true
  r: 64
  lora_alpha: 128
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "lm_head"
  modules_to_save:
    - "embed_tokens"
    - "lm_head"
```

### DeepSpeed Configuration

The framework supports DeepSpeed ZeRO stages 2 and 3:

- `ds_zero2.json`: Recommended for multi-GPU training
- `ds_zero3_offload.json`: For memory-constrained environments

## Extension Points

This framework is designed for extensibility:

### Adding New Tasks

Extend the dataset classes in `datasets/video_dataset.py`:

```python
class VideoQADataset(VideoTemporalDataset):
    """Dataset for Video Question Answering."""
    pass
```

### Adding New Rewards

Register new reward functions:

```python
from rewards import register_reward

@register_reward("my_reward")
class MyReward(nn.Module):
    def forward(self, pred, gt):
        # Custom reward logic
        pass
```

### Replacing Vision Encoder

The vision encoder can be replaced by modifying the model loading in trainers.

## Citation

If you use this framework, please cite:

```bibtex
@misc{qwen3-vl-grpo-vtg,
  title={Qwen3-VL-GRPO-VTG: Video Temporal Grounding with GRPO},
  author={Lwt235},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Lwt235/Qwen3-VL-GRPO-VTG}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base vision-language model
- [TRL](https://github.com/huggingface/trl) for reinforcement learning training
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for distributed training
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning