# Fix for generation_config.json Loading Error with LoRA Adapters

## Problem

When using `use_temporal_tokens=True` with a LoRA adapter checkpoint for inference, the following error occurs:

```
OSError: outputs/sft/checkpoint-final does not appear to have a file named generation_config.json
```

## Root Cause

The issue is in `inference/video_infer.py` at line 207-209. The code attempts to load `generation_config.json` from the LoRA adapter directory, but LoRA adapters typically don't contain this file - it should be loaded from the base model instead.

### Code Before Fix

```python
# Default generation config
self.generation_config = GenerationConfig.from_pretrained(
    model_path_str  # Always uses the model_path, even for LoRA adapters
)
```

When `model_path_str` points to a LoRA adapter directory (e.g., `outputs/sft/checkpoint-final`), this directory doesn't have `generation_config.json`, causing the error.

## Solution

The fix ensures that when a LoRA adapter is detected, the `generation_config.json` is loaded from the base model path instead of the adapter path.

### Code After Fix

```python
# Default generation config
# Load from base model if using LoRA adapter, otherwise from model_path
config_path = base_model_path if is_lora_adapter and base_model_path else model_path_str
self.generation_config = GenerationConfig.from_pretrained(
    config_path
)
```

### Changes Made

1. **Line 113**: Initialize `base_model_path = None` at the start of `_load_model()`
2. **Lines 207-212**: Select the correct path for loading generation config:
   - For LoRA adapters: Use `base_model_path` (e.g., `./ckpts/Qwen3-VL-2B-Instruct`)
   - For non-LoRA models: Use `model_path_str` (the original model path)

## How It Works

### LoRA Adapter Loading Flow

1. **Detect LoRA Adapter**: Check for `adapter_config.json` in the model directory
2. **Extract Base Model Path**: Read `base_model_name_or_path` from adapter config
3. **Load Components**:
   - Processor/Tokenizer: Load from adapter directory (contains tokenizer with temporal tokens if trained with them)
   - Base Model: Load from base model path
   - Temporal Tokens: Add to tokenizer if enabled
   - LoRA Weights: Load from adapter directory
4. **Load Generation Config**: **Now correctly loads from base model path** ✓

### File Structure

```
./ckpts/Qwen3-VL-2B-Instruct/      # Base model
├── config.json
├── generation_config.json          # ✓ Contains generation config
├── model.safetensors
└── ...

./outputs/sft/checkpoint-final/     # LoRA adapter
├── adapter_config.json             # Points to base model
├── adapter_model.safetensors       # LoRA weights
├── tokenizer.json                  # Tokenizer (may include temporal tokens)
└── ...                             # ✗ No generation_config.json
```

## Testing

A test case has been added in `tests/test_inference.py` to verify that:

1. When a LoRA adapter is used, `GenerationConfig.from_pretrained()` is called with the base model path
2. When a non-LoRA model is used, `GenerationConfig.from_pretrained()` is called with the model path

## Usage Example

Now the following code works correctly:

```python
from inference import VideoTemporalInference

# This now works when outputs/sft/checkpoint-final is a LoRA adapter
engine = VideoTemporalInference(
    model_path="./outputs/sft/checkpoint-final",  # LoRA adapter
    use_temporal_tokens=True,  # Enable temporal tokens
)

result = engine.predict(
    video_path="./data/videos/example.mp4",
    query="the paper bill has an image of a woman holding up a paper",
    video_start=404.2,
    video_end=429.1,
    duration=24.9,
)

print(f"Segment: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
```

## Related Files

- `inference/video_infer.py`: Main fix
- `tests/test_inference.py`: Test coverage
- `utils/common.py`: `get_adapter_info()` utility function

## Benefits

1. **Fixes the reported error**: LoRA adapter inference now works with temporal tokens
2. **Maintains backward compatibility**: Non-LoRA models work as before
3. **Minimal changes**: Only 4 lines modified
4. **Proper error handling**: Still validates that base_model_path exists in adapter config
