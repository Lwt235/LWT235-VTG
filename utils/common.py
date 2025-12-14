"""
Common utility functions for Video Temporal Localization Framework.

Provides general-purpose utilities used across the framework.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic algorithms which may
            significantly impact training performance. Set to False for
            faster training when exact reproducibility is not required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if deterministic:
        # Enable deterministic behavior (may significantly impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to find the best algorithm for performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Specific device string (e.g., "cuda:0", "cpu").
                If None, auto-detect the best available device.
    
    Returns:
        torch.device instance.
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
    
    Returns:
        Dictionary with parameter counts.
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        "total": total if not trainable_only else trainable,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / (trainable + frozen) if (trainable + frozen) > 0 else 0,
    }


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a configuration file (YAML or JSON).
    
    Args:
        config_path: Path to configuration file.
    
    Returns:
        OmegaConf DictConfig object.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return OmegaConf.create(config)


def merge_configs(*configs: Union[Dict, DictConfig]) -> DictConfig:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    
    Args:
        configs: Configuration dictionaries to merge.
    
    Returns:
        Merged OmegaConf DictConfig.
    """
    merged = OmegaConf.create({})
    
    for config in configs:
        if config is not None:
            if isinstance(config, dict):
                config = OmegaConf.create(config)
            merged = OmegaConf.merge(merged, config)
    
    return merged


def save_config(config: Union[Dict, DictConfig], path: Union[str, Path]):
    """
    Save a configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format.
        precision: Decimal precision.
    
    Returns:
        Formatted number string.
    """
    for suffix, divisor in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(num) >= divisor:
            return f"{num / divisor:.{precision}f}{suffix}"
    return str(num)


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory stats.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total - reserved
    
    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": free / 1e9,
        "utilization": allocated / total if total > 0 else 0,
    }


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string to seconds.
    
    Args:
        timestamp: Timestamp string (e.g., "01:23:45.678" or "1:23.45").
    
    Returns:
        Time in seconds.
    """
    parts = timestamp.split(":")
    
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(timestamp)


def seconds_to_timestamp(seconds: float, include_hours: bool = True) -> str:
    """
    Convert seconds to timestamp string.
    
    Args:
        seconds: Time in seconds.
        include_hours: Whether to include hours in output.
    
    Returns:
        Timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"


def normalize_timestamp(
    start: float,
    end: float,
    duration: float,
) -> tuple:
    """
    Normalize timestamp to [0, 1] range.
    
    Args:
        start: Start time in seconds.
        end: End time in seconds.
        duration: Total duration in seconds.
    
    Returns:
        Tuple of (normalized_start, normalized_end).
    """
    if duration <= 0:
        raise ValueError("Duration must be positive")
    
    norm_start = max(0.0, min(1.0, start / duration))
    norm_end = max(0.0, min(1.0, end / duration))
    
    return norm_start, norm_end


def denormalize_timestamp(
    norm_start: float,
    norm_end: float,
    duration: float,
) -> tuple:
    """
    Denormalize timestamp from [0, 1] range to seconds.
    
    Args:
        norm_start: Normalized start time.
        norm_end: Normalized end time.
        duration: Total duration in seconds.
    
    Returns:
        Tuple of (start_seconds, end_seconds).
    """
    return norm_start * duration, norm_end * duration


def get_adapter_info(model_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Check if a model path contains a LoRA adapter and return adapter info.

    This function checks for the presence of `adapter_config.json` in the
    model directory, which indicates a PEFT/LoRA adapter checkpoint.

    Args:
        model_path: Path to model checkpoint directory.

    Returns:
        Adapter config dictionary if it's a LoRA adapter, None otherwise.
    """
    import json
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            return json.load(f)
    return None
