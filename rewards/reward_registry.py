"""
Reward Registry for Video Temporal Localization.

Provides a registry for reward functions and composite reward computation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from utils.logging_utils import get_logger
from .temporal_iou import TemporalIoU
from .segment_overlap import SegmentOverlap
from .step_consistency import StepConsistency

logger = get_logger(__name__)


# Global registry
_REWARD_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_reward(name: str) -> Callable:
    """
    Decorator to register a reward class.
    
    Args:
        name: Name to register the reward under.
    
    Returns:
        Decorator function.
    
    Example:
        @register_reward("my_reward")
        class MyReward(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        _REWARD_REGISTRY[name] = cls
        return cls
    return decorator


def get_reward_class(name: str) -> Type[nn.Module]:
    """
    Get a registered reward class by name.
    
    Args:
        name: Registered name of the reward.
    
    Returns:
        Reward class.
    
    Raises:
        KeyError: If reward is not registered.
    """
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Reward '{name}' not found. Available: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name]


class RewardRegistry:
    """
    Registry for managing reward functions.
    
    Supports registration, lookup, and instantiation of reward modules.
    """
    
    def __init__(self):
        """Initialize the registry with default rewards."""
        self._rewards: Dict[str, Type[nn.Module]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register built-in reward functions."""
        self.register("temporal_iou", TemporalIoU)
        self.register("segment_overlap", SegmentOverlap)
        self.register("step_consistency", StepConsistency)
    
    def register(self, name: str, reward_class: Type[nn.Module]):
        """
        Register a reward class.
        
        Args:
            name: Name to register under.
            reward_class: Reward class to register.
        """
        self._rewards[name] = reward_class
        logger.debug(f"Registered reward: {name}")
    
    def get(self, name: str) -> Type[nn.Module]:
        """
        Get a reward class by name.
        
        Args:
            name: Name of the reward.
        
        Returns:
            Reward class.
        """
        if name not in self._rewards:
            raise KeyError(f"Reward '{name}' not found. Available: {self.list()}")
        return self._rewards[name]
    
    def create(self, name: str, **kwargs) -> nn.Module:
        """
        Create an instance of a registered reward.
        
        Args:
            name: Name of the reward.
            **kwargs: Arguments to pass to the reward constructor.
        
        Returns:
            Instantiated reward module.
        """
        reward_class = self.get(name)
        return reward_class(**kwargs)
    
    def list(self) -> List[str]:
        """List all registered reward names."""
        return list(self._rewards.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._rewards
    
    def __getitem__(self, name: str) -> Type[nn.Module]:
        return self.get(name)


class CompositeReward(nn.Module):
    """
    Composite reward combining multiple reward functions.
    
    Supports weighted combination and normalization of rewards.
    """
    
    def __init__(
        self,
        rewards: Dict[str, Dict[str, Any]],
        normalize: bool = True,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        scale: float = 1.0,
        registry: Optional[RewardRegistry] = None,
    ):
        """
        Initialize the composite reward.
        
        Args:
            rewards: Dictionary mapping reward names to their configs.
                Each config should have 'weight' and optional reward-specific params.
                Example: {"temporal_iou": {"weight": 1.0, "scale": 1.0}}
            normalize: Whether to normalize combined rewards.
            clip_min: Minimum reward value.
            clip_max: Maximum reward value.
            scale: Final scale factor.
            registry: Reward registry to use. Uses default if None.
        """
        super().__init__()
        
        self.normalize = normalize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scale = scale
        
        self.registry = registry or get_default_registry()
        
        # Initialize reward modules
        self.reward_modules = nn.ModuleDict()
        self.weights = {}
        
        total_weight = 0.0
        for name, config in rewards.items():
            if not config.get("enabled", True):
                continue
            
            weight = config.get("weight", 1.0)
            self.weights[name] = weight
            total_weight += weight
            
            # Get reward-specific params (exclude 'weight' and 'enabled')
            reward_params = {
                k: v for k, v in config.items() 
                if k not in ["weight", "enabled"]
            }
            
            # Create reward module
            self.reward_modules[name] = self.registry.create(name, **reward_params)
        
        # Normalize weights
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Initialized composite reward with: {list(self.weights.keys())}")
    
    def forward(
        self,
        pred_timestamps: torch.Tensor,
        gt_timestamps: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        normalized: bool = True,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute composite reward.
        
        Args:
            pred_timestamps: Predicted [start, end] pairs, shape (batch, 2).
            gt_timestamps: Ground truth [start, end] pairs, shape (batch, 2).
            durations: Video durations for denormalization.
            normalized: Whether timestamps are normalized to [0, 1].
            return_components: If True, also return individual reward components.
        
        Returns:
            Combined reward tensor of shape (batch,), and optionally component rewards.
        """
        batch_size = pred_timestamps.shape[0]
        device = pred_timestamps.device
        
        combined_reward = torch.zeros(batch_size, device=device)
        components = {}
        
        for name, module in self.reward_modules.items():
            weight = self.weights.get(name, 1.0)
            
            reward = module(
                pred_timestamps=pred_timestamps,
                gt_timestamps=gt_timestamps,
                durations=durations,
                normalized=normalized,
            )
            
            components[name] = reward
            combined_reward = combined_reward + weight * reward
        
        # Normalize across batch
        if self.normalize and batch_size > 1:
            mean = combined_reward.mean()
            std = combined_reward.std() + 1e-8
            combined_reward = (combined_reward - mean) / std
        
        # Clip
        combined_reward = torch.clamp(combined_reward, self.clip_min, self.clip_max)
        
        # Scale
        combined_reward = combined_reward * self.scale
        
        if return_components:
            return combined_reward, components
        return combined_reward
    
    def compute_from_text(
        self,
        predictions: List[str],
        ground_truths: List[Tuple[float, float]],
        durations: Optional[List[float]] = None,
        parse_fn: Optional[Callable] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute composite reward from text predictions.
        
        Args:
            predictions: List of prediction strings.
            ground_truths: List of (start, end) ground truth tuples.
            durations: List of video durations.
            parse_fn: Function to parse timestamps from text.
            return_components: If True, also return individual components.
        
        Returns:
            Combined reward tensor, and optionally component rewards.
        """
        batch_size = len(predictions)
        
        combined_reward = torch.zeros(batch_size)
        components = {}
        
        for name, module in self.reward_modules.items():
            weight = self.weights.get(name, 1.0)
            
            if hasattr(module, "compute_from_text"):
                reward = module.compute_from_text(
                    predictions=predictions,
                    ground_truths=ground_truths,
                    durations=durations,
                    parse_fn=parse_fn,
                )
            else:
                # Skip if module doesn't support text input
                logger.warning(f"Reward '{name}' doesn't support text input")
                reward = torch.zeros(batch_size)
            
            components[name] = reward
            combined_reward = combined_reward + weight * reward
        
        # Normalize
        if self.normalize and batch_size > 1:
            mean = combined_reward.mean()
            std = combined_reward.std() + 1e-8
            combined_reward = (combined_reward - mean) / std
        
        # Clip and scale
        combined_reward = torch.clamp(combined_reward, self.clip_min, self.clip_max)
        combined_reward = combined_reward * self.scale
        
        if return_components:
            return combined_reward, components
        return combined_reward


# Default global registry instance
_default_registry: Optional[RewardRegistry] = None


def get_default_registry() -> RewardRegistry:
    """Get or create the default reward registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RewardRegistry()
    return _default_registry


def create_composite_reward(
    config: Dict[str, Any],
    registry: Optional[RewardRegistry] = None,
) -> CompositeReward:
    """
    Create a composite reward from configuration.
    
    Args:
        config: Reward configuration dictionary.
        registry: Optional custom registry.
    
    Returns:
        Configured CompositeReward instance.
    
    Example config:
        {
            "temporal_iou": {"enabled": True, "weight": 1.0},
            "segment_overlap": {"enabled": True, "weight": 0.5, "mode": "recall"},
            "processing": {"normalize": True, "clip_min": -10, "clip_max": 10, "scale": 1.0}
        }
    """
    # Extract processing config
    processing = config.pop("processing", {})
    
    # Remove disabled rewards
    reward_configs = {
        k: v for k, v in config.items()
        if isinstance(v, dict) and v.get("enabled", True)
    }
    
    return CompositeReward(
        rewards=reward_configs,
        normalize=processing.get("normalize", True),
        clip_min=processing.get("clip_min", -10.0),
        clip_max=processing.get("clip_max", 10.0),
        scale=processing.get("scale", 1.0),
        registry=registry,
    )
