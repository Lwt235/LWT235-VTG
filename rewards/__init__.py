"""
Reward functions for Video Temporal Localization.
"""

from .temporal_iou import TemporalIoU, temporal_iou
from .segment_overlap import SegmentOverlap, segment_overlap
from .step_consistency import StepConsistency, step_consistency
from .reward_registry import (
    RewardRegistry,
    CompositeReward,
    get_default_registry,
    register_reward,
)

__all__ = [
    "TemporalIoU",
    "temporal_iou",
    "SegmentOverlap",
    "segment_overlap",
    "StepConsistency",
    "step_consistency",
    "RewardRegistry",
    "CompositeReward",
    "get_default_registry",
    "register_reward",
]
