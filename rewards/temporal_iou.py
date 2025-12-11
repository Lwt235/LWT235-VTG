"""
Temporal IoU Reward for Video Temporal Localization.

Computes the temporal Intersection over Union between predicted
and ground truth time segments.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def temporal_iou(
    pred_start: Union[float, torch.Tensor],
    pred_end: Union[float, torch.Tensor],
    gt_start: Union[float, torch.Tensor],
    gt_end: Union[float, torch.Tensor],
    eps: float = 1e-8,
) -> Union[float, torch.Tensor]:
    """
    Compute temporal Intersection over Union.
    
    Args:
        pred_start: Predicted start time(s).
        pred_end: Predicted end time(s).
        gt_start: Ground truth start time(s).
        gt_end: Ground truth end time(s).
        eps: Small value to avoid division by zero.
    
    Returns:
        Temporal IoU score(s) in range [0, 1].
    """
    # Ensure pred_end >= pred_start
    if isinstance(pred_start, torch.Tensor):
        pred_end = torch.maximum(pred_end, pred_start)
    else:
        pred_end = max(pred_end, pred_start)
    
    # Compute intersection
    inter_start = max(pred_start, gt_start) if not isinstance(pred_start, torch.Tensor) else torch.maximum(pred_start, gt_start)
    inter_end = min(pred_end, gt_end) if not isinstance(pred_end, torch.Tensor) else torch.minimum(pred_end, gt_end)
    
    if isinstance(inter_start, torch.Tensor):
        intersection = torch.clamp(inter_end - inter_start, min=0)
    else:
        intersection = max(0, inter_end - inter_start)
    
    # Compute union
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    union = pred_duration + gt_duration - intersection
    
    # Compute IoU
    if isinstance(union, torch.Tensor):
        iou = intersection / (union + eps)
    else:
        iou = intersection / (union + eps) if union > 0 else 0.0
    
    return iou


class TemporalIoU(nn.Module):
    """
    Temporal IoU reward module.
    
    Computes IoU-based rewards for temporal grounding predictions.
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        threshold: float = 0.0,
        binary: bool = False,
        binary_threshold: float = 0.5,
        name: str = "temporal_iou",
    ):
        """
        Initialize the TemporalIoU reward.
        
        Args:
            scale: Scale factor for the reward.
            threshold: Minimum IoU to get positive reward.
            binary: If True, return binary reward based on threshold.
            binary_threshold: Threshold for binary reward.
            name: Name identifier for logging.
        """
        super().__init__()
        self.scale = scale
        self.threshold = threshold
        self.binary = binary
        self.binary_threshold = binary_threshold
        self.name = name
    
    def forward(
        self,
        pred_timestamps: torch.Tensor,
        gt_timestamps: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute temporal IoU rewards.
        
        Args:
            pred_timestamps: Predicted [start, end] pairs, shape (batch, 2).
            gt_timestamps: Ground truth [start, end] pairs, shape (batch, 2).
            durations: Video durations for denormalization (optional).
            normalized: Whether timestamps are normalized to [0, 1].
        
        Returns:
            Reward tensor of shape (batch,).
        """
        batch_size = pred_timestamps.shape[0]
        device = pred_timestamps.device
        
        # Extract start and end times
        pred_start = pred_timestamps[:, 0]
        pred_end = pred_timestamps[:, 1]
        gt_start = gt_timestamps[:, 0]
        gt_end = gt_timestamps[:, 1]
        
        # Denormalize if needed
        if not normalized and durations is not None:
            pred_start = pred_start * durations
            pred_end = pred_end * durations
            gt_start = gt_start * durations
            gt_end = gt_end * durations
        
        # Compute IoU
        iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
        
        # Apply threshold
        if self.threshold > 0:
            iou = torch.clamp(iou - self.threshold, min=0) / (1 - self.threshold)
        
        # Binary conversion if needed
        if self.binary:
            iou = (iou >= self.binary_threshold).float()
        
        # Scale
        reward = iou * self.scale
        
        return reward
    
    def compute_from_text(
        self,
        predictions: List[str],
        ground_truths: List[Tuple[float, float]],
        durations: Optional[List[float]] = None,
        parse_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Compute rewards from text predictions.
        
        Args:
            predictions: List of predicted text responses.
            ground_truths: List of (start, end) ground truth tuples.
            durations: List of video durations.
            parse_fn: Function to parse timestamps from text.
        
        Returns:
            Reward tensor of shape (batch,).
        """
        if parse_fn is None:
            parse_fn = self._default_parse
        
        rewards = []
        
        for i, (pred_text, gt) in enumerate(zip(predictions, ground_truths)):
            try:
                pred_start, pred_end = parse_fn(pred_text)
                gt_start, gt_end = gt
                
                # Normalize if duration is provided
                if durations is not None:
                    duration = durations[i]
                    pred_start = pred_start / duration
                    pred_end = pred_end / duration
                    gt_start = gt_start / duration
                    gt_end = gt_end / duration
                
                iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
                
                if self.binary:
                    iou = 1.0 if iou >= self.binary_threshold else 0.0
                
                rewards.append(iou * self.scale)
                
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse prediction '{pred_text}': {e}")
                rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    @staticmethod
    def _default_parse(text: str) -> Tuple[float, float]:
        """
        Default parser for timestamp text.
        
        Handles formats like:
        - "0.25 to 0.75"
        - "start: 0.25, end: 0.75"
        - "(0.25, 0.75)"
        """
        import re
        
        # Try common patterns
        patterns = [
            r"([\d.]+)\s*(?:to|->|-|,)\s*([\d.]+)",  # "0.25 to 0.75"
            r"start[:\s]*([\d.]+).*end[:\s]*([\d.]+)",  # "start: 0.25, end: 0.75"
            r"\(([\d.]+)[,\s]+([\d.]+)\)",  # "(0.25, 0.75)"
            r"\[([\d.]+)[,\s]+([\d.]+)\]",  # "[0.25, 0.75]"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                return start, end
        
        raise ValueError(f"Could not parse timestamps from: {text}")


def batch_temporal_iou(
    pred_timestamps: torch.Tensor,
    gt_timestamps: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute batch temporal IoU efficiently.
    
    Args:
        pred_timestamps: Shape (batch, 2) with [start, end] pairs.
        gt_timestamps: Shape (batch, 2) with [start, end] pairs.
        eps: Small value to avoid division by zero.
    
    Returns:
        IoU tensor of shape (batch,).
    """
    pred_start = pred_timestamps[:, 0]
    pred_end = torch.maximum(pred_timestamps[:, 1], pred_start)
    gt_start = gt_timestamps[:, 0]
    gt_end = gt_timestamps[:, 1]
    
    inter_start = torch.maximum(pred_start, gt_start)
    inter_end = torch.minimum(pred_end, gt_end)
    intersection = torch.clamp(inter_end - inter_start, min=0)
    
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    union = pred_duration + gt_duration - intersection
    
    iou = intersection / (union + eps)
    return iou
