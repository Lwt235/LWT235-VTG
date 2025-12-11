"""
Segment Overlap Reward for Video Temporal Localization.

Computes overlap-based rewards measuring how much of the ground truth
segment is covered by the prediction.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def segment_overlap(
    pred_start: Union[float, torch.Tensor],
    pred_end: Union[float, torch.Tensor],
    gt_start: Union[float, torch.Tensor],
    gt_end: Union[float, torch.Tensor],
    mode: str = "recall",
    eps: float = 1e-8,
) -> Union[float, torch.Tensor]:
    """
    Compute segment overlap metrics.
    
    Args:
        pred_start: Predicted start time(s).
        pred_end: Predicted end time(s).
        gt_start: Ground truth start time(s).
        gt_end: Ground truth end time(s).
        mode: Overlap mode:
            - "recall": intersection / gt_duration (coverage of GT)
            - "precision": intersection / pred_duration (accuracy of pred)
            - "f1": harmonic mean of recall and precision
        eps: Small value to avoid division by zero.
    
    Returns:
        Overlap score(s) in range [0, 1].
    """
    # Ensure pred_end >= pred_start
    if isinstance(pred_start, torch.Tensor):
        pred_end = torch.maximum(pred_end, pred_start)
    else:
        pred_end = max(pred_end, pred_start)
    
    # Compute intersection
    if isinstance(pred_start, torch.Tensor):
        inter_start = torch.maximum(pred_start, gt_start)
        inter_end = torch.minimum(pred_end, gt_end)
        intersection = torch.clamp(inter_end - inter_start, min=0)
    else:
        inter_start = max(pred_start, gt_start)
        inter_end = min(pred_end, gt_end)
        intersection = max(0, inter_end - inter_start)
    
    # Compute durations
    pred_duration = pred_end - pred_start
    gt_duration = gt_end - gt_start
    
    if mode == "recall":
        # What fraction of GT is covered
        if isinstance(gt_duration, torch.Tensor):
            overlap = intersection / (gt_duration + eps)
        else:
            overlap = intersection / (gt_duration + eps) if gt_duration > 0 else 0.0
            
    elif mode == "precision":
        # What fraction of prediction overlaps with GT
        if isinstance(pred_duration, torch.Tensor):
            overlap = intersection / (pred_duration + eps)
        else:
            overlap = intersection / (pred_duration + eps) if pred_duration > 0 else 0.0
            
    elif mode == "f1":
        # Harmonic mean of recall and precision
        if isinstance(gt_duration, torch.Tensor):
            recall = intersection / (gt_duration + eps)
            precision = intersection / (pred_duration + eps)
            overlap = 2 * recall * precision / (recall + precision + eps)
        else:
            recall = intersection / (gt_duration + eps) if gt_duration > 0 else 0.0
            precision = intersection / (pred_duration + eps) if pred_duration > 0 else 0.0
            overlap = 2 * recall * precision / (recall + precision + eps) if (recall + precision) > 0 else 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'recall', 'precision', or 'f1'.")
    
    return overlap


class SegmentOverlap(nn.Module):
    """
    Segment overlap reward module.
    
    Computes overlap-based rewards for temporal grounding predictions.
    """
    
    def __init__(
        self,
        mode: str = "recall",
        scale: float = 1.0,
        threshold: float = 0.0,
        name: str = "segment_overlap",
    ):
        """
        Initialize the SegmentOverlap reward.
        
        Args:
            mode: Overlap mode ("recall", "precision", "f1").
            scale: Scale factor for the reward.
            threshold: Minimum overlap to get positive reward.
            name: Name identifier for logging.
        """
        super().__init__()
        self.mode = mode
        self.scale = scale
        self.threshold = threshold
        self.name = name
    
    def forward(
        self,
        pred_timestamps: torch.Tensor,
        gt_timestamps: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """
        Compute segment overlap rewards.
        
        Args:
            pred_timestamps: Predicted [start, end] pairs, shape (batch, 2).
            gt_timestamps: Ground truth [start, end] pairs, shape (batch, 2).
            durations: Video durations for denormalization (optional).
            normalized: Whether timestamps are normalized to [0, 1].
        
        Returns:
            Reward tensor of shape (batch,).
        """
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
        
        # Compute overlap
        overlap = segment_overlap(
            pred_start, pred_end, gt_start, gt_end,
            mode=self.mode,
        )
        
        # Apply threshold
        if self.threshold > 0:
            overlap = torch.clamp(overlap - self.threshold, min=0) / (1 - self.threshold)
        
        # Scale
        reward = overlap * self.scale
        
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
                
                overlap = segment_overlap(
                    pred_start, pred_end, gt_start, gt_end,
                    mode=self.mode,
                )
                
                rewards.append(overlap * self.scale)
                
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse prediction '{pred_text}': {e}")
                rewards.append(0.0)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    @staticmethod
    def _default_parse(text: str) -> Tuple[float, float]:
        """Default parser for timestamp text."""
        import re
        
        patterns = [
            r"([\d.]+)\s*(?:to|->|-|,)\s*([\d.]+)",
            r"start[:\s]*([\d.]+).*end[:\s]*([\d.]+)",
            r"\(([\d.]+)[,\s]+([\d.]+)\)",
            r"\[([\d.]+)[,\s]+([\d.]+)\]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                return start, end
        
        raise ValueError(f"Could not parse timestamps from: {text}")


def batch_segment_overlap(
    pred_timestamps: torch.Tensor,
    gt_timestamps: torch.Tensor,
    mode: str = "recall",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute batch segment overlap efficiently.
    
    Args:
        pred_timestamps: Shape (batch, 2) with [start, end] pairs.
        gt_timestamps: Shape (batch, 2) with [start, end] pairs.
        mode: Overlap mode ("recall", "precision", "f1").
        eps: Small value to avoid division by zero.
    
    Returns:
        Overlap tensor of shape (batch,).
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
    
    if mode == "recall":
        overlap = intersection / (gt_duration + eps)
    elif mode == "precision":
        overlap = intersection / (pred_duration + eps)
    elif mode == "f1":
        recall = intersection / (gt_duration + eps)
        precision = intersection / (pred_duration + eps)
        overlap = 2 * recall * precision / (recall + precision + eps)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return overlap
