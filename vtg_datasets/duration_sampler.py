"""
Duration-Based Batch Sampler for Video Temporal Localization.

Provides batch sampling strategies that group videos by total duration
to achieve more stable GPU memory utilization during training.
"""

from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DurationBasedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups videos by total duration.
    
    This sampler creates batches where the sum of video durations is 
    approximately equal to a target value, which helps stabilize GPU 
    memory usage when videos have varying lengths.
    
    The sampler uses a greedy bin-packing approach:
    1. Optionally shuffle the dataset indices
    2. Iterate through samples, adding to current batch until target duration is reached
    3. When target is reached or exceeded, yield the batch and start a new one
    """
    
    def __init__(
        self,
        durations: List[float],
        target_batch_duration: float,
        max_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the duration-based batch sampler.
        
        Args:
            durations: List of video durations (in seconds) for each sample in the dataset.
            target_batch_duration: Target total duration (in seconds) for each batch.
            max_batch_size: Maximum number of samples per batch (optional constraint).
            min_batch_size: Minimum number of samples per batch.
            shuffle: Whether to shuffle the dataset each epoch.
            drop_last: Whether to drop the last incomplete batch.
            seed: Random seed for shuffling (for reproducibility).
        """
        super().__init__(None)
        
        if len(durations) == 0:
            raise ValueError("durations cannot be empty")
        if target_batch_duration <= 0:
            raise ValueError("target_batch_duration must be positive")
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if max_batch_size is not None and max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        
        self.durations = durations
        self.target_batch_duration = target_batch_duration
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Track epoch for reproducible shuffling
        self._epoch = 0
        
        # Log statistics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        expected_batch_size = target_batch_duration / avg_duration
        
        logger.info(
            f"DurationBasedBatchSampler initialized: "
            f"samples={len(durations)}, target_duration={target_batch_duration:.1f}s, "
            f"avg_sample_duration={avg_duration:.1f}s, "
            f"expected_batch_size={expected_batch_size:.1f}"
        )
        logger.info(
            f"Duration range: min={min_duration:.1f}s, max={max_duration:.1f}s"
        )
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number for deterministic shuffling.
        
        Args:
            epoch: Current epoch number.
        """
        self._epoch = epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches of sample indices.
        
        Yields:
            Lists of sample indices for each batch.
        """
        # Create list of indices
        indices = list(range(len(self.durations)))
        
        # Shuffle if requested
        if self.shuffle:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed + self._epoch)
            else:
                g.manual_seed(self._epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        
        # Build batches using greedy bin-packing
        current_batch: List[int] = []
        current_duration = 0.0
        
        for idx in indices:
            sample_duration = self.durations[idx]
            
            # Check if adding this sample would exceed constraints
            would_exceed_duration = (
                current_duration + sample_duration > self.target_batch_duration
                and len(current_batch) >= self.min_batch_size
            )
            would_exceed_size = (
                self.max_batch_size is not None 
                and len(current_batch) >= self.max_batch_size
            )
            
            if current_batch and (would_exceed_duration or would_exceed_size):
                # Yield current batch and start new one
                yield current_batch
                current_batch = [idx]
                current_duration = sample_duration
            else:
                # Add to current batch
                current_batch.append(idx)
                current_duration += sample_duration
        
        # Handle remaining samples
        if current_batch:
            if len(current_batch) >= self.min_batch_size or not self.drop_last:
                yield current_batch
    
    def __len__(self) -> int:
        """
        Return the estimated number of batches.
        
        Note: This is an approximation since batch sizes vary.
        """
        total_duration = sum(self.durations)
        estimated_batches = max(1, int(total_duration / self.target_batch_duration))
        
        # Account for constraints
        if self.max_batch_size is not None:
            min_batches = (len(self.durations) + self.max_batch_size - 1) // self.max_batch_size
            estimated_batches = max(estimated_batches, min_batches)
        
        return estimated_batches


def create_duration_based_batch_sampler(
    dataset,
    target_batch_duration: float,
    max_batch_size: Optional[int] = None,
    min_batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> DurationBasedBatchSampler:
    """
    Create a duration-based batch sampler from a dataset.
    
    Args:
        dataset: A VideoTemporalDataset (or subclass) with loaded samples.
        target_batch_duration: Target total duration per batch in seconds.
        max_batch_size: Maximum number of samples per batch.
        min_batch_size: Minimum number of samples per batch.
        shuffle: Whether to shuffle the dataset each epoch.
        drop_last: Whether to drop the last incomplete batch.
        seed: Random seed for reproducibility.
    
    Returns:
        Configured DurationBasedBatchSampler instance.
    """
    # Extract durations from dataset
    durations = []
    for sample in dataset.samples:
        duration = sample.get("duration", 0.0)
        if duration <= 0:
            logger.warning(f"Sample has invalid duration: {duration}, using 1.0")
            duration = 1.0
        durations.append(duration)
    
    return DurationBasedBatchSampler(
        durations=durations,
        target_batch_duration=target_batch_duration,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
