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


def _get_world_size() -> int:
    """Get the number of GPUs/processes in distributed training."""
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def _get_rank() -> int:
    """Get the rank of current GPU/process in distributed training."""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class DurationBasedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups videos by total duration.
    
    This sampler creates batches where the sum of video durations is 
    approximately equal to a target value, which helps stabilize GPU 
    memory usage when videos have varying lengths.
    
    The sampler uses a sorted greedy bin-packing approach:
    1. Sort videos by duration (descending) to group similar-length videos
    2. Apply greedy bin-packing on sorted videos until target duration is reached
    3. Shuffle the order of batches (not videos within) for training randomness
    
    This approach ensures:
    - Each batch has similar total duration (controlled by target_batch_duration)
    - Each batch contains videos of similar length (due to sorted grouping)
    - GPU memory usage is more consistent across batches (similar-length videos
      produce similar numbers of frames)
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
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
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
            num_replicas: Number of processes participating in distributed training.
                If None, will use torch.distributed.get_world_size().
            rank: Rank of the current process in distributed training.
                If None, will use torch.distributed.get_rank().
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
        
        # Distributed training support
        if num_replicas is None:
            num_replicas = _get_world_size()
        if rank is None:
            rank = _get_rank()
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, should be in [0, {num_replicas})")
        
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Track epoch for reproducible shuffling
        self._epoch = 0
        
        # Log statistics (only on rank 0 to avoid duplicate logs)
        if self.rank == 0:
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
            if self.num_replicas > 1:
                logger.info(
                    f"Distributed training enabled: num_replicas={self.num_replicas}"
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
        
        The batching strategy ensures more consistent GPU memory usage:
        1. Sort all videos by duration (descending) to group similar-length videos
        2. Apply greedy bin-packing on sorted videos to create balanced batches
        3. Shuffle the order of batches (not videos within batches) for randomness
        4. In distributed training, partition batches across GPUs
        
        This approach ensures that:
        - Each batch has similar total duration (controlled by target_batch_duration)
        - Each batch has a similar NUMBER of videos (because similar-length videos are grouped)
        - GPU memory usage is more consistent across batches
        - Each GPU gets a disjoint subset of batches in distributed training
        
        Yields:
            Lists of sample indices for each batch.
        """
        # Create list of (index, duration) pairs and sort by duration (descending)
        # Sorting by duration ensures videos of similar length are grouped together,
        # which leads to more consistent batch sizes and GPU memory usage.
        indexed_durations = list(enumerate(self.durations))
        indexed_durations.sort(key=lambda x: x[1], reverse=True)
        
        # Build batches using greedy bin-packing on sorted videos
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_duration = 0.0
        
        for idx, sample_duration in indexed_durations:
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
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = [idx]
                current_duration = sample_duration
            else:
                # Add to current batch
                current_batch.append(idx)
                current_duration += sample_duration
        
        # Handle remaining samples
        if current_batch:
            if len(current_batch) >= self.min_batch_size or not self.drop_last:
                batches.append(current_batch)
        
        # Shuffle the order of batches if requested
        # This maintains randomness in training order while keeping batches balanced
        if self.shuffle and batches:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed + self._epoch)
            else:
                g.manual_seed(self._epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        
        # In distributed training, partition batches across GPUs using round-robin distribution
        # Each GPU gets every num_replicas-th batch starting from its rank
        # Example with 2 GPUs and 10 batches:
        #   GPU 0 (rank=0): batches [0, 2, 4, 6, 8]
        #   GPU 1 (rank=1): batches [1, 3, 5, 7, 9]
        # This ensures load balancing and no data overlap between GPUs
        if self.num_replicas > 1:
            batches = [batches[i] for i in range(self.rank, len(batches), self.num_replicas)]
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """
        Return the estimated number of batches for this GPU/process.
        
        For accurate progress bar calculation, this method estimates the total
        batches based on durations and constraints, then divides by num_replicas
        for distributed training.
        
        Note: This is an estimation since actual batch sizes vary based on video durations.
        """
        # For accurate batch count, we need to compute actual batches
        # This is important for correct progress bar display
        # Since batch sizes vary based on duration, estimation can be inaccurate
        
        # Simple estimation based on total duration (fast but may be inaccurate)
        total_duration = sum(self.durations)
        estimated_batches = max(1, int(total_duration / self.target_batch_duration))
        
        # Account for max_batch_size constraint
        if self.max_batch_size is not None:
            min_batches = (len(self.durations) + self.max_batch_size - 1) // self.max_batch_size
            estimated_batches = max(estimated_batches, min_batches)
        
        # In distributed training, each GPU processes a subset of batches
        if self.num_replicas > 1:
            # Each GPU gets approximately 1/num_replicas of the batches
            estimated_batches = (estimated_batches + self.num_replicas - 1) // self.num_replicas
        
        return max(1, estimated_batches)


def create_duration_based_batch_sampler(
    dataset,
    target_batch_duration: float,
    max_batch_size: Optional[int] = None,
    min_batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> DurationBasedBatchSampler:
    """
    Create a duration-based batch sampler from a dataset.
    
    Args:
        dataset: A VideoTemporalDataset (or subclass) with a `samples` attribute
            containing a list of dictionaries. Each dictionary should have a
            'duration' key with the video duration in seconds.
        target_batch_duration: Target total duration per batch in seconds.
        max_batch_size: Maximum number of samples per batch.
        min_batch_size: Minimum number of samples per batch.
        shuffle: Whether to shuffle the dataset each epoch.
        drop_last: Whether to drop the last incomplete batch.
        seed: Random seed for reproducibility.
        num_replicas: Number of processes in distributed training. If None, auto-detected.
        rank: Rank of current process in distributed training. If None, auto-detected.
    
    Returns:
        Configured DurationBasedBatchSampler instance.
    
    Raises:
        AttributeError: If dataset doesn't have a `samples` attribute.
        TypeError: If samples are not a list of dictionaries.
    """
    # Validate dataset interface
    if not hasattr(dataset, "samples"):
        raise AttributeError(
            "Dataset must have a 'samples' attribute containing sample dictionaries. "
            "Expected VideoTemporalDataset or compatible dataset."
        )
    
    # Extract durations from dataset
    durations = []
    for idx, sample in enumerate(dataset.samples):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample at index {idx} must be a dictionary, got {type(sample).__name__}"
            )
        duration = sample.get("duration", 0.0)
        if duration <= 0:
            logger.warning(f"Sample at index {idx} has invalid duration: {duration}, using 1.0")
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
        num_replicas=num_replicas,
        rank=rank,
    )
