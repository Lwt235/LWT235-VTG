"""
Training callbacks for Video Temporal Localization.

Provides custom callbacks for logging batch statistics and other training information.
"""

from typing import Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BatchLoggingCallback(TrainerCallback):
    """
    Callback to log batch size and duration information during training.
    
    This callback logs:
    - Actual batch size for each step
    - Total duration of videos in each batch (if available)
    - Number of samples processed per GPU
    - Aggregate statistics over logging intervals
    """
    
    def __init__(self, log_batch_stats: bool = True):
        """
        Initialize the batch logging callback.
        
        Args:
            log_batch_stats: Whether to log batch statistics at each logging step.
        """
        self.log_batch_stats = log_batch_stats
        self.batch_sizes: List[int] = []
    
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        Called at the beginning of each training step.
        
        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Additional arguments (may include 'inputs').
        
        Returns:
            Updated trainer control.
        """
        # Extract batch information if available
        inputs = kwargs.get("inputs", None)
        if inputs is not None and self.log_batch_stats:
            # Get batch size from inputs
            batch_size = None
            if isinstance(inputs, dict):
                # Try to get batch size from any tensor in the batch
                for key, value in inputs.items():
                    if hasattr(value, "shape") and len(value.shape) > 0:
                        batch_size = value.shape[0]
                        break
            
            if batch_size is not None:
                self.batch_sizes.append(batch_size)
        
        return control
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TrainerControl:
        """
        Called when logging occurs.
        
        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            logs: Dictionary of logged values.
            **kwargs: Additional arguments.
        
        Returns:
            Updated trainer control.
        """
        if self.log_batch_stats and self.batch_sizes:
            # Compute statistics
            avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
            min_batch_size = min(self.batch_sizes)
            max_batch_size = max(self.batch_sizes)
            
            # Log batch statistics
            if logs is not None:
                logs["batch_size_avg"] = avg_batch_size
                logs["batch_size_min"] = min_batch_size
                logs["batch_size_max"] = max_batch_size
            
            # Log to console (only on main process)
            if state.is_world_process_zero:
                logger.info(
                    f"Step {state.global_step}: "
                    f"batch_size avg={avg_batch_size:.1f}, "
                    f"min={min_batch_size}, max={max_batch_size}"
                )
            
            # Reset statistics
            self.batch_sizes.clear()
        
        return control
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        Called at the beginning of training.
        
        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Additional arguments (may include 'train_dataloader').
        
        Returns:
            Updated trainer control.
        """
        if state.is_world_process_zero:
            # Calculate effective batch size once for readability
            effective_batch_size = (
                args.per_device_train_batch_size * 
                args.gradient_accumulation_steps * 
                args.world_size
            )
            
            logger.info("=" * 80)
            logger.info("Training Configuration:")
            logger.info(f"  Per-device batch size: {args.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
            logger.info(f"  Effective batch size: {effective_batch_size}")
            logger.info(f"  World size (GPUs): {args.world_size}")
            logger.info(f"  Total optimization steps: {state.max_steps}")
            
            # Log dataloader information if available
            train_dataloader = kwargs.get("train_dataloader", None)
            if train_dataloader is not None:
                if hasattr(train_dataloader, "batch_sampler"):
                    batch_sampler = train_dataloader.batch_sampler
                    if hasattr(batch_sampler, "target_batch_duration"):
                        logger.info(f"  Duration-based batching enabled:")
                        logger.info(f"    Target batch duration: {batch_sampler.target_batch_duration:.1f}s")
                        if hasattr(batch_sampler, "max_batch_size") and batch_sampler.max_batch_size is not None:
                            logger.info(f"    Max batch size: {batch_sampler.max_batch_size}")
                        if hasattr(batch_sampler, "min_batch_size"):
                            logger.info(f"    Min batch size: {batch_sampler.min_batch_size}")
                        if hasattr(batch_sampler, "num_replicas") and batch_sampler.num_replicas > 1:
                            logger.info(f"    Distributed: {batch_sampler.num_replicas} replicas")
            
            logger.info("=" * 80)
        
        return control
