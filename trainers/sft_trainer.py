"""
Supervised Fine-Tuning Trainer for Video Temporal Localization.

Provides SFT training with DeepSpeed, mixed precision, and gradient accumulation.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from omegaconf import DictConfig, OmegaConf

from utils.logging_utils import get_logger
from utils.common import load_config, merge_configs, count_parameters
from utils.temporal_tokens import (
    add_temporal_tokens_to_tokenizer,
    resize_model_embeddings_for_temporal_tokens,
    get_temporal_token_ids,
)

logger = get_logger(__name__)


class VideoTemporalSFTTrainer(Trainer):
    """
    SFT Trainer for video temporal grounding.

    Extends HuggingFace Trainer with video-specific functionality.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        processor: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        mm_projector_lr: Optional[float] = None,
        vision_tower_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
        duration_batching_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the SFT trainer.

        Args:
            model: The model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            tokenizer: Tokenizer for text processing.
            data_collator: Data collator for batching.
            processor: Qwen VL processor.
            compute_metrics: Function to compute metrics.
            callbacks: List of trainer callbacks.
            mm_projector_lr: Learning rate for multimodal projector.
            vision_tower_lr: Learning rate for vision tower.
            head_lr: Learning rate for task head.
            duration_batching_config: Configuration for duration-based batch sampling.
            **kwargs: Additional arguments for base Trainer.
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # Use processing_class instead of deprecated tokenizer
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )

        self.processor = processor
        self.mm_projector_lr = mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.head_lr = head_lr
        self.duration_batching_config = duration_batching_config

    def get_train_dataloader(self) -> DataLoader:
        """
        Create the training dataloader with optional duration-based batch sampling.

        When duration_batching_config is provided and enabled, uses
        DurationBasedBatchSampler to group videos by total duration instead
        of a fixed batch size.

        Returns:
            DataLoader: Training data loader.
        """
        if self.duration_batching_config and self.duration_batching_config.get("enabled", False):
            from vtg_datasets.duration_sampler import create_duration_based_batch_sampler

            # Get current epoch for reproducible shuffling
            current_epoch = int(self.state.epoch) if hasattr(self, "state") and self.state and self.state.epoch is not None else 0

            # For distributed training, pass distributed parameters to the sampler
            # SFT trainer uses TrainingArguments which provides world_size and process_index
            # These will be auto-detected by the sampler if not provided, but we pass
            # them explicitly to ensure consistency with HuggingFace Trainer's settings
            num_replicas = self.args.world_size if self.args.world_size > 1 else None
            rank = self.args.process_index if self.args.world_size > 1 else None

            # Create duration-based batch sampler
            batch_sampler = create_duration_based_batch_sampler(
                dataset=self.train_dataset,
                target_batch_duration=self.duration_batching_config.get("target_batch_duration", 60.0),
                max_batch_size=self.duration_batching_config.get("max_batch_size"),
                min_batch_size=self.duration_batching_config.get("min_batch_size", 1),
                shuffle=True,  # Always shuffle training data
                drop_last=self.duration_batching_config.get("drop_last", False),
                seed=self.args.data_seed,
                num_replicas=num_replicas,
                rank=rank,
            )
            batch_sampler.set_epoch(current_epoch)

            # Log only on main process (rank 0 or not distributed)
            if rank in [None, 0]:
                logger.info("Using duration-based batch sampling for training")
                if num_replicas and num_replicas > 1:
                    logger.info(f"Distributed training: {num_replicas} GPUs, rank {rank}")

            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        # Fall back to default behavior
        return super().get_train_dataloader()

    def create_optimizer(self):
        """
        Create optimizer with different learning rates for different model parts.
        """
        if self.optimizer is not None:
            return self.optimizer

        # Get base learning rate
        base_lr = self.args.learning_rate

        # Separate parameters by module type
        param_groups = []

        # Identify parameter groups
        vision_params = []
        projector_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "visual" in name.lower() or "vision" in name.lower():
                if "project" in name.lower():
                    projector_params.append(param)
                else:
                    vision_params.append(param)
            elif "lm_head" in name.lower() or "head" in name.lower():
                head_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different learning rates
        if vision_params and self.vision_tower_lr is not None:
            param_groups.append({
                "params": vision_params,
                "lr": self.vision_tower_lr,
                "name": "vision_tower",
            })

        if projector_params and self.mm_projector_lr is not None:
            param_groups.append({
                "params": projector_params,
                "lr": self.mm_projector_lr,
                "name": "mm_projector",
            })

        if head_params and self.head_lr is not None:
            param_groups.append({
                "params": head_params,
                "lr": self.head_lr,
                "name": "head",
            })

        # Add remaining parameters with base learning rate
        remaining_params = []
        if vision_params and self.vision_tower_lr is None:
            remaining_params.extend(vision_params)
        if projector_params and self.mm_projector_lr is None:
            remaining_params.extend(projector_params)
        if head_params and self.head_lr is None:
            remaining_params.extend(head_params)
        remaining_params.extend(other_params)

        if remaining_params:
            param_groups.append({
                "params": remaining_params,
                "lr": base_lr,
                "name": "default",
            })

        # Log parameter groups
        for group in param_groups:
            num_params = sum(p.numel() for p in group["params"])
            logger.info(f"Parameter group '{group['name']}': {num_params:,} params, lr={group['lr']}")

        # Create optimizer
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        return self.optimizer

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute the training loss.

        Args:
            model: The model.
            inputs: Input batch.
            return_outputs: Whether to return model outputs.
            num_items_in_batch: Number of items in the batch.

        Returns:
            Loss tensor, and optionally model outputs.
        """
        # Create a copy to avoid modifying the original inputs
        inputs = dict(inputs)

        # Extract labels if present
        # labels = inputs.pop("labels", None)

        # Remove metadata that shouldn't go to model
        inputs.pop("temporal_bins", None)
        inputs.pop("sample_indices", None)

        # Forward pass
        outputs = model(**inputs)

        # if labels is not None:
        #     # Compute loss with labels
        #     logits = outputs.logits

        #     # Shift for causal LM
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()

        #     # Flatten
        #     loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)),
        #         shift_labels.view(-1),
        #     )
        # else:
        #     # Use model's computed loss
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        """
        Save the model, tokenizer, and processor.

        Args:
            output_dir: Output directory.
            _internal_call: Whether this is an internal call.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save model (handles PEFT automatically)
        super().save_model(output_dir, _internal_call=_internal_call)

        # Save tokenizer (processing_class) to preserve temporal tokens
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Save processor if available
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")


def create_sft_trainer(
    config: Union[str, Path, Dict, DictConfig],
    data_config: Optional[Union[str, Path, Dict, DictConfig]] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
) -> VideoTemporalSFTTrainer:
    """
    Create an SFT trainer from configuration.

    Args:
        config: Training configuration (path or dict).
        data_config: Data configuration (path or dict).
        train_dataset: Pre-built training dataset.
        eval_dataset: Pre-built evaluation dataset.
        callbacks: Additional callbacks.

    Returns:
        Configured VideoTemporalSFTTrainer instance.
    """
    # Load configurations
    if isinstance(config, (str, Path)):
        config = load_config(config)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)

    if data_config is not None:
        if isinstance(data_config, (str, Path)):
            data_config = load_config(data_config)
        elif isinstance(data_config, dict):
            data_config = OmegaConf.create(data_config)

    # Load model and processor
    model_config = config.get("model", {})
    model_name = model_config.get("name_or_path", "Qwen/Qwen3-VL-4B-Instruct")

    logger.info(f"Loading model from {model_name}")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # Check if temporal tokens should be used
    temporal_config = {}
    if data_config is not None:
        temporal_config = data_config.get("temporal", {})
    use_temporal_tokens = temporal_config.get("use_temporal_tokens", False)

    # Add temporal tokens to tokenizer if enabled
    if use_temporal_tokens:
        logger.info("Adding temporal tokens (<0>~<999>) to tokenizer")
        add_temporal_tokens_to_tokenizer(tokenizer)

    # Load model
    torch_dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )

    # Sync model config with tokenizer special tokens to avoid mismatch warnings
    # This ensures consistency between the tokenizer and model configurations
    configs_to_sync = [model.config]
    if hasattr(model, "generation_config"):
        configs_to_sync.append(model.generation_config)

    for cfg in configs_to_sync:
        if tokenizer.pad_token_id is not None:
            cfg.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            cfg.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            cfg.bos_token_id = tokenizer.bos_token_id

    # Initialize temporal token embeddings if enabled
    if use_temporal_tokens:
        logger.info("Initializing temporal token embeddings with sinusoidal encoding")
        init_strategy = temporal_config.get("embedding_init_strategy", "sinusoidal")
        resize_model_embeddings_for_temporal_tokens(model, tokenizer, init_strategy)

    # Apply LoRA if configured
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        logger.info("Applying LoRA configuration")

        # Get target modules from config and convert to list (OmegaConf -> Python)
        target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
        if hasattr(target_modules, "__iter__") and not isinstance(target_modules, (str, list)):
            target_modules = list(target_modules)

        # When using temporal tokens, use trainable_token_indices for efficient training
        # This method only trains the specific new tokens without modifying the full embedding matrix.
        # Reference: https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
        # "Efficiently train tokens alongside LoRA"
        trainable_token_indices = None
        if use_temporal_tokens:
            # Get the token IDs for temporal tokens
            temporal_token_ids = get_temporal_token_ids(tokenizer)
            # Use trainable_token_indices parameter to efficiently train new tokens
            # This saves memory compared to adding embed_tokens to target_modules
            trainable_token_indices = {"embed_tokens": temporal_token_ids}
            logger.info(f"Temporal tokens enabled: trainable_token_indices with {len(temporal_token_ids)} tokens")

        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            trainable_token_indices=trainable_token_indices,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Create training arguments
    # Convert OmegaConf to native Python types for JSON serialization compatibility (e.g., W&B)
    training_config = config.get("training", {})
    if isinstance(training_config, DictConfig):
        training_config = OmegaConf.to_container(training_config, resolve=True)

    # Get gradient_checkpointing_kwargs (already converted to native Python dict above)
    gradient_checkpointing_kwargs = training_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    training_args = TrainingArguments(
        output_dir=training_config.get("output_dir", "./outputs/sft"),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        learning_rate=training_config.get("learning_rate", 1e-4),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        warmup_steps=training_config.get("warmup_steps", 0),
        optim=training_config.get("optim", "adamw_torch"),
        adam_beta1=training_config.get("adam_beta1", 0.9),
        adam_beta2=training_config.get("adam_beta2", 0.999),
        adam_epsilon=training_config.get("adam_epsilon", 1e-8),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        logging_dir=training_config.get("logging_dir", "./outputs/sft/logs"),
        logging_steps=training_config.get("logging_steps", 10),
        logging_strategy=training_config.get("logging_strategy", "steps"),
        report_to=training_config.get("report_to", "tensorboard"),
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        ddp_find_unused_parameters=training_config.get("ddp_find_unused_parameters", False),
        deepspeed=config.get("deepspeed"),
    )

    # Create datasets if not provided
    if train_dataset is None and data_config is not None:
        from vtg_datasets.video_dataset import VideoTemporalSFTDataset

        dataset_config = data_config.get("dataset", {})
        video_config = data_config.get("video", {})
        temporal_cfg = data_config.get("temporal", {})

        train_dataset = VideoTemporalSFTDataset(
            annotation_file=dataset_config.get("annotation_file"),
            video_dir=dataset_config.get("video_dir"),
            processor=processor,
            tokenizer=tokenizer,
            max_frames=video_config.get("max_frames", 32),
            use_relative_timestamps=temporal_cfg.get("use_relative_timestamps", True),
            num_bins=temporal_cfg.get("num_bins", 100),
            use_temporal_tokens=use_temporal_tokens,
            min_pixels=video_config.get("min_pixels"),
            max_pixels=video_config.get("max_pixels"),
            total_pixels=video_config.get("total_pixels"),
        )

    if eval_dataset is None and data_config is not None:
        validation_config = data_config.get("validation", {})
        if validation_config.get("annotation_file"):
            from vtg_datasets.video_dataset import VideoTemporalSFTDataset

            temporal_cfg = data_config.get("temporal", {})
            video_cfg = data_config.get("video", {})
            eval_dataset = VideoTemporalSFTDataset(
                annotation_file=validation_config.get("annotation_file"),
                video_dir=data_config.get("dataset", {}).get("video_dir"),
                processor=processor,
                tokenizer=tokenizer,
                max_frames=video_cfg.get("max_frames", 32),
                use_relative_timestamps=temporal_cfg.get("use_relative_timestamps", True),
                num_bins=temporal_cfg.get("num_bins", 100),
                use_temporal_tokens=use_temporal_tokens,
                min_pixels=video_cfg.get("min_pixels"),
                max_pixels=video_cfg.get("max_pixels"),
                total_pixels=video_cfg.get("total_pixels"),
            )

    # Create data collator
    from vtg_datasets.collate_fns import create_sft_collator

    data_collator = create_sft_collator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=training_config.get("max_length", 2048),
    )

    # Get duration batching configuration from data config
    duration_batching_config = None
    if data_config is not None:
        db_config = data_config.get("duration_batching", {})
        if db_config.get("enabled", False):
            # Convert OmegaConf to native Python dict if needed
            if isinstance(db_config, DictConfig):
                duration_batching_config = OmegaConf.to_container(db_config, resolve=True)
            else:
                duration_batching_config = dict(db_config)

    # Setup callbacks
    from trainers.callbacks import BatchLoggingCallback
    
    all_callbacks = []
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Add batch logging callback if duration batching is enabled
    if duration_batching_config:
        all_callbacks.append(BatchLoggingCallback(log_batch_stats=True))

    # Create trainer
    trainer = VideoTemporalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        processor=processor,
        callbacks=all_callbacks if all_callbacks else None,
        mm_projector_lr=training_config.get("mm_projector_lr"),
        vision_tower_lr=training_config.get("vision_tower_lr"),
        head_lr=training_config.get("head_lr"),
        duration_batching_config=duration_batching_config,
    )

    # Log training info
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['trainable']:,} trainable, {param_info['frozen']:,} frozen")

    return trainer
