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
            **kwargs: Additional arguments for base Trainer.
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )
        
        self.processor = processor
        self.mm_projector_lr = mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.head_lr = head_lr
    
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
        labels = inputs.pop("labels", None)
        
        # Remove metadata that shouldn't go to model
        inputs.pop("temporal_bins", None)
        inputs.pop("sample_indices", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        if labels is not None:
            # Compute loss with labels
            logits = outputs.logits
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            # Use model's computed loss
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
        Save the model and processor.
        
        Args:
            output_dir: Output directory.
            _internal_call: Whether this is an internal call.
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model (handles PEFT automatically)
        super().save_model(output_dir, _internal_call=_internal_call)
        
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
        dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    
    # Initialize temporal token embeddings if enabled
    if use_temporal_tokens:
        logger.info("Initializing temporal token embeddings with sinusoidal encoding")
        init_strategy = temporal_config.get("embedding_init_strategy", "sinusoidal")
        resize_model_embeddings_for_temporal_tokens(model, tokenizer, init_strategy)
    
    # Apply LoRA if configured
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        logger.info("Applying LoRA configuration")
        
        # Get target modules from config
        target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
        
        # When using temporal tokens, ensure embed_tokens and lm_head are included
        # in target_modules for proper adaptation to new token embeddings
        if use_temporal_tokens:
            if isinstance(target_modules, list):
                target_modules = list(target_modules)  # Make a copy
            else:
                target_modules = list(target_modules)
            
            if "embed_tokens" not in target_modules:
                target_modules.append("embed_tokens")
            if "lm_head" not in target_modules:
                target_modules.append("lm_head")
            logger.info(f"Temporal tokens enabled: target_modules = {target_modules}")
        
        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Create training arguments
    training_config = config.get("training", {})
    
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
        gradient_checkpointing_kwargs=training_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
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
        from datasets.video_dataset import VideoTemporalSFTDataset
        
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
        )
    
    if eval_dataset is None and data_config is not None:
        validation_config = data_config.get("validation", {})
        if validation_config.get("annotation_file"):
            from datasets.video_dataset import VideoTemporalSFTDataset
            
            temporal_cfg = data_config.get("temporal", {})
            eval_dataset = VideoTemporalSFTDataset(
                annotation_file=validation_config.get("annotation_file"),
                video_dir=data_config.get("dataset", {}).get("video_dir"),
                processor=processor,
                tokenizer=tokenizer,
                max_frames=data_config.get("video", {}).get("max_frames", 32),
                use_relative_timestamps=temporal_cfg.get("use_relative_timestamps", True),
                num_bins=temporal_cfg.get("num_bins", 100),
                use_temporal_tokens=use_temporal_tokens,
            )
    
    # Create data collator
    from datasets.collate_fns import create_sft_collator
    
    data_collator = create_sft_collator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=training_config.get("max_length", 2048),
    )
    
    # Create trainer
    trainer = VideoTemporalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        processor=processor,
        callbacks=callbacks,
        mm_projector_lr=training_config.get("mm_projector_lr"),
        vision_tower_lr=training_config.get("vision_tower_lr"),
        head_lr=training_config.get("head_lr"),
    )
    
    # Log training info
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['trainable']:,} trainable, {param_info['frozen']:,} frozen")
    
    return trainer
