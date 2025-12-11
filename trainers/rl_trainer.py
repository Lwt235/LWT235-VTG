"""
Reinforcement Learning Trainer for Video Temporal Localization.

Provides GRPO/R1 training with reward functions for temporal grounding.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from trl import GRPOConfig, GRPOTrainer
from omegaconf import DictConfig, OmegaConf

from utils.logging_utils import get_logger
from utils.common import load_config, merge_configs, count_parameters
from rewards import CompositeReward, create_composite_reward

logger = get_logger(__name__)


class VideoTemporalRLTrainer:
    """
    RL Trainer for video temporal grounding using GRPO.
    
    Implements Group Relative Policy Optimization with temporal grounding rewards.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel] = None,
        reward_model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        processor: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[DictConfig] = None,
        data_collator: Optional[Callable] = None,
    ):
        """
        Initialize the RL trainer.
        
        Args:
            model: The model to train.
            ref_model: Reference model for KL divergence.
            reward_model: Reward model/function.
            tokenizer: Tokenizer for text processing.
            processor: Qwen VL processor.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            config: Training configuration.
            data_collator: Data collator for batching.
        """
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or OmegaConf.create({})
        self.data_collator = data_collator
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float("-inf")
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Generation config
        grpo_config = self.config.get("grpo", {})
        gen_config = grpo_config.get("generation", {})
        
        self.generation_config = GenerationConfig(
            max_new_tokens=gen_config.get("max_new_tokens", 64),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            do_sample=gen_config.get("do_sample", True),
            num_beams=gen_config.get("num_beams", 1),
        )
        
        self.num_generations = grpo_config.get("num_generations", 4)
    
    def generate_responses(
        self,
        batch: Dict[str, torch.Tensor],
        num_generations: Optional[int] = None,
    ) -> List[str]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            batch: Input batch with prompts.
            num_generations: Number of generations per prompt.
        
        Returns:
            List of generated response strings.
        """
        num_generations = num_generations or self.num_generations
        
        input_ids = batch["prompt_input_ids"].to(self.device)
        attention_mask = batch["prompt_attention_mask"].to(self.device)
        
        batch_size = input_ids.shape[0]
        all_responses = []
        
        # Generate for each prompt
        for _ in range(num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode responses (excluding prompt)
            prompt_length = input_ids.shape[1]
            response_ids = outputs[:, prompt_length:]
            
            responses = self.tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
            )
            
            all_responses.extend(responses)
        
        return all_responses
    
    def compute_rewards(
        self,
        responses: List[str],
        ground_truths: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        Args:
            responses: List of generated response strings.
            ground_truths: List of ground truth dictionaries.
        
        Returns:
            Reward tensor of shape (num_responses,).
        """
        # Extract ground truth timestamps
        gt_timestamps = []
        durations = []
        
        for gt in ground_truths:
            gt_timestamps.append(tuple(gt["normalized_timestamp"]))
            durations.append(gt["duration"])
        
        # Repeat ground truths for multiple generations
        num_generations = len(responses) // len(ground_truths)
        gt_timestamps = gt_timestamps * num_generations
        durations = durations * num_generations
        
        # Compute rewards using reward model
        if self.reward_model is not None:
            rewards = self.reward_model.compute_from_text(
                predictions=responses,
                ground_truths=gt_timestamps,
                durations=durations,
            )
        else:
            # Default to temporal IoU
            from rewards import TemporalIoU
            reward_fn = TemporalIoU()
            rewards = reward_fn.compute_from_text(
                predictions=responses,
                ground_truths=gt_timestamps,
                durations=durations,
            )
        
        return rewards.to(self.device)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute advantages using GRPO-style group relative estimation.
        
        Args:
            rewards: Reward tensor of shape (num_responses,).
            batch_size: Original batch size (before generation expansion).
        
        Returns:
            Advantage tensor of shape (num_responses,).
        """
        config = self.config.get("grpo", {}).get("advantage", {})
        baseline_type = config.get("baseline_type", "mean")
        normalize = config.get("normalize", True)
        clip_range = config.get("clip_range", 10.0)
        
        # Reshape to (batch_size, num_generations)
        rewards_grouped = rewards.view(self.num_generations, batch_size).T
        
        # Compute baseline
        if baseline_type == "mean":
            baseline = rewards_grouped.mean(dim=1, keepdim=True)
        elif baseline_type == "none":
            baseline = 0.0
        else:
            baseline = rewards_grouped.mean(dim=1, keepdim=True)
        
        # Compute advantages
        advantages = rewards_grouped - baseline
        
        # Normalize
        if normalize:
            std = advantages.std() + 1e-8
            advantages = advantages / std
        
        # Clip
        advantages = torch.clamp(advantages, -clip_range, clip_range)
        
        # Flatten back
        advantages = advantages.T.contiguous().view(-1)
        
        return advantages
    
    def compute_policy_loss(
        self,
        model_outputs: Any,
        response_ids: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO-style policy loss.
        
        Args:
            model_outputs: Model forward outputs.
            response_ids: Generated response token IDs.
            advantages: Advantage estimates.
            old_log_probs: Log probabilities from generation.
        
        Returns:
            Policy loss tensor.
        """
        config = self.config.get("grpo", {}).get("policy", {})
        clip_range = config.get("clip_range", 0.2)
        
        # Get log probabilities for responses
        logits = model_outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        new_log_probs = torch.gather(
            log_probs[:, :-1],
            dim=-1,
            index=response_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        
        # Sum over sequence
        new_log_probs = new_log_probs.sum(dim=-1)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_kl_penalty(
        self,
        model_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty.
        
        Args:
            model_log_probs: Log probabilities from current model.
            ref_log_probs: Log probabilities from reference model.
        
        Returns:
            KL divergence tensor.
        """
        kl = (ref_log_probs.exp() * (ref_log_probs - model_log_probs)).sum(dim=-1)
        return kl.mean()
    
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch.
        
        Returns:
            Dictionary of metrics.
        """
        self.model.train()
        
        batch_size = len(batch["ground_truths"])
        
        # Generate responses
        responses = self.generate_responses(batch)
        
        # Compute rewards
        rewards = self.compute_rewards(responses, batch["ground_truths"])
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, batch_size)
        
        # Forward pass for policy loss
        # Note: This is a simplified implementation for the initial framework.
        # A full GRPO implementation would:
        # 1. Track old log probabilities during generation
        # 2. Compute new log probabilities for the generated responses
        # 3. Apply PPO-style clipping with the advantage estimates
        # 4. Add KL penalty between current and reference policy
        # The TRL library's GRPOTrainer can be used for full functionality.
        # TODO: Implement complete policy gradient updates in future iterations
        
        metrics = {
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }
        
        self.global_step += 1
        
        return metrics
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the training loop.
        
        Args:
            num_epochs: Number of training epochs.
            max_steps: Maximum training steps.
        """
        training_config = self.config.get("training", {})
        num_epochs = num_epochs or training_config.get("num_train_epochs", 1)
        max_steps = max_steps or training_config.get("max_steps", -1)
        
        logger.info(f"Starting RL training for {num_epochs} epochs")
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=training_config.get("per_device_train_batch_size", 1),
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=training_config.get("dataloader_num_workers", 4),
        )
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = {"reward_mean": 0, "reward_std": 0}
            num_batches = 0
            
            for batch in dataloader:
                if max_steps > 0 and self.global_step >= max_steps:
                    break
                
                metrics = self.train_step(batch)
                
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                num_batches += 1
                
                if self.global_step % training_config.get("logging_steps", 10) == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"reward={metrics['reward_mean']:.4f}"
                    )
            
            # Log epoch summary
            for k in epoch_metrics:
                epoch_metrics[k] /= max(num_batches, 1)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: {epoch_metrics}")
            
            # Save checkpoint
            if training_config.get("save_strategy") == "epoch":
                self.save_checkpoint(
                    os.path.join(
                        training_config.get("output_dir", "./outputs/rl"),
                        f"checkpoint-epoch-{epoch + 1}",
                    )
                )
    
    def save_checkpoint(self, output_dir: str):
        """
        Save a training checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer and processor
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
        }
        torch.save(state, os.path.join(output_dir, "trainer_state.pt"))
        
        logger.info(f"Checkpoint saved to {output_dir}")


def create_rl_trainer(
    config: Union[str, Path, Dict, DictConfig],
    data_config: Optional[Union[str, Path, Dict, DictConfig]] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
) -> VideoTemporalRLTrainer:
    """
    Create an RL trainer from configuration.
    
    Args:
        config: Training configuration (path or dict).
        data_config: Data configuration (path or dict).
        train_dataset: Pre-built training dataset.
        eval_dataset: Pre-built evaluation dataset.
    
    Returns:
        Configured VideoTemporalRLTrainer instance.
    """
    # Load configurations
    if isinstance(config, (str, Path)):
        config = load_config(config)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)
    
    if data_config is None:
        data_config_path = config.get("data", {}).get("config_path")
        if data_config_path:
            data_config = load_config(data_config_path)
    elif isinstance(data_config, (str, Path)):
        data_config = load_config(data_config)
    elif isinstance(data_config, dict):
        data_config = OmegaConf.create(data_config)
    
    # Load model and processor
    model_config = config.get("model", {})
    model_name = model_config.get("name_or_path", "./outputs/sft/checkpoint-best")
    
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
    
    # Load model
    torch_dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    
    # Apply LoRA if configured
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False) and not isinstance(model, PeftModel):
        logger.info("Applying LoRA configuration")
        
        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 128),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load reference model if needed
    ref_model_config = config.get("ref_model", {})
    ref_model = None
    
    if ref_model_config.get("name_or_path"):
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_config["name_or_path"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    # Create reward model
    reward_config = config.get("rewards", {})
    reward_model = create_composite_reward(dict(reward_config))
    
    # Create datasets if not provided
    if train_dataset is None and data_config is not None:
        from datasets.video_dataset import VideoTemporalRLDataset
        
        dataset_config = data_config.get("dataset", {})
        video_config = data_config.get("video", {})
        temporal_config = data_config.get("temporal", {})
        
        train_dataset = VideoTemporalRLDataset(
            annotation_file=dataset_config.get("annotation_file"),
            video_dir=dataset_config.get("video_dir"),
            processor=processor,
            tokenizer=tokenizer,
            max_frames=video_config.get("max_frames", 32),
            use_relative_timestamps=temporal_config.get("use_relative_timestamps", True),
            num_bins=temporal_config.get("num_bins", 100),
        )
    
    # Create data collator
    from datasets.collate_fns import create_rl_collator
    
    data_collator = create_rl_collator(
        processor=processor,
        tokenizer=tokenizer,
    )
    
    # Create trainer
    trainer = VideoTemporalRLTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        data_collator=data_collator,
    )
    
    # Log training info
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['trainable']:,} trainable, {param_info['frozen']:,} frozen")
    
    return trainer
