#!/usr/bin/env python
"""
RL Training Entry Point for Video Temporal Localization.

Usage:
    python train_rl.py --config configs/training/rl_config.yaml \
                       --data_config configs/data/video_config.yaml \
                       --model_path ./outputs/sft/checkpoint-best \
                       --output_dir ./outputs/rl
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from omegaconf import OmegaConf

from utils.logging_utils import setup_logger, get_logger
from utils.common import seed_everything, load_config
from trainers import create_rl_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning (GRPO) for Video Temporal Localization"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/rl_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/video_config.yaml",
        help="Path to data configuration file",
    )
    
    # Model path
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to SFT checkpoint (overrides config)",
    )
    
    # Override options
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    
    # Local rank for distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # Apply command-line overrides
    if args.model_path:
        config.model.name_or_path = args.model_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.deepspeed:
        config.deepspeed = args.deepspeed
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.seed:
        config.training.seed = args.seed
    
    # Setup output directory
    output_dir = config.training.get("output_dir", "./outputs/rl")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        name="rl_training",
        level="INFO",
        log_file=os.path.join(output_dir, "training.log"),
    )
    
    # Set seed for reproducibility
    seed = config.training.get("seed", 42)
    seed_everything(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    logger.info(f"Configuration saved to {config_save_path}")
    
    data_config_save_path = os.path.join(output_dir, "data_config.yaml")
    OmegaConf.save(data_config, data_config_save_path)
    
    # Create trainer
    logger.info("Creating RL trainer...")
    trainer = create_rl_trainer(
        config=config,
        data_config=data_config,
    )
    
    # Start training
    logger.info("Starting RL (GRPO) training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_checkpoint(os.path.join(output_dir, "checkpoint-final"))
    
    logger.info("RL Training complete!")


if __name__ == "__main__":
    main()
