#!/usr/bin/env python
"""
SFT Training Entry Point for Video Temporal Localization.

Usage:
    python train_sft.py --config configs/training/sft_config.yaml \
                        --data_config configs/data/video_config.yaml \
                        --output_dir ./outputs/sft
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
from utils.common import seed_everything, load_config, merge_configs
from trainers import create_sft_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for Video Temporal Localization"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/sft_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/video_config.yaml",
        help="Path to data configuration file",
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
        "--batch_size",
        type=int,
        default=None,
        help="Per-device batch size (overrides config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
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
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.deepspeed:
        config.deepspeed = args.deepspeed
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.seed:
        config.training.seed = args.seed
    
    # Setup output directory
    output_dir = config.training.get("output_dir", "./outputs/sft")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(
        name="sft_training",
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
    logger.info("Creating SFT trainer...")
    trainer = create_sft_trainer(
        config=config,
        data_config=data_config,
    )
    
    # Start training
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
