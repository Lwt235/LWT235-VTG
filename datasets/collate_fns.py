"""
Collate Functions for Video Temporal Localization.

Provides collators for batching video samples with proper padding and masking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SFTCollator:
    """
    Collator for Supervised Fine-Tuning.
    
    Handles batching of video-text samples with proper padding.
    """
    
    processor: Any
    tokenizer: Any
    max_length: int = 2048
    padding: Union[bool, str] = True
    truncation: bool = True
    return_tensors: str = "pt"
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries.
        
        Returns:
            Batched tensors ready for model input.
        """
        # Extract messages and metadata
        messages_list = [sample["messages"] for sample in batch]
        
        # Process with Qwen VL processor
        try:
            # Apply chat template and process
            texts = []
            for messages in messages_list:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            
            # Process videos and text together
            # Note: This requires qwen_vl_utils for video processing
            from qwen_vl_utils import process_vision_info
            
            image_inputs = []
            video_inputs = []
            
            for messages in messages_list:
                images, videos = process_vision_info(messages)
                image_inputs.append(images)
                video_inputs.append(videos)
            
            # Tokenize
            model_inputs = self.processor(
                text=texts,
                images=image_inputs if any(image_inputs) else None,
                videos=video_inputs if any(video_inputs) else None,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
            
        except ImportError:
            logger.warning(
                "qwen_vl_utils not available. Using basic text tokenization."
            )
            texts = []
            for messages in messages_list:
                # Simple concatenation without vision processing
                text_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    text_parts.append(f"{role}: {content}")
                texts.append("\n".join(text_parts))
            
            model_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
            )
        
        # Create labels (shift input_ids for causal LM)
        labels = model_inputs["input_ids"].clone()
        
        # Mask padding tokens in labels
        if "attention_mask" in model_inputs:
            labels[model_inputs["attention_mask"] == 0] = -100
        
        # Mask prompt tokens (only train on response)
        # This requires finding the response boundary
        # For simplicity, we'll mask based on a heuristic
        model_inputs["labels"] = labels
        
        # Add metadata
        model_inputs["temporal_bins"] = torch.tensor(
            [sample["temporal_bins"] for sample in batch],
            dtype=torch.long,
        )
        
        model_inputs["sample_indices"] = torch.tensor(
            [sample["sample_idx"] for sample in batch],
            dtype=torch.long,
        )
        
        return model_inputs


@dataclass
class RLCollator:
    """
    Collator for Reinforcement Learning (GRPO).
    
    Handles batching with support for multiple generations per prompt.
    """
    
    processor: Any
    tokenizer: Any
    max_length: int = 2048
    max_prompt_length: int = 1024
    padding: Union[bool, str] = True
    truncation: bool = True
    return_tensors: str = "pt"
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples for RL training.
        
        Args:
            batch: List of sample dictionaries.
        
        Returns:
            Batched data ready for RL training.
        """
        messages_list = [sample["messages"] for sample in batch]
        
        # Process prompts only (without responses)
        try:
            texts = []
            for messages in messages_list:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)
            
            from qwen_vl_utils import process_vision_info
            
            video_inputs = []
            for messages in messages_list:
                _, videos = process_vision_info(messages)
                video_inputs.append(videos)
            
            prompt_inputs = self.processor(
                text=texts,
                videos=video_inputs if any(video_inputs) else None,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_prompt_length,
                return_tensors=self.return_tensors,
            )
            
        except ImportError:
            logger.warning(
                "qwen_vl_utils not available. Using basic text tokenization."
            )
            texts = []
            for messages in messages_list:
                text_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    text_parts.append(f"{role}: {content}")
                texts.append("\n".join(text_parts))
            
            prompt_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_prompt_length,
                return_tensors=self.return_tensors,
            )
        
        # Extract ground truth for reward computation
        ground_truths = [sample["ground_truth"] for sample in batch]
        
        result = {
            "prompt_input_ids": prompt_inputs["input_ids"],
            "prompt_attention_mask": prompt_inputs["attention_mask"],
            "ground_truths": ground_truths,
            "prompts": [sample["prompt"] for sample in batch],
            "queries": [sample["query"] for sample in batch],
            "sample_indices": torch.tensor(
                [sample["sample_idx"] for sample in batch],
                dtype=torch.long,
            ),
        }
        
        # Include pixel values if available
        if "pixel_values" in prompt_inputs:
            result["pixel_values"] = prompt_inputs["pixel_values"]
        if "pixel_values_videos" in prompt_inputs:
            result["pixel_values_videos"] = prompt_inputs["pixel_values_videos"]
        if "video_grid_thw" in prompt_inputs:
            result["video_grid_thw"] = prompt_inputs["video_grid_thw"]
        
        return result


def create_sft_collator(
    processor: Any,
    tokenizer: Any,
    max_length: int = 2048,
    **kwargs,
) -> SFTCollator:
    """
    Create an SFT collator with the given configuration.
    
    Args:
        processor: Qwen VL processor.
        tokenizer: Tokenizer.
        max_length: Maximum sequence length.
        **kwargs: Additional arguments for SFTCollator.
    
    Returns:
        Configured SFTCollator instance.
    """
    return SFTCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        **kwargs,
    )


def create_rl_collator(
    processor: Any,
    tokenizer: Any,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    **kwargs,
) -> RLCollator:
    """
    Create an RL collator with the given configuration.
    
    Args:
        processor: Qwen VL processor.
        tokenizer: Tokenizer.
        max_length: Maximum sequence length.
        max_prompt_length: Maximum prompt length.
        **kwargs: Additional arguments for RLCollator.
    
    Returns:
        Configured RLCollator instance.
    """
    return RLCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        **kwargs,
    )


def pad_temporal_sequences(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Pad temporal sequences to the same length.
    
    Args:
        sequences: List of 1D tensors.
        padding_value: Value to use for padding.
        max_length: Maximum length (uses longest sequence if None).
    
    Returns:
        Padded tensor of shape (batch_size, max_length).
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = torch.full(
        (len(sequences), max_length),
        padding_value,
        dtype=sequences[0].dtype,
    )
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    
    return padded


def create_temporal_mask(
    lengths: List[int],
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Create attention mask for temporal sequences.
    
    Args:
        lengths: List of sequence lengths.
        max_length: Maximum length (uses longest if None).
    
    Returns:
        Boolean mask of shape (batch_size, max_length).
    """
    if max_length is None:
        max_length = max(lengths)
    
    mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask
