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
            video_metadata_list = []
            all_video_kwargs = {}
            
            for messages in messages_list:
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                image_inputs.append(images)
                
                if videos is not None and len(videos) > 0:
                    video_data, video_meta = zip(*videos)
                    video_inputs.append(list(video_data))
                    video_metadata_list.append(list(video_meta))
                else:
                    video_inputs.append(None)
                    video_metadata_list.append(None)
                
                # Merge video_kwargs (they should be the same for all samples)
                if video_kwargs:
                    all_video_kwargs.update(video_kwargs)
            
            # Flatten video inputs and metadata for batched processing
            flat_videos = []
            flat_video_metadata = []
            for vids, metas in zip(video_inputs, video_metadata_list):
                if vids is not None and metas is not None:
                    flat_videos.extend(vids)
                    flat_video_metadata.extend(metas)
            
            # Tokenize
            # Note: Disable truncation when processing videos to avoid mismatch
            # between video token placeholders in text and actual video inputs.
            # The Qwen VL processor cannot properly handle truncation of video tokens.
            has_videos = bool(flat_videos)
            
            # Build processor kwargs - only include max_length when truncation is enabled
            # to avoid warning about max_length being ignored
            processor_kwargs = {
                "text": texts,
                "images": image_inputs if any(image_inputs) else None,
                "videos": flat_videos if flat_videos else None,
                "video_metadata": flat_video_metadata if flat_video_metadata else None,
                "padding": self.padding,
                "truncation": False if has_videos else self.truncation,
                "return_tensors": self.return_tensors,
                "do_resize": False,
            }
            # Only pass max_length when truncation is enabled
            if not has_videos and self.truncation:
                processor_kwargs["max_length"] = self.max_length
            processor_kwargs.update(all_video_kwargs)
            
            model_inputs = self.processor(**processor_kwargs)
            
        except ImportError:
            logger.warning(
                "qwen_vl_utils not available. Using basic text tokenization. "
                "Install qwen_vl_utils for full video processing functionality. "
                "Without it, video content will not be processed correctly."
            )
            texts = []
            for messages in messages_list:
                # Simple concatenation without vision processing
                # Note: This fallback loses video information and should only
                # be used for testing or when video processing is not required.
                text_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if isinstance(content, list):
                        # Extract text and note video placeholders
                        text_content = []
                        for c in content:
                            if c.get("type") == "text":
                                text_content.append(c.get("text", ""))
                            elif c.get("type") == "video":
                                text_content.append(f"[VIDEO: {c.get('video', 'unknown')}]")
                        content = " ".join(text_content)
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
        
        # Mask prompt tokens (only train on assistant response)
        # Calculate the length of prompt for each sample to mask non-response tokens
        for i, messages in enumerate(messages_list):
            # Apply chat template to just the user message to find prompt length
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            
            try:
                # Get prompt text (without assistant response)
                prompt_text = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Adds assistant prefix
                )
                
                # Tokenize prompt to get its length
                # Use same processor/tokenizer for consistency
                prompt_ids = self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                )
                prompt_length = len(prompt_ids)
                
                # Mask all tokens in the prompt (before assistant response)
                # Handle edge case: if prompt_length >= seq_len, entire sequence is prompt
                if prompt_length >= labels.size(1):
                    logger.warning(
                        f"Sample {i}: Prompt length ({prompt_length}) >= sequence length "
                        f"({labels.size(1)}). Skipping masking to preserve at least some tokens."
                    )
                elif prompt_length > 0:
                    labels[i, :prompt_length] = -100
                    
            except Exception as e:
                # Fallback: If we can't determine prompt length, don't mask
                # This ensures training continues even if there's an issue
                logger.warning(
                    f"Could not determine prompt length for sample {i}: {e}. "
                    "Loss will be calculated on all tokens."
                )
        
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
            video_metadata_list = []
            all_video_kwargs = {}
            
            for messages in messages_list:
                _, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                
                if videos is not None and len(videos) > 0:
                    video_data, video_meta = zip(*videos)
                    video_inputs.append(list(video_data))
                    video_metadata_list.append(list(video_meta))
                else:
                    video_inputs.append(None)
                    video_metadata_list.append(None)
                
                # Merge video_kwargs (they should be the same for all samples)
                if video_kwargs:
                    all_video_kwargs.update(video_kwargs)
            
            # Flatten video inputs and metadata for batched processing
            flat_videos = []
            flat_video_metadata = []
            for vids, metas in zip(video_inputs, video_metadata_list):
                if vids is not None and metas is not None:
                    flat_videos.extend(vids)
                    flat_video_metadata.extend(metas)
            
            # Note: Disable truncation when processing videos to avoid mismatch
            # between video token placeholders in text and actual video inputs.
            has_videos = bool(flat_videos)
            
            # Build processor kwargs - only include max_length when truncation is enabled
            # to avoid warning about max_length being ignored
            processor_kwargs = {
                "text": texts,
                "videos": flat_videos if flat_videos else None,
                "video_metadata": flat_video_metadata if flat_video_metadata else None,
                "padding": self.padding,
                "truncation": False if has_videos else self.truncation,
                "return_tensors": self.return_tensors,
                "do_resize": False,
            }
            # Only pass max_length when truncation is enabled
            if not has_videos and self.truncation:
                processor_kwargs["max_length"] = self.max_prompt_length
            processor_kwargs.update(all_video_kwargs)
            
            prompt_inputs = self.processor(**processor_kwargs)
            
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
