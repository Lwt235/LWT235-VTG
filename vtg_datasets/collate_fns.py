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
    temporal_loss_only: bool = False
    
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
        
        # Apply temporal loss masking if enabled
        # This masks all tokens except those within the temporal response
        # (i.e., all tokens from <|box_start|> to <|box_end|> inclusive,
        # which typically contains the temporal tokens like <250>)
        if self.temporal_loss_only:
            labels = self._mask_non_temporal_tokens(labels)
        
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
    
    def _mask_non_temporal_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask all tokens except temporal response tokens.
        
        Only keeps labels for tokens between <|box_start|> and <|box_end|> (inclusive).
        This includes the box markers themselves and any tokens in between
        (typically the temporal tokens like <250>, <500>).
        All other tokens are set to -100 (ignored in loss calculation).
        
        Args:
            labels: Tensor of shape (batch_size, seq_len) with token IDs.
            
        Returns:
            Modified labels tensor with non-temporal tokens masked to -100.
        """
        # Get token IDs for box markers
        box_start_token = "<|box_start|>"
        box_end_token = "<|box_end|>"
        
        box_start_id = self.tokenizer.convert_tokens_to_ids(box_start_token)
        box_end_id = self.tokenizer.convert_tokens_to_ids(box_end_token)
        
        # Check if tokens exist in vocabulary by verifying they don't map to UNK
        # and that they can be converted back to the original token string
        unk_id = getattr(self.tokenizer, 'unk_token_id', None)
        box_start_valid = (
            unk_id is None or box_start_id != unk_id
        ) and self.tokenizer.convert_ids_to_tokens(box_start_id) == box_start_token
        box_end_valid = (
            unk_id is None or box_end_id != unk_id
        ) and self.tokenizer.convert_ids_to_tokens(box_end_id) == box_end_token
        
        if not box_start_valid or not box_end_valid:
            logger.warning(
                f"Box tokens not found in tokenizer vocabulary. "
                f"box_start_id={box_start_id}, box_end_id={box_end_id}. "
                f"Falling back to full sequence loss."
            )
            return labels
        
        batch_size, seq_len = labels.shape
        masked_labels = torch.full_like(labels, -100)
        
        for i in range(batch_size):
            seq = labels[i]
            
            # Find positions of box_start and box_end tokens
            box_start_positions = (seq == box_start_id).nonzero(as_tuple=True)[0]
            box_end_positions = (seq == box_end_id).nonzero(as_tuple=True)[0]
            
            if len(box_start_positions) == 0 or len(box_end_positions) == 0:
                # No temporal response found, keep full sequence loss for this sample
                masked_labels[i] = seq
                continue
            
            # Use the first occurrence of box_start and the first box_end after it
            box_start_pos = box_start_positions[0].item()
            
            # Find the first box_end after box_start
            valid_end_positions = box_end_positions[box_end_positions > box_start_pos]
            if len(valid_end_positions) == 0:
                # No matching box_end found, keep full sequence loss
                masked_labels[i] = seq
                continue
            
            box_end_pos = valid_end_positions[0].item()
            
            # Keep labels only for tokens from box_start to box_end (inclusive)
            temporal_range = slice(box_start_pos, box_end_pos + 1)
            masked_labels[i, temporal_range] = seq[temporal_range]
        
        return masked_labels


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
