"""
Video Dataset for Temporal Localization.

Provides dataset classes for loading and processing video data
for temporal grounding tasks.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from utils.logging_utils import get_logger
from utils.data_validation import REQUIRED_FIELDS, OPTIONAL_KNOWN_FIELDS
from utils.temporal_tokens import (
    format_temporal_response,
    timestamp_to_temporal_tokens,
    NUM_TEMPORAL_TOKENS,
)

logger = get_logger(__name__)


class VideoTemporalDataset(Dataset):
    """
    Base dataset for video temporal grounding.

    Loads video-text pairs with temporal annotations from JSONL files.
    """

    def __init__(
        self,
        annotation_file: Union[str, Path],
        video_dir: Optional[Union[str, Path]] = None,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_frames: int = 32,
        use_relative_timestamps: bool = True,
        num_bins: int = 100,
        use_temporal_tokens: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            annotation_file: Path to JSONL annotation file.
            video_dir: Base directory for video files.
            processor: Qwen VL processor for video preprocessing.
            tokenizer: Tokenizer for text processing.
            max_frames: Maximum number of frames to sample.
            use_relative_timestamps: Whether to normalize timestamps to [0, 1].
            num_bins: Number of temporal bins for discretization.
            use_temporal_tokens: Whether to use temporal tokens (<0>~<999>) for output.
            transform: Optional transform to apply to samples.
        """
        self.annotation_file = Path(annotation_file)
        self.video_dir = Path(video_dir) if video_dir else None
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.use_relative_timestamps = use_relative_timestamps
        self.num_bins = num_bins
        self.use_temporal_tokens = use_temporal_tokens
        self.transform = transform

        # Load annotations
        self.samples = self._load_annotations()
        logger.info(f"Loaded {len(self.samples)} samples from {annotation_file}")

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load and parse annotation file.

        Supports two formats:
        1. JSON array format: A single JSON array containing annotation objects
        2. JSONL format: One JSON object per line (legacy format)
        """
        samples = []

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        # First, try to detect the file format by reading the first non-empty character
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            first_char = None
            for char in iter(lambda: f.read(1), ''):
                if char and not char.isspace():
                    first_char = char
                    break

        if first_char is None:
            return samples

        # If starts with '[', try JSON array format
        if first_char == '[':
            try:
                with open(self.annotation_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # JSON array format
                    for idx, sample in enumerate(data, start=1):
                        if isinstance(sample, dict):
                            sample_copy = sample.copy()
                            sample_copy["_line_num"] = idx
                            samples.append(sample_copy)
                        else:
                            logger.warning(f"Item {idx}: Expected dict, got {type(sample).__name__}")
                    return samples
            except json.JSONDecodeError:
                # Failed to parse as JSON array, fall back to JSONL
                pass

        # Fall back to JSONL format (one JSON per line) - memory efficient line-by-line reading
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    sample["_line_num"] = line_num
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Get video path
        video_path = sample["video"]
        if self.video_dir:
            video_path = self.video_dir / video_path
        else:
            video_path = Path(video_path)

        # Get temporal info
        duration = sample["duration"]
        timestamp = sample["timestamp"]
        start_time, end_time = timestamp[0], timestamp[1]

        # Apply video trimming if specified
        video_start = sample.get("video_start")
        video_end = sample.get("video_end")

        if video_start is not None:
            start_time = max(0, start_time - video_start)
            end_time = max(0, end_time - video_start)
            duration = duration - video_start
            if video_end is not None:
                duration = min(duration, video_end - video_start)
        elif video_end is not None:
            duration = min(duration, video_end)

        # Ensure duration is positive to avoid errors in temporal token conversion
        if duration <= 0:
            # Fall back to original duration from sample if trimming resulted in invalid duration
            # Use max with 1e-6 to ensure a positive value even if original duration is also invalid
            original_duration = sample["duration"]
            duration = max(original_duration if original_duration > 0 else 1e-6, 1e-6)
            # Clamp timestamps to valid range
            start_time = max(0, min(start_time, duration))
            end_time = max(0, min(end_time, duration))

        # Normalize timestamps if needed
        if self.use_relative_timestamps and duration > 0:
            norm_start = start_time / duration
            norm_end = end_time / duration
        else:
            norm_start = start_time
            norm_end = end_time

        # Discretize to bins
        start_bin = int(norm_start * self.num_bins)
        end_bin = int(norm_end * self.num_bins)
        start_bin = max(0, min(self.num_bins - 1, start_bin))
        end_bin = max(0, min(self.num_bins - 1, end_bin))

        result = {
            "video_path": str(video_path),
            "video_start": video_start,
            "video_end": video_end,
            "query": sample["sentence"],
            "duration": duration,
            "timestamp": [start_time, end_time],
            "normalized_timestamp": [norm_start, norm_end],
            "temporal_bins": [start_bin, end_bin],
            "sample_idx": idx,
        }

        # Add temporal tokens if enabled
        if self.use_temporal_tokens:
            start_token, end_token = timestamp_to_temporal_tokens(
                start_time, end_time, duration, NUM_TEMPORAL_TOKENS
            )
            result["temporal_tokens"] = [start_token, end_token]
            result["temporal_response"] = format_temporal_response(
                start_time, end_time, duration, NUM_TEMPORAL_TOKENS
            )

        # Include additional metadata fields (any fields beyond the required ones)
        # Note: kwargs field was a misunderstanding - additional fields are stored directly
        # Uses constants from utils.data_validation for consistency
        required_fields = set(REQUIRED_FIELDS) | {"_line_num"}
        optional_known_fields = {"video_start", "video_end"}
        all_known_fields = required_fields | optional_known_fields

        metadata = {}
        for key, value in sample.items():
            if key not in all_known_fields:
                metadata[key] = value

        if metadata:
            result["metadata"] = metadata

        if self.transform:
            result = self.transform(result)

        return result

    def get_video_messages(
        self,
        video_path: str,
        query: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create message format for Qwen VL model.

        Args:
            video_path: Path to video file.
            query: Text query.
            video_start: Optional start time for trimming.
            video_end: Optional end time for trimming.

        Returns:
            List of messages in Qwen VL format.
        """
        video_content = {
            "type": "video",
            "video": video_path,
        }

        if video_start is not None:
            video_content["video_start"] = video_start
        if video_end is not None:
            video_content["video_end"] = video_end

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": query},
                ],
            }
        ]

        return messages


class VideoTemporalSFTDataset(VideoTemporalDataset):
    """
    Dataset for Supervised Fine-Tuning on video temporal grounding.

    Extends VideoTemporalDataset with prompt formatting and label generation.
    """

    # Default prompt template
    DEFAULT_PROMPT = (
        "Given the video, please identify the start and end time of the moment "
        "described by the following query: \"{query}\"\n"
        "Provide the answer in the format: [start_time, end_time]"
    )

    # Prompt template for temporal tokens mode
    TEMPORAL_TOKEN_PROMPT = (
        "Given the video, please identify the start and end time of the moment "
        "described by the following query: \"{query}\"\n"
        "Provide the answer using temporal tokens in the format: [start, end]"
    )

    def __init__(
        self,
        annotation_file: Union[str, Path],
        video_dir: Optional[Union[str, Path]] = None,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_frames: int = 32,
        use_relative_timestamps: bool = True,
        num_bins: int = 100,
        use_temporal_tokens: bool = False,
        prompt_template: Optional[str] = None,
        response_template: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the SFT dataset.

        Args:
            annotation_file: Path to JSONL annotation file.
            video_dir: Base directory for video files.
            processor: Qwen VL processor for video preprocessing.
            tokenizer: Tokenizer for text processing.
            max_frames: Maximum number of frames to sample.
            use_relative_timestamps: Whether to normalize timestamps to [0, 1].
            num_bins: Number of temporal bins for discretization.
            use_temporal_tokens: Whether to use temporal tokens (<0>~<999>) for output.
            prompt_template: Custom prompt template with {query} placeholder.
            response_template: Custom response template with {start} and {end} placeholders.
            transform: Optional transform to apply to samples.
        """
        super().__init__(
            annotation_file=annotation_file,
            video_dir=video_dir,
            processor=processor,
            tokenizer=tokenizer,
            max_frames=max_frames,
            use_relative_timestamps=use_relative_timestamps,
            num_bins=num_bins,
            use_temporal_tokens=use_temporal_tokens,
            transform=transform,
        )

        # Set prompt and response templates based on temporal tokens mode
        if use_temporal_tokens:
            self.prompt_template = prompt_template or self.TEMPORAL_TOKEN_PROMPT
            # Response template not used with temporal tokens (generated dynamically)
            self.response_template = None
        else:
            self.prompt_template = prompt_template or self.DEFAULT_PROMPT
            self.response_template = response_template or "<|box_start|><{start:.2f}><{end:.2f}><|box_end|>"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base sample
        sample = super().__getitem__(idx)

        # Format prompt
        prompt = self.prompt_template.format(query=sample["query"])

        # Format response (ground truth)
        if self.use_temporal_tokens:
            # Use temporal tokens from base dataset
            response = sample.get("temporal_response", "")
        else:
            if self.use_relative_timestamps:
                start, end = sample["normalized_timestamp"]
            else:
                start, end = sample["timestamp"]
            response = self.response_template.format(start=start, end=end)

        sample["prompt"] = prompt
        sample["response"] = response

        # Create messages for Qwen VL
        sample["messages"] = self._create_messages(sample, prompt, response)

        return sample

    def _create_messages(
        self,
        sample: Dict[str, Any],
        prompt: str,
        response: str,
    ) -> List[Dict[str, Any]]:
        """Create training messages with video, prompt, and response."""
        video_content = {
            "type": "video",
            "video": sample["video_path"],
        }

        if sample.get("video_start") is not None:
            video_content["video_start"] = sample["video_start"]
        if sample.get("video_end") is not None:
            video_content["video_end"] = sample["video_end"]

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return messages


class VideoTemporalRLDataset(VideoTemporalDataset):
    """
    Dataset for Reinforcement Learning on video temporal grounding.

    Provides samples for GRPO/R1 training with reward computation support.
    """

    DEFAULT_PROMPT = (
        "Watch the video and identify when the following event occurs: \"{query}\"\n"
        "Respond with the start and end times of the relevant segment."
    )

    TEMPORAL_TOKEN_PROMPT = (
        "Watch the video and identify when the following event occurs: \"{query}\"\n"
        "Respond with the temporal tokens indicating start and end times."
    )

    def __init__(
        self,
        annotation_file: Union[str, Path],
        video_dir: Optional[Union[str, Path]] = None,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_frames: int = 32,
        use_relative_timestamps: bool = True,
        num_bins: int = 100,
        use_temporal_tokens: bool = False,
        prompt_template: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the RL dataset.

        Args:
            annotation_file: Path to JSONL annotation file.
            video_dir: Base directory for video files.
            processor: Qwen VL processor for video preprocessing.
            tokenizer: Tokenizer for text processing.
            max_frames: Maximum number of frames to sample.
            use_relative_timestamps: Whether to normalize timestamps to [0, 1].
            num_bins: Number of temporal bins for discretization.
            use_temporal_tokens: Whether to use temporal tokens (<0>~<999>) for output.
            prompt_template: Custom prompt template with {query} placeholder.
            transform: Optional transform to apply to samples.
        """
        super().__init__(
            annotation_file=annotation_file,
            video_dir=video_dir,
            processor=processor,
            tokenizer=tokenizer,
            max_frames=max_frames,
            use_relative_timestamps=use_relative_timestamps,
            num_bins=num_bins,
            use_temporal_tokens=use_temporal_tokens,
            transform=transform,
        )

        # Set prompt template based on temporal tokens mode
        if use_temporal_tokens:
            self.prompt_template = prompt_template or self.TEMPORAL_TOKEN_PROMPT
        else:
            self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base sample
        sample = super().__getitem__(idx)

        # Format prompt
        prompt = self.prompt_template.format(query=sample["query"])
        sample["prompt"] = prompt

        # Store ground truth for reward computation
        ground_truth = {
            "timestamp": sample["timestamp"],
            "normalized_timestamp": sample["normalized_timestamp"],
            "duration": sample["duration"],
        }

        # Add temporal tokens info if enabled
        if self.use_temporal_tokens and "temporal_tokens" in sample:
            ground_truth["temporal_tokens"] = sample["temporal_tokens"]

        sample["ground_truth"] = ground_truth

        # Create messages for generation
        sample["messages"] = self._create_prompt_messages(sample, prompt)

        return sample

    def _create_prompt_messages(
        self,
        sample: Dict[str, Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Create prompt messages for generation (without response)."""
        video_content = {
            "type": "video",
            "video": sample["video_path"],
        }

        if sample.get("video_start") is not None:
            video_content["video_start"] = sample["video_start"]
        if sample.get("video_end") is not None:
            video_content["video_end"] = sample["video_end"]

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        return messages
