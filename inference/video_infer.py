"""
Video Inference for Temporal Localization.

Provides inference capabilities with visualization for temporal grounding.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel

from utils.logging_utils import get_logger
from utils.common import load_config, seconds_to_timestamp
from utils.temporal_tokens import (
    parse_temporal_response,
    add_temporal_tokens_to_tokenizer,
    resize_model_embeddings_for_temporal_tokens,
    NUM_TEMPORAL_TOKENS,
)
from qwen_vl_utils import process_vision_info

logger = get_logger(__name__)


class VideoTemporalInference:
    """
    Inference engine for video temporal grounding.

    Supports single video and batch inference with visualization.
    """

    DEFAULT_PROMPT = (
        "Given the video, please identify the start and end time of the moment "
        "described by the following query: \"{query}\"\n"
        "Provide the answer in the format: <start_time> to <end_time>"
    )

    TEMPORAL_TOKEN_PROMPT = (
        "Given the video, please identify the start and end time of the moment "
        "described by the following query: \"{query}\"\n"
        "Provide the answer using temporal tokens in the format: <start><end>"
    )

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        prompt_template: Optional[str] = None,
        use_flash_attention: bool = True,
        use_temporal_tokens: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        total_pixels: Optional[int] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to model checkpoint or Hugging Face model ID.
            device: Device to run inference on ("cuda", "cpu", etc.).
            torch_dtype: Torch data type for model.
            prompt_template: Custom prompt template with {query} placeholder.
            use_flash_attention: Whether to use flash attention.
            use_temporal_tokens: Whether to use temporal tokens (<0>~<999>) for output.
            min_pixels: Minimum pixels per video frame (for GPU memory control).
            max_pixels: Maximum pixels per video frame (for GPU memory control).
            total_pixels: Total pixels across all video frames (for GPU memory control).
        """
        self.model_path = Path(model_path)
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_temporal_tokens = use_temporal_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

        # Set prompt template based on mode
        if use_temporal_tokens:
            self.prompt_template = prompt_template or self.TEMPORAL_TOKEN_PROMPT
        else:
            self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and processor
        self._load_model(use_flash_attention)

        logger.info(f"Inference engine initialized on {self.device}")

    def _load_model(self, use_flash_attention: bool):
        """Load model, processor, and tokenizer."""
        model_path_str = str(self.model_path)

        logger.info(f"Loading model from {model_path_str}")

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path_str,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_str,
            trust_remote_code=True,
        )

        # Add temporal tokens if enabled
        if self.use_temporal_tokens:
            logger.info("Adding temporal tokens (<0>~<999>) to tokenizer")
            add_temporal_tokens_to_tokenizer(self.tokenizer)

        # Load model
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path_str,
            dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map=self.device,
        )

        # Initialize temporal token embeddings if enabled
        if self.use_temporal_tokens:
            logger.info("Initializing temporal token embeddings with sinusoidal encoding")
            resize_model_embeddings_for_temporal_tokens(
                self.model, self.tokenizer, "sinusoidal"
            )

        self.model.eval()

        # Default generation config
        self.generation_config = GenerationConfig.from_pretrained(
            model_path_str
        )

    def predict(
        self,
        video_path: Union[str, Path],
        query: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        duration: Optional[float] = None,
        fps: Optional[float] = 2,
        generation_config: Optional[GenerationConfig] = None,
        return_raw_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict temporal segment for a single video-query pair.

        Args:
            video_path: Path to video file.
            query: Text query describing the moment.
            video_start: Optional start time to trim video.
            video_end: Optional end time to trim video.
            duration: Video duration in seconds (required for temporal tokens mode).
            generation_config: Custom generation configuration.
            return_raw_output: Whether to include raw model output.

        Returns:
            Dictionary containing prediction results.
        """
        video_path = str(video_path)
        gen_config = generation_config or self.generation_config

        # Create prompt
        prompt = self.prompt_template.format(query=query)

        # Create messages
        video_content = {"type": "video", "video": video_path, "fps": fps}
        if video_start is not None:
            video_content["video_start"] = video_start
        if video_end is not None:
            video_content["video_end"] = video_end

        # Add pixel limit settings for GPU memory control
        if self.min_pixels is not None:
            video_content["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            video_content["max_pixels"] = self.max_pixels
        if self.total_pixels is not None:
            video_content["total_pixels"] = self.total_pixels

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process input
        try:


            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

            if video_inputs is not None:
                video_inputs, video_metadata = zip(*video_inputs)
                video_inputs, video_metadata = list(video_inputs), list(video_metadata)
            else:
                video_metadata = None

            inputs = self.processor(
                text=[text],
                videos=video_inputs,
                video_metadata=video_metadata,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs
            )

        except ImportError:
            logger.warning(
                "qwen_vl_utils not available. Video processing is disabled. "
                "Install qwen_vl_utils for full video processing functionality: "
                "pip install qwen-vl-utils"
            )
            # Fallback to text-only processing - video content will not be used
            text = f"Video: {video_path}\nQuery: {prompt}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
            )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )

        # Decode
        prompt_length = inputs["input_ids"].shape[1]
        response_ids = outputs[0, prompt_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Parse timestamps
        try:
            if self.use_temporal_tokens:
                # Parse temporal tokens
                if duration is None:
                    # Try to estimate duration from video_start/video_end
                    if video_start is not None and video_end is not None:
                        duration = video_end - video_start
                    else:
                        # Default duration - will result in normalized values
                        duration = 1.0
                        logger.warning(
                            "No duration provided for temporal token parsing. "
                            "Using duration=1.0 (normalized values)."
                        )
                result_times = parse_temporal_response(response_text, duration)
                if result_times is not None:
                    start_time, end_time = result_times
                    success = True
                else:
                    start_time, end_time = 0.0, 0.0
                    success = False
            else:
                start_time, end_time = self._parse_timestamps(response_text)
                success = True
        except ValueError:
            start_time, end_time = 0.0, 0.0
            success = False
            logger.warning(f"Failed to parse timestamps from: {response_text}")

        result = {
            "video_path": video_path,
            "query": query,
            "response": response_text,
            "start_time": start_time,
            "end_time": end_time,
            "success": success,
        }

        if return_raw_output:
            result["raw_output_ids"] = response_ids.cpu()

        return result

    def predict_batch(
        self,
        video_paths: List[Union[str, Path]],
        queries: List[str],
        video_starts: Optional[List[Optional[float]]] = None,
        video_ends: Optional[List[Optional[float]]] = None,
        durations: Optional[List[Optional[float]]] = None,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict temporal segments for multiple video-query pairs.

        Args:
            video_paths: List of video file paths.
            queries: List of text queries.
            video_starts: Optional list of start times.
            video_ends: Optional list of end times.
            durations: Optional list of video durations (required for temporal tokens).
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.

        Returns:
            List of prediction result dictionaries.
        """
        n_samples = len(video_paths)

        if video_starts is None:
            video_starts = [None] * n_samples
        if video_ends is None:
            video_ends = [None] * n_samples
        if durations is None:
            durations = [None] * n_samples

        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, n_samples, batch_size), desc="Inferring")
        else:
            iterator = range(0, n_samples, batch_size)

        for i in iterator:
            batch_end = min(i + batch_size, n_samples)

            # Process batch sequentially for now
            # (Batch processing with videos is complex due to variable lengths)
            for j in range(i, batch_end):
                result = self.predict(
                    video_path=video_paths[j],
                    query=queries[j],
                    video_start=video_starts[j],
                    video_end=video_ends[j],
                    duration=durations[j],
                )
                results.append(result)

        return results

    def _parse_timestamps(self, text: str) -> Tuple[float, float]:
        """
        Parse timestamps from model output.

        Args:
            text: Model output text.

        Returns:
            Tuple of (start_time, end_time).
        """
        patterns = [
            r"([\d.]+)\s*(?:to|->|-|,)\s*([\d.]+)",
            r"start[:\s]*([\d.]+).*end[:\s]*([\d.]+)",
            r"\(([\d.]+)[,\s]+([\d.]+)\)",
            r"\[([\d.]+)[,\s]+([\d.]+)\]",
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                return start, end

        raise ValueError(f"Could not parse timestamps from: {text}")


def visualize_temporal_grounding(
    video_path: Union[str, Path],
    predictions: List[Dict[str, Any]],
    ground_truths: Optional[List[Tuple[float, float]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    duration: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
) -> Optional[Any]:
    """
    Visualize temporal grounding predictions.

    Args:
        video_path: Path to video file.
        predictions: List of prediction dictionaries.
        ground_truths: Optional list of (start, end) ground truth tuples.
        output_path: Optional path to save visualization.
        duration: Video duration (auto-detected if None).
        figsize: Figure size.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if available, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None

    # Get video duration if not provided
    if duration is None:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 60.0
            cap.release()
        except Exception:
            duration = 60.0  # Default duration

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up timeline
    ax.set_xlim(0, duration)
    ax.set_ylim(0, len(predictions) + 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Predictions")
    ax.set_title(f"Temporal Grounding: {Path(video_path).name}")

    # Plot predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

    for i, (pred, color) in enumerate(zip(predictions, colors)):
        y_pos = i + 0.5

        start = pred.get("start_time", 0)
        end = pred.get("end_time", 0)

        # Plot prediction bar
        ax.barh(
            y_pos,
            end - start,
            left=start,
            height=0.4,
            color=color,
            alpha=0.7,
            label=pred.get("query", f"Query {i+1}")[:30],
        )

        # Add text label
        ax.text(
            start,
            y_pos + 0.25,
            f"{start:.2f}s - {end:.2f}s",
            fontsize=8,
            va="bottom",
        )

    # Plot ground truths if available
    if ground_truths:
        for i, (gt_start, gt_end) in enumerate(ground_truths):
            y_pos = i + 0.5
            ax.barh(
                y_pos,
                gt_end - gt_start,
                left=gt_start,
                height=0.2,
                color="black",
                alpha=0.3,
                hatch="//",
            )

    # Add legend
    pred_patch = mpatches.Patch(color="tab:blue", alpha=0.7, label="Predictions")
    legend_handles = [pred_patch]

    if ground_truths:
        gt_patch = mpatches.Patch(
            facecolor="black",
            alpha=0.3,
            hatch="//",
            label="Ground Truth",
        )
        legend_handles.append(gt_patch)

    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")

    if show:
        plt.show()

    return fig


def extract_video_frames(
    video_path: Union[str, Path],
    start_time: float,
    end_time: float,
    num_frames: int = 8,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video segment.

    Args:
        video_path: Path to video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        num_frames: Number of frames to extract.
        output_dir: Optional directory to save frames.

    Returns:
        List of frame arrays (RGB).
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available for frame extraction")
        return []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        logger.error("Could not get video FPS")
        return []

    # Calculate frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame

    if total_frames <= 0:
        return []

    # Sample frame indices
    if num_frames >= total_frames:
        frame_indices = list(range(start_frame, end_frame))
    else:
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # Save if output directory specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_path), frame)

    cap.release()

    return frames


def main():
    """Command-line interface for video inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Video Temporal Grounding Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query describing the moment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)",
    )

    args = parser.parse_args()

    # Create inference engine
    engine = VideoTemporalInference(
        model_path=args.model_path,
        device=args.device,
    )

    # Run prediction
    result = engine.predict(
        video_path=args.video,
        query=args.query,
    )

    # Print result
    print("\n" + "=" * 50)
    print("Video Temporal Grounding Result")
    print("=" * 50)
    print(f"Video: {result['video_path']}")
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Start Time: {result['start_time']:.2f}s")
    print(f"End Time: {result['end_time']:.2f}s")
    print(f"Success: {result['success']}")
    print("=" * 50)

    # Visualize if requested
    if args.output:
        visualize_temporal_grounding(
            video_path=args.video,
            predictions=[result],
            output_path=args.output,
            show=False,
        )


if __name__ == "__main__":
    main()
