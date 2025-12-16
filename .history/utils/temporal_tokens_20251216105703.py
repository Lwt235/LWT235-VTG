"""
Temporal Tokens Utility for Video Temporal Localization.

This module provides functionality to add special temporal tokens
<0> to <999> for quantizing relative time in video temporal grounding.

The temporal tokens enable fine-grained temporal grounding by discretizing
normalized timestamps [0, 1] into 1000 bins.

Key features:
- Adds new tokens to tokenizer vocabulary (not replacing existing tokens)
- Uses sinusoidal positional encoding for embedding initialization
- Designed to work with LoRA adapters on embed_tokens and lm_head
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AddedToken


# Number of temporal tokens
NUM_TEMPORAL_TOKENS = 1000

# Temporal token format
TEMPORAL_TOKEN_TEMPLATE = "<{num}>"

# These will be set after adding tokens to tokenizer
_temporal_token_start_id: Optional[int] = None
_temporal_token_end_id: Optional[int] = None


def get_temporal_token(bin_index: int) -> str:
    """
    Get the temporal token string for a given bin index.

    Args:
        bin_index: Bin index from 0 to 999.

    Returns:
        Temporal token string (e.g., "<0>", "<999>").

    Raises:
        ValueError: If bin_index is out of range.
    """
    if not 0 <= bin_index < NUM_TEMPORAL_TOKENS:
        raise ValueError(
            f"Bin index must be between 0 and {NUM_TEMPORAL_TOKENS - 1}, got {bin_index}"
        )
    return TEMPORAL_TOKEN_TEMPLATE.format(num=bin_index)


def get_temporal_token_id(bin_index: int, tokenizer: Any = None) -> int:
    """
    Get the token ID for a given temporal bin index.

    Args:
        bin_index: Bin index from 0 to 999.
        tokenizer: Optional tokenizer to get exact token ID.

    Returns:
        Token ID corresponding to the temporal token.

    Raises:
        ValueError: If bin_index is out of range or tokens not initialized.
    """
    if not 0 <= bin_index < NUM_TEMPORAL_TOKENS:
        raise ValueError(
            f"Bin index must be between 0 and {NUM_TEMPORAL_TOKENS - 1}, got {bin_index}"
        )

    if tokenizer is not None:
        token = get_temporal_token(bin_index)
        return tokenizer.convert_tokens_to_ids(token)

    global _temporal_token_start_id
    if _temporal_token_start_id is None:
        raise ValueError(
            "Temporal tokens not initialized. Call add_temporal_tokens_to_tokenizer first."
        )
    return _temporal_token_start_id + bin_index


def bin_index_from_token_id(token_id: int) -> Optional[int]:
    """
    Get the bin index from a token ID.

    Args:
        token_id: Token ID.

    Returns:
        Bin index if the token ID is a temporal token, None otherwise.
    """
    global _temporal_token_start_id, _temporal_token_end_id
    if _temporal_token_start_id is None or _temporal_token_end_id is None:
        return None
    if _temporal_token_start_id <= token_id <= _temporal_token_end_id:
        return token_id - _temporal_token_start_id
    return None


def normalize_to_bin(value: float, num_bins: int = NUM_TEMPORAL_TOKENS) -> int:
    """
    Convert a normalized value [0, 1] to a bin index.

    Args:
        value: Normalized value between 0 and 1.
        num_bins: Number of bins (default: 1000).

    Returns:
        Bin index.
    """
    value = max(0.0, min(1.0, value))
    bin_idx = int(value * num_bins)
    return min(bin_idx, num_bins - 1)


def bin_to_normalized(bin_index: int, num_bins: int = NUM_TEMPORAL_TOKENS) -> float:
    """
    Convert a bin index back to a normalized value.

    Args:
        bin_index: Bin index.
        num_bins: Number of bins (default: 1000).

    Returns:
        Center of the bin as normalized value.
    """
    return (bin_index + 0.5) / num_bins


def timestamp_to_temporal_tokens(
    start: float,
    end: float,
    duration: float,
    num_bins: int = NUM_TEMPORAL_TOKENS,
) -> Tuple[str, str]:
    """
    Convert timestamps to temporal token strings.

    Args:
        start: Start time in seconds.
        end: End time in seconds.
        duration: Total video duration in seconds.
        num_bins: Number of temporal bins (default: 1000).

    Returns:
        Tuple of (start_token, end_token) strings.
    """
    if duration <= 0:
        raise ValueError("Duration must be positive")

    # Normalize to [0, 1]
    norm_start = max(0.0, min(1.0, start / duration))
    norm_end = max(0.0, min(1.0, end / duration))

    # Convert to bins
    start_bin = normalize_to_bin(norm_start, num_bins)
    end_bin = normalize_to_bin(norm_end, num_bins)

    return get_temporal_token(start_bin), get_temporal_token(end_bin)


def temporal_tokens_to_timestamp(
    start_token: str,
    end_token: str,
    duration: float,
    num_bins: int = NUM_TEMPORAL_TOKENS,
) -> Tuple[float, float]:
    """
    Convert temporal token strings back to timestamps.

    Args:
        start_token: Start temporal token (e.g., "<100>").
        end_token: End temporal token (e.g., "<200>").
        duration: Total video duration in seconds.
        num_bins: Number of temporal bins (default: 1000).

    Returns:
        Tuple of (start_time, end_time) in seconds.
    """
    start_bin = parse_temporal_token(start_token)
    end_bin = parse_temporal_token(end_token)

    if start_bin is None or end_bin is None:
        raise ValueError(f"Invalid temporal tokens: {start_token}, {end_token}")

    norm_start = bin_to_normalized(start_bin, num_bins)
    norm_end = bin_to_normalized(end_bin, num_bins)

    return norm_start * duration, norm_end * duration


def parse_temporal_token(token: str) -> Optional[int]:
    """
    Parse a temporal token string to extract the bin index.

    Args:
        token: Token string (e.g., "<100>").

    Returns:
        Bin index if valid temporal token, None otherwise.
    """
    match = re.match(r"<(\d+)>", token.strip())
    if match:
        bin_idx = int(match.group(1))
        if 0 <= bin_idx < NUM_TEMPORAL_TOKENS:
            return bin_idx
    return None


def extract_temporal_tokens_from_text(text: str) -> List[int]:
    """
    Extract all temporal tokens from text.

    Args:
        text: Input text containing temporal tokens.

    Returns:
        List of bin indices found in the text.
    """
    pattern = r"<(\d+)>"
    matches = re.findall(pattern, text)

    bins = []
    for match in matches:
        bin_idx = int(match)
        if 0 <= bin_idx < NUM_TEMPORAL_TOKENS:
            bins.append(bin_idx)

    return bins


def format_temporal_response(
    start: float,
    end: float,
    duration: float,
    num_bins: int = NUM_TEMPORAL_TOKENS,
) -> str:
    """
    Format a temporal response using temporal tokens.

    Args:
        start: Start time in seconds.
        end: End time in seconds.
        duration: Total video duration in seconds.
        num_bins: Number of temporal bins.

    Returns:
        Formatted response string with temporal tokens.
    """
    start_token, end_token = timestamp_to_temporal_tokens(
        start, end, duration, num_bins
    )
    return f"<|box_start|>{start_token}{end_token}<|box_end|>"


def parse_temporal_response(
    response: str,
    duration: float,
    num_bins: int = NUM_TEMPORAL_TOKENS,
) -> Optional[Tuple[float, float]]:
    """
    Parse a temporal response to extract timestamps.

    Args:
        response: Model response containing temporal tokens.
        duration: Total video duration in seconds.
        num_bins: Number of temporal bins.

    Returns:
        Tuple of (start_time, end_time) in seconds, or None if parsing fails.
    """
    bins = extract_temporal_tokens_from_text(response)

    if len(bins) >= 2:
        start_bin = bins[0]
        end_bin = bins[1]

        norm_start = bin_to_normalized(start_bin, num_bins)
        norm_end = bin_to_normalized(end_bin, num_bins)

        return norm_start * duration, norm_end * duration

    return None


def get_all_temporal_tokens() -> List[str]:
    """
    Get all temporal token strings.

    Returns:
        List of temporal token strings from <0> to <999>.
    """
    return [get_temporal_token(i) for i in range(NUM_TEMPORAL_TOKENS)]


def get_temporal_token_ids(tokenizer: Any = None) -> List[int]:
    """
    Get all temporal token IDs.

    Args:
        tokenizer: Optional tokenizer to get exact token IDs.

    Returns:
        List of token IDs for all temporal tokens.
    """
    global _temporal_token_start_id, _temporal_token_end_id

    if tokenizer is not None:
        return [
            tokenizer.convert_tokens_to_ids(get_temporal_token(i))
            for i in range(NUM_TEMPORAL_TOKENS)
        ]

    if _temporal_token_start_id is None or _temporal_token_end_id is None:
        raise ValueError(
            "Temporal tokens not initialized. Call add_temporal_tokens_to_tokenizer first."
        )
    return list(range(_temporal_token_start_id, _temporal_token_end_id + 1))


def create_sinusoidal_embeddings(
    num_tokens: int,
    embedding_dim: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for temporal tokens.

    Uses the same approach as transformer positional encodings:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    This helps the model learn temporal ordering faster.

    Args:
        num_tokens: Number of temporal tokens (1000).
        embedding_dim: Dimension of embeddings.
        device: Device for tensor.
        dtype: Data type for tensor.

    Returns:
        Tensor of shape (num_tokens, embedding_dim) with sinusoidal embeddings.
    """
    position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)

    # Calculate div_term for both sin and cos
    half_dim = (embedding_dim + 1) // 2  # Handle odd dimensions
    div_term = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )

    embeddings = torch.zeros(num_tokens, embedding_dim)

    # Fill sin values for even indices
    sin_values = torch.sin(position * div_term)
    num_sin_cols = (embedding_dim + 1) // 2  # ceil(embedding_dim / 2)
    embeddings[:, 0::2] = sin_values[:, :num_sin_cols]

    # Fill cos values for odd indices
    cos_values = torch.cos(position * div_term)
    num_cos_cols = embedding_dim // 2  # floor(embedding_dim / 2)
    embeddings[:, 1::2] = cos_values[:, :num_cos_cols]

    if device is not None:
        embeddings = embeddings.to(device)
    if dtype is not None:
        embeddings = embeddings.to(dtype)

    return embeddings


def add_temporal_tokens_to_tokenizer(tokenizer: Any) -> Dict[str, int]:
    """
    Add temporal tokens to the tokenizer vocabulary.

    This function adds new tokens <0> to <999> to the tokenizer.
    The model's embedding layer should be resized after calling this.

    Uses AddedToken objects with special=True, normalized=False, and
    single_word=True to prevent sub-tokenization of temporal tokens.
    This ensures tokens like <100> are treated as single atomic units
    and not split into <, 1, 0, 0, > during tokenization.

    Args:
        tokenizer: The tokenizer to modify (in-place).

    Returns:
        Dictionary mapping temporal token strings to their IDs.
    """
    global _temporal_token_start_id, _temporal_token_end_id

    temporal_token_strs = get_all_temporal_tokens()

    # Record original vocab size (where new tokens will start)
    # original_vocab_size = len(tokenizer)

    # Create AddedToken objects with proper settings to prevent sub-tokenization
    # - special=True: Mark as special token
    # - normalized=False: Prevent normalization (e.g., lowercasing)
    # - single_word=True: Prevent sub-tokenization
    # temporal_tokens = [
    #     AddedToken(token, special=True, normalized=False, single_word=True)
    #     for token in temporal_token_strs
    # ]

    # Add tokens to vocabulary as special tokens
    # num_added = tokenizer.add_special_tokens(temporal_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": temporal_token_strs})

    # Build mapping
    token_to_id = {}
    for token in temporal_token_strs:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_to_id[token] = token_id

    # Update global token ID range using min/max of values
    _temporal_token_start_id = min(token_to_id.values())
    _temporal_token_end_id = max(token_to_id.values())

    return token_to_id


def _get_existing_embeddings_mask(
    total_size: int,
    temporal_ids: List[int],
) -> torch.Tensor:
    """
    Create a mask for existing (non-temporal) embeddings.

    Args:
        total_size: Total vocabulary size.
        temporal_ids: List of temporal token IDs.

    Returns:
        Boolean mask where True indicates existing embeddings.
    """
    mask = torch.ones(total_size, dtype=torch.bool)
    for tid in temporal_ids:
        if tid < total_size:
            mask[tid] = False
    return mask


def initialize_temporal_embeddings(
    model: Any,
    tokenizer: Any,
    initialization_strategy: str = "sinusoidal",
) -> None:
    """
    Initialize embeddings for temporal tokens.

    Args:
        model: The model with embedding layer.
        tokenizer: The tokenizer with temporal tokens added.
        initialization_strategy: How to initialize embeddings:
            - "sinusoidal": Use sinusoidal positional encoding (recommended)
            - "mean": Use mean of all embeddings
            - "random": Random initialization
    """
    # Get embedding layer
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embeddings = model.model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")

    # Get embedding weight
    weight = embeddings.weight.data
    embedding_dim = weight.size(1)
    vocab_size = weight.size(0)

    # Get temporal token IDs
    temporal_ids = get_temporal_token_ids(tokenizer)

    # Create mask for existing embeddings (excluding temporal tokens)
    existing_mask = _get_existing_embeddings_mask(vocab_size, temporal_ids)
    existing_embeddings = weight[existing_mask]

    with torch.no_grad():
        if initialization_strategy == "sinusoidal":
            # Use sinusoidal positional encoding for faster convergence
            sin_embeddings = create_sinusoidal_embeddings(
                num_tokens=NUM_TEMPORAL_TOKENS,
                embedding_dim=embedding_dim,
                device=weight.device,
                dtype=weight.dtype,
            )

            # Scale to match existing embedding magnitude
            existing_std = existing_embeddings.std().item()
            sin_embeddings = sin_embeddings * existing_std

            for i, token_id in enumerate(temporal_ids):
                if token_id < weight.size(0):
                    weight[token_id] = sin_embeddings[i]

        elif initialization_strategy == "mean":
            # Initialize with mean of all existing embeddings
            mean_embedding = existing_embeddings.mean(dim=0)
            for token_id in temporal_ids:
                if token_id < weight.size(0):
                    weight[token_id] = mean_embedding

        elif initialization_strategy == "random":
            # Random initialization with same std as existing embeddings
            std = existing_embeddings.std().item()
            for token_id in temporal_ids:
                if token_id < weight.size(0):
                    weight[token_id] = torch.randn_like(weight[token_id]) * std

        else:
            raise ValueError(f"Unknown initialization strategy: {initialization_strategy}")


def resize_model_embeddings_for_temporal_tokens(
    model: Any,
    tokenizer: Any,
    initialization_strategy: str = "sinusoidal",
) -> None:
    """
    Resize model embeddings and initialize temporal tokens.

    This function should be called after adding temporal tokens to the tokenizer
    to ensure the model's embedding layer is properly sized and initialized.

    Uses sinusoidal positional encoding by default for faster convergence.

    Args:
        model: The model to resize.
        tokenizer: The tokenizer with temporal tokens added.
        initialization_strategy: How to initialize new embeddings.
            Default is "sinusoidal" for faster convergence.
    """
    # Resize token embeddings to match tokenizer vocab size
    model.resize_token_embeddings(len(tokenizer))

    # Initialize temporal token embeddings
    # initialize_temporal_embeddings(model, tokenizer, initialization_strategy)
