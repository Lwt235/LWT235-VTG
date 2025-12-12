"""
Temporal Tokens Utility for Video Temporal Localization.

This module provides functionality to replace infrequently used tokens in
Qwen3-VL vocabulary (IDs 150643 to 151642) with special temporal tokens
<0> to <999> for quantizing relative time.

The temporal tokens enable fine-grained temporal grounding by discretizing
normalized timestamps [0, 1] into 1000 bins.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# Token ID range to replace (1000 tokens)
TEMPORAL_TOKEN_START_ID = 150643
TEMPORAL_TOKEN_END_ID = 151642  # inclusive
NUM_TEMPORAL_TOKENS = 1000

# Temporal token format
TEMPORAL_TOKEN_TEMPLATE = "<{num}>"


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


def get_temporal_token_id(bin_index: int) -> int:
    """
    Get the token ID for a given temporal bin index.

    Args:
        bin_index: Bin index from 0 to 999.

    Returns:
        Token ID corresponding to the temporal token.

    Raises:
        ValueError: If bin_index is out of range.
    """
    if not 0 <= bin_index < NUM_TEMPORAL_TOKENS:
        raise ValueError(
            f"Bin index must be between 0 and {NUM_TEMPORAL_TOKENS - 1}, got {bin_index}"
        )
    return TEMPORAL_TOKEN_START_ID + bin_index


def bin_index_from_token_id(token_id: int) -> Optional[int]:
    """
    Get the bin index from a token ID.

    Args:
        token_id: Token ID.

    Returns:
        Bin index if the token ID is a temporal token, None otherwise.
    """
    if TEMPORAL_TOKEN_START_ID <= token_id <= TEMPORAL_TOKEN_END_ID:
        return token_id - TEMPORAL_TOKEN_START_ID
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
    return f"{start_token}{end_token}"


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


def get_temporal_token_ids() -> List[int]:
    """
    Get all temporal token IDs.

    Returns:
        List of token IDs from TEMPORAL_TOKEN_START_ID to TEMPORAL_TOKEN_END_ID.
    """
    return list(range(TEMPORAL_TOKEN_START_ID, TEMPORAL_TOKEN_END_ID + 1))


def add_temporal_tokens_to_tokenizer(
    tokenizer: Any,
    reinitialize_embeddings: bool = True,
) -> Dict[str, int]:
    """
    Add temporal tokens to the tokenizer vocabulary.

    This function replaces the tokens at IDs 150643-151642 with temporal tokens
    <0> to <999>. Note that this modifies the tokenizer in-place.

    Args:
        tokenizer: The tokenizer to modify.
        reinitialize_embeddings: Whether to mark for embedding reinitialization.

    Returns:
        Dictionary mapping temporal token strings to their IDs.
    """
    temporal_tokens = get_all_temporal_tokens()

    # Add tokens to vocabulary
    # Note: We're replacing existing tokens, not adding new ones
    # The tokenizer should recognize these patterns after adding

    # For transformers tokenizers, we use add_tokens
    num_added = tokenizer.add_tokens(temporal_tokens, special_tokens=True)

    # Build mapping
    token_to_id = {}
    for i, token in enumerate(temporal_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_to_id[token] = token_id

    return token_to_id


def initialize_temporal_embeddings(
    model: Any,
    tokenizer: Any,
    initialization_strategy: str = "mean",
) -> None:
    """
    Initialize embeddings for temporal tokens.

    Args:
        model: The model with embedding layer.
        tokenizer: The tokenizer.
        initialization_strategy: How to initialize embeddings:
            - "mean": Use mean of all embeddings
            - "random": Random initialization
            - "copy_existing": Copy from existing tokens at those IDs
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

    # Get temporal token IDs
    temporal_ids = get_temporal_token_ids()

    with torch.no_grad():
        if initialization_strategy == "mean":
            # Initialize with mean of all embeddings
            mean_embedding = weight.mean(dim=0)
            for token_id in temporal_ids:
                if token_id < weight.size(0):
                    weight[token_id] = mean_embedding

        elif initialization_strategy == "random":
            # Random initialization with same std as existing embeddings
            std = weight.std().item()
            for token_id in temporal_ids:
                if token_id < weight.size(0):
                    weight[token_id] = torch.randn_like(weight[token_id]) * std

        elif initialization_strategy == "copy_existing":
            # Keep existing embeddings (no change needed)
            pass

        else:
            raise ValueError(f"Unknown initialization strategy: {initialization_strategy}")


def resize_model_embeddings_for_temporal_tokens(
    model: Any,
    tokenizer: Any,
    initialization_strategy: str = "mean",
) -> None:
    """
    Resize model embeddings and initialize temporal tokens.

    This function should be called after adding temporal tokens to the tokenizer
    to ensure the model's embedding layer is properly sized and initialized.

    Args:
        model: The model to resize.
        tokenizer: The tokenizer with temporal tokens added.
        initialization_strategy: How to initialize new embeddings.
    """
    # Resize token embeddings to match tokenizer vocab size
    model.resize_token_embeddings(len(tokenizer))

    # Initialize temporal token embeddings
    initialize_temporal_embeddings(model, tokenizer, initialization_strategy)
