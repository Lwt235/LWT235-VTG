"""
Tests for temporal tokens module.
"""

import pytest

from utils.temporal_tokens import (
    get_temporal_token,
    get_temporal_token_id,
    bin_index_from_token_id,
    normalize_to_bin,
    bin_to_normalized,
    timestamp_to_temporal_tokens,
    temporal_tokens_to_timestamp,
    parse_temporal_token,
    extract_temporal_tokens_from_text,
    format_temporal_response,
    parse_temporal_response,
    get_all_temporal_tokens,
    get_temporal_token_ids,
    TEMPORAL_TOKEN_START_ID,
    TEMPORAL_TOKEN_END_ID,
    NUM_TEMPORAL_TOKENS,
)


class TestTemporalTokenConstants:
    """Tests for temporal token constants."""

    def test_token_range(self):
        """Test that the token range is correct."""
        assert TEMPORAL_TOKEN_START_ID == 150643
        assert TEMPORAL_TOKEN_END_ID == 151642
        assert NUM_TEMPORAL_TOKENS == 1000
        assert TEMPORAL_TOKEN_END_ID - TEMPORAL_TOKEN_START_ID + 1 == NUM_TEMPORAL_TOKENS


class TestGetTemporalToken:
    """Tests for get_temporal_token function."""

    def test_first_token(self):
        """Test first temporal token."""
        assert get_temporal_token(0) == "<0>"

    def test_last_token(self):
        """Test last temporal token."""
        assert get_temporal_token(999) == "<999>"

    def test_middle_token(self):
        """Test middle temporal tokens."""
        assert get_temporal_token(100) == "<100>"
        assert get_temporal_token(500) == "<500>"

    def test_out_of_range_negative(self):
        """Test that negative indices raise ValueError."""
        with pytest.raises(ValueError):
            get_temporal_token(-1)

    def test_out_of_range_large(self):
        """Test that indices >= 1000 raise ValueError."""
        with pytest.raises(ValueError):
            get_temporal_token(1000)


class TestGetTemporalTokenId:
    """Tests for get_temporal_token_id function."""

    def test_first_token_id(self):
        """Test first token ID."""
        assert get_temporal_token_id(0) == TEMPORAL_TOKEN_START_ID

    def test_last_token_id(self):
        """Test last token ID."""
        assert get_temporal_token_id(999) == TEMPORAL_TOKEN_END_ID

    def test_middle_token_id(self):
        """Test middle token IDs."""
        assert get_temporal_token_id(100) == TEMPORAL_TOKEN_START_ID + 100

    def test_out_of_range(self):
        """Test out of range indices."""
        with pytest.raises(ValueError):
            get_temporal_token_id(-1)
        with pytest.raises(ValueError):
            get_temporal_token_id(1000)


class TestBinIndexFromTokenId:
    """Tests for bin_index_from_token_id function."""

    def test_first_token_id(self):
        """Test first token ID returns bin 0."""
        assert bin_index_from_token_id(TEMPORAL_TOKEN_START_ID) == 0

    def test_last_token_id(self):
        """Test last token ID returns bin 999."""
        assert bin_index_from_token_id(TEMPORAL_TOKEN_END_ID) == 999

    def test_non_temporal_token(self):
        """Test non-temporal token returns None."""
        assert bin_index_from_token_id(100) is None
        assert bin_index_from_token_id(TEMPORAL_TOKEN_START_ID - 1) is None
        assert bin_index_from_token_id(TEMPORAL_TOKEN_END_ID + 1) is None


class TestNormalizeToBin:
    """Tests for normalize_to_bin function."""

    def test_zero(self):
        """Test that 0.0 maps to bin 0."""
        assert normalize_to_bin(0.0) == 0

    def test_one(self):
        """Test that 1.0 maps to bin 999."""
        assert normalize_to_bin(1.0) == 999

    def test_middle(self):
        """Test middle values."""
        assert normalize_to_bin(0.5) == 500

    def test_clipping_negative(self):
        """Test that negative values are clipped to 0."""
        assert normalize_to_bin(-0.5) == 0

    def test_clipping_large(self):
        """Test that values > 1 are clipped to 999."""
        assert normalize_to_bin(1.5) == 999

    def test_custom_bins(self):
        """Test with custom number of bins."""
        assert normalize_to_bin(0.5, num_bins=100) == 50


class TestBinToNormalized:
    """Tests for bin_to_normalized function."""

    def test_first_bin(self):
        """Test first bin returns center value."""
        result = bin_to_normalized(0)
        assert 0.0 < result < 0.001

    def test_last_bin(self):
        """Test last bin returns near 1.0."""
        result = bin_to_normalized(999)
        assert 0.999 < result < 1.0

    def test_middle_bin(self):
        """Test middle bin returns approximately 0.5."""
        result = bin_to_normalized(500)
        assert 0.49 < result < 0.51


class TestTimestampToTemporalTokens:
    """Tests for timestamp_to_temporal_tokens function."""

    def test_basic_conversion(self):
        """Test basic timestamp to token conversion."""
        start_token, end_token = timestamp_to_temporal_tokens(
            start=5.0, end=10.0, duration=20.0
        )
        # 5/20 = 0.25 -> bin 250, 10/20 = 0.5 -> bin 500
        assert start_token == "<250>"
        assert end_token == "<500>"

    def test_full_video(self):
        """Test full video duration."""
        start_token, end_token = timestamp_to_temporal_tokens(
            start=0.0, end=100.0, duration=100.0
        )
        assert start_token == "<0>"
        assert end_token == "<999>"

    def test_invalid_duration(self):
        """Test that zero or negative duration raises error."""
        with pytest.raises(ValueError):
            timestamp_to_temporal_tokens(0.0, 10.0, duration=0.0)
        with pytest.raises(ValueError):
            timestamp_to_temporal_tokens(0.0, 10.0, duration=-1.0)


class TestTemporalTokensToTimestamp:
    """Tests for temporal_tokens_to_timestamp function."""

    def test_basic_conversion(self):
        """Test basic token to timestamp conversion."""
        start, end = temporal_tokens_to_timestamp("<250>", "<500>", duration=20.0)
        # bin 250 -> 0.2505, bin 500 -> 0.5005
        # With duration 20.0: 5.01, 10.01
        assert 4.9 < start < 5.2
        assert 9.9 < end < 10.2

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original_start, original_end = 5.0, 10.0
        duration = 20.0

        start_token, end_token = timestamp_to_temporal_tokens(
            original_start, original_end, duration
        )
        recovered_start, recovered_end = temporal_tokens_to_timestamp(
            start_token, end_token, duration
        )

        # Should be close to original (within bin resolution)
        assert abs(recovered_start - original_start) < 0.1
        assert abs(recovered_end - original_end) < 0.1


class TestParseTemporalToken:
    """Tests for parse_temporal_token function."""

    def test_valid_token(self):
        """Test parsing valid tokens."""
        assert parse_temporal_token("<0>") == 0
        assert parse_temporal_token("<500>") == 500
        assert parse_temporal_token("<999>") == 999

    def test_with_whitespace(self):
        """Test parsing tokens with whitespace."""
        assert parse_temporal_token("  <100>  ") == 100

    def test_invalid_format(self):
        """Test that invalid formats return None."""
        assert parse_temporal_token("100") is None
        assert parse_temporal_token("<abc>") is None
        assert parse_temporal_token("") is None

    def test_out_of_range(self):
        """Test that out-of-range values return None."""
        assert parse_temporal_token("<1000>") is None
        assert parse_temporal_token("<-1>") is None


class TestExtractTemporalTokensFromText:
    """Tests for extract_temporal_tokens_from_text function."""

    def test_simple_extraction(self):
        """Test extracting tokens from simple text."""
        text = "The segment is <100><200>"
        bins = extract_temporal_tokens_from_text(text)
        assert bins == [100, 200]

    def test_multiple_tokens(self):
        """Test extracting multiple tokens."""
        text = "<0> to <500> and <999>"
        bins = extract_temporal_tokens_from_text(text)
        assert bins == [0, 500, 999]

    def test_no_tokens(self):
        """Test text with no tokens."""
        text = "No tokens here"
        bins = extract_temporal_tokens_from_text(text)
        assert bins == []

    def test_out_of_range_ignored(self):
        """Test that out-of-range tokens are ignored."""
        text = "<100><1000><200>"
        bins = extract_temporal_tokens_from_text(text)
        assert bins == [100, 200]


class TestFormatTemporalResponse:
    """Tests for format_temporal_response function."""

    def test_basic_format(self):
        """Test basic response formatting."""
        response = format_temporal_response(5.0, 10.0, duration=20.0)
        assert response == "<250><500>"

    def test_edge_cases(self):
        """Test edge case formatting."""
        response = format_temporal_response(0.0, 100.0, duration=100.0)
        assert response == "<0><999>"


class TestParseTemporalResponse:
    """Tests for parse_temporal_response function."""

    def test_basic_parse(self):
        """Test basic response parsing."""
        result = parse_temporal_response("<250><500>", duration=20.0)
        assert result is not None
        start, end = result
        assert 4.9 < start < 5.2
        assert 9.9 < end < 10.2

    def test_invalid_response(self):
        """Test that invalid responses return None."""
        result = parse_temporal_response("no tokens here", duration=20.0)
        assert result is None

    def test_single_token(self):
        """Test that single token returns None (need at least 2)."""
        result = parse_temporal_response("<100>", duration=20.0)
        assert result is None


class TestGetAllTemporalTokens:
    """Tests for get_all_temporal_tokens function."""

    def test_count(self):
        """Test that we get 1000 tokens."""
        tokens = get_all_temporal_tokens()
        assert len(tokens) == 1000

    def test_first_and_last(self):
        """Test first and last tokens."""
        tokens = get_all_temporal_tokens()
        assert tokens[0] == "<0>"
        assert tokens[-1] == "<999>"


class TestGetTemporalTokenIds:
    """Tests for get_temporal_token_ids function."""

    def test_count(self):
        """Test that we get 1000 IDs."""
        ids = get_temporal_token_ids()
        assert len(ids) == 1000

    def test_first_and_last(self):
        """Test first and last IDs."""
        ids = get_temporal_token_ids()
        assert ids[0] == TEMPORAL_TOKEN_START_ID
        assert ids[-1] == TEMPORAL_TOKEN_END_ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
