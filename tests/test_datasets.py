"""
Tests for dataset modules.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from vtg_datasets.video_dataset import (
    VideoTemporalDataset,
    VideoTemporalSFTDataset,
    VideoTemporalRLDataset,
)
from vtg_datasets.collate_fns import (
    pad_temporal_sequences,
    create_temporal_mask,
)


class TestVideoDataset:
    """Tests for VideoTemporalDataset."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 30.0,
                    "timestamp": [5.0, 10.0],
                    "sentence": "A person opens the door",
                },
                {
                    "video": "./videos/test2.mp4",
                    "duration": 45.0,
                    "timestamp": [20.0, 35.0],
                    "sentence": "The cat jumps on the table",
                    "difficulty": 0.8,
                    "qid": "test_query_001",
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_dataset_loading(self, sample_annotations):
        """Test loading dataset from annotations."""
        dataset = VideoTemporalDataset(annotation_file=sample_annotations)

        assert len(dataset) == 2

    def test_getitem(self, sample_annotations):
        """Test getting single item."""
        dataset = VideoTemporalDataset(annotation_file=sample_annotations)

        item = dataset[0]

        assert "video_path" in item
        assert "query" in item
        assert "timestamp" in item
        assert "normalized_timestamp" in item
        assert "temporal_bins" in item

    def test_normalized_timestamps(self, sample_annotations):
        """Test timestamp normalization."""
        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            use_relative_timestamps=True,
        )

        item = dataset[0]

        # 5/30 = 0.167, 10/30 = 0.333
        assert 0.16 < item["normalized_timestamp"][0] < 0.17
        assert 0.33 < item["normalized_timestamp"][1] < 0.34

    def test_temporal_bins(self, sample_annotations):
        """Test temporal binning."""
        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            num_bins=100,
        )

        item = dataset[0]

        # Normalized timestamps are [5/30, 10/30] = [0.167, 0.333]
        # Binned: [16, 33]
        assert 15 <= item["temporal_bins"][0] <= 17
        assert 32 <= item["temporal_bins"][1] <= 34


class TestJSONArrayFormat:
    """Tests for JSON array format support."""

    @pytest.fixture
    def json_array_annotations(self):
        """Create sample annotation file in JSON array format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.json"

            samples = [
                {
                    "video": "timerft_data/ZMlcrfpgiIY.mp4",
                    "duration": 154.31378074801725,
                    "timestamp": [101.0, 137.0],
                    "sentence": "The video shows a room with a bed and window.",
                    "qid": "my|cosmo|ZMlcrfpgiIY|test",
                    "video_start": None,
                    "video_end": None,
                    "difficulty": 24.083333333333336,
                    "pred": [108.0, 116.67],
                },
                {
                    "video": "timerft_data/cdjYrjRtUNA.mp4",
                    "duration": 114.2141,
                    "timestamp": [28.0, 34.0],
                    "sentence": "A black Chevrolet HHR LT SUV is parked in a snowy lot.",
                    "qid": "my|cosmo|cdjYrjRtUNA|test",
                    "video_start": None,
                    "video_end": None,
                    "difficulty": 14.705882352941178,
                    "pred": [0.0, 33.0],
                },
            ]

            with open(annotation_path, "w") as f:
                json.dump(samples, f, indent=4)

            yield annotation_path

    def test_json_array_loading(self, json_array_annotations):
        """Test loading dataset from JSON array format."""
        dataset = VideoTemporalDataset(annotation_file=json_array_annotations)

        assert len(dataset) == 2

    def test_json_array_item_contents(self, json_array_annotations):
        """Test contents of items loaded from JSON array format."""
        dataset = VideoTemporalDataset(annotation_file=json_array_annotations)

        item = dataset[0]

        assert "video_path" in item
        assert "query" in item
        assert "timestamp" in item
        assert item["query"] == "The video shows a room with a bed and window."
        assert item["timestamp"] == [101.0, 137.0]

    def test_json_array_sft_dataset(self, json_array_annotations):
        """Test SFT dataset with JSON array format."""
        dataset = VideoTemporalSFTDataset(annotation_file=json_array_annotations)

        item = dataset[0]

        assert "prompt" in item
        assert "response" in item
        assert "messages" in item

    def test_json_array_rl_dataset(self, json_array_annotations):
        """Test RL dataset with JSON array format."""
        dataset = VideoTemporalRLDataset(annotation_file=json_array_annotations)

        item = dataset[0]

        assert "prompt" in item
        assert "ground_truth" in item
        assert "messages" in item


class TestSFTDataset:
    """Tests for SFT dataset."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 30.0,
                    "timestamp": [5.0, 10.0],
                    "sentence": "A person opens the door",
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_sft_dataset_format(self, sample_annotations):
        """Test SFT dataset output format."""
        dataset = VideoTemporalSFTDataset(annotation_file=sample_annotations)

        item = dataset[0]

        assert "prompt" in item
        assert "response" in item
        assert "messages" in item
        assert len(item["messages"]) == 2  # User and assistant


class TestRLDataset:
    """Tests for RL dataset."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 30.0,
                    "timestamp": [5.0, 10.0],
                    "sentence": "A person opens the door",
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_rl_dataset_format(self, sample_annotations):
        """Test RL dataset output format."""
        dataset = VideoTemporalRLDataset(annotation_file=sample_annotations)

        item = dataset[0]

        assert "prompt" in item
        assert "ground_truth" in item
        assert "messages" in item
        assert len(item["messages"]) == 1  # Only user (no response)


class TestCollateFunctions:
    """Tests for collate functions."""

    def test_pad_temporal_sequences(self):
        """Test padding temporal sequences."""
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]

        padded = pad_temporal_sequences(sequences)

        assert padded.shape == (3, 4)
        assert padded[1, 2] == 0  # Padding
        assert padded[1, 3] == 0  # Padding

    def test_create_temporal_mask(self):
        """Test creating temporal mask."""
        lengths = [3, 2, 4]

        mask = create_temporal_mask(lengths)

        assert mask.shape == (3, 4)
        assert mask[0, 0] == True
        assert mask[0, 2] == True
        assert mask[0, 3] == False
        assert mask[1, 1] == True
        assert mask[1, 2] == False


class TestTemporalTokensInDataset:
    """Tests for temporal tokens support in datasets."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 20.0,
                    "timestamp": [5.0, 10.0],
                    "sentence": "A person opens the door",
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_base_dataset_with_temporal_tokens(self, sample_annotations):
        """Test base dataset with temporal tokens enabled."""
        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            use_temporal_tokens=True,
        )

        item = dataset[0]

        # Check temporal tokens are present
        assert "temporal_tokens" in item
        assert len(item["temporal_tokens"]) == 2
        # 5/20 = 0.25 -> bin 250, 10/20 = 0.5 -> bin 500
        assert item["temporal_tokens"][0] == "<250>"
        assert item["temporal_tokens"][1] == "<500>"

        # Check temporal response is present
        assert "temporal_response" in item
        assert item["temporal_response"] == "<|box_start|><250><500><|box_end|>"

    def test_sft_dataset_with_temporal_tokens(self, sample_annotations):
        """Test SFT dataset with temporal tokens enabled."""
        dataset = VideoTemporalSFTDataset(
            annotation_file=sample_annotations,
            use_temporal_tokens=True,
        )

        item = dataset[0]

        # Response should use temporal tokens format
        assert "response" in item
        assert item["response"] == "<|box_start|><250><500><|box_end|>"

        # Prompt should mention temporal tokens
        assert "temporal tokens" in item["prompt"].lower()

    def test_rl_dataset_with_temporal_tokens(self, sample_annotations):
        """Test RL dataset with temporal tokens enabled."""
        dataset = VideoTemporalRLDataset(
            annotation_file=sample_annotations,
            use_temporal_tokens=True,
        )

        item = dataset[0]

        # Ground truth should include temporal tokens
        assert "ground_truth" in item
        assert "temporal_tokens" in item["ground_truth"]
        assert item["ground_truth"]["temporal_tokens"] == ["<250>", "<500>"]


class TestEdgeCases:
    """Tests for edge case handling in datasets."""

    @pytest.fixture
    def zero_duration_annotations(self):
        """Create sample annotation file with edge case durations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            # Sample where video_start equals duration (would result in zero duration)
            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 30.0,
                    "timestamp": [5.0, 10.0],
                    "sentence": "A person opens the door",
                    "video_start": 30.0,  # This would make duration = 0
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_dataset_handles_zero_duration(self, zero_duration_annotations):
        """Test that dataset handles cases where trimming would cause zero duration."""
        dataset = VideoTemporalDataset(
            annotation_file=zero_duration_annotations,
            use_temporal_tokens=True,
        )

        # Should not raise an error
        item = dataset[0]

        # Duration should be positive
        assert item["duration"] > 0

        # Temporal tokens should still be generated
        assert "temporal_tokens" in item
        assert len(item["temporal_tokens"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
