"""
Tests for dataset modules.
"""

import json
import statistics
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


class TestVideoMetadata:
    """Tests for video metadata handling in datasets."""

    @pytest.fixture
    def annotations_with_video_metadata(self):
        """Create sample annotation file with video_start and video_end metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "train.jsonl"

            # Sample with video_start and video_end metadata
            # In the annotation file, duration and timestamp are already adjusted during annotation creation:
            # - duration is the actual video segment duration (30.0 seconds here)
            # - timestamp values are relative to segment start (not absolute video time)
            # video_start/video_end are metadata for video loading only
            samples = [
                {
                    "video": "./videos/test1.mp4",
                    "duration": 30.0,  # Already the trimmed duration
                    "timestamp": [5.0, 10.0],  # Already offset from video_start
                    "sentence": "A person opens the door",
                    "video_start": 10.0,  # Metadata for video loading
                    "video_end": 40.0,  # Metadata for video loading
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_video_metadata_passed_through(self, annotations_with_video_metadata):
        """Test that video_start and video_end metadata is preserved in dataset output."""
        dataset = VideoTemporalDataset(
            annotation_file=annotations_with_video_metadata,
            use_temporal_tokens=True,
        )

        item = dataset[0]

        # Duration and timestamp should be used as-is (already adjusted in annotation)
        assert item["duration"] == 30.0
        assert item["timestamp"] == [5.0, 10.0]

        # video_start and video_end should be passed through as metadata for video loading
        assert item["video_start"] == 10.0
        assert item["video_end"] == 40.0

        # Temporal tokens should be generated based on the duration and timestamp
        assert "temporal_tokens" in item
        assert len(item["temporal_tokens"]) == 2


class TestPixelLimits:
    """Tests for video frame resolution limit parameters (min_pixels, max_pixels, total_pixels)."""

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

    def test_base_dataset_with_pixel_limits(self, sample_annotations):
        """Test VideoTemporalDataset correctly stores pixel limit parameters."""
        min_pixels = 4 * 32 * 32  # 4096
        max_pixels = 256 * 32 * 32  # 262144
        total_pixels = 20480 * 32 * 32  # 20971520

        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
        )

        assert dataset.min_pixels == min_pixels
        assert dataset.max_pixels == max_pixels
        assert dataset.total_pixels == total_pixels

    def test_base_dataset_get_video_messages_with_pixel_limits(self, sample_annotations):
        """Test get_video_messages includes pixel limit settings."""
        min_pixels = 4096
        max_pixels = 262144
        total_pixels = 20971520

        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
        )

        messages = dataset.get_video_messages(
            video_path="./videos/test.mp4",
            query="Test query",
        )

        video_content = messages[0]["content"][0]
        assert video_content["type"] == "video"
        assert video_content["min_pixels"] == min_pixels
        assert video_content["max_pixels"] == max_pixels
        assert video_content["total_pixels"] == total_pixels

    def test_base_dataset_no_pixel_limits(self, sample_annotations):
        """Test that pixel limits are not included when not specified."""
        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
        )

        messages = dataset.get_video_messages(
            video_path="./videos/test.mp4",
            query="Test query",
        )

        video_content = messages[0]["content"][0]
        assert video_content["type"] == "video"
        assert "min_pixels" not in video_content
        assert "max_pixels" not in video_content
        assert "total_pixels" not in video_content

    def test_sft_dataset_with_pixel_limits(self, sample_annotations):
        """Test VideoTemporalSFTDataset includes pixel limits in messages."""
        min_pixels = 4096
        max_pixels = 262144
        total_pixels = 20971520

        dataset = VideoTemporalSFTDataset(
            annotation_file=sample_annotations,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
        )

        item = dataset[0]
        messages = item["messages"]

        video_content = messages[0]["content"][0]
        assert video_content["type"] == "video"
        assert video_content["min_pixels"] == min_pixels
        assert video_content["max_pixels"] == max_pixels
        assert video_content["total_pixels"] == total_pixels

    def test_rl_dataset_with_pixel_limits(self, sample_annotations):
        """Test VideoTemporalRLDataset includes pixel limits in messages."""
        min_pixels = 4096
        max_pixels = 262144
        total_pixels = 20971520

        dataset = VideoTemporalRLDataset(
            annotation_file=sample_annotations,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
        )

        item = dataset[0]
        messages = item["messages"]

        video_content = messages[0]["content"][0]
        assert video_content["type"] == "video"
        assert video_content["min_pixels"] == min_pixels
        assert video_content["max_pixels"] == max_pixels
        assert video_content["total_pixels"] == total_pixels

    def test_partial_pixel_limits(self, sample_annotations):
        """Test that partial pixel limits work correctly."""
        # Only set max_pixels
        dataset = VideoTemporalDataset(
            annotation_file=sample_annotations,
            max_pixels=262144,
        )

        messages = dataset.get_video_messages(
            video_path="./videos/test.mp4",
            query="Test query",
        )

        video_content = messages[0]["content"][0]
        assert "min_pixels" not in video_content
        assert video_content["max_pixels"] == 262144
        assert "total_pixels" not in video_content


class TestDurationBasedBatchSampler:
    """Tests for DurationBasedBatchSampler."""

    def test_basic_batching(self):
        """Test basic duration-based batching."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        # Create samples with varying durations
        durations = [10.0, 20.0, 15.0, 25.0, 30.0]  # Total: 100s
        target_duration = 40.0  # Should create ~2-3 batches

        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,  # Disable shuffle for predictable test
        )

        batches = list(sampler)

        # Verify all samples are included
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(durations)))

    def test_respects_min_batch_size(self):
        """Test that min_batch_size constraint is respected."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        durations = [50.0, 60.0, 40.0]  # Each sample exceeds target
        target_duration = 30.0
        min_batch_size = 2

        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            min_batch_size=min_batch_size,
            shuffle=False,
        )

        batches = list(sampler)

        # Each batch should have at least min_batch_size samples
        # (except possibly the last batch if drop_last=False)
        for batch in batches[:-1]:
            assert len(batch) >= min_batch_size

    def test_respects_max_batch_size(self):
        """Test that max_batch_size constraint is respected."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        durations = [5.0] * 10  # Short videos
        target_duration = 100.0  # Would include all without max constraint
        max_batch_size = 3

        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            max_batch_size=max_batch_size,
            shuffle=False,
        )

        batches = list(sampler)

        # Each batch should have at most max_batch_size samples
        for batch in batches:
            assert len(batch) <= max_batch_size

    def test_shuffle_reproducibility(self):
        """Test that shuffling is reproducible with seed."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        durations = [10.0, 20.0, 15.0, 25.0, 30.0]

        sampler1 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=40.0,
            shuffle=True,
            seed=42,
        )

        sampler2 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=40.0,
            shuffle=True,
            seed=42,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        assert batches1 == batches2

    def test_set_epoch(self):
        """Test that set_epoch changes shuffle order."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        durations = [10.0, 20.0, 15.0, 25.0, 30.0]

        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=40.0,
            shuffle=True,
            seed=42,
        )

        sampler.set_epoch(0)
        batches_epoch0 = list(sampler)

        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)

        # Batches should be different for different epochs
        all_indices_0 = [idx for batch in batches_epoch0 for idx in batch]
        all_indices_1 = [idx for batch in batches_epoch1 for idx in batch]
        assert all_indices_0 != all_indices_1

    def test_drop_last(self):
        """Test drop_last behavior."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        durations = [30.0, 30.0, 10.0]  # Last sample alone
        target_duration = 50.0
        min_batch_size = 2

        # With drop_last=True
        sampler_drop = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            min_batch_size=min_batch_size,
            drop_last=True,
            shuffle=False,
        )

        # With drop_last=False
        sampler_keep = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            min_batch_size=min_batch_size,
            drop_last=False,
            shuffle=False,
        )

        batches_drop = list(sampler_drop)
        batches_keep = list(sampler_keep)

        # drop_last should have fewer samples covered
        indices_drop = [idx for batch in batches_drop for idx in batch]
        indices_keep = [idx for batch in batches_keep for idx in batch]
        assert len(indices_keep) >= len(indices_drop)

    def test_validation_errors(self):
        """Test that invalid configurations raise errors."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        # Empty durations
        with pytest.raises(ValueError, match="empty"):
            DurationBasedBatchSampler(
                durations=[],
                target_batch_duration=30.0,
            )

        # Non-positive target duration
        with pytest.raises(ValueError, match="positive"):
            DurationBasedBatchSampler(
                durations=[10.0, 20.0],
                target_batch_duration=0.0,
            )

        # Invalid min_batch_size
        with pytest.raises(ValueError, match="min_batch_size"):
            DurationBasedBatchSampler(
                durations=[10.0, 20.0],
                target_batch_duration=30.0,
                min_batch_size=0,
            )

        # max_batch_size < min_batch_size
        with pytest.raises(ValueError, match="max_batch_size"):
            DurationBasedBatchSampler(
                durations=[10.0, 20.0],
                target_batch_duration=30.0,
                min_batch_size=3,
                max_batch_size=2,
            )

    def test_consistent_batch_sizes_with_varied_durations(self):
        """Test that batches have consistent sizes when durations vary significantly.
        
        This test verifies the fix for GPU memory stability: with varied durations,
        the sampler should group similar-length videos together, resulting in
        batches with similar number of samples (not just similar total duration).
        """
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        # Maximum acceptable coefficient of variation for batch sizes
        # A lower CV indicates more consistent batch sizes
        MAX_CV_THRESHOLD = 1.5

        # Create a mix of short and long videos
        # This simulates a real dataset with varying video lengths
        durations = (
            [10.0] * 20 +   # 20 short videos (10s each)
            [50.0] * 10 +   # 10 medium videos (50s each)
            [200.0] * 5     # 5 long videos (200s each)
        )  # Total: 35 videos

        target_duration = 100.0
        
        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=True,
            seed=42,
        )

        batches = list(sampler)
        batch_sizes = [len(batch) for batch in batches]

        # All samples should be included
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(len(durations)))

        # Calculate batch size variance
        # With the fix, batch sizes should be more consistent
        # because similar-length videos are grouped together
        if len(batch_sizes) > 1:
            mean_size = statistics.mean(batch_sizes)
            std_size = statistics.stdev(batch_sizes)
            cv = std_size / mean_size if mean_size > 0 else 0  # Coefficient of variation
            
            # The coefficient of variation should be reasonable
            # (not too high, indicating consistent batch sizes)
            # Before fix: CV could be very high due to random grouping
            # After fix: CV should be lower due to sorted grouping
            assert cv < MAX_CV_THRESHOLD, f"Batch size variance too high: CV={cv:.2f}, sizes={batch_sizes}"

    def test_sorts_by_duration_for_consistent_memory(self):
        """Test that videos are sorted by duration before batching.
        
        This ensures that batches contain videos of similar length,
        leading to more consistent GPU memory usage.
        """
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler

        # Maximum acceptable ratio between longest and shortest video in a batch
        # A lower ratio indicates more consistent video lengths within each batch
        MAX_DURATION_RATIO = 5.0

        # Create durations in random order
        durations = [100.0, 10.0, 50.0, 200.0, 20.0, 150.0, 30.0, 180.0]
        target_duration = 100.0

        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,  # Disable shuffle to see sorting effect
        )

        batches = list(sampler)

        # With sorting by duration (descending), long videos should form their own batches
        # and short videos should be grouped together
        for batch in batches:
            batch_durations = [durations[idx] for idx in batch]
            # Each batch should have relatively similar durations
            # (because videos are sorted before batching)
            if len(batch_durations) > 1:
                max_dur = max(batch_durations)
                min_dur = min(batch_durations)
                # The ratio between max and min should not be too extreme
                # (videos of vastly different lengths shouldn't be in same batch)
                assert max_dur / min_dur < MAX_DURATION_RATIO, (
                    f"Batch contains videos with vastly different durations: "
                    f"max={max_dur}, min={min_dur}, ratio={max_dur/min_dur:.1f}"
                )


class TestCreateDurationBasedBatchSampler:
    """Tests for create_duration_based_batch_sampler factory function."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation file with varying durations."""
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
                },
                {
                    "video": "./videos/test3.mp4",
                    "duration": 20.0,
                    "timestamp": [5.0, 15.0],
                    "sentence": "A dog barks",
                },
            ]

            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            yield annotation_path

    def test_create_from_dataset(self, sample_annotations):
        """Test creating batch sampler from dataset."""
        from vtg_datasets.duration_sampler import create_duration_based_batch_sampler

        dataset = VideoTemporalDataset(annotation_file=sample_annotations)

        sampler = create_duration_based_batch_sampler(
            dataset=dataset,
            target_batch_duration=60.0,
            shuffle=False,
        )

        batches = list(sampler)

        # Verify all samples are included
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_extracts_durations_correctly(self, sample_annotations):
        """Test that durations are correctly extracted from dataset."""
        from vtg_datasets.duration_sampler import create_duration_based_batch_sampler

        dataset = VideoTemporalDataset(annotation_file=sample_annotations)

        sampler = create_duration_based_batch_sampler(
            dataset=dataset,
            target_batch_duration=60.0,
        )

        # Check that durations match the dataset
        assert sampler.durations == [30.0, 45.0, 20.0]

    def test_invalid_dataset_no_samples_attribute(self):
        """Test error when dataset doesn't have samples attribute."""
        from vtg_datasets.duration_sampler import create_duration_based_batch_sampler

        class InvalidDataset:
            pass

        with pytest.raises(AttributeError, match="samples"):
            create_duration_based_batch_sampler(
                dataset=InvalidDataset(),
                target_batch_duration=60.0,
            )

    def test_invalid_dataset_samples_not_dicts(self):
        """Test error when samples are not dictionaries."""
        from vtg_datasets.duration_sampler import create_duration_based_batch_sampler

        class InvalidDataset:
            samples = ["not", "dicts"]

        with pytest.raises(TypeError, match="dictionary"):
            create_duration_based_batch_sampler(
                dataset=InvalidDataset(),
                target_batch_duration=60.0,
            )


class TestDistributedBatching:
    """Tests for distributed training support in DurationBasedBatchSampler."""
    
    def test_distributed_batch_partitioning(self):
        """Test that batches are correctly partitioned across multiple GPUs."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler
        
        durations = [10.0] * 20  # 20 videos, each 10s
        target_duration = 40.0   # Each batch should have ~4 videos
        
        # Simulate 2 GPUs
        sampler_rank0 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,
            num_replicas=2,
            rank=0,
        )
        
        sampler_rank1 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,
            num_replicas=2,
            rank=1,
        )
        
        batches_rank0 = list(sampler_rank0)
        batches_rank1 = list(sampler_rank1)
        
        # Collect all indices from both GPUs
        indices_rank0 = [idx for batch in batches_rank0 for idx in batch]
        indices_rank1 = [idx for batch in batches_rank1 for idx in batch]
        
        # Each GPU should have different indices (no overlap)
        assert set(indices_rank0).isdisjoint(set(indices_rank1))
        
        # Together they should cover all samples
        all_indices = sorted(indices_rank0 + indices_rank1)
        assert all_indices == list(range(len(durations)))
    
    def test_distributed_length_calculation(self):
        """Test that __len__ returns correct count per GPU."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler
        
        durations = [10.0] * 20
        target_duration = 40.0
        
        # Single GPU
        sampler_single = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,
        )
        
        # Multi GPU
        sampler_rank0 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,
            num_replicas=2,
            rank=0,
        )
        
        sampler_rank1 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=False,
            num_replicas=2,
            rank=1,
        )
        
        # Length should be approximately half for each GPU
        len_single = len(sampler_single)
        len_rank0 = len(sampler_rank0)
        len_rank1 = len(sampler_rank1)
        
        # Each GPU should process roughly half the batches
        assert len_rank0 <= len_single
        assert len_rank1 <= len_single
        assert len_rank0 + len_rank1 >= len_single  # May have +1 due to rounding
    
    def test_distributed_with_shuffle(self):
        """Test distributed training with shuffling enabled."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler
        
        durations = [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 18.0, 22.0]
        target_duration = 40.0
        
        sampler_rank0 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=True,
            seed=42,
            num_replicas=2,
            rank=0,
        )
        
        sampler_rank1 = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=True,
            seed=42,
            num_replicas=2,
            rank=1,
        )
        
        batches_rank0 = list(sampler_rank0)
        batches_rank1 = list(sampler_rank1)
        
        # Collect all indices
        indices_rank0 = [idx for batch in batches_rank0 for idx in batch]
        indices_rank1 = [idx for batch in batches_rank1 for idx in batch]
        
        # Verify no overlap and complete coverage
        assert set(indices_rank0).isdisjoint(set(indices_rank1))
        all_indices = sorted(indices_rank0 + indices_rank1)
        assert all_indices == list(range(len(durations)))
    
    def test_distributed_set_epoch(self):
        """Test that set_epoch produces different shuffling across epochs."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler
        
        durations = [10.0] * 12
        target_duration = 40.0
        
        sampler = DurationBasedBatchSampler(
            durations=durations,
            target_batch_duration=target_duration,
            shuffle=True,
            seed=42,
            num_replicas=2,
            rank=0,
        )
        
        # Get batches for epoch 0
        sampler.set_epoch(0)
        batches_epoch0 = list(sampler)
        indices_epoch0 = [idx for batch in batches_epoch0 for idx in batch]
        
        # Get batches for epoch 1
        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)
        indices_epoch1 = [idx for batch in batches_epoch1 for idx in batch]
        
        # Different epochs should produce different ordering
        assert indices_epoch0 != indices_epoch1
    
    def test_invalid_rank_raises_error(self):
        """Test that invalid rank raises ValueError."""
        from vtg_datasets.duration_sampler import DurationBasedBatchSampler
        
        durations = [10.0] * 5
        
        # Rank >= num_replicas should raise error
        with pytest.raises(ValueError, match="Invalid rank"):
            DurationBasedBatchSampler(
                durations=durations,
                target_batch_duration=40.0,
                num_replicas=2,
                rank=2,
            )
        
        # Negative rank should raise error
        with pytest.raises(ValueError, match="Invalid rank"):
            DurationBasedBatchSampler(
                durations=durations,
                target_batch_duration=40.0,
                num_replicas=2,
                rank=-1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
