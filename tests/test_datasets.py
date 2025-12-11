"""
Tests for dataset modules.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from datasets.video_dataset import (
    VideoTemporalDataset,
    VideoTemporalSFTDataset,
    VideoTemporalRLDataset,
)
from datasets.collate_fns import (
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
                    "kwargs": {"difficulty": "hard"},
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
