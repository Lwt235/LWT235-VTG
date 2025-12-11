"""
Tests for utility functions.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from utils.common import (
    seed_everything,
    get_device,
    count_parameters,
    load_config,
    merge_configs,
    save_config,
    format_time,
    format_number,
    timestamp_to_seconds,
    seconds_to_timestamp,
    normalize_timestamp,
    denormalize_timestamp,
)
from utils.data_validation import DataValidator, validate_annotation_file
from utils.logging_utils import setup_logger, get_logger


class TestCommonUtils:
    """Tests for common utility functions."""
    
    def test_seed_everything(self):
        """Test seed setting."""
        seed_everything(42)
        
        # Check torch seed
        a = torch.rand(10)
        seed_everything(42)
        b = torch.rand(10)
        
        assert torch.allclose(a, b)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]
        
        # Test explicit device
        device = get_device("cpu")
        assert device == torch.device("cpu")
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        
        counts = count_parameters(model)
        
        assert counts["trainable"] == 10 * 5 + 5  # weights + bias
        assert counts["frozen"] == 0
        
        # Freeze model
        for p in model.parameters():
            p.requires_grad = False
        
        counts = count_parameters(model, trainable_only=False)
        assert counts["frozen"] == 10 * 5 + 5
    
    def test_format_time(self):
        """Test time formatting."""
        assert format_time(65) == "1m 5s"
        assert format_time(3665) == "1h 1m 5s"
        assert format_time(30) == "30s"
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1000) == "1.00K"
        assert format_number(1500000) == "1.50M"
        assert format_number(2000000000) == "2.00B"
        assert format_number(500) == "500"
    
    def test_timestamp_conversions(self):
        """Test timestamp conversions."""
        # timestamp_to_seconds
        assert timestamp_to_seconds("1:30") == 90.0
        assert timestamp_to_seconds("1:30:45") == 5445.0
        assert timestamp_to_seconds("45.5") == 45.5
        
        # seconds_to_timestamp
        assert seconds_to_timestamp(90.0, include_hours=False) == "01:30.000"
        assert seconds_to_timestamp(5445.0) == "01:30:45.000"
    
    def test_normalize_denormalize(self):
        """Test timestamp normalization."""
        norm_start, norm_end = normalize_timestamp(10.0, 20.0, 100.0)
        assert norm_start == 0.1
        assert norm_end == 0.2
        
        start, end = denormalize_timestamp(0.1, 0.2, 100.0)
        assert start == 10.0
        assert end == 20.0
    
    def test_config_operations(self):
        """Test config load/save/merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Save config
            config = {"a": 1, "b": {"c": 2}}
            save_config(config, config_path)
            
            # Load config
            loaded = load_config(config_path)
            assert loaded.a == 1
            assert loaded.b.c == 2
            
            # Merge configs
            config2 = {"b": {"c": 3, "d": 4}}
            merged = merge_configs(loaded, config2)
            assert merged.a == 1
            assert merged.b.c == 3
            assert merged.b.d == 4


class TestDataValidation:
    """Tests for data validation."""
    
    def test_valid_sample(self):
        """Test validation of valid sample."""
        validator = DataValidator()
        
        sample = {
            "video": "test.mp4",
            "duration": 30.0,
            "timestamp": [10.0, 20.0],
            "sentence": "A person walks into the room",
        }
        
        assert validator.validate_sample(sample)
    
    def test_missing_required_field(self):
        """Test validation with missing required field."""
        validator = DataValidator(strict_mode=False)
        
        sample = {
            "video": "test.mp4",
            "duration": 30.0,
            # Missing timestamp and sentence
        }
        
        assert not validator.validate_sample(sample)
    
    def test_invalid_timestamp(self):
        """Test validation with invalid timestamp."""
        validator = DataValidator(strict_mode=False)
        
        # End before start
        sample = {
            "video": "test.mp4",
            "duration": 30.0,
            "timestamp": [20.0, 10.0],  # Invalid
            "sentence": "Test",
        }
        
        assert not validator.validate_sample(sample)
    
    def test_validate_file(self):
        """Test file validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "annotations.jsonl"
            
            # Write test annotations
            samples = [
                {"video": "v1.mp4", "duration": 30.0, "timestamp": [5.0, 10.0], "sentence": "Test 1"},
                {"video": "v2.mp4", "duration": 25.0, "timestamp": [2.0, 8.0], "sentence": "Test 2"},
            ]
            
            with open(annotation_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            validator = DataValidator()
            total, valid, errors = validator.validate_file(annotation_path)
            
            assert total == 2
            assert valid == 2
            assert errors == 0


class TestLogging:
    """Tests for logging utilities."""
    
    def test_setup_logger(self):
        """Test logger setup."""
        logger = setup_logger("test_logger", level="DEBUG")
        
        assert logger.name == "test_logger"
        assert logger.level == 10  # DEBUG
    
    def test_get_logger(self):
        """Test getting logger."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")
        
        # Should return same logger
        assert logger1 is logger2
    
    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            logger = setup_logger("file_test", log_file=log_file)
            logger.info("Test message")
            
            # Check file was created and contains message
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
