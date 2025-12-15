"""
Tests for trainer modules.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager

from trainers.sft_trainer import VideoTemporalSFTTrainer


class TestVideoTemporalSFTTrainer:
    """Tests for VideoTemporalSFTTrainer."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer instance for testing."""
        # Mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock()]
        
        # Mock args
        mock_args = Mock()
        mock_args.output_dir = "/tmp/test_output"
        mock_args.local_rank = -1
        
        # Mock tokenizer (processing_class)
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.save_pretrained = Mock()
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={})
        
        # Create trainer instance
        trainer = VideoTemporalSFTTrainer(
            model=mock_model,
            args=mock_args,
            train_dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )
        
        return trainer

    @contextmanager
    def _patch_parent_save_model(self):
        """Context manager to patch parent class save_model method."""
        with patch('transformers.Trainer.save_model'):
            yield

    def test_save_model_saves_tokenizer(self, mock_trainer, temp_dir):
        """Test that save_model saves the tokenizer (processing_class)."""
        with self._patch_parent_save_model():
            mock_trainer.save_model(output_dir=temp_dir)
        
        # Verify that tokenizer was saved
        assert mock_trainer.processing_class.save_pretrained.called
        assert mock_trainer.processing_class.save_pretrained.call_args[0][0] == temp_dir

    def test_save_model_saves_processor(self, mock_trainer, temp_dir):
        """Test that save_model saves the processor."""
        with self._patch_parent_save_model():
            mock_trainer.save_model(output_dir=temp_dir)
        
        # Verify that processor was saved
        assert mock_trainer.processor.save_pretrained.called
        assert mock_trainer.processor.save_pretrained.call_args[0][0] == temp_dir

    def test_save_model_handles_none_tokenizer(self, mock_trainer, temp_dir):
        """Test that save_model handles None tokenizer gracefully."""
        # Set tokenizer to None using the public property
        mock_trainer.processing_class = None
        
        with self._patch_parent_save_model():
            # Should not raise an error
            mock_trainer.save_model(output_dir=temp_dir)

    def test_save_model_handles_none_processor(self, mock_trainer, temp_dir):
        """Test that save_model handles None processor gracefully."""
        # Set processor to None
        mock_trainer.processor = None
        
        with self._patch_parent_save_model():
            # Should not raise an error
            mock_trainer.save_model(output_dir=temp_dir)


class TestTemporalTokensSaving:
    """Integration tests for temporal tokens saving."""

    def test_temporal_tokens_preserved_after_save_load(self):
        """
        Test that temporal tokens are preserved after saving and loading.
        
        This is an integration test that would require actual model loading,
        so we skip it in unit tests.
        """
        pytest.skip("Integration test - requires actual model loading")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
