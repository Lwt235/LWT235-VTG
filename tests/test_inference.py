"""
Tests for inference module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.video_infer import VideoTemporalInference


class TestVideoInferenceLoRAAdapter:
    """Tests for VideoTemporalInference with LoRA adapters."""

    @patch('inference.video_infer.Qwen3VLForConditionalGeneration')
    @patch('inference.video_infer.AutoProcessor')
    @patch('inference.video_infer.AutoTokenizer')
    @patch('inference.video_infer.PeftModel')
    @patch('inference.video_infer.GenerationConfig')
    def test_generation_config_loaded_from_base_model_for_lora(
        self,
        mock_generation_config,
        mock_peft_model,
        mock_tokenizer,
        mock_processor,
        mock_model_class,
    ):
        """
        Test that GenerationConfig is loaded from base model when using LoRA adapter.
        
        This tests the fix for the issue where generation_config.json was not found
        in LoRA adapter directories.
        """
        # Create temporary directories for base model and adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            base_model_dir = Path(tmpdir) / "base_model"
            adapter_dir = Path(tmpdir) / "adapter"
            base_model_dir.mkdir()
            adapter_dir.mkdir()

            # Create adapter_config.json to simulate LoRA adapter
            adapter_config = {
                "base_model_name_or_path": str(base_model_dir),
                "peft_type": "LORA",
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)

            # Mock the from_pretrained methods
            mock_processor.from_pretrained = MagicMock(return_value=MagicMock())
            mock_tokenizer.from_pretrained = MagicMock(return_value=MagicMock())
            mock_model_instance = MagicMock()
            mock_model_instance.eval = MagicMock(return_value=None)
            mock_model_class.from_pretrained = MagicMock(return_value=mock_model_instance)
            mock_peft_model.from_pretrained = MagicMock(return_value=mock_model_instance)
            mock_generation_config.from_pretrained = MagicMock(return_value=MagicMock())

            # Create inference engine with LoRA adapter path
            # This should not raise an error
            try:
                engine = VideoTemporalInference(
                    model_path=adapter_dir,
                    use_temporal_tokens=False,
                    use_flash_attention=False,
                )
            except Exception as e:
                pytest.fail(f"Initialization failed with: {e}")

            # Verify that GenerationConfig.from_pretrained was called with base model path
            mock_generation_config.from_pretrained.assert_called_once()
            called_path = mock_generation_config.from_pretrained.call_args[0][0]
            
            # The fix should ensure generation_config is loaded from base_model_path
            assert called_path == str(base_model_dir), (
                f"GenerationConfig should be loaded from base model path {base_model_dir}, "
                f"but was called with {called_path}"
            )

    @patch('inference.video_infer.Qwen3VLForConditionalGeneration')
    @patch('inference.video_infer.AutoProcessor')
    @patch('inference.video_infer.AutoTokenizer')
    @patch('inference.video_infer.GenerationConfig')
    def test_generation_config_loaded_from_model_path_for_non_lora(
        self,
        mock_generation_config,
        mock_tokenizer,
        mock_processor,
        mock_model_class,
    ):
        """
        Test that GenerationConfig is loaded from model path when not using LoRA adapter.
        """
        # Create temporary directory for non-LoRA model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            # No adapter_config.json, so this is not a LoRA adapter

            # Mock the from_pretrained methods
            mock_processor.from_pretrained = MagicMock(return_value=MagicMock())
            mock_tokenizer.from_pretrained = MagicMock(return_value=MagicMock())
            mock_model_instance = MagicMock()
            mock_model_instance.eval = MagicMock(return_value=None)
            mock_model_class.from_pretrained = MagicMock(return_value=mock_model_instance)
            mock_generation_config.from_pretrained = MagicMock(return_value=MagicMock())

            # Create inference engine with non-LoRA model path
            try:
                engine = VideoTemporalInference(
                    model_path=model_dir,
                    use_temporal_tokens=False,
                    use_flash_attention=False,
                )
            except Exception as e:
                pytest.fail(f"Initialization failed with: {e}")

            # Verify that GenerationConfig.from_pretrained was called with model path
            mock_generation_config.from_pretrained.assert_called_once()
            called_path = mock_generation_config.from_pretrained.call_args[0][0]
            
            # For non-LoRA models, generation_config should be loaded from model_path
            assert called_path == str(model_dir), (
                f"GenerationConfig should be loaded from model path {model_dir}, "
                f"but was called with {called_path}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
