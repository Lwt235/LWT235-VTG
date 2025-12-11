"""
Tests for reward functions.
"""

import pytest
import torch

from rewards.temporal_iou import TemporalIoU, temporal_iou, batch_temporal_iou
from rewards.segment_overlap import SegmentOverlap, segment_overlap, batch_segment_overlap
from rewards.step_consistency import StepConsistency, step_consistency
from rewards.reward_registry import RewardRegistry, CompositeReward


class TestTemporalIoU:
    """Tests for Temporal IoU reward."""
    
    def test_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        iou = temporal_iou(0.2, 0.8, 0.2, 0.8)
        assert abs(iou - 1.0) < 1e-6
    
    def test_no_overlap(self):
        """Test IoU with no overlap."""
        iou = temporal_iou(0.0, 0.2, 0.5, 0.8)
        assert abs(iou) < 1e-6
    
    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        # Pred: [0.2, 0.6], GT: [0.4, 0.8]
        # Intersection: [0.4, 0.6] = 0.2
        # Union: [0.2, 0.8] = 0.6
        # IoU = 0.2 / 0.6 = 0.333...
        iou = temporal_iou(0.2, 0.6, 0.4, 0.8)
        expected = 0.2 / 0.6
        assert abs(iou - expected) < 1e-6
    
    def test_contained_segment(self):
        """Test IoU when one segment contains the other."""
        # Pred contains GT
        iou = temporal_iou(0.1, 0.9, 0.3, 0.7)
        # Intersection: [0.3, 0.7] = 0.4
        # Union: [0.1, 0.9] = 0.8
        expected = 0.4 / 0.8
        assert abs(iou - expected) < 1e-6
    
    def test_batch_temporal_iou(self):
        """Test batch IoU computation."""
        pred = torch.tensor([[0.2, 0.8], [0.0, 0.5]])
        gt = torch.tensor([[0.2, 0.8], [0.5, 1.0]])
        
        iou = batch_temporal_iou(pred, gt)
        
        assert abs(iou[0].item() - 1.0) < 1e-6  # Perfect overlap
        assert abs(iou[1].item()) < 1e-6  # No overlap
    
    def test_temporal_iou_module(self):
        """Test TemporalIoU module."""
        module = TemporalIoU(scale=2.0)
        
        pred = torch.tensor([[0.2, 0.8]])
        gt = torch.tensor([[0.2, 0.8]])
        
        reward = module(pred, gt)
        
        assert abs(reward[0].item() - 2.0) < 1e-6  # Scale applied
    
    def test_parse_text(self):
        """Test text parsing."""
        module = TemporalIoU()
        
        # Test various formats
        predictions = [
            "0.25 to 0.75",
            "start: 0.25, end: 0.75",
            "(0.25, 0.75)",
            "[0.25, 0.75]",
        ]
        ground_truths = [(0.25, 0.75)] * 4
        
        rewards = module.compute_from_text(predictions, ground_truths)
        
        for i, reward in enumerate(rewards):
            assert abs(reward - 1.0) < 1e-6, f"Failed for format {i}"


class TestSegmentOverlap:
    """Tests for Segment Overlap reward."""
    
    def test_recall_mode(self):
        """Test recall mode (coverage of GT)."""
        # Pred covers half of GT
        overlap = segment_overlap(0.0, 0.5, 0.0, 1.0, mode="recall")
        assert abs(overlap - 0.5) < 1e-6
    
    def test_precision_mode(self):
        """Test precision mode (accuracy of pred)."""
        # Pred [0.0, 1.0], GT [0.0, 0.5]
        # Intersection: 0.5, Pred duration: 1.0
        overlap = segment_overlap(0.0, 1.0, 0.0, 0.5, mode="precision")
        assert abs(overlap - 0.5) < 1e-6
    
    def test_f1_mode(self):
        """Test F1 mode."""
        # Pred [0.0, 0.5], GT [0.25, 0.75]
        # Intersection: 0.25
        # Recall: 0.25 / 0.5 = 0.5
        # Precision: 0.25 / 0.5 = 0.5
        # F1: 2 * 0.5 * 0.5 / 1.0 = 0.5
        overlap = segment_overlap(0.0, 0.5, 0.25, 0.75, mode="f1")
        assert abs(overlap - 0.5) < 1e-6
    
    def test_perfect_overlap(self):
        """Test perfect overlap."""
        overlap = segment_overlap(0.2, 0.8, 0.2, 0.8, mode="recall")
        assert abs(overlap - 1.0) < 1e-6


class TestStepConsistency:
    """Tests for Step Consistency reward."""
    
    def test_valid_prediction(self):
        """Test with valid prediction."""
        scores = step_consistency(["0.25 to 0.75"])
        assert scores[0] > 0.5
    
    def test_invalid_format(self):
        """Test with invalid format."""
        scores = step_consistency(["invalid text"])
        assert scores[0] == 0.0
    
    def test_invalid_order(self):
        """Test with start > end."""
        module = StepConsistency()
        pred = torch.tensor([[0.8, 0.2]])  # start > end
        
        reward = module(pred)
        
        assert reward[0].item() == 0.0


class TestRewardRegistry:
    """Tests for Reward Registry."""
    
    def test_register_and_get(self):
        """Test registering and getting rewards."""
        registry = RewardRegistry()
        
        assert "temporal_iou" in registry
        assert "segment_overlap" in registry
        
        reward_cls = registry.get("temporal_iou")
        assert reward_cls == TemporalIoU
    
    def test_create_reward(self):
        """Test creating reward instance."""
        registry = RewardRegistry()
        
        reward = registry.create("temporal_iou", scale=2.0)
        
        assert isinstance(reward, TemporalIoU)
        assert reward.scale == 2.0


class TestCompositeReward:
    """Tests for Composite Reward."""
    
    def test_composite_reward(self):
        """Test composite reward computation."""
        composite = CompositeReward({
            "temporal_iou": {"weight": 1.0},
            "segment_overlap": {"weight": 1.0, "mode": "recall"},
        })
        
        pred = torch.tensor([[0.2, 0.8]])
        gt = torch.tensor([[0.2, 0.8]])
        
        reward = composite(pred, gt)
        
        # Both should give 1.0, so average is 1.0 (but normalized)
        assert reward.shape[0] == 1
    
    def test_disabled_reward(self):
        """Test that disabled rewards are not used."""
        composite = CompositeReward({
            "temporal_iou": {"weight": 1.0, "enabled": True},
            "segment_overlap": {"weight": 1.0, "enabled": False},
        })
        
        # Only temporal_iou should be in the modules
        assert len(composite.reward_modules) == 1
        assert "temporal_iou" in composite.reward_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
