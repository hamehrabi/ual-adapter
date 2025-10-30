"""
Tests for Dimension Projection module
"""

import pytest
import torch
import numpy as np

from ual_adapter.core.projection import DimensionProjector


class TestDimensionProjector:
    """Test dimension projection functionality."""
    
    def test_svd_projection_same_dimension(self):
        """Test SVD projection with same dimensions."""
        projector = DimensionProjector(variance_threshold=0.95)
        
        # Create sample LoRA matrices
        rank = 16
        dim = 768
        lora_a = torch.randn(rank, dim)
        lora_b = torch.randn(dim, rank)
        
        # Project to same dimension
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, dim, dim, method="svd"
        )
        
        assert proj_a.shape == (rank, dim)
        assert proj_b.shape == (dim, rank)
    
    def test_svd_projection_upscaling(self):
        """Test SVD projection when upscaling dimensions."""
        projector = DimensionProjector(variance_threshold=0.95)
        
        # Create sample LoRA matrices
        rank = 16
        source_dim = 768
        target_dim = 2048
        
        lora_a = torch.randn(rank, source_dim)
        lora_b = torch.randn(source_dim, rank)
        
        # Project to larger dimension
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, target_dim, target_dim, method="svd"
        )
        
        # Check dimensions
        assert proj_a.shape[1] == target_dim
        assert proj_b.shape[0] == target_dim
    
    def test_svd_projection_downscaling(self):
        """Test SVD projection when downscaling dimensions."""
        projector = DimensionProjector(variance_threshold=0.95)
        
        # Create sample LoRA matrices
        rank = 16
        source_dim = 2048
        target_dim = 768
        
        lora_a = torch.randn(rank, source_dim)
        lora_b = torch.randn(source_dim, rank)
        
        # Project to smaller dimension
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, target_dim, target_dim, method="svd"
        )
        
        # Check dimensions
        assert proj_a.shape[1] == target_dim
        assert proj_b.shape[0] == target_dim
    
    def test_truncate_projection(self):
        """Test truncation-based projection."""
        projector = DimensionProjector()
        
        # Test upscaling
        lora_a = torch.randn(16, 768)
        lora_b = torch.randn(768, 16)
        
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, 1024, 1024, method="truncate"
        )
        
        assert proj_a.shape == (16, 1024)
        assert proj_b.shape == (1024, 16)
        
        # Check that original values are preserved
        assert torch.allclose(proj_a[:, :768], lora_a)
        assert torch.allclose(proj_b[:768, :], lora_b)
    
    def test_interpolate_projection(self):
        """Test interpolation-based projection."""
        projector = DimensionProjector()
        
        lora_a = torch.randn(16, 768)
        lora_b = torch.randn(768, 16)
        
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, 1024, 1024, method="interpolate"
        )
        
        assert proj_a.shape == (16, 1024)
        assert proj_b.shape == (1024, 16)
    
    def test_variance_preservation(self):
        """Test that variance threshold is respected."""
        projector = DimensionProjector(variance_threshold=0.90)
        
        # Create LoRA matrices with known rank structure
        rank = 16
        dim = 768
        
        # Create low-rank matrices
        U = torch.randn(dim, rank)
        V = torch.randn(rank, dim)
        
        lora_b = U
        lora_a = V
        
        # Project
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, dim, dim, method="svd"
        )
        
        # Reconstruct and check variance
        original = torch.matmul(lora_b, lora_a)
        reconstructed = torch.matmul(proj_b, proj_a)
        
        original_var = torch.var(original)
        reconstruction_var = torch.var(reconstructed)
        
        # Variance should be mostly preserved
        variance_ratio = reconstruction_var / original_var
        assert variance_ratio > 0.5  # Reasonable threshold for test
    
    def test_scale_factor_computation(self):
        """Test adaptive scaling computation."""
        projector = DimensionProjector()
        
        # Test various dimension combinations
        test_cases = [
            ((768, 768), (768, 768), 1.0),
            ((768, 768), (2048, 2048), 0.612),  # sqrt(768/2048)
            ((2048, 2048), (768, 768), 1.633),  # sqrt(2048/768)
        ]
        
        for source, target, expected_scale in test_cases:
            scale = projector._compute_scale_factor(source, target)
            assert abs(scale - expected_scale) < 0.01
    
    def test_projection_quality_analysis(self):
        """Test projection quality metrics."""
        projector = DimensionProjector()
        
        # Create test matrices
        lora_a = torch.randn(16, 768)
        lora_b = torch.randn(768, 16)
        original_delta = torch.matmul(lora_b, lora_a)
        
        # Project to same dimension (should have high quality)
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, 768, 768, method="svd"
        )
        
        metrics = projector.analyze_projection_quality(
            original_delta, proj_a, proj_b
        )
        
        assert "reconstruction_error" in metrics
        assert "rank_preservation" in metrics
        assert "norm_ratio" in metrics
        assert metrics["reconstruction_error"] < 0.5  # Should be small for same-dim
    
    def test_extreme_dimension_ratios(self):
        """Test projection with extreme dimension ratios."""
        projector = DimensionProjector()
        
        # Very small to very large
        lora_a = torch.randn(8, 128)
        lora_b = torch.randn(128, 8)
        
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, 4096, 4096, method="svd"
        )
        
        assert proj_a.shape[1] == 4096
        assert proj_b.shape[0] == 4096
        
        # Very large to very small
        lora_a = torch.randn(32, 4096)
        lora_b = torch.randn(4096, 32)
        
        proj_a, proj_b = projector.project_adapter(
            lora_a, lora_b, 128, 128, method="svd"
        )
        
        assert proj_a.shape[1] == 128
        assert proj_b.shape[0] == 128


class TestProjectionMethods:
    """Test different projection methods."""
    
    def test_method_consistency(self):
        """Test that all methods produce valid outputs."""
        projector = DimensionProjector()
        
        lora_a = torch.randn(16, 768)
        lora_b = torch.randn(768, 16)
        target_dim = 1024
        
        methods = ["svd", "truncate", "interpolate"]
        
        for method in methods:
            proj_a, proj_b = projector.project_adapter(
                lora_a, lora_b, target_dim, target_dim, method=method
            )
            
            assert proj_a.shape == (16, target_dim)
            assert proj_b.shape == (target_dim, 16)
            assert not torch.isnan(proj_a).any()
            assert not torch.isnan(proj_b).any()
    
    def test_invalid_method(self):
        """Test error handling for invalid projection method."""
        projector = DimensionProjector()
        
        lora_a = torch.randn(16, 768)
        lora_b = torch.randn(768, 16)
        
        with pytest.raises(ValueError, match="Unknown projection method"):
            projector.project_adapter(
                lora_a, lora_b, 1024, 1024, method="invalid_method"
            )
