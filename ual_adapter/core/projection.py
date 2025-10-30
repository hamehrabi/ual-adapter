"""
Dimension-Adaptive Projection System

Handles dimension mismatches when transferring adapters between models
with different hidden dimensions using SVD-based projection.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
from loguru import logger


class DimensionProjector:
    """
    Handles dimension adaptation for LoRA weights using SVD projection.
    
    This allows adapters trained on models with one dimension (e.g., 768)
    to be transferred to models with different dimensions (e.g., 2048).
    """
    
    def __init__(self, variance_threshold: float = 0.95):
        """
        Initialize the dimension projector.
        
        Args:
            variance_threshold: Percentage of variance to preserve (0-1)
        """
        self.variance_threshold = variance_threshold
        
    def project_adapter(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        target_in_dim: int,
        target_out_dim: int,
        method: str = "svd"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project LoRA adapter to target dimensions.
        
        Args:
            lora_a: LoRA A matrix (rank x in_features)
            lora_b: LoRA B matrix (out_features x rank)
            target_in_dim: Target input dimension
            target_out_dim: Target output dimension
            method: Projection method ("svd", "truncate", "interpolate")
            
        Returns:
            Tuple of projected (A, B) matrices
        """
        if method == "svd":
            return self._svd_projection(
                lora_a, lora_b, target_in_dim, target_out_dim
            )
        elif method == "truncate":
            return self._truncate_projection(
                lora_a, lora_b, target_in_dim, target_out_dim
            )
        elif method == "interpolate":
            return self._interpolate_projection(
                lora_a, lora_b, target_in_dim, target_out_dim
            )
        else:
            raise ValueError(f"Unknown projection method: {method}")
    
    def _svd_projection(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        target_in_dim: int,
        target_out_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SVD-based projection preserving the most important subspace.
        """
        # Compute the full delta matrix
        delta = torch.matmul(lora_b, lora_a).cpu().numpy()
        source_out_dim, source_in_dim = delta.shape

        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(delta, full_matrices=False)

        # Determine rank to preserve
        # Always preserve the original LoRA rank to maintain adapter capacity
        original_rank = lora_a.shape[0]
        preserved_rank = min(original_rank, len(S))
        
        logger.debug(
            f"Preserving rank {preserved_rank}/{len(S)} "
            f"({self.variance_threshold*100:.1f}% variance)"
        )
        
        # Truncate to preserved rank
        U_trunc = U[:, :preserved_rank]
        S_trunc = S[:preserved_rank]
        Vt_trunc = Vt[:preserved_rank, :]
        
        # Adapt to target dimensions
        U_adapted = self._resize_matrix(U_trunc, (target_out_dim, preserved_rank))
        Vt_adapted = self._resize_matrix(Vt_trunc, (preserved_rank, target_in_dim))
        
        # Apply adaptive scaling
        scale_factor = self._compute_scale_factor(
            delta.shape, (target_out_dim, target_in_dim)
        )
        S_scaled = S_trunc * scale_factor
        
        # Reconstruct as LoRA factors
        sqrt_S = np.sqrt(S_scaled)
        lora_b_new = U_adapted * sqrt_S[np.newaxis, :]
        lora_a_new = sqrt_S[:, np.newaxis] * Vt_adapted
        
        # Convert back to torch tensors
        device = lora_a.device
        dtype = lora_a.dtype
        
        return (
            torch.from_numpy(lora_a_new).to(device=device, dtype=dtype),
            torch.from_numpy(lora_b_new).to(device=device, dtype=dtype)
        )
    
    def _truncate_projection(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        target_in_dim: int,
        target_out_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple truncation or zero-padding projection.
        """
        rank = lora_a.shape[0]
        
        # Handle A matrix (rank x in_features)
        if target_in_dim > lora_a.shape[1]:
            # Pad with zeros
            lora_a_new = torch.zeros(rank, target_in_dim, dtype=lora_a.dtype)
            lora_a_new[:, :lora_a.shape[1]] = lora_a
        else:
            # Truncate
            lora_a_new = lora_a[:, :target_in_dim]
        
        # Handle B matrix (out_features x rank)
        if target_out_dim > lora_b.shape[0]:
            # Pad with zeros
            lora_b_new = torch.zeros(target_out_dim, rank, dtype=lora_b.dtype)
            lora_b_new[:lora_b.shape[0], :] = lora_b
        else:
            # Truncate
            lora_b_new = lora_b[:target_out_dim, :]

        # Return without scaling to preserve exact values
        # This maintains the original LoRA weights in the overlapping region
        return lora_a_new, lora_b_new
    
    def _interpolate_projection(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        target_in_dim: int,
        target_out_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolation-based projection for smooth resizing.
        """
        import torch.nn.functional as F
        
        # Reshape for interpolation
        rank = lora_a.shape[0]
        
        # Interpolate A matrix
        lora_a_reshaped = lora_a.unsqueeze(0).unsqueeze(0)  # 1x1xRxD
        lora_a_new = F.interpolate(
            lora_a_reshaped,
            size=(rank, target_in_dim),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Interpolate B matrix
        lora_b_reshaped = lora_b.unsqueeze(0).unsqueeze(0)  # 1x1xDxR
        lora_b_new = F.interpolate(
            lora_b_reshaped,
            size=(target_out_dim, rank),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return lora_a_new, lora_b_new
    
    def _resize_matrix(
        self,
        matrix: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize a matrix to target shape by truncation or padding.
        """
        current_shape = matrix.shape
        resized = np.zeros(target_shape)
        
        # Copy the overlapping region
        min_rows = min(current_shape[0], target_shape[0])
        min_cols = min(current_shape[1], target_shape[1])
        resized[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
        
        return resized
    
    def _compute_scale_factor(
        self,
        source_shape: Tuple[int, int],
        target_shape: Tuple[int, int]
    ) -> float:
        """
        Compute adaptive scaling factor to maintain update magnitude.
        """
        source_dim = np.sqrt(source_shape[0] * source_shape[1])
        target_dim = np.sqrt(target_shape[0] * target_shape[1])
        
        # Scale inversely with dimension ratio
        return np.sqrt(source_dim / target_dim)
    
    def analyze_projection_quality(
        self,
        original_delta: torch.Tensor,
        projected_a: torch.Tensor,
        projected_b: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze the quality of a projection.
        
        Returns metrics like reconstruction error, rank preservation, etc.
        """
        # Reconstruct delta from projected factors
        reconstructed = torch.matmul(projected_b, projected_a)
        
        # Compute metrics
        metrics = {}
        
        # Reconstruction error (if dimensions match)
        if original_delta.shape == reconstructed.shape:
            mse = torch.mean((original_delta - reconstructed) ** 2).item()
            relative_error = mse / torch.mean(original_delta ** 2).item()
            metrics["reconstruction_error"] = relative_error
        
        # Rank analysis
        original_rank = torch.linalg.matrix_rank(original_delta).item()
        reconstructed_rank = torch.linalg.matrix_rank(reconstructed).item()
        metrics["rank_preservation"] = reconstructed_rank / original_rank
        
        # Norm preservation
        original_norm = torch.norm(original_delta, 'fro').item()
        reconstructed_norm = torch.norm(reconstructed, 'fro').item()
        metrics["norm_ratio"] = reconstructed_norm / original_norm
        
        # Subspace alignment (principal angle)
        if original_delta.shape == reconstructed.shape:
            U_orig, _, _ = torch.svd(original_delta)
            U_recon, _, _ = torch.svd(reconstructed)
            
            # Compute cosine of principal angles
            cos_angles = torch.diagonal(
                torch.matmul(U_orig[:, :5].T, U_recon[:, :5])
            )
            metrics["subspace_alignment"] = torch.mean(cos_angles).item()
        
        return metrics
