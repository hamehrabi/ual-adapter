"""
Base Model Binder

Abstract base class for architecture-specific weight mapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
from loguru import logger


class ModelBinder(ABC):
    """
    Abstract base class for model-specific binders.
    
    Binders handle the mapping between AIR semantic roles and
    model-specific parameter names.
    """
    
    def __init__(self):
        """Initialize the binder."""
        self.architecture_name = self.__class__.__name__.replace("Binder", "").lower()
        self.role_mappings = self._define_mappings()
    
    @abstractmethod
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Define the mappings between semantic roles and model-specific names.
        
        Returns:
            Dictionary mapping semantic roles to model patterns
        """
        pass
    
    def map_weights(
        self,
        weights: Dict[str, torch.Tensor],
        target_model_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Map weights to target model naming convention.
        
        Args:
            weights: Dictionary of weights with AIR or source names
            target_model_info: Information about the target model
            
        Returns:
            Dictionary with mapped weight names
        """
        mapped_weights = {}
        unmapped = []
        
        for weight_name, weight_tensor in weights.items():
            mapped_name = self._map_single_weight(weight_name, target_model_info)
            
            if mapped_name:
                mapped_weights[mapped_name] = weight_tensor
                logger.debug(f"Mapped {weight_name} -> {mapped_name}")
            else:
                unmapped.append(weight_name)
        
        if unmapped:
            logger.warning(f"Could not map {len(unmapped)} weights: {unmapped[:5]}...")
        
        logger.info(
            f"Mapped {len(mapped_weights)}/{len(weights)} weights "
            f"for {self.architecture_name}"
        )
        
        return mapped_weights
    
    def _map_single_weight(
        self,
        weight_name: str,
        target_model_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Map a single weight name to target convention.
        
        Args:
            weight_name: Source weight name
            target_model_info: Target model information
            
        Returns:
            Mapped name or None if cannot map
        """
        # Extract layer index if present
        import re
        layer_match = re.search(r"layer_(\d+)", weight_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0
        
        # Extract semantic role
        for role, mapping in self.role_mappings.items():
            if role in weight_name:
                # Format the pattern with layer index
                pattern = mapping.get("pattern", "")
                if pattern:
                    return pattern.format(layer=layer_idx)
        
        return None
    
    def get_fused_modules(self) -> List[str]:
        """
        Get list of fused modules (e.g., combined QKV projections).
        
        Returns:
            List of fused module patterns
        """
        fused = []
        for role, mapping in self.role_mappings.items():
            if mapping.get("fused", False):
                fused.append(mapping.get("pattern", ""))
        
        return list(set(fused))  # Remove duplicates
    
    def supports_architecture(self, architecture: str) -> bool:
        """
        Check if this binder supports a given architecture.
        
        Args:
            architecture: Architecture name to check
            
        Returns:
            True if supported
        """
        return architecture.lower() in [
            self.architecture_name,
            self.architecture_name.replace("_", ""),
            self.architecture_name.replace("-", "")
        ]
