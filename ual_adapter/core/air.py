"""
Architecture-Agnostic Intermediate Representation (AIR) Format

This module handles the conversion of model-specific LoRA weights to a portable
format that can be transferred across different architectures.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
import safetensors.torch
from loguru import logger


@dataclass
class AIRMetadata:
    """Metadata for AIR format adapters."""
    
    version: str = "1.0.0"
    source_model: str = ""
    source_architecture: str = ""
    source_dimensions: Dict[str, int] = None
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    training_config: Dict[str, Any] = None
    domain: str = ""
    description: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if self.source_dimensions is None:
            self.source_dimensions = {}
        if self.training_config is None:
            self.training_config = {}


class AIRFormat:
    """
    Handles conversion between model-specific LoRA weights and portable AIR format.
    
    The AIR format uses semantic role-based naming instead of model-specific
    parameter names, enabling cross-architecture transfer.
    """
    
    # Semantic role definitions
    ATTENTION_ROLES = {
        "attention_query": ["q_proj", "query", "q_lin", "Wqkv.q", "c_attn"],
        "attention_key": ["k_proj", "key", "k_lin", "Wqkv.k"],
        "attention_value": ["v_proj", "value", "v_lin", "Wqkv.v"],
        "attention_output": ["o_proj", "out_proj", "dense"],
    }
    
    MLP_ROLES = {
        "mlp_up": ["up_proj", "w1", "c_fc", "intermediate.dense", "mlp.fc1"],
        "mlp_down": ["down_proj", "w2", "c_proj", "output.dense", "mlp.fc2"],
        "mlp_gate": ["gate_proj", "w3", "gate", "mlp.gate"],
    }
    
    def __init__(self):
        """Initialize AIR format handler."""
        self.role_mappings = {**self.ATTENTION_ROLES, **self.MLP_ROLES}
        
    def export_to_air(
        self,
        lora_weights: Dict[str, torch.Tensor],
        metadata: AIRMetadata,
        output_path: str
    ) -> None:
        """
        Export LoRA weights to AIR format.
        
        Args:
            lora_weights: Dictionary of LoRA weights with model-specific names
            metadata: Metadata about the adapter
            output_path: Path to save the AIR file
        """
        air_weights = {}
        unmapped_modules = []
        
        # Convert model-specific names to semantic roles
        for param_name, weight in lora_weights.items():
            role = self._get_semantic_role(param_name)
            
            if role:
                # Extract layer index
                layer_idx = self._extract_layer_index(param_name)
                air_key = f"layer_{layer_idx}.{role}"
                air_weights[air_key] = weight
                logger.debug(f"Mapped {param_name} -> {air_key}")
            else:
                unmapped_modules.append(param_name)
                logger.warning(f"Could not map parameter: {param_name}")
        
        if unmapped_modules:
            logger.info(f"Unmapped modules: {unmapped_modules}")
        
        # Save weights and metadata
        save_data = {
            "weights": air_weights,
            "metadata": asdict(metadata),
            "unmapped": unmapped_modules,
        }
        
        # Use safetensors for efficient storage
        safetensors.torch.save_file(
            air_weights,
            f"{output_path}.weights.safetensors",
            metadata={"format": "air", "version": metadata.version}
        )
        
        # Save metadata separately
        with open(f"{output_path}.metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)
            
        logger.info(f"Exported adapter to AIR format: {output_path}")
        
    def import_from_air(
        self,
        air_path: str,
        target_model_info: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], AIRMetadata]:
        """
        Import AIR format adapter for a target model.
        
        Args:
            air_path: Path to the AIR format files
            target_model_info: Information about the target model architecture
            
        Returns:
            Tuple of (converted weights dict, metadata)
        """
        # Load weights
        air_weights = safetensors.torch.load_file(f"{air_path}.weights.safetensors")
        
        # Load metadata
        with open(f"{air_path}.metadata.json", "r") as f:
            metadata_dict = json.load(f)
            metadata = AIRMetadata(**metadata_dict)
        
        # Convert to target model naming
        target_weights = {}
        for air_key, weight in air_weights.items():
            target_name = self._air_to_model_specific(
                air_key,
                target_model_info["architecture"]
            )
            if target_name:
                target_weights[target_name] = weight
                logger.debug(f"Mapped {air_key} -> {target_name}")
        
        logger.info(f"Imported {len(target_weights)} weights from AIR format")
        return target_weights, metadata
    
    def _get_semantic_role(self, param_name: str) -> Optional[str]:
        """Map a model-specific parameter name to a semantic role."""
        param_lower = param_name.lower()
        
        for role, patterns in self.role_mappings.items():
            for pattern in patterns:
                if pattern in param_lower:
                    return role
        
        return None
    
    def _extract_layer_index(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        import re
        
        # Common patterns for layer indices
        patterns = [
            r"layer[s]?\.(\d+)",
            r"h\.(\d+)",
            r"block[s]?\.(\d+)",
            r"transformer\.block\.(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        
        return 0  # Default to layer 0 if no index found
    
    def _air_to_model_specific(
        self,
        air_key: str,
        target_architecture: str
    ) -> Optional[str]:
        """Convert AIR role to target model-specific naming."""
        # Parse AIR key
        parts = air_key.split(".")
        if len(parts) != 2:
            return None
            
        layer_part, role = parts
        layer_idx = int(layer_part.split("_")[1])
        
        # Get architecture-specific naming
        binder = self._get_architecture_binder(target_architecture)
        if not binder:
            return None
            
        return binder.get(role, {}).get("pattern", "").format(layer=layer_idx)
    
    def _get_architecture_binder(self, architecture: str) -> Dict[str, Dict]:
        """Get the naming patterns for a specific architecture."""
        # This would load from config files in production
        binders = {
            "gpt2": {
                "attention_query": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "attention_key": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "attention_value": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "mlp_up": {"pattern": "transformer.h.{layer}.mlp.c_fc"},
                "mlp_down": {"pattern": "transformer.h.{layer}.mlp.c_proj"},
            },
            "llama": {
                "attention_query": {"pattern": "model.layers.{layer}.self_attn.q_proj"},
                "attention_key": {"pattern": "model.layers.{layer}.self_attn.k_proj"},
                "attention_value": {"pattern": "model.layers.{layer}.self_attn.v_proj"},
                "attention_output": {"pattern": "model.layers.{layer}.self_attn.o_proj"},
                "mlp_up": {"pattern": "model.layers.{layer}.mlp.up_proj"},
                "mlp_down": {"pattern": "model.layers.{layer}.mlp.down_proj"},
                "mlp_gate": {"pattern": "model.layers.{layer}.mlp.gate_proj"},
            },
            "pythia": {
                "attention_query": {"pattern": "gpt_neox.layers.{layer}.attention.query"},
                "attention_key": {"pattern": "gpt_neox.layers.{layer}.attention.key"},
                "attention_value": {"pattern": "gpt_neox.layers.{layer}.attention.value"},
                "mlp_up": {"pattern": "gpt_neox.layers.{layer}.mlp.dense_h_to_4h"},
                "mlp_down": {"pattern": "gpt_neox.layers.{layer}.mlp.dense_4h_to_h"},
            },
            "qwen": {
                "attention_query": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "attention_key": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "attention_value": {"pattern": "transformer.h.{layer}.attn.c_attn"},
                "attention_output": {"pattern": "transformer.h.{layer}.attn.c_proj"},
                "mlp_up": {"pattern": "transformer.h.{layer}.mlp.w1"},
                "mlp_down": {"pattern": "transformer.h.{layer}.mlp.w2"},
                "mlp_gate": {"pattern": "transformer.h.{layer}.mlp.c_proj"},
            },
        }

        return binders.get(architecture.lower(), {})
