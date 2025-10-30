"""
Universal Adapter Main Module

Core class that orchestrates architecture-agnostic LoRA transfer
across different model families.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

from ual_adapter.core.air import AIRFormat, AIRMetadata
from ual_adapter.core.projection import DimensionProjector
from ual_adapter.core.dispatcher import LoRADispatcher
from ual_adapter.utils.model_utils import ModelAnalyzer
from ual_adapter.binders.base import ModelBinder
from ual_adapter.binders.registry import BinderRegistry


class UniversalAdapter:
    """
    Main class for Universal Adapter LoRA functionality.
    
    Handles training, exporting, importing, and transferring LoRA adapters
    across different model architectures.
    """
    
    def __init__(
        self,
        base_model: Union[PreTrainedModel, nn.Module],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "auto"
    ):
        """
        Initialize Universal Adapter.
        
        Args:
            base_model: The base model to work with
            tokenizer: Optional tokenizer for the model
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        
        # Initialize components
        self.air_format = AIRFormat()
        self.projector = DimensionProjector()
        self.analyzer = ModelAnalyzer(base_model)
        self.binder_registry = BinderRegistry()
        
        # Storage for adapters
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.adapter_metadata: Dict[str, AIRMetadata] = {}
        
        # Analyze model architecture
        self.model_info = self.analyzer.analyze()
        logger.info(
            f"Initialized UAL for {self.model_info['architecture']} model "
            f"with {self.model_info['num_parameters']:,} parameters"
        )
    
    def train_adapter(
        self,
        adapter_name: str,
        training_data: Union[List[str], Any],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train a new LoRA adapter.
        
        Args:
            adapter_name: Name for the adapter
            training_data: Training data (texts or dataset)
            rank: LoRA rank
            alpha: LoRA alpha scaling parameter
            dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (auto-detected if None)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            **training_kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training adapter '{adapter_name}'...")
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = self.analyzer.get_lora_target_modules()
            logger.info(f"Auto-detected target modules: {target_modules}")
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules
        )
        
        # Apply LoRA to model
        peft_model = get_peft_model(self.base_model, peft_config)
        
        # Training would happen here (simplified for example)
        # In production, use HuggingFace Trainer or custom training loop
        logger.info("Training adapter... (simplified implementation)")
        
        # Extract LoRA weights
        lora_weights = {}
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                lora_weights[name] = param.detach().clone()
        
        # Store adapter
        self.adapters[adapter_name] = lora_weights
        
        # Create metadata
        metadata = AIRMetadata(
            source_model=self.model_info.get("model_name", "unknown"),
            source_architecture=self.model_info["architecture"],
            source_dimensions={
                "hidden_size": self.model_info["hidden_size"],
                "num_layers": self.model_info["num_layers"],
            },
            adapter_rank=rank,
            adapter_alpha=alpha,
            training_config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
            domain=adapter_name,
            created_at=datetime.now().isoformat()
        )
        self.adapter_metadata[adapter_name] = metadata
        
        logger.info(f"✅ Trained adapter '{adapter_name}' with {len(lora_weights)} weights")
        
        return {
            "adapter_name": adapter_name,
            "num_weights": len(lora_weights),
            "total_parameters": sum(w.numel() for w in lora_weights.values()),
            "metadata": metadata
        }
    
    def export_adapter(
        self,
        adapter_name: str,
        output_path: str
    ) -> None:
        """
        Export adapter to portable AIR format.
        
        Args:
            adapter_name: Name of adapter to export
            output_path: Path to save AIR files
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        weights = self.adapters[adapter_name]
        metadata = self.adapter_metadata[adapter_name]
        
        self.air_format.export_to_air(weights, metadata, output_path)
        logger.info(f"✅ Exported adapter '{adapter_name}' to {output_path}")
    
    def import_adapter(
        self,
        air_path: str,
        adapter_name: Optional[str] = None,
        projection_method: str = "svd",
        auto_project: bool = True
    ) -> Dict[str, Any]:
        """
        Import adapter from AIR format.
        
        Args:
            air_path: Path to AIR format files
            adapter_name: Name for imported adapter (uses domain from metadata if None)
            projection_method: Method for dimension projection
            auto_project: Whether to automatically project dimensions
            
        Returns:
            Import results dictionary
        """
        # Load from AIR format
        weights, metadata = self.air_format.import_from_air(
            air_path,
            self.model_info
        )
        
        # Use domain from metadata if no name specified
        if adapter_name is None:
            adapter_name = metadata.domain or "imported"
        
        # Check if dimension projection needed
        source_hidden = metadata.source_dimensions.get("hidden_size")
        target_hidden = self.model_info["hidden_size"]
        
        if source_hidden != target_hidden and auto_project:
            logger.info(
                f"Projecting dimensions: {source_hidden} -> {target_hidden}"
            )
            weights = self._project_weights(
                weights,
                source_hidden,
                target_hidden,
                method=projection_method
            )
        
        # Store adapter
        self.adapters[adapter_name] = weights
        self.adapter_metadata[adapter_name] = metadata
        
        results = {
            "adapter_name": adapter_name,
            "source_architecture": metadata.source_architecture,
            "target_architecture": self.model_info["architecture"],
            "dimension_projected": source_hidden != target_hidden,
            "num_weights": len(weights),
            "attachment_rate": self._calculate_attachment_rate(weights)
        }
        
        logger.info(
            f"✅ Imported adapter '{adapter_name}' "
            f"({results['attachment_rate']:.1%} attachment rate)"
        )
        
        return results
    
    def transfer_to_model(
        self,
        adapter_name: str,
        target_model: Union[PreTrainedModel, nn.Module],
        projection_method: str = "svd",
        verify_compatibility: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Transfer adapter to a different model architecture.
        
        Args:
            adapter_name: Name of adapter to transfer
            target_model: Target model to transfer to
            projection_method: Method for dimension projection
            verify_compatibility: Whether to verify architecture compatibility
            
        Returns:
            Tuple of (transferred weights, transfer report)
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # Analyze target model
        target_analyzer = ModelAnalyzer(target_model)
        target_info = target_analyzer.analyze()
        
        logger.info(
            f"Transferring '{adapter_name}' from {self.model_info['architecture']} "
            f"to {target_info['architecture']}"
        )
        
        # Get appropriate binder
        binder = self.binder_registry.get_binder(target_info["architecture"])
        
        # Project dimensions if needed
        weights = self.adapters[adapter_name]
        if self.model_info["hidden_size"] != target_info["hidden_size"]:
            weights = self._project_weights(
                weights,
                self.model_info["hidden_size"],
                target_info["hidden_size"],
                method=projection_method
            )
        
        # Apply binder mappings
        transferred_weights = binder.map_weights(weights, target_info)
        
        # Generate transfer report
        report = {
            "source_architecture": self.model_info["architecture"],
            "target_architecture": target_info["architecture"],
            "dimension_change": f"{self.model_info['hidden_size']} -> {target_info['hidden_size']}",
            "weights_transferred": len(transferred_weights),
            "attachment_rate": self._calculate_attachment_rate(transferred_weights),
            "projection_method": projection_method if self.model_info["hidden_size"] != target_info["hidden_size"] else None
        }
        
        logger.info(
            f"✅ Transfer complete: {report['weights_transferred']} weights, "
            f"{report['attachment_rate']:.1%} attachment rate"
        )
        
        return transferred_weights, report
    
    def _project_weights(
        self,
        weights: Dict[str, torch.Tensor],
        source_dim: int,
        target_dim: int,
        method: str = "svd"
    ) -> Dict[str, torch.Tensor]:
        """Project adapter weights to different dimensions."""
        projected = {}
        
        for name, weight in weights.items():
            if "lora_A" in name or "lora_B" in name:
                # Determine if this is A or B matrix
                is_a_matrix = "lora_A" in name
                
                if is_a_matrix:
                    # A matrix: rank x in_features
                    target_in = target_dim
                    target_out = weight.shape[0]  # Keep rank same
                else:
                    # B matrix: out_features x rank
                    target_in = weight.shape[1]  # Keep rank same
                    target_out = target_dim
                
                # Project using appropriate method
                if len(weight.shape) == 2:
                    # Find corresponding A/B pair
                    pair_name = name.replace("lora_A", "lora_B") if is_a_matrix else name.replace("lora_B", "lora_A")
                    if pair_name in weights:
                        # Project together for better results
                        a_weight = weight if is_a_matrix else weights[pair_name]
                        b_weight = weights[pair_name] if is_a_matrix else weight
                        
                        proj_a, proj_b = self.projector.project_adapter(
                            a_weight, b_weight, target_dim, target_dim, method
                        )
                        
                        if is_a_matrix:
                            projected[name] = proj_a
                        else:
                            projected[name] = proj_b
                    else:
                        # Single matrix projection
                        projected[name] = self._resize_tensor(
                            weight, (target_out, target_in)
                        )
            else:
                # Non-LoRA weight, keep as is
                projected[name] = weight
        
        return projected
    
    def _resize_tensor(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Resize tensor by truncation or padding."""
        resized = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Copy overlapping region
        slices = tuple(
            slice(0, min(tensor.shape[i], target_shape[i]))
            for i in range(len(target_shape))
        )
        resized[slices] = tensor[slices]
        
        return resized
    
    def _calculate_attachment_rate(
        self,
        weights: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate the attachment success rate for weights."""
        if not weights:
            return 0.0
        
        # Simple heuristic: check if weights have reasonable magnitudes
        valid_weights = 0
        for weight in weights.values():
            if weight.abs().mean() > 1e-6:  # Not effectively zero
                valid_weights += 1
        
        return valid_weights / len(weights)
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all loaded adapters with their information."""
        adapters_info = []
        
        for name in self.adapters:
            info = {
                "name": name,
                "num_weights": len(self.adapters[name]),
                "total_parameters": sum(
                    w.numel() for w in self.adapters[name].values()
                ),
            }
            
            if name in self.adapter_metadata:
                metadata = self.adapter_metadata[name]
                info.update({
                    "source_model": metadata.source_model,
                    "source_architecture": metadata.source_architecture,
                    "rank": metadata.adapter_rank,
                    "created_at": metadata.created_at,
                })
            
            adapters_info.append(info)
        
        return adapters_info
    
    def clear_adapters(self) -> None:
        """Clear all loaded adapters."""
        self.adapters.clear()
        self.adapter_metadata.clear()
        logger.info("Cleared all adapters")
