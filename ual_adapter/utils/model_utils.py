"""
Model Analysis Utilities

Tools for analyzing model architectures and detecting LoRA-compatible modules.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from loguru import logger


class ModelAnalyzer:
    """
    Analyzes model architectures to extract useful information for UAL.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model analyzer.
        
        Args:
            model: The model to analyze
        """
        self.model = model
        self._architecture = None
        self._module_map = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "architecture": self._detect_architecture(),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_layers": self._count_layers(),
            "hidden_size": self._detect_hidden_size(),
            "module_types": self._get_module_types(),
            "attention_modules": self._find_attention_modules(),
            "mlp_modules": self._find_mlp_modules(),
        }
        
        # Add model name if available
        if hasattr(self.model, "name_or_path"):
            info["model_name"] = self.model.name_or_path
        elif hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path"):
            info["model_name"] = self.model.config._name_or_path
        
        return info
    
    def get_lora_target_modules(self) -> List[str]:
        """
        Auto-detect the best modules for LoRA application.
        
        Returns:
            List of module names/patterns for LoRA targets
        """
        model_type = self._detect_architecture().lower()
        
        # Get all module names
        all_modules = [name for name, _ in self.model.named_modules()]
        
        # Architecture-specific patterns
        architecture_patterns = {
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gptj": ["q_proj", "v_proj"],
            "gpt_neox": ["query_key_value", "dense"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "bert": ["query", "key", "value", "dense"],
            "roberta": ["query", "key", "value", "dense"],
            "t5": ["q", "k", "v", "o", "wi", "wo"],
            "phi": ["Wqkv", "out_proj", "fc1", "fc2"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["c_attn", "c_proj", "w1", "w2"],
        }
        
        # Try to find architecture-specific patterns
        target_modules = []
        for arch, patterns in architecture_patterns.items():
            if arch in model_type:
                for pattern in patterns:
                    # Check if pattern exists in model
                    for module_name in all_modules:
                        if pattern in module_name and '.' not in module_name.replace(pattern, ''):
                            if pattern not in target_modules:
                                target_modules.append(pattern)
                                
                if target_modules:
                    logger.debug(f"Found architecture-specific patterns for {arch}")
                    break
        
        # If no specific patterns found, use generic attention module detection
        if not target_modules:
            target_modules = self._generic_lora_detection()
        
        # Final fallback
        if not target_modules:
            logger.warning("Could not detect specific modules, using Linear layers")
            target_modules = ["Linear"]
        
        return list(set(target_modules))  # Remove duplicates
    
    def _generic_lora_detection(self) -> List[str]:
        """Generic detection of LoRA-compatible modules."""
        target_modules = []
        
        for name, module in self.model.named_modules():
            # Look for linear layers in attention/mlp blocks
            if isinstance(module, nn.Linear):
                name_lower = name.lower()
                
                # Check for attention-related patterns
                attention_keywords = ["attention", "attn", "query", "key", "value", "proj"]
                mlp_keywords = ["mlp", "ffn", "feedforward", "fc", "dense"]
                
                is_attention = any(keyword in name_lower for keyword in attention_keywords)
                is_mlp = any(keyword in name_lower for keyword in mlp_keywords)
                
                if is_attention or is_mlp:
                    # Extract the final component
                    final_component = name.split('.')[-1]
                    if final_component not in target_modules:
                        target_modules.append(final_component)
        
        return target_modules
    
    def _detect_architecture(self) -> str:
        """Detect the model architecture."""
        if self._architecture:
            return self._architecture
        
        # Check model class name
        model_class = self.model.__class__.__name__.lower()
        
        # Common architecture mappings
        architecture_map = {
            "gpt2": "gpt2",
            "gptj": "gptj",
            "gptneox": "gpt_neox",
            "llama": "llama",
            "opt": "opt",
            "bloom": "bloom",
            "bert": "bert",
            "roberta": "roberta",
            "t5": "t5",
            "phi": "phi",
            "mistral": "mistral",
            "qwen": "qwen",
            "pythia": "pythia",
            "falcon": "falcon",
            "mpt": "mpt",
        }
        
        for key, arch in architecture_map.items():
            if key in model_class:
                self._architecture = arch
                return arch
        
        # Check config if available
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "model_type"):
                self._architecture = config.model_type
                return config.model_type
        
        # Default
        self._architecture = "unknown"
        return "unknown"
    
    def _detect_hidden_size(self) -> int:
        """Detect the model's hidden size."""
        # Try config first
        if hasattr(self.model, "config"):
            config = self.model.config
            
            # Common attribute names for hidden size
            size_attrs = ["hidden_size", "d_model", "n_embd", "dim"]
            for attr in size_attrs:
                if hasattr(config, attr):
                    return getattr(config, attr)
        
        # Try to infer from linear layers
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Assume first dimension is often hidden size
                return module.weight.shape[0]
        
        return 768  # Default fallback
    
    def _count_layers(self) -> int:
        """Count the number of transformer layers."""
        # Try config first
        if hasattr(self.model, "config"):
            config = self.model.config
            
            # Common attribute names for layer count
            layer_attrs = ["num_hidden_layers", "n_layers", "n_layer", "num_layers"]
            for attr in layer_attrs:
                if hasattr(config, attr):
                    return getattr(config, attr)
        
        # Count by module patterns
        layer_count = 0
        for name in self.model.state_dict().keys():
            # Look for layer indices
            import re
            patterns = [r"layer\.(\d+)", r"layers\.(\d+)", r"h\.(\d+)", r"block\.(\d+)"]
            
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_count = max(layer_count, layer_idx + 1)
        
        return layer_count if layer_count > 0 else 12  # Default fallback
    
    def _get_module_types(self) -> Dict[str, int]:
        """Get counts of different module types."""
        module_types = {}
        
        for module in self.model.modules():
            module_type = module.__class__.__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
        
        return module_types
    
    def _find_attention_modules(self) -> List[str]:
        """Find attention-related modules."""
        attention_modules = []
        
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ["attention", "attn", "self_attn"]):
                if isinstance(module, nn.Linear):
                    attention_modules.append(name)
        
        return attention_modules
    
    def _find_mlp_modules(self) -> List[str]:
        """Find MLP/FFN modules."""
        mlp_modules = []
        
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ["mlp", "ffn", "feedforward", "fc"]):
                if isinstance(module, nn.Linear):
                    mlp_modules.append(name)
        
        return mlp_modules
    
    def get_module_dimensions(self, module_name: str) -> Optional[Tuple[int, int]]:
        """
        Get input and output dimensions for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Tuple of (in_features, out_features) or None
        """
        try:
            module = self.model
            for part in module_name.split('.'):
                module = getattr(module, part)
            
            if isinstance(module, nn.Linear):
                return (module.in_features, module.out_features)
            elif hasattr(module, "weight"):
                weight_shape = module.weight.shape
                if len(weight_shape) == 2:
                    return (weight_shape[1], weight_shape[0])
        except AttributeError:
            pass
        
        return None
    
    def validate_lora_compatibility(
        self,
        target_modules: List[str]
    ) -> Dict[str, bool]:
        """
        Validate if target modules are compatible with LoRA.
        
        Args:
            target_modules: List of module names/patterns to check
            
        Returns:
            Dictionary mapping module patterns to compatibility status
        """
        compatibility = {}
        
        for pattern in target_modules:
            found = False
            for name, module in self.model.named_modules():
                if pattern in name:
                    # Check if module is compatible (Linear layer)
                    if isinstance(module, nn.Linear):
                        found = True
                        break
            
            compatibility[pattern] = found
        
        return compatibility
