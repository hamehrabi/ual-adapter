"""
Binder Registry

Manages registration and retrieval of architecture-specific binders.
"""

from typing import Dict, Optional, Type
from loguru import logger

from ual_adapter.binders.base import ModelBinder
from ual_adapter.binders.architectures import (
    GPT2Binder,
    LLaMABinder,
    PythiaBinder,
    QwenBinder,
    MistralBinder,
    PhiBinder,
    BERTBinder,
    T5Binder,
    GenericBinder
)


class BinderRegistry:
    """
    Registry for model binders.
    
    Manages the mapping between architecture names and their
    corresponding binder implementations.
    """
    
    def __init__(self):
        """Initialize the binder registry."""
        self._binders: Dict[str, Type[ModelBinder]] = {}
        self._instances: Dict[str, ModelBinder] = {}
        
        # Register default binders
        self._register_default_binders()
    
    def _register_default_binders(self) -> None:
        """Register the default set of binders."""
        default_binders = [
            ("gpt2", GPT2Binder),
            ("gptj", GPT2Binder),  # Similar structure
            ("gpt-j", GPT2Binder),
            ("llama", LLaMABinder),
            ("llama2", LLaMABinder),
            ("llama3", LLaMABinder),
            ("codellama", LLaMABinder),
            ("pythia", PythiaBinder),
            ("gpt-neox", PythiaBinder),
            ("gptneox", PythiaBinder),
            ("qwen", QwenBinder),
            ("qwen2", QwenBinder),
            ("mistral", MistralBinder),
            ("mixtral", MistralBinder),
            ("phi", PhiBinder),
            ("phi2", PhiBinder),
            ("phi3", PhiBinder),
            ("bert", BERTBinder),
            ("roberta", BERTBinder),  # Similar structure
            ("distilbert", BERTBinder),
            ("t5", T5Binder),
            ("t5-base", T5Binder),
            ("mt5", T5Binder),
            ("generic", GenericBinder),
            ("unknown", GenericBinder),
        ]
        
        for arch_name, binder_class in default_binders:
            self.register(arch_name, binder_class)
        
        logger.info(f"Registered {len(self._binders)} default binders")
    
    def register(
        self,
        architecture: str,
        binder_class: Type[ModelBinder]
    ) -> None:
        """
        Register a new binder for an architecture.
        
        Args:
            architecture: Architecture name (case-insensitive)
            binder_class: Binder class to register
        """
        arch_lower = architecture.lower()
        self._binders[arch_lower] = binder_class
        logger.debug(f"Registered binder for '{arch_lower}'")
    
    def get_binder(self, architecture: str) -> ModelBinder:
        """
        Get a binder instance for the specified architecture.
        
        Args:
            architecture: Architecture name
            
        Returns:
            Binder instance for the architecture
        """
        arch_lower = architecture.lower()
        
        # Check if we have a cached instance
        if arch_lower in self._instances:
            return self._instances[arch_lower]
        
        # Look for exact match
        if arch_lower in self._binders:
            binder_class = self._binders[arch_lower]
        else:
            # Try to find partial match
            binder_class = self._find_compatible_binder(arch_lower)
        
        if binder_class:
            instance = binder_class()
            self._instances[arch_lower] = instance
            logger.info(f"Using {binder_class.__name__} for '{architecture}'")
            return instance
        else:
            # Fall back to generic binder
            logger.warning(
                f"No specific binder found for '{architecture}', "
                "using generic binder"
            )
            instance = GenericBinder()
            self._instances[arch_lower] = instance
            return instance
    
    def _find_compatible_binder(
        self,
        architecture: str
    ) -> Optional[Type[ModelBinder]]:
        """
        Find a compatible binder for an architecture.
        
        Args:
            architecture: Architecture name to match
            
        Returns:
            Compatible binder class or None
        """
        # Try substring matching
        for registered_arch, binder_class in self._binders.items():
            if registered_arch in architecture or architecture in registered_arch:
                logger.debug(
                    f"Found compatible binder '{registered_arch}' "
                    f"for '{architecture}'"
                )
                return binder_class
        
        # Try removing version numbers
        import re
        arch_base = re.sub(r'[-_]\d+', '', architecture)
        if arch_base != architecture:
            return self._find_compatible_binder(arch_base)
        
        return None
    
    def list_supported_architectures(self) -> list:
        """
        List all supported architectures.
        
        Returns:
            List of architecture names
        """
        return sorted(self._binders.keys())
    
    def is_architecture_supported(self, architecture: str) -> bool:
        """
        Check if an architecture is supported.
        
        Args:
            architecture: Architecture name to check
            
        Returns:
            True if architecture is supported
        """
        arch_lower = architecture.lower()
        
        # Check exact match
        if arch_lower in self._binders:
            return True
        
        # Check if we can find a compatible binder
        return self._find_compatible_binder(arch_lower) is not None
