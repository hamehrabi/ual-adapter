"""
Tests for AIR Format module
"""

import pytest
import torch
import json
import os
from pathlib import Path

from ual_adapter.core.air import AIRFormat, AIRMetadata


class TestAIRFormat:
    """Test AIR format functionality."""
    
    def test_metadata_creation(self):
        """Test AIR metadata creation."""
        metadata = AIRMetadata(
            source_model="test-model",
            source_architecture="gpt2",
            source_dimensions={"hidden_size": 768, "num_layers": 12},
            adapter_rank=16,
            adapter_alpha=16.0,
            domain="medical"
        )
        
        assert metadata.source_model == "test-model"
        assert metadata.source_architecture == "gpt2"
        assert metadata.adapter_rank == 16
        assert metadata.domain == "medical"
    
    def test_export_to_air(self, sample_lora_weights, temp_dir):
        """Test exporting weights to AIR format."""
        air_format = AIRFormat()
        metadata = AIRMetadata(
            source_model="test-model",
            source_architecture="test",
            adapter_rank=16
        )
        
        output_path = os.path.join(temp_dir, "test_adapter")
        air_format.export_to_air(sample_lora_weights, metadata, output_path)
        
        # Check files were created
        assert os.path.exists(f"{output_path}.weights.safetensors")
        assert os.path.exists(f"{output_path}.metadata.json")
        
        # Check metadata content
        with open(f"{output_path}.metadata.json", "r") as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["source_model"] == "test-model"
        assert saved_metadata["adapter_rank"] == 16
    
    def test_import_from_air(self, sample_lora_weights, temp_dir):
        """Test importing weights from AIR format."""
        air_format = AIRFormat()
        metadata = AIRMetadata(
            source_model="test-model",
            source_architecture="gpt2",
            source_dimensions={"hidden_size": 768},
            adapter_rank=16
        )
        
        # Export first
        output_path = os.path.join(temp_dir, "test_adapter")
        air_format.export_to_air(sample_lora_weights, metadata, output_path)
        
        # Import back
        target_model_info = {
            "architecture": "llama",
            "hidden_size": 768
        }
        imported_weights, imported_metadata = air_format.import_from_air(
            output_path,
            target_model_info
        )
        
        assert imported_metadata.source_model == "test-model"
        assert len(imported_weights) > 0
    
    def test_semantic_role_mapping(self):
        """Test semantic role detection."""
        air_format = AIRFormat()
        
        # Test various parameter names
        test_cases = [
            ("model.layers.0.self_attn.q_proj", "attention_query"),
            ("transformer.h.0.attn.c_attn", "attention_query"),
            ("gpt_neox.layers.0.attention.query", "attention_query"),
            ("model.layers.0.mlp.up_proj", "mlp_up"),
            ("transformer.h.0.mlp.c_fc", "mlp_up"),
        ]
        
        for param_name, expected_role in test_cases:
            role = air_format._get_semantic_role(param_name)
            assert role == expected_role, f"Failed for {param_name}"
    
    def test_layer_index_extraction(self):
        """Test layer index extraction from parameter names."""
        air_format = AIRFormat()
        
        test_cases = [
            ("model.layers.5.self_attn.q_proj", 5),
            ("transformer.h.10.attn.c_attn", 10),
            ("bert.encoder.layer.3.attention.query", 3),
            ("no_layer_info.weight", 0),
        ]
        
        for param_name, expected_idx in test_cases:
            idx = air_format._extract_layer_index(param_name)
            assert idx == expected_idx, f"Failed for {param_name}"
    
    def test_architecture_binder_retrieval(self):
        """Test getting architecture-specific binders."""
        air_format = AIRFormat()
        
        # Test known architectures
        gpt2_binder = air_format._get_architecture_binder("gpt2")
        assert "attention_query" in gpt2_binder
        
        llama_binder = air_format._get_architecture_binder("llama")
        assert "attention_query" in llama_binder
        assert llama_binder["attention_query"]["pattern"] != gpt2_binder["attention_query"]["pattern"]
        
        # Test unknown architecture
        unknown_binder = air_format._get_architecture_binder("unknown_arch")
        assert unknown_binder == {}


class TestAIRInteroperability:
    """Test AIR format interoperability across architectures."""
    
    def test_cross_architecture_mapping(self, sample_lora_weights, temp_dir):
        """Test mapping weights across different architectures."""
        air_format = AIRFormat()
        
        # Create metadata for GPT-2 source
        metadata = AIRMetadata(
            source_model="gpt2",
            source_architecture="gpt2",
            source_dimensions={"hidden_size": 768},
            adapter_rank=16
        )
        
        # Export from GPT-2 format
        output_path = os.path.join(temp_dir, "cross_arch_adapter")
        air_format.export_to_air(sample_lora_weights, metadata, output_path)
        
        # Import to different architectures
        architectures = ["llama", "pythia", "qwen"]
        
        for target_arch in architectures:
            target_info = {
                "architecture": target_arch,
                "hidden_size": 768
            }
            
            imported_weights, _ = air_format.import_from_air(
                output_path,
                target_info
            )
            
            # Should successfully import some weights
            assert len(imported_weights) > 0, f"Failed for {target_arch}"
    
    def test_dimension_compatibility_info(self):
        """Test that dimension information is preserved in metadata."""
        metadata = AIRMetadata(
            source_dimensions={"hidden_size": 768, "num_layers": 12, "vocab_size": 50257}
        )
        
        assert metadata.source_dimensions["hidden_size"] == 768
        assert metadata.source_dimensions["num_layers"] == 12
        assert metadata.source_dimensions["vocab_size"] == 50257
