"""
Test configuration and fixtures
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any
import tempfile
import os


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_layers=12):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Simulate transformer layers
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.ModuleDict({
                        'q_proj': nn.Linear(hidden_size, hidden_size),
                        'k_proj': nn.Linear(hidden_size, hidden_size),
                        'v_proj': nn.Linear(hidden_size, hidden_size),
                        'o_proj': nn.Linear(hidden_size, hidden_size),
                    }),
                    'mlp': nn.ModuleDict({
                        'up_proj': nn.Linear(hidden_size, 4 * hidden_size),
                        'down_proj': nn.Linear(4 * hidden_size, hidden_size),
                    })
                })
                for _ in range(num_layers)
            ])
            
            # Add config attribute
            self.config = type('Config', (), {
                'hidden_size': hidden_size,
                'num_hidden_layers': num_layers,
                'model_type': 'test_transformer'
            })()
    
        def forward(self, x):
            return x
    
    return SimpleTransformer()


@pytest.fixture
def sample_lora_weights():
    """Create sample LoRA weights."""
    weights = {
        'layers.0.attention.q_proj.lora_A': torch.randn(16, 768),
        'layers.0.attention.q_proj.lora_B': torch.randn(768, 16),
        'layers.0.attention.v_proj.lora_A': torch.randn(16, 768),
        'layers.0.attention.v_proj.lora_B': torch.randn(768, 16),
        'layers.0.mlp.up_proj.lora_A': torch.randn(16, 768),
        'layers.0.mlp.up_proj.lora_B': torch.randn(3072, 16),
    }
    return weights


@pytest.fixture
def sample_training_texts():
    """Sample training texts for testing."""
    return [
        "The patient presents with acute symptoms.",
        "Legal analysis of the contract terms.",
        "Machine learning model optimization techniques.",
        "Financial market analysis and predictions.",
        "Clinical trial results show significant improvement.",
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
            
        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            if isinstance(texts, str):
                texts = [texts]
            
            batch_encoding = {
                'input_ids': torch.randint(0, 1000, (len(texts), 128)),
                'attention_mask': torch.ones(len(texts), 128)
            }
            
            if kwargs.get('return_tensors') == 'pt':
                return batch_encoding
            return batch_encoding
        
        def encode(self, text, **kwargs):
            return torch.randint(0, 1000, (128,)).tolist()
        
        def decode(self, ids, **kwargs):
            return "Decoded text output"
    
    return MockTokenizer()


# Test environment configuration
TEST_CONFIG = {
    "test_models": [
        "gpt2",
        "llama",
        "pythia",
        "qwen"
    ],
    "test_dimensions": [
        (768, 768),    # Same dimension
        (768, 2048),   # Upscaling
        (2048, 768),   # Downscaling
        (768, 896),    # Different dimension
    ],
    "test_ranks": [4, 8, 16, 32],
    "test_projection_methods": ["svd", "truncate", "interpolate"]
}
