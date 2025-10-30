"""
Architecture-Specific Model Binders

Implementations for GPT-2, LLaMA, Pythia, and other model families.
"""

from typing import Dict, Any
from ual_adapter.binders.base import ModelBinder


class GPT2Binder(ModelBinder):
    """Binder for GPT-2 family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "q"
            },
            "attention_key": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "k"
            },
            "attention_value": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "v"
            },
            "attention_output": {
                "pattern": "transformer.h.{layer}.attn.c_proj",
                "fused": False
            },
            "mlp_up": {
                "pattern": "transformer.h.{layer}.mlp.c_fc",
                "fused": False
            },
            "mlp_down": {
                "pattern": "transformer.h.{layer}.mlp.c_proj",
                "fused": False
            }
        }


class LLaMABinder(ModelBinder):
    """Binder for LLaMA family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "model.layers.{layer}.self_attn.q_proj",
                "fused": False
            },
            "attention_key": {
                "pattern": "model.layers.{layer}.self_attn.k_proj",
                "fused": False
            },
            "attention_value": {
                "pattern": "model.layers.{layer}.self_attn.v_proj",
                "fused": False
            },
            "attention_output": {
                "pattern": "model.layers.{layer}.self_attn.o_proj",
                "fused": False
            },
            "mlp_up": {
                "pattern": "model.layers.{layer}.mlp.up_proj",
                "fused": False
            },
            "mlp_down": {
                "pattern": "model.layers.{layer}.mlp.down_proj",
                "fused": False
            },
            "mlp_gate": {
                "pattern": "model.layers.{layer}.mlp.gate_proj",
                "fused": False
            }
        }


class PythiaBinder(ModelBinder):
    """Binder for Pythia/GPT-NeoX family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "gpt_neox.layers.{layer}.attention.query_key_value",
                "fused": True,
                "slice": "q"
            },
            "attention_key": {
                "pattern": "gpt_neox.layers.{layer}.attention.query_key_value",
                "fused": True,
                "slice": "k"
            },
            "attention_value": {
                "pattern": "gpt_neox.layers.{layer}.attention.query_key_value",
                "fused": True,
                "slice": "v"
            },
            "attention_output": {
                "pattern": "gpt_neox.layers.{layer}.attention.dense",
                "fused": False
            },
            "mlp_up": {
                "pattern": "gpt_neox.layers.{layer}.mlp.dense_h_to_4h",
                "fused": False
            },
            "mlp_down": {
                "pattern": "gpt_neox.layers.{layer}.mlp.dense_4h_to_h",
                "fused": False
            }
        }


class QwenBinder(ModelBinder):
    """Binder for Qwen family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "q"
            },
            "attention_key": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "k"
            },
            "attention_value": {
                "pattern": "transformer.h.{layer}.attn.c_attn",
                "fused": True,
                "slice": "v"
            },
            "attention_output": {
                "pattern": "transformer.h.{layer}.attn.c_proj",
                "fused": False
            },
            "mlp_up": {
                "pattern": "transformer.h.{layer}.mlp.w1",
                "fused": False
            },
            "mlp_down": {
                "pattern": "transformer.h.{layer}.mlp.w2",
                "fused": False
            }
        }


class MistralBinder(ModelBinder):
    """Binder for Mistral family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "model.layers.{layer}.self_attn.q_proj",
                "fused": False
            },
            "attention_key": {
                "pattern": "model.layers.{layer}.self_attn.k_proj",
                "fused": False
            },
            "attention_value": {
                "pattern": "model.layers.{layer}.self_attn.v_proj",
                "fused": False
            },
            "attention_output": {
                "pattern": "model.layers.{layer}.self_attn.o_proj",
                "fused": False
            },
            "mlp_up": {
                "pattern": "model.layers.{layer}.mlp.up_proj",
                "fused": False
            },
            "mlp_down": {
                "pattern": "model.layers.{layer}.mlp.down_proj",
                "fused": False
            },
            "mlp_gate": {
                "pattern": "model.layers.{layer}.mlp.gate_proj",
                "fused": False
            }
        }


class PhiBinder(ModelBinder):
    """Binder for Microsoft Phi family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "model.layers.{layer}.mixer.Wqkv",
                "fused": True,
                "slice": "q"
            },
            "attention_key": {
                "pattern": "model.layers.{layer}.mixer.Wqkv",
                "fused": True,
                "slice": "k"
            },
            "attention_value": {
                "pattern": "model.layers.{layer}.mixer.Wqkv",
                "fused": True,
                "slice": "v"
            },
            "attention_output": {
                "pattern": "model.layers.{layer}.mixer.out_proj",
                "fused": False
            },
            "mlp_up": {
                "pattern": "model.layers.{layer}.mlp.fc1",
                "fused": False
            },
            "mlp_down": {
                "pattern": "model.layers.{layer}.mlp.fc2",
                "fused": False
            }
        }


class BERTBinder(ModelBinder):
    """Binder for BERT family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "bert.encoder.layer.{layer}.attention.self.query",
                "fused": False
            },
            "attention_key": {
                "pattern": "bert.encoder.layer.{layer}.attention.self.key",
                "fused": False
            },
            "attention_value": {
                "pattern": "bert.encoder.layer.{layer}.attention.self.value",
                "fused": False
            },
            "attention_output": {
                "pattern": "bert.encoder.layer.{layer}.attention.output.dense",
                "fused": False
            },
            "mlp_up": {
                "pattern": "bert.encoder.layer.{layer}.intermediate.dense",
                "fused": False
            },
            "mlp_down": {
                "pattern": "bert.encoder.layer.{layer}.output.dense",
                "fused": False
            }
        }


class T5Binder(ModelBinder):
    """Binder for T5 family models."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            "attention_query": {
                "pattern": "encoder.block.{layer}.layer.0.SelfAttention.q",
                "fused": False
            },
            "attention_key": {
                "pattern": "encoder.block.{layer}.layer.0.SelfAttention.k",
                "fused": False
            },
            "attention_value": {
                "pattern": "encoder.block.{layer}.layer.0.SelfAttention.v",
                "fused": False
            },
            "attention_output": {
                "pattern": "encoder.block.{layer}.layer.0.SelfAttention.o",
                "fused": False
            },
            "mlp_up": {
                "pattern": "encoder.block.{layer}.layer.1.DenseReluDense.wi",
                "fused": False
            },
            "mlp_down": {
                "pattern": "encoder.block.{layer}.layer.1.DenseReluDense.wo",
                "fused": False
            }
        }


class GenericBinder(ModelBinder):
    """Generic binder for unknown architectures."""
    
    def _define_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Generic mappings that work for many architectures."""
        return {
            "attention_query": {
                "pattern": "layers.{layer}.attention.q_proj",
                "fused": False,
                "alternatives": ["q_proj", "query", "q_lin"]
            },
            "attention_key": {
                "pattern": "layers.{layer}.attention.k_proj",
                "fused": False,
                "alternatives": ["k_proj", "key", "k_lin"]
            },
            "attention_value": {
                "pattern": "layers.{layer}.attention.v_proj",
                "fused": False,
                "alternatives": ["v_proj", "value", "v_lin"]
            },
            "attention_output": {
                "pattern": "layers.{layer}.attention.o_proj",
                "fused": False,
                "alternatives": ["o_proj", "out_proj", "dense"]
            },
            "mlp_up": {
                "pattern": "layers.{layer}.mlp.up_proj",
                "fused": False,
                "alternatives": ["up_proj", "fc1", "w1", "dense_h_to_4h"]
            },
            "mlp_down": {
                "pattern": "layers.{layer}.mlp.down_proj",
                "fused": False,
                "alternatives": ["down_proj", "fc2", "w2", "dense_4h_to_h"]
            }
        }
