# Universal Adapter LoRA (UAL)

Article: [Universal Adapter LoRA (UAL) for Architecture-Agnostic Transfer](https://ai.vixra.org/abs/2510.0074)

A Python package for creating portable, architecture-agnostic LoRA adapters that can be transferred across different model families without retraining.

## Features

- **Architecture-Agnostic Transfer**: Train once, deploy everywhere across GPT-2, LLaMA, Pythia, Qwen, and more
- **Intelligent LoRA Dispatcher**: Automatically routes queries to the most suitable domain adapter
- **Dimension-Adaptive Projection**: Handles arbitrary dimension mismatches through SVD
- **Multi-Agent Support**: Deploy heterogeneous models with shared expertise
- **Production-Ready**: Clean, testable code with comprehensive error handling

## Installation

```bash
pip install ual-adapter
```

Or install from source:

```bash
git clone https://github.com/hamehrabi/ual-adapter.git
cd ual-adapter
pip install -e .
```

## Quick Start

```python
from ual_adapter import UniversalAdapter, LoRADispatcher
from transformers import AutoModel, AutoTokenizer

# Load your base model
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create UAL adapter
ual = UniversalAdapter(model, tokenizer)

# Train a domain-specific LoRA
medical_texts = ["Medical text 1", "Medical text 2", ...]
ual.train_adapter("medical", medical_texts)

# Export to AIR format (portable)
ual.export_adapter("medical", "medical_adapter.air")

# Transfer to different model
target_model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B")
target_ual = UniversalAdapter(target_model)
target_ual.import_adapter("medical_adapter.air")

# Use with intelligent dispatcher
dispatcher = LoRADispatcher(target_ual)
response = dispatcher.generate("What are the symptoms of diabetes?")
```

## Architecture

The package consists of several key components:

1. **AIR Format**: Architecture-Agnostic Intermediate Representation for portable adapters
2. **Model Binders**: Family-aware mappings for different architectures
3. **Dimension Projection**: SVD-based adaptation for dimension mismatches
4. **LoRA Dispatcher**: Intelligent routing based on query embeddings
5. **Training Pipeline**: Efficient adapter training with automatic target detection

## Documentation

Full documentation available at [https://ual-adapter.readthedocs.io](https://ual-adapter.readthedocs.io)

## Contributing

If you're interested in contributing, please reach out via email: [mehrabi.hamed@outlook.com](mailto:mehrabi.hamed@outlook.com)

## License

MIT License
