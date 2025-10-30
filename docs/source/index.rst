Welcome to UAL Adapter Documentation
=====================================

**Universal Adapter LoRA (UAL)** is a Python package for creating portable, architecture-agnostic LoRA adapters that can be transferred across different model families without retraining.

.. image:: https://img.shields.io/pypi/v/ual-adapter
   :target: https://pypi.org/project/ual-adapter/
   :alt: PyPI Version

.. image:: https://img.shields.io/github/license/hamehrabi/ual-adapter
   :target: https://github.com/hamehrabi/ual-adapter/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

Key Features
------------

* **Architecture-Agnostic Transfer**: Train once, deploy everywhere across GPT-2, LLaMA, Pythia, Qwen, and more
* **Intelligent LoRA Dispatcher**: Automatically routes queries to the most suitable domain adapter
* **Dimension-Adaptive Projection**: Handles arbitrary dimension mismatches through SVD
* **Multi-Agent Support**: Deploy heterogeneous models with shared expertise
* **Production-Ready**: Clean, testable code with comprehensive error handling

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install ual-adapter

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   architecture
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/air_format
   concepts/model_binders
   concepts/dimension_projection
   concepts/lora_dispatcher

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/adapter
   api/air
   api/dispatcher
   api/projection
   api/binders
   api/trainer
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   advanced/custom_architectures
   advanced/multi_domain
   advanced/optimization
   advanced/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
