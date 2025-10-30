Architecture Overview
====================

UAL Adapter is built on a modular architecture that enables seamless transfer of LoRA adapters across different model families.

System Components
-----------------

.. image:: _static/architecture_diagram.svg
   :align: center
   :alt: UAL Adapter Architecture

The system consists of five main components:

1. AIR Format
~~~~~~~~~~~~~

**Architecture-Agnostic Intermediate Representation**

The AIR format is the core innovation that enables cross-architecture transfer. It:

* Maps model-specific parameter names to universal semantic roles
* Stores LoRA weights in a portable format
* Preserves metadata about training configuration
* Enables dimension-aware reconstruction

**Semantic Roles:**

* ``attention_query`` - Query projection in attention
* ``attention_key`` - Key projection in attention
* ``attention_value`` - Value projection in attention
* ``attention_output`` - Output projection in attention
* ``mlp_up`` - MLP up-projection
* ``mlp_down`` - MLP down-projection
* ``mlp_gate`` - MLP gate projection (for gated architectures)

2. Model Binders
~~~~~~~~~~~~~~~~

**Architecture-Specific Mappings**

Binders provide the mapping between model-specific parameter names and universal semantic roles.

Supported architectures:

* **GPT-2** family (GPT-2, GPT-Neo)
* **LLaMA** family (LLaMA, LLaMA-2, TinyLlama)
* **Pythia** family (EleutherAI models)
* **Qwen** family (Qwen, Qwen-2)
* **BERT** family (BERT, RoBERTa)
* **T5** family (T5, Flan-T5)

Each binder defines:

* Parameter name patterns with layer indexing
* Dimension information (in_features, out_features)
* Special handling for fused layers (e.g., GPT-2's c_attn)

3. Dimension Projection
~~~~~~~~~~~~~~~~~~~~~~~

**SVD-Based Dimension Adaptation**

When transferring between models with different dimensions, UAL uses three projection methods:

**SVD Projection (Recommended)**

.. code-block:: python

   # Projects through singular value decomposition
   # Preserves variance while adapting dimensions
   projector.project(lora_weights, target_dim, method="svd")

**Truncate Projection**

.. code-block:: python

   # Simple truncation or zero-padding
   # Fast but may lose information
   projector.project(lora_weights, target_dim, method="truncate")

**Interpolate Projection**

.. code-block:: python

   # Bilinear interpolation for smooth adaptation
   # Good for moderate dimension changes
   projector.project(lora_weights, target_dim, method="interpolate")

4. LoRA Dispatcher
~~~~~~~~~~~~~~~~~~

**Intelligent Multi-Domain Routing**

The dispatcher enables multi-agent systems with domain-specific adapters:

.. code-block:: python

   Query → Embedding → Router → Domain Selection → Adapter Application

**Router Training:**

* Uses sentence embeddings for semantic understanding
* Trains multi-class logistic regression for classification
* Provides confidence scores for each domain
* Supports confidence thresholds for fallback behavior

5. Training Pipeline
~~~~~~~~~~~~~~~~~~~~

**Efficient LoRA Training**

The training component provides:

* Automatic target module detection
* Gradient accumulation support
* Learning rate scheduling
* Checkpointing and resuming
* Comprehensive logging

Data Flow
---------

Training Phase
~~~~~~~~~~~~~~

1. Load base model and tokenizer
2. Create UniversalAdapter instance
3. Detect trainable modules automatically
4. Apply LoRA to target modules
5. Train on domain-specific data
6. Export to AIR format

Transfer Phase
~~~~~~~~~~~~~~

1. Load AIR format adapter
2. Parse semantic roles and weights
3. Detect target model architecture
4. Map semantic roles to target parameters
5. Project dimensions if needed
6. Apply adapted LoRA weights

Inference Phase
~~~~~~~~~~~~~~~

**Single Domain:**

1. Import adapter to target model
2. Run inference with adapted model

**Multi-Domain:**

1. Register multiple domain adapters
2. Train dispatcher router
3. Query routing at inference time
4. Dynamic adapter selection
5. Generate with selected adapter

Design Principles
-----------------

**Architecture Agnostic**

No hardcoded assumptions about model structure. Everything goes through semantic role mapping.

**Dimension Adaptive**

Handles any dimension mismatch through intelligent projection methods.

**Modular & Extensible**

Easy to add new architectures by implementing binders.

**Production Ready**

Comprehensive error handling, logging, and type hints throughout.

**Testable**

High test coverage with unit and integration tests.

Performance Considerations
--------------------------

**Memory Efficiency**

LoRA reduces parameters by 10,000x compared to full fine-tuning:

* Full fine-tuning: 100M+ parameters
* LoRA (rank=16): ~10K parameters

**Compute Efficiency**

Training is 3-5x faster than full fine-tuning:

* No backward pass through full model
* Only adapter parameters updated
* Smaller optimizer state

**Storage Efficiency**

AIR files are compact:

* Typical size: 1-10 MB per adapter
* vs 500MB+ for full model checkpoints

Limitations
-----------

* Transfer quality depends on architecture similarity
* Very large dimension mismatches may reduce performance
* Router accuracy depends on training examples quality
* Some architecture-specific optimizations may not transfer

Next Steps
----------

* Learn about :doc:`concepts/air_format` in detail
* Understand :doc:`concepts/dimension_projection` methods
* Explore :doc:`concepts/lora_dispatcher` for multi-domain use
* Read :doc:`advanced/custom_architectures` to add new models
