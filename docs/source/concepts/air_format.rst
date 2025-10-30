AIR Format Specification
========================

The Architecture-Agnostic Intermediate Representation (AIR) format is the cornerstone of UAL Adapter's portability.

Format Structure
----------------

An AIR file is a JSON file with the following structure:

.. code-block:: json

   {
     "metadata": {
       "version": "0.1.0",
       "source_architecture": "gpt2",
       "domain": "medical",
       "rank": 16,
       "alpha": 32,
       "created_at": "2025-01-15T10:30:00",
       "training_config": {...}
     },
     "adapters": [
       {
         "semantic_role": "attention_query",
         "layer_index": 0,
         "lora_A": [[...], ...],
         "lora_B": [[...], ...],
         "original_shape": [768, 768],
         "rank": 16
       },
       ...
     ]
   }

Semantic Roles
--------------

UAL defines universal semantic roles for model parameters:

Attention Roles
~~~~~~~~~~~~~~~

* ``attention_query`` - Query projection (Q in attention)
* ``attention_key`` - Key projection (K in attention)
* ``attention_value`` - Value projection (V in attention)
* ``attention_output`` - Output projection after attention

MLP Roles
~~~~~~~~~

* ``mlp_up`` - Up-projection in feed-forward network
* ``mlp_down`` - Down-projection in feed-forward network
* ``mlp_gate`` - Gate projection (for gated architectures like LLaMA)

Layer Indexing
--------------

Layers are indexed starting from 0. The format preserves layer information to enable:

* Layer-specific transfer strategies
* Selective layer adaptation
* Cross-layer analysis

LoRA Weight Storage
-------------------

Each adapter stores two matrices:

* ``lora_A``: Low-rank matrix A (rank × in_features)
* ``lora_B``: Low-rank matrix B (out_features × rank)

The effective weight update is: ``ΔW = (alpha/rank) * B @ A``

Metadata Fields
---------------

version
~~~~~~~
AIR format version for compatibility checking.

source_architecture
~~~~~~~~~~~~~~~~~~~
Original model architecture (e.g., "gpt2", "llama").

domain
~~~~~~
Domain identifier for the adapter.

rank
~~~~
LoRA rank used during training.

alpha
~~~~~
LoRA scaling factor.

created_at
~~~~~~~~~~
ISO 8601 timestamp of creation.

training_config
~~~~~~~~~~~~~~~
Additional training configuration (optional).

Compatibility
-------------

AIR format is designed for forward compatibility:

* New fields can be added without breaking old readers
* Version checking enables format evolution
* Architecture-agnostic design supports new models

Import Process
--------------

When importing an AIR adapter:

1. Parse JSON structure
2. Validate format version
3. Map semantic roles to target architecture
4. Check dimension compatibility
5. Project dimensions if needed
6. Apply LoRA weights to model

Export Process
--------------

When exporting to AIR:

1. Extract LoRA weights from model
2. Detect source architecture
3. Map parameters to semantic roles
4. Extract layer indices
5. Collect metadata
6. Serialize to JSON

Best Practices
--------------

* Include comprehensive metadata for reproducibility
* Document training configuration
* Version your AIR files
* Test import/export roundtrip
* Validate dimension compatibility before deployment
