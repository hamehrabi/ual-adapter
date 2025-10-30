Model Binders
=============

Binders provide the mapping between architecture-specific parameter names and universal semantic roles.

Architecture Detection
----------------------

UAL automatically detects model architecture from:

* Model class name
* Configuration attributes
* Parameter name patterns

Binder Structure
----------------

Each binder defines:

.. code-block:: python

   {
       "attention_query": {
           "pattern": "model.layers.{layer}.self_attn.q_proj",
           "in_features": 4096,
           "out_features": 4096
       },
       ...
   }

Pattern Matching
----------------

Patterns use ``{layer}`` placeholder for layer indexing:

.. code-block:: python

   # Matches: model.layers.0.self_attn.q_proj
   #          model.layers.1.self_attn.q_proj
   #          ...
   pattern = "model.layers.{layer}.self_attn.q_proj"

Supported Architectures
-----------------------

GPT-2 Family
~~~~~~~~~~~~

* Parameter prefix: ``transformer.h.{layer}``
* Fused attention: ``c_attn`` for Q/K/V
* MLP: ``c_fc`` (up), ``c_proj`` (down)

LLaMA Family
~~~~~~~~~~~~

* Parameter prefix: ``model.layers.{layer}``
* Separate Q/K/V projections
* Gated MLP with ``gate_proj``

Qwen Family
~~~~~~~~~~~

* Similar to GPT-2 structure
* Fused attention with ``c_attn``
* Custom MLP naming

Custom Binders
--------------

To add a new architecture:

.. code-block:: python

   from ual_adapter.binders import BaseBinder

   class MyModelBinder(BaseBinder):
       def get_attention_query_pattern(self, layer: int) -> str:
           return f"model.layer.{layer}.attention.query"

       def get_dimensions(self, param_name: str) -> tuple:
           # Return (in_features, out_features)
           return (768, 768)
