UniversalAdapter
================

.. automodule:: ual_adapter.core.adapter
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``UniversalAdapter`` class is the main entry point for working with UAL Adapter. It provides methods for training, exporting, and importing LoRA adapters.

Class Definition
----------------

.. autoclass:: ual_adapter.core.adapter.UniversalAdapter
   :members:
   :special-members: __init__
   :exclude-members: __weakref__

Methods
-------

Training Methods
~~~~~~~~~~~~~~~~

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.train_adapter

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.apply_lora

Export/Import Methods
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.export_adapter

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.import_adapter

Utility Methods
~~~~~~~~~~~~~~~

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.get_adapter_info

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.list_adapters

.. automethod:: ual_adapter.core.adapter.UniversalAdapter.remove_adapter

Usage Examples
--------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   from ual_adapter import UniversalAdapter
   from transformers import AutoModel, AutoTokenizer

   model = AutoModel.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")

   ual = UniversalAdapter(model, tokenizer)

   # Train adapter
   ual.train_adapter(
       domain_name="medical",
       texts=training_texts,
       rank=16,
       alpha=32
   )

Export and Import
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export trained adapter
   ual.export_adapter("medical", "medical.air")

   # Import to different model
   target_model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B")
   target_ual = UniversalAdapter(target_model)
   target_ual.import_adapter("medical.air")

Managing Adapters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List all loaded adapters
   adapters = ual.list_adapters()

   # Get adapter information
   info = ual.get_adapter_info("medical")

   # Remove adapter
   ual.remove_adapter("medical")

See Also
--------

* :doc:`air` - AIR format implementation
* :doc:`dispatcher` - Multi-domain dispatcher
* :doc:`trainer` - Training utilities
