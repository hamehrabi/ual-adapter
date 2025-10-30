Adding Custom Architectures
===========================

Guide for adding support for new model architectures.

Creating a Binder
-----------------

.. code-block:: python

   from ual_adapter.binders import BaseBinder

   class MyModelBinder(BaseBinder):
       \"\"\"Binder for MyCustomModel architecture.\"\"\"

       architecture_name = "my_model"

       def get_parameter_mapping(self) -> dict:
           return {
               "attention_query": {
                   "pattern": "encoder.layer.{layer}.attention.self.query",
                   "in_features": 768,
                   "out_features": 768
               },
               "attention_key": {
                   "pattern": "encoder.layer.{layer}.attention.self.key",
                   "in_features": 768,
                   "out_features": 768
               },
               # ... more mappings
           }

Registering the Binder
----------------------

.. code-block:: python

   from ual_adapter.binders import BinderRegistry

   registry = BinderRegistry()
   registry.register("my_model", MyModelBinder)

Testing the Binder
------------------

.. code-block:: python

   # Test detection
   from transformers import AutoModel
   model = AutoModel.from_pretrained("path/to/my_model")

   from ual_adapter.utils import detect_architecture
   arch = detect_architecture(model)
   assert arch == "my_model"

   # Test parameter mapping
   ual = UniversalAdapter(model, tokenizer)
   ual.export_adapter("test", "test.air")

   # Verify AIR format
   import json
   with open("test.air") as f:
       air = json.load(f)
   assert air["metadata"]["source_architecture"] == "my_model"

Contributing
------------

To contribute your binder to UAL Adapter:

1. Fork the repository
2. Add binder to ``ual_adapter/binders/architectures.py``
3. Add tests to ``tests/test_binders.py``
4. Submit pull request
