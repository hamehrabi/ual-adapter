Basic Adapter Transfer
======================

This tutorial shows how to train a LoRA adapter on one model and transfer it to another.

Step 1: Prepare Training Data
------------------------------

.. code-block:: python

   training_texts = [
       "Example text 1",
       "Example text 2",
       # ... more examples
   ]

Step 2: Train Adapter
---------------------

.. code-block:: python

   from ual_adapter import UniversalAdapter
   from transformers import AutoModel, AutoTokenizer

   # Load source model
   model = AutoModel.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")

   # Create adapter
   ual = UniversalAdapter(model, tokenizer)

   # Train
   ual.train_adapter(
       domain_name="my_domain",
       texts=training_texts,
       rank=16,
       alpha=32,
       epochs=3
   )

Step 3: Export to AIR
---------------------

.. code-block:: python

   ual.export_adapter("my_domain", "adapter.air")

Step 4: Transfer to Target Model
---------------------------------

.. code-block:: python

   # Load target model
   target_model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B")
   target_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")

   # Import adapter
   target_ual = UniversalAdapter(target_model, target_tokenizer)
   target_ual.import_adapter("adapter.air")

   # Use adapted model
   outputs = target_model(**target_tokenizer("Test query", return_tensors="pt"))

Complete Example
----------------

See :doc:`../examples/complete_example` for full code.
