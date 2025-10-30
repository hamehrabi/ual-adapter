Quick Start Guide
=================

This guide will help you get started with UAL Adapter in minutes.

Basic Workflow
--------------

The typical UAL Adapter workflow consists of three main steps:

1. **Train** a LoRA adapter on your source model
2. **Export** the adapter to AIR format
3. **Import** and use the adapter on a different target model

Step 1: Train an Adapter
-------------------------

First, train a LoRA adapter on your source model:

.. code-block:: python

   from ual_adapter import UniversalAdapter
   from transformers import AutoModel, AutoTokenizer

   # Load source model (e.g., GPT-2)
   model = AutoModel.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")

   # Create adapter
   ual = UniversalAdapter(model, tokenizer)

   # Prepare training data
   training_texts = [
       "Your training text here...",
       "More training examples...",
       # ... more texts
   ]

   # Train domain-specific adapter
   ual.train_adapter(
       domain_name="medical",
       texts=training_texts,
       rank=16,
       alpha=32,
       epochs=3,
       batch_size=4
   )

Step 2: Export to AIR Format
-----------------------------

Export your trained adapter to a portable AIR file:

.. code-block:: python

   # Export adapter
   ual.export_adapter("medical", "medical_adapter.air")

The AIR format is architecture-agnostic and can be transferred to any supported model.

Step 3: Transfer to Target Model
---------------------------------

Import the adapter into a different model architecture:

.. code-block:: python

   from transformers import AutoModel

   # Load target model (e.g., LLaMA)
   target_model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B")
   target_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")

   # Create adapter for target model
   target_ual = UniversalAdapter(target_model, target_tokenizer)

   # Import AIR adapter
   target_ual.import_adapter("medical_adapter.air")

   # Use the adapted model
   outputs = target_model(**tokenizer("Medical query here", return_tensors="pt"))

Multi-Domain Dispatcher
-----------------------

For advanced use cases with multiple domains:

.. code-block:: python

   from ual_adapter import LoRADispatcher

   # Create dispatcher
   dispatcher = LoRADispatcher(
       base_model=target_model,
       tokenizer=target_tokenizer,
       encoder_model="all-MiniLM-L6-v2"
   )

   # Register multiple domain adapters
   dispatcher.register_domain(
       domain_name="medical",
       adapter_path="medical_adapter.air",
       examples=[
           "What are the symptoms of diabetes?",
           "How does insulin work?",
           # ... more medical examples
       ]
   )

   dispatcher.register_domain(
       domain_name="legal",
       adapter_path="legal_adapter.air",
       examples=[
           "What is contract law?",
           "Explain liability clauses.",
           # ... more legal examples
       ]
   )

   # Train router
   dispatcher.train_router()

   # Automatic domain routing
   response = dispatcher.generate(
       "What are the side effects of aspirin?",
       max_length=100
   )

Command-Line Interface
----------------------

UAL Adapter provides a CLI for common operations:

Train an adapter:

.. code-block:: bash

   ual-adapter train \\
       --model gpt2 \\
       --data training_data.txt \\
       --domain medical \\
       --output medical_adapter.air

Transfer an adapter:

.. code-block:: bash

   ual-adapter transfer \\
       --source medical_adapter.air \\
       --target-model TinyLlama/TinyLlama-1.1B \\
       --output medical_llama.air

Use dispatcher:

.. code-block:: bash

   ual-adapter dispatch \\
       --model TinyLlama/TinyLlama-1.1B \\
       --domains medical:medical.air legal:legal.air \\
       --query "What is informed consent?"

Next Steps
----------

* Read the :doc:`architecture` guide to understand how UAL works
* Explore :doc:`tutorials/index` for detailed examples
* Check :doc:`api/adapter` for the complete API reference
* Learn about :doc:`concepts/air_format` for technical details
