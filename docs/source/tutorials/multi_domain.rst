Multi-Domain Dispatcher
=======================

Learn how to use the dispatcher for multiple domain adapters.

Overview
--------

The dispatcher automatically routes queries to the appropriate domain adapter.

Step 1: Train Multiple Adapters
--------------------------------

Train separate adapters for each domain following the basic transfer tutorial.

Step 2: Create Dispatcher
--------------------------

.. code-block:: python

   from ual_adapter import LoRADispatcher

   dispatcher = LoRADispatcher(
       base_model=model,
       tokenizer=tokenizer,
       confidence_threshold=0.7
   )

Step 3: Register Domains
-------------------------

.. code-block:: python

   # Medical domain
   dispatcher.register_domain(
       domain_name="medical",
       adapter_path="medical.air",
       examples=[
           "What are symptoms of diabetes?",
           "How does insulin work?",
           "Explain heart disease"
       ]
   )

   # Legal domain
   dispatcher.register_domain(
       domain_name="legal",
       adapter_path="legal.air",
       examples=[
           "What is a contract?",
           "Explain liability",
           "Define tort law"
       ]
   )

Step 4: Train Router
--------------------

.. code-block:: python

   dispatcher.train_router()

Step 5: Use for Inference
--------------------------

.. code-block:: python

   # Automatic routing
   response = dispatcher.generate(
       "What are the side effects of aspirin?",
       max_length=100
   )

   # Manual routing
   domain, conf, scores = dispatcher.route_query(
       "Explain informed consent"
   )
   print(f"Selected domain: {domain} (confidence: {conf:.2f})")
