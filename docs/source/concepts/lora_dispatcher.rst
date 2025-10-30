LoRA Dispatcher
===============

The dispatcher enables intelligent routing between multiple domain-specific adapters.

Architecture
------------

.. code-block:: text

   Query → Sentence Encoder → Domain Router → Adapter Selection → Generation

Components
----------

Sentence Encoder
~~~~~~~~~~~~~~~~

Converts text to semantic embeddings:

* Default: ``all-MiniLM-L6-v2``
* Can use any SentenceTransformer model
* Embeddings cached for performance

Domain Router
~~~~~~~~~~~~~

Multi-class logistic regression classifier:

* Trained on domain examples
* Outputs confidence scores
* Supports confidence thresholds

Domain Registry
~~~~~~~~~~~~~~~

Manages multiple adapters:

* Domain name → adapter mapping
* Example queries per domain
* Metadata and configuration

Usage Pattern
-------------

1. Register Domains
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dispatcher.register_domain(
       domain_name="medical",
       adapter_path="medical.air",
       examples=[
           "What are symptoms of flu?",
           "How does insulin work?",
       ]
   )

2. Train Router
~~~~~~~~~~~~~~~

.. code-block:: python

   dispatcher.train_router()

3. Route Queries
~~~~~~~~~~~~~~~~

.. code-block:: python

   domain, confidence, scores = dispatcher.route_query(
       "What is diabetes?"
   )
   # Returns: ("medical", 0.95, {"medical": 0.95, "legal": 0.05})

Confidence Thresholds
---------------------

Set minimum confidence for domain selection:

.. code-block:: python

   dispatcher = LoRADispatcher(
       base_model=model,
       confidence_threshold=0.7
   )

If confidence < threshold:
* Returns None (no domain)
* Can fallback to default adapter
* Prevents incorrect routing

Embedding Cache
---------------

Cache embeddings for faster routing:

.. code-block:: python

   dispatcher = LoRADispatcher(
       base_model=model,
       cache_embeddings=True
   )

Benefits:
* Faster repeated queries
* Reduced compute
* Consistent routing

Domain Overlap Analysis
-----------------------

Analyze similarity between domains:

.. code-block:: python

   overlap = dispatcher.analyze_domain_overlap()
   # Returns: {
   #   ("medical", "legal"): 0.15,
   #   ("medical", "tech"): 0.08,
   #   ...
   # }

Use to:
* Identify ambiguous domains
* Improve example selection
* Optimize domain definitions
