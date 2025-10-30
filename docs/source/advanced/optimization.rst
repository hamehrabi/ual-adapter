Performance Optimization
========================

Tips for optimizing UAL Adapter performance.

Memory Optimization
-------------------

Use Lower Ranks
~~~~~~~~~~~~~~~

.. code-block:: python

   # Reduces memory by 50%
   ual.train_adapter(domain="example", texts=data, rank=8)  # vs rank=16

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.gradient_checkpointing_enable()

Speed Optimization
------------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple queries at once
   results = dispatcher.batch_generate(queries, max_length=50)

Cache Embeddings
~~~~~~~~~~~~~~~~

.. code-block:: python

   dispatcher = LoRADispatcher(cache_embeddings=True)

Projection Method
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Truncate is fastest
   projector.project(weights, dim, method="truncate")

Quality vs Speed Tradeoffs
---------------------------

* SVD: Best quality, slower
* Interpolate: Good balance
* Truncate: Fastest, may lose quality
