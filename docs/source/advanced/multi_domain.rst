Multi-Domain Best Practices
===========================

Advanced patterns for multi-domain systems.

Domain Design
-------------

Clear Boundaries
~~~~~~~~~~~~~~~~

Define domains with minimal overlap:

* Medical: Health, symptoms, treatments
* Legal: Laws, contracts, regulations
* Technical: Programming, systems, engineering

Good Examples
~~~~~~~~~~~~~

Provide diverse, representative examples:

.. code-block:: python

   examples = [
       "Direct question",
       "Conceptual query",
       "Practical application",
       "Edge case scenario"
   ]

Router Optimization
-------------------

Confidence Tuning
~~~~~~~~~~~~~~~~~

Adjust threshold based on accuracy:

.. code-block:: python

   # High precision (fewer false positives)
   dispatcher = LoRADispatcher(confidence_threshold=0.9)

   # High recall (fewer false negatives)
   dispatcher = LoRADispatcher(confidence_threshold=0.5)

Performance Monitoring
----------------------

Track routing decisions:

.. code-block:: python

   # Enable logging
   import logging
   logging.basicConfig(level=logging.INFO)

   # Monitor decisions
   domain, conf, scores = dispatcher.route_query(query)
   logger.info(f"Routed to {domain} with {conf:.2f} confidence")
