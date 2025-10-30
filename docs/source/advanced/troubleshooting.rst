Troubleshooting Guide
=====================

Common issues and solutions.

Import Errors
-------------

Dimension Mismatch
~~~~~~~~~~~~~~~~~~

**Problem:** Dimensions don't match between models

**Solution:** Use SVD projection

.. code-block:: python

   projector = DimensionProjector(method="svd", variance_threshold=0.95)

Architecture Not Detected
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Model architecture not recognized

**Solution:** Add custom binder or specify manually

.. code-block:: python

   ual = UniversalAdapter(model, tokenizer, architecture="custom")

Training Issues
---------------

OOM Errors
~~~~~~~~~~

**Problem:** Out of memory during training

**Solutions:**

1. Reduce batch size
2. Use gradient accumulation
3. Lower LoRA rank
4. Enable gradient checkpointing

Slow Training
~~~~~~~~~~~~~

**Problem:** Training takes too long

**Solutions:**

1. Increase batch size
2. Use fewer target modules
3. Reduce dataset size
4. Use GPU if available

Dispatcher Issues
-----------------

Poor Routing Accuracy
~~~~~~~~~~~~~~~~~~~~~

**Problem:** Queries routed to wrong domains

**Solutions:**

1. Add more diverse examples
2. Adjust confidence threshold
3. Analyze domain overlap
4. Retrain router with better examples

Low Confidence Scores
~~~~~~~~~~~~~~~~~~~~~

**Problem:** All confidence scores below threshold

**Solutions:**

1. Lower confidence threshold
2. Improve domain examples
3. Add fallback domain
4. Check query preprocessing

Quality Issues
--------------

Poor Transfer Quality
~~~~~~~~~~~~~~~~~~~~~

**Problem:** Transferred adapter performs poorly

**Solutions:**

1. Use SVD projection method
2. Increase LoRA rank
3. Check dimension compatibility
4. Verify source adapter quality

Getting Help
------------

If issues persist:

1. Enable debug logging
2. Check GitHub issues
3. Provide minimal reproduction
4. Include error messages and logs
