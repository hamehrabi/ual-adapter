Dimension Projection
====================

Dimension projection enables transferring adapters between models with different dimensions.

Why Projection is Needed
-------------------------

Different models have different dimensions:

* GPT-2: 768 dimensions
* LLaMA-7B: 4096 dimensions
* LLaMA-13B: 5120 dimensions

Direct transfer requires dimension adaptation.

Projection Methods
------------------

SVD Projection
~~~~~~~~~~~~~~

**Recommended for most use cases**

Uses Singular Value Decomposition to preserve maximum variance:

.. code-block:: python

   # Reconstruct: ΔW = B @ A
   # SVD: ΔW = U @ S @ Vt
   # Project: A' = S[:r] @ Vt[:r]
   #          B' = U[:, :r]

Advantages:
* Preserves semantic information
* Handles both upscaling and downscaling
* Maintains rank automatically

Truncate Projection
~~~~~~~~~~~~~~~~~~~

**Fast but may lose information**

Simply truncates or zero-pads dimensions:

.. code-block:: python

   # Downscaling: A' = A[:, :target_dim]
   # Upscaling: A' = pad(A, target_dim)

Use when:
* Dimensions are similar
* Speed is critical
* Information loss is acceptable

Interpolate Projection
~~~~~~~~~~~~~~~~~~~~~~

**Smooth adaptation**

Uses bilinear interpolation:

.. code-block:: python

   A_new = F.interpolate(A, size=target_dim)

Good for:
* Moderate dimension changes
* Preserving smoothness
* Image-like weight patterns

Rank Preservation
-----------------

UAL preserves the original LoRA rank during projection:

* Maintains adapter capacity
* Prevents information loss
* Ensures consistent behavior

Quality Analysis
----------------

Projection quality metrics:

* Variance preserved
* Frobenius norm ratio
* Effective rank

.. code-block:: python

   quality = projector.analyze_projection_quality(
       original_weights,
       projected_weights
   )
   # Returns: {
   #   "variance_preserved": 0.95,
   #   "frobenius_ratio": 0.98,
   #   "effective_rank": 15
   # }
