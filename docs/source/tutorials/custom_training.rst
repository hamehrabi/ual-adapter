Custom Training Configuration
=============================

Advanced training options and customization.

Training Parameters
-------------------

rank
~~~~

LoRA rank (number of low-rank dimensions):

* Lower (4-8): Faster, less parameters
* Medium (16-32): Good balance
* Higher (64+): More capacity, slower

alpha
~~~~~

LoRA scaling factor:

* Typically 2× rank
* Higher = stronger adaptation
* Lower = more conservative

Custom Target Modules
----------------------

.. code-block:: python

   ual.train_adapter(
       domain_name="custom",
       texts=training_texts,
       target_modules=["q_proj", "v_proj"],  # Only Q and V
       rank=16
   )

Gradient Accumulation
---------------------

.. code-block:: python

   ual.train_adapter(
       domain_name="large_batch",
       texts=training_texts,
       batch_size=1,
       gradient_accumulation_steps=8  # Effective batch size: 8
   )

Learning Rate Scheduling
------------------------

.. code-block:: python

   ual.train_adapter(
       domain_name="scheduled",
       texts=training_texts,
       learning_rate=1e-4,
       warmup_steps=100,
       lr_scheduler_type="cosine"
   )

Checkpointing
-------------

.. code-block:: python

   ual.train_adapter(
       domain_name="checkpointed",
       texts=training_texts,
       checkpoint_every=500,
       checkpoint_dir="./checkpoints"
   )
