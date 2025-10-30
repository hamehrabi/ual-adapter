Installation
============

Requirements
------------

* Python 3.8 or higher
* PyTorch 2.0 or higher
* transformers 4.30 or higher

Installation Methods
--------------------

Via pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install ual-adapter

From Source
~~~~~~~~~~~

Install the development version from GitHub:

.. code-block:: bash

   git clone https://github.com/hamehrabi/ual-adapter.git
   cd ual-adapter
   pip install -e .

For Development
~~~~~~~~~~~~~~~

If you want to contribute to UAL Adapter:

.. code-block:: bash

   git clone https://github.com/hamehrabi/ual-adapter.git
   cd ual-adapter
   pip install -e ".[dev]"

This installs additional development dependencies including:

* pytest for testing
* black for code formatting
* mypy for type checking
* sphinx for documentation

Docker
~~~~~~

Use the provided Dockerfile for containerized deployment:

.. code-block:: bash

   docker build -t ual-adapter .
   docker run -it ual-adapter

Verify Installation
-------------------

After installation, verify that UAL Adapter is working correctly:

.. code-block:: python

   import ual_adapter
   print(ual_adapter.__version__)

   # Quick test
   from ual_adapter import UniversalAdapter
   from transformers import AutoModel, AutoTokenizer

   model = AutoModel.from_pretrained("gpt2")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   ual = UniversalAdapter(model, tokenizer)
   print("UAL Adapter installed successfully!")

GPU Support
-----------

UAL Adapter automatically uses GPU if available. Ensure you have:

* CUDA 11.8 or higher
* PyTorch with CUDA support

.. code-block:: bash

   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'ual_adapter'**

Make sure you installed the package correctly:

.. code-block:: bash

   pip list | grep ual-adapter

**Version conflicts with transformers**

UAL Adapter requires transformers >= 4.30. Upgrade if needed:

.. code-block:: bash

   pip install --upgrade transformers

**CUDA out of memory**

If you encounter OOM errors during training:

* Reduce batch size
* Use gradient accumulation
* Enable gradient checkpointing
* Use smaller LoRA ranks

Getting Help
~~~~~~~~~~~~

If you encounter issues:

* Check the `Troubleshooting Guide <advanced/troubleshooting.html>`_
* Search existing `GitHub Issues <https://github.com/hamehrabi/ual-adapter/issues>`_
* Open a new issue with reproduction steps
