Changelog
=========

Version 0.1.0 (2025-01-15)
--------------------------

Initial release of UAL Adapter.

Features
~~~~~~~~

* Architecture-agnostic LoRA adapter transfer
* AIR (Architecture-Agnostic Intermediate Representation) format
* Support for GPT-2, LLaMA, Pythia, Qwen architectures
* Dimension projection with SVD, truncate, and interpolate methods
* Intelligent LoRA dispatcher with multi-domain routing
* Command-line interface
* Comprehensive test suite
* Docker support

Known Limitations
~~~~~~~~~~~~~~~~~

* Transfer quality varies with architecture similarity
* Large dimension mismatches may reduce performance
* Router requires quality training examples

Future Plans
~~~~~~~~~~~~

* Add support for more architectures
* Improve projection quality metrics
* Add quantization support
* Enhance CLI with more features
