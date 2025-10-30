Contributing
============

We welcome contributions to UAL Adapter!

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/hamehrabi/ual-adapter.git
   cd ual-adapter
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest tests/
   pytest --cov=ual_adapter tests/

Code Style
----------

We use:

* Black for formatting
* mypy for type checking
* pylint for linting

.. code-block:: bash

   black ual_adapter/
   mypy ual_adapter/
   pylint ual_adapter/

Contributing Guidelines
-----------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run test suite
6. Submit pull request

Areas for Contribution
----------------------

* New architecture binders
* Performance optimizations
* Documentation improvements
* Bug fixes
* Example notebooks

Code of Conduct
---------------

Be respectful and constructive in all interactions.
