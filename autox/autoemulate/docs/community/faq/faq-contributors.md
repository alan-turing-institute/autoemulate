# First-Time Contributors' Frequently Asked Questions

## Technical Questions

1. How is the AutoEmulate project structured?
   <!-- An introduction to the project's architecture and where contributors can find key components. -->
   * The key component is the `AutoEmulate` class in `autoemulate/core/compare.py`, which is the main class for setting up and comparing emulators, visualising and summarising results and saving models.
   * All other modules in `autoemulate/` are supporting modules for the main class, such as data splitting, model processing, hyperparameter searching, plotting, saving, etc.
   * `autoemulate/emulators/` contains the emulator models, which all inherit from a number of base classes captured in `base.py`.
   * `autoemulate/simulations/` contains simple example simulations.
   * `tests/` contains tests for the package.
   * `data/` contains example datasets.
   * `docs/` contains the documentation source files. We use `jupyter-book` to build the documentation.

2. How do I set up my development environment for AutoEmulate?
   <!-- Steps to configure a local development environment, including any necessary tools or dependencies. -->
   See the 'Install from source for development' section of the [installation](../../installation.md) page.

3. How do I run tests for AutoEmulate?
   <!-- Instructions on how to execute the project's test suite to ensure changes do not introduce regressions. -->
   * We use `pytest` to run the tests. To run all tests:

   ```bash
   pytest
   ```

   * To run tests with print statements:

   ```bash
   pytest -s
   ```

   * To run a specific test module:

   ```bash
   pytest tests/test_example.py
   ```

   * To run a specific test:

   ```bash
   pytest tests/test_example.py::test_function
   ```

## Community and Support

1. Where can I ask questions if I'm stuck?
   <!-- Information on where to find support, such as community forums, chat channels, or mailing lists. -->
   * We use [Discussion on GitHub](https://github.com/alan-turing-institute/autoemulate/discussions) for questions and general discussion.

2. Is there a code of conduct for contributors?
   <!-- Details on the project's code of conduct, expectations for respectful and constructive interaction, and how to report violations. -->
   * Yes, it's [here](../code-of-conduct.md).

3. How can I get involved in decision-making or project planning as a contributor?
   <!-- Explanation of how the project governance works, ways to participate in project roadmap discussions, and opportunities for contributors to influence development priorities. -->
   * We use GitHub [Discussions](https://github.com/alan-turing-institute/autoemulate/discussions) for general discussion and [Issues](https://github.com/alan-turing-institute/autoemulate/issues) for project planning and development.