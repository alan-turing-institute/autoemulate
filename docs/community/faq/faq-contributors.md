# First-Time Contributors' Frequently Asked Questions

## Technical Questions

1. How is the AutoEmulate project structured?
   <!-- An introduction to the project's architecture and where contributors can find key components. -->
   * The key component is the `AutoEmulate` class in `autoemulate/compare.py`, which is the main class for setting up and comparing emulators, visualising and summarising results, saving models, and applications such as sensitivity analysis.
   * All other modules in `autoemulate/` are supporting modules for the main class, such as data splitting, model processing, hyperparameter searching, plotting, saving, etc.
   * `autoemulate/emulators/` contains the emulator models, which are implemented as [scikit-learn estimators](https://scikit-learn.org/1.5/developers/develop.html). Architectures for deep learning models are in `autoemulate/emulators/neural_networks/`, which feed into the emulators via [skorch](https://skorch.readthedocs.io/en/latest/?badge=latest).
   * Emulators need to be registered in the model registry in `autoemulate/emulators/__init__.py` to be available in `AutoEmulate`.
   * `autoemulate/simulations/` contains simple example simulations.
   * `tests/` contains tests for the package.
   * `data/` contains example datasets.
   * `docs/` contains the documentation source files. We use `jupyter-book` to build the documentation.

2. How do I set up my development environment for AutoEmulate?
   <!-- Steps to configure a local development environment, including any necessary tools or dependencies. -->
   * Ensure have poetry installed. If not, install it following the [official instructions](https://python-poetry.org/docs/).
   * Fork and clone the repository.

   ```bash
   git clone https://github.com/alan-turing-institute/autoemulate.git
   cd autoemulate
   ```

   * Install the dependencies:

   ```bash
   poetry install
   ```

   * If needed, enter the shell (optional when working using an IDE which recognises poetry environments):

   ```bash
   poetry shell
   ```

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

2. How does AutoEmulate handle contributions related to security issues?
   <!-- Guidelines on reporting security vulnerabilities and how they are addressed by the project. -->

3. Is there a code of conduct for contributors?
   <!-- Details on the project's code of conduct, expectations for respectful and constructive interaction, and how to report violations. -->

4. How can I get involved in decision-making or project planning as a contributor?
   <!-- Explanation of how the project governance works, ways to participate in project roadmap discussions, and opportunities for contributors to influence development priorities. -->

## Beyond Code Contributions

1. Can I contribute without coding, for example, through design, marketing, or community management?
   <!-- Overview of non-code contribution opportunities, including outreach efforts, event organisation, or community moderation. -->

2. How does the project recognise or reward contributions?
   <!-- Information on acknowledgment of contributions through all-contributors. -->

3. Are there regular meetings or forums where contributors can discuss the project?
   <!-- Schedule and formats of any regular contributor meetings, forums for discussion, or channels for real-time communication among contributors. -->
