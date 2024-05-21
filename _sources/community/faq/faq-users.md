# First-Time Users' Frequently Asked Questions

## General Questions

1. What is `AutoEmulate`?
   <!-- A brief description of what the package does, its main features, and its intended use case. -->
   - A Python package that makes it easy to build emulators for complex simulations. It takes a set of simulation inputs `X` and outputs `y`, and automatically fits, optimises and evaluates various machine learning models to find the best emulator model. The emulator model can then be used as a drop-in replacement for the simulation, but will be much faster and computationally cheaper to evaluate. 

2. How do I install `AutoEmulate`?
   <!-- Step-by-step instructions on installing the package, including any dependencies that might be required. -->
   - See the [installation guide](../../getting-started/installation.md) for detailed instructions.

3. What are the prerequisites for using `AutoEmulate`?
   <!-- Information on the knowledge or data required to effectively use AutoEmulate, such as familiarity with Python, machine learning concepts, or specific data formats. -->
   - `AutoEmulate` is designed to be easy to use. The user has to first generate a dataset of simulation inputs `X` and outputs `y`, and optimally have a basic understanding of Python and machine learning concepts.

## Usage Questions

1. How do I start using `AutoEmulate` with my simulation?
   <!-- A simple example to get a new user started, possibly pointing to more detailed tutorials or documentation. -->
   - See the [getting started guide](../../getting-started/quickstart.ipynb) or a more [in-depth tutorial](../../tutorials/01_start.ipynb).

2. What kind of data does `AutoEmulate` need to build an emulator?
   <!-- Clarification on the types of datasets suitable for analysis, including data formats and recommended data sizes. -->

   - `AutoEmulate` takes simulation inputs `X` and simulation outputs `y` to build an emulator.`X` is an ndarray of shape `(n_samples, n_parameters)` and `y` is an ndarray of shape `(n_samples, n_outputs)`. Each sample here is a simulation run, so each row of `X` corresponds to a set of input parameters and each row of `y` corresponds to the corresponding simulation output. Currently, all inputs and outputs should be numeric, and we don't support missing data.

   - All models work with multi-output data. We have optimised `AutoEmulate` to work with smaller datasets (in the order of hundreds to thousands of samples). Training emulators with large datasets (hundreds of thousands of samples) may currently require a long time and is not recommended.

3. How do I interpret the results from `AutoEmulate`?
   <!-- Guidance on understanding the output of the software, including any metrics or visualizations it produces. -->
   - See the [tutorial](../../tutorials/01_start.ipynb) for an example of how to interpret the results from `AutoEmulate`. Briefly, `X` and `y` are first split into training and test sets. Cross-validation and/or hyperparameter optimisation are performed on the training data. After comparing the results from different emulators, the user can evaluate the chosen emulator on the test set with `AutoEmulate.evaluate_model()`, and plot test set predictions with `AutoEmulate.plot_model()`, see [autoemulate.compare](../../reference/compare.rst) module for details.

   - An important thing to note is that the emulator can only be as good as the data it was trained on. Therefore, the experimental design (on which points the simulation was evaluated) is key to obtaining a good emulator.

4. Can I use `AutoEmulate` for commercial purposes?
   <!-- Information on licensing and any restrictions on use. -->
   - Yes. It's licensed under the MIT license, which allows for commercial use. See the [license](../../../LICENSE) for more information.

## Advanced Usage

1. Does AutoEmulate support parallel processing or high-performance computing (HPC) environments?
   <!-- Details on the software's capabilities to leverage multi-threading, distributed computing, or HPC resources to speed up computations. -->
   - Yes, [AutoEmulate.setup()](../../reference/compare.rst) has an `n_jobs` parameter which allows to parallelise cross-validation and hyperparameter optimisation.

2. Can AutoEmulate be integrated with other data analysis or simulation tools?
   <!-- Information on APIs, file formats, or protocols that facilitate the integration of AutoEmulate with other software ecosystems. -->
   - `AutoEmulate` takes simple `X` and `y` ndarrays as input, and returns emulator models that can be saved and loaded with `joblib`. All emulators are written as scikit learn estimators, so they can be used like any other scikit learn model in a pipeline.

## Data Handling

1. What are the best practices for data preprocessing before using `AutoEmulate`?
   <!-- Tips and recommendations on preparing data, including normalisation, dealing with missing values, or data segmentation. -->
   - The user will typically run their simulation on a selected set of input parameters (-> experimental design) using a latin hypercube or other sampling method. `AutoEmulate` currently needs all inputs to be numeric and we don't support missing data. By default, `AutoEmulate` will scale the input data to zero mean and unit variance, and there's the option to do dimensionality reduction in `setup()`.

2. How does AutoEmulate handle large datasets?
   <!-- Advice on managing large-scale data analyses, potential memory management features, or ways to streamline processing. -->
   - `AutoEmulate` is optimised to work with smaller datasets (in the order of hundreds to thousands of samples). Training emulators with large datasets (hundreds of thousands of samples) may currently require a long time and is not recommended. Emulators are created because it's expensive to evaluate the simulation, so we expect most users to have a relatively small dataset.

## Troubleshooting

1. What common issues might I encounter when using `AutoEmulate`, and how can I solve them?
   <!-- A list of frequently encountered problems with suggested solutions, possibly linked to a more extensive troubleshooting guide. -->
   - `AutoEmulate.setup()` has a `log_to_file` option to log all warnings/errors to a file. It also has a `verbose` option to print more information to the console. If you encounter an error, please open an issue (see below).

2. How can I report a bug or request a feature in `AutoEmulate`?
   <!-- Instructions on the proper channels for reporting issues or suggesting enhancements, including any templates or information to include. -->
   - You can report a bug or request a new feature through the [issue templates](https://github.com/alan-turing-institute/autoemulate/issues/new/choose) in our GitHub repository. Head on over there and choose one of the templates for your purpose and get started.

## Community and Learning Resources

1. Are there any community projects or collaborations using `AutoEmulate` I can join or learn from?
   <!-- Information on community-led projects, study groups, or collaborative research initiatives involving AutoEmulate. -->
   - Reach out to Martin ([email](mailto:mstoffel@turing.ac.uk)) or Kalle ([email](mailto:kwesterline@turing.ac.uk)) for more information.

2. Where can I find tutorials or case studies on using `AutoEmulate`?
   <!-- Directions to comprehensive learning materials, such as video tutorials (if we want to record that), written guides, or published research papers using AutoEmulate. -->
   - See the [tutorial](../../tutorials/01_start.ipynb) for a comprehensive guide on using the package.

3. How can I stay updated on new releases or updates to AutoEmulate?
   <!-- Guidance on subscribing to newsletters when/if we will have that, community calls if we start that, following the project on social media if we want to create those platforms, or joining community forums/Slack once we have that ready... -->
   - Watch the [AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate).

4. What support options are available if I need help with AutoEmulate?
   <!-- Overview of support resources, including documentation, community forums/Slack when we have that ready... -->
   - Please open an issue or contact the maintainer ([email](mailto:mstoffel@turing.ac.uk)) directly.
