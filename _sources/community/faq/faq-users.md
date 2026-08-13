# First-Time Users' Frequently Asked Questions

## General Questions

1. What is `AutoEmulate`?
   <!-- A brief description of what the package does, its main features, and its intended use case. -->
   - A Python package that makes it easy to create emulators for complex simulations. It takes a set of simulation inputs `X` and outputs `y`, and automatically fits, optimises and evaluates various machine learning models to find the best emulator model. The emulator model can then be used as a drop-in replacement for the simulation, but will be much faster and computationally cheaper to evaluate. We have also implemented global sensitivity analysis as a common emulator application and working towards making `AutoEmulate` a true end-to-end package for building emulators.

2. How do I know whether `AutoEmulate` is the right tool for me?
   - You need to build an emulator for a simulation.
   - You want to do global sensitivity analysis
   - Your inputs `x` and outputs `y` are numeric and complete (we don't support missing data yet).
   - You have one or more input parameters and one or more output variables.
   - You have a small-ish dataset in the order of hundreds to few thousands of samples. All default emulator parameters and search spaces are optimised for smaller datasets.

3. Does `AutoEmulate` support multi-output data?
   - Yes, some models support multi-output data. When instantiating `AutoEmulate` with multioutput data, the tool will automatically restrict the search space to models that can handle it.

4. Does `AutoEmulate` support temporal or spatial data?
   - Not explicitly. AutoEmulate currently expects 2D data `[n_simulations, n_outputs]`. The second dimension could be temporal or spatial indexes but it will not explicitly model spatial or temporal correlations. This is a feature we hope to add in the future. 

5. `AutoEmulate` takes a long time to run on my dataset, why?
   - The package fits a lot of models, in particular when hyperparameters are optimised. With say 8 default models and 5-fold cross-validation, this amounts to 40 model fits. With the addition of hyperparameter optimisation (n_iter=20), this results in 800 model fits. Some models such as Gaussian Processes will take a long time to run on a CPU. 

## Usage Questions

1. What data do I need to provide to `AutoEmulate` to build an emulator?
   <!-- A simple example to get a new user started, possibly pointing to more detailed tutorials or documentation. -->
   - You'll need two input objects: `x` and `y`. `x` is an ndarray / torch tensor of shape `(n_samples, n_parameters)` and `y` is an ndarray / torch tensor of shape `(n_samples, n_outputs)`. Each sample here is a simulation run, so each row of `x` corresponds to a set of input parameters and each row of `y` corresponds to the corresponding simulation output. You'll usually have created `x` using Latin hypercube sampling or similar methods, and `y` by running the simulation on these `x` inputs.

2. Can I use `AutoEmulate` for commercial purposes?
   <!-- Information on licensing and any restrictions on use. -->
   - Yes. It's licensed under the MIT license, which allows for commercial use. See the [license](../../../LICENSE) for more information.

## Data Handling

1. What are the best practices for data preprocessing before using `AutoEmulate`?
   <!-- Tips and recommendations on preparing data, including normalisation, dealing with missing values, or data segmentation. -->
   - The user will typically run their simulation on a selected set of input parameters (-> experimental design) using a latin hypercube or other sampling method. `AutoEmulate` currently needs all inputs to be numeric and we don't support missing data. By default, `AutoEmulate` will scale the input and output data to zero mean and unit variance. There's also the option to do dimensionality reduction (see the dimensionality reduction tutorial).

## Community and Learning Resources

1. Where can I find tutorials or case studies on using `AutoEmulate`?
   <!-- Directions to comprehensive learning materials, such as video tutorials (if we want to record that), written guides, or published research papers using AutoEmulate. -->
   - See the [tutorial](../../tutorials/01_start.ipynb) for a comprehensive guide on using the package. Case studies are coming soon.

2. How can I stay updated on new releases or updates to AutoEmulate?
   <!-- Guidance on subscribing to newsletters when/if we will have that, community calls if we start that, following the project on social media if we want to create those platforms, or joining community forums/Slack once we have that ready... -->
   - Watch the [AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate).

3. What support options are available if I need help with AutoEmulate?
   <!-- Overview of support resources, including documentation, community forums/Slack when we have that ready... -->
   - Please open an issue or start a discussion on [GitHub](https://github.com/alan-turing-institute/autoemulate).
