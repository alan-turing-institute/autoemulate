---
title: "AutoEmulate: A Python package for semi-automated emulation"
tags:
    - Python
    - Surrogate modelling
    - Emulation
    - Simulation
    - Machine Learning
    - Gaussian Processes
    - Neural Processes
authors:
    - name: Martin A. Stoffel
      orcid: 0000-0003-4030-3543
      affiliation: 1
      corresponding: true
    - name: Bryan M. Li
      orcid: 0000-0003-3144-4838
      affiliation: 1, 2
    - name: Kalle Westerling
      orcid: 0000-0002-2014-332X
      affiliation: 1
    - name: Sophie Arana
      orcid: 0000-0001-9708-7058
      affiliation: 1
    - name: Max Balmus
      orcid: 0000-0002-6003-0178
      affiliation: 1, 3
    - name: Eric Daub
      orcid: 0000-0002-8499-0720
      affiliation: 1
    - name: Steve Niederer
      orcid: 0000-0002-4612-6982
      affiliation: 1, 3
affiliations:
    - name: The Alan Turing Institute, London, United Kingdom
      index: 1
    - name: University of Edinburgh, Edinburgh, United Kingdom
      index: 2
    - name: Imperial College London, London, United Kingdom
      index: 3
date: 28 November 2024
bibliography: paper.bib
---

# Summary

Simulations are ubiquitous in research and application, but are often too slow and computationally expensive to deeply explore the underlying system. One solution is to create efficient emulators (also surrogate- or meta-models) to approximate simulations, but this requires substantial expertise. Here, we present AutoEmulate, a low-code, AutoML-style python package for emulation. AutoEmulate makes it easy to fit and compare emulators, abstracting away the need for extensive machine learning (ML) experimentation. The package includes a range of emulators, from Gaussian Processes, Support Vector Machines and Gradient Boosting Models to novel, experimental deep learning emulators such as Neural Processes [@garnelo_conditional_2018]. It also implements global sensitivity analysis as a common emulator application, which quantifies the relative contribution of different inputs to the output variance. Through community feedback and collaboration, we aim for AutoEmulate to evolve into an end-to-end tool for most emulation problems.

# Statement of need

To understand complex real-world systems, researchers and engineers often construct computer simulations. These can be computationally expensive and take minutes, hours or even days to run. For tasks like optimisation, sensitivity analysis or uncertainty quantification where thousands or even millions of runs are needed, a solution has long been to approximate simulations with efficient emulators, which can be orders of magnitudes faster [@forrester_recent_2009; @kudela_recent_2022]. Emulation is becoming increasingly widespread, ranging from engineering [@yondo_review_2018], architecture [@westermann_surrogate_2019], biomedical [@strocchi_cell_2023] and climate science [@bounceur_global_2015], to agent-based models [@angione_using_2022].

A typical emulation pipeline involves three steps: 1. Evaluating the simulation at a small, strategically chosen set of inputs using techniques such as Latin Hypercube Sampling [@mckay_comparison_1979] to create a representative dataset, 2. constructing a high-accuracy emulator using that dataset, which involves model selection, hyperparameter optimisation and evaluation and 3. applying the emulator to tasks such as prediction, sensitivity analysis, or optimisation. A key challenge is the emulator construction, which requires machine learning expertise and familiarity with an evolving ecosystem of models and tools - creating a significant barrier for researchers focused on studying the underlying system rather than building emulators.

AutoEmulate automates emulator building, with the goal to eventually streamline the whole emulation pipeline. For people new to ML, AutoEmulate compares, optimises and evaluates a range of models to create an efficient emulator for their simulation in just a few lines of code. For experienced surrogate modellers, AutoEmulate provides a reference set of cutting-edge emulators to quickly benchmark new models against. The package includes classic emulators such as Radial Basis Functions and Gaussian Processes, established ML models like Gradient Boosting and Support Vector Machines, as well as experimental deep learning emulators such as [Conditional Neural Processes](https://yanndubs.github.io/Neural-Process-Family/text/Intro.html) [@garnelo_conditional_2018]. AutoEmulate is built to be extensible. Emulators follow the popular [scikit-learn estimator template](https://scikit-learn.org/1.5/developers/develop.html#rolling-your-own-estimator) and PyTorch [@paszke_pytorch_2019] deep learning models are supported with little overhead using a [skorch](https://skorch.readthedocs.io/en/stable/) [@tietz_skorch_2017] interface.

AutoEmulate fills a gap in the current landscape of surrogate modeling tools as it’s both highly accessible for newcomers while providing cutting-edge emulators for experienced surrogate modelers. In contrast, existing libraries either focus on lower level implementations of specific models, like GPflow [@matthews_gpflow_2017] and GPytorch, or provide multiple emulators and applications but require to manually pre-process data, compare emulators and optimise hyperparameters like SMT in Python [@saves_smt_2024] or [Surrogates.jl](https://docs.sciml.ai/Surrogates/latest/) in Julia.

# Pipeline

The inputs for AutoEmulate are `X` and `y`, where `X` is a 2D array (e.g. numpy-array, Pandas DataFrame) containing simulation parameters in columns and their values in rows, and `y` is an array containing the corresponding simulation outputs. A dataset `X`, `y` is usually obtained by constructing a set of parameters `X` using sampling techniques like Latin Hypercube Sampling [@mckay_comparison_1979] and evaluating the simulation on these inputs to obtain outputs `y`. With `X` and `y`, we can create an emulator with AutoEmulate in just a few lines of code.

```python
from autoemulate.compare import AutoEmulate

ae = AutoEmulate()
ae.setup(X, y)                            # customise pipeline
ae.compare()                              # runs the pipeline
```

Under the hood, AutoEmulate runs a complete ML pipeline. It splits the data into training and test sets, standardises inputs, fits a set of user-specified emulators, compares them using cross-validation and optionally optimises hyperparameters using pre-defined search spaces. All these steps can be customised in `setup()`. After running `compare()`, the cross-validation results can be visualised and summarised.

```python
ae.plot_cv()                              # visualise results
ae.summarise_cv()                         # cv scores for each model
```

: Average cross-validation scores

| Model | Short Name | RMSE | R² |
|-------|------------|------|-----|
| Gaussian Process | gp | 0.1027 | 0.9851 |
| Random Forest | rf | 0.1511 | 0.9677 |
| Gradient Boosting | gb | 0.1566 | 0.9642 |
| Conditional Neural Process | cnp | 0.1915 | 0.9465 |
| Radial Basis Functions | rbf | 0.3518 | 0.7670 |
| Support Vector Machines | svm | 0.4924 | 0.6635 |
| LightGBM | lgbm | 0.6044 | 0.4930 |
| Second Order Polynomial | sop | 0.8378 | 0.0297 |

After comparing cross-validation metrics and plots, an emulator can be selected and evaluated on the held-out test set (defaults to 20% of the data).

```python
emulator = ae.get_model("GaussianProcess") # get fitted emulator
ae.evaluate(emulator)                      # calculate test set scores
ae.plot_eval(emulator, input_index=[0, 1]) # plot predictions
```

![Test set predictions](eval_2.png)

Finally, the emulator can be refitted on the combined training and test set data before applying it. It's now ready to be used as an efficient replacement for the original simulation, being able to generate tens of thousands of new data points in negligible time using `predict()`. Lastly, we implemented global sensitivity analysis, which requires a large number of samples from the emulator to quantify how each simulation parameter and their interactions contribute to the output variance.

```python
emulator = ae.refit(emulator)               # refit using full data
emulator.predict(X)                         # emulate!
ae.sensitivity_analysis(emulator)           # global SA with Sobol indices
```

# Acknowledgements

We thank the Turing Research and Innovation Cluster in Digital Twins for supporting this work. We also thank Keith Worden, Ieva Kazlauskaite, Rosie Williams, Robert Arthern, Peter Yatsyshin, Christopher Burr and James Byrne for discussions and Marina Strocci, Zack Xuereb Conti, Richard Wilkinson for discussions and providing datasets.

# References