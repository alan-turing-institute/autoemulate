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
      orcid: 0000-0002-2014-332X
      affiliation: 1
    - name: Max Balmus
      orcid: 0000-0002-6003-0178
      affiliation: 1, 3
    - name: Eric Daub
      orcid: 
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

Simulations are ubiquitous in research and application, but are often too slow and computationally expensive to deeply explore the underlying system. One solution is to create efficient emulators (also surrogate- or meta-models) to approximate simulations, but this requires substantial expertise. Here, we present AutoEmulate, a low-code, AutoML-style python package for emulation. AutoEmulate makes it easy to fit and compare emulators, abstracting away the need for extensive machine learning (ML) experimentation. The package includes a range of emulators, from Gaussian Processes, Support Vector Machines and Gradient Boosting Models to novel, experimental deep learning emulators such as Neural Processes [@garnelo_conditional_2018]. AutoEmulate also implements global sensitivity analysis as a common emulator application, and we aim to add other applications in the future. Finally, AutoEmulate is designed to be easy to contribute to by being modular, integrated with the scikit-learn ecosystem [@pedregosa_scikit-learn_2011], and well documented. We aim to iterate based on user feedback to make AutoEmulate a tool for end-to-end emulation across fields.

# Statement of need

To understand complex real-world systems, researchers and engineers often construct computer simulations. These can be computationally expensive and take minutes, hours or even days to run. For tasks like optimisation, sensitivity analysis or uncertainty quantification where thousands or even millions of runs are needed, a solution has long been to approximate simulations with efficient emulators, which can be orders of magnitudes faster [@forrester_recent_2009, @kudela_recent_2022]. Emulation is becoming increasingly widespread, ranging from engineering [@yondo_review_2018], biomedical [@strocci_cell_2023] and climate science [@bounceur_global_2015] to agent-based models [@angione_using_2022]. A typical emulation pipeline involves three steps: 1. Sample selection, which involves evaluating the simulation at a small, strategically chosen set of inputs to create a representative dataset., 2. constructing a high-accuracy emulator using that dataset, which involves model selection, hyperparameter optimisation and evaluation and 3. applying the emulator to tasks such as prediction, sensitivity analysis, or optimisation. Building an emulator in particular is a key challenge for non-experts, as it can involve substantial machine learning experimentation, all within an ever increasing ecosystem of models and packages. This puts a substantial burden on practitioners whose main focus is to explore the underlying system, not building the emulator.

AutoEmulate automates emulator building, with the goal to eventually streamline the whole emulation pipeline. For people new to ML, AutoEmulate compares, optimises and evaluates a range of models to create an efficient emulator for their simulation in just a few lines of code. For experienced surrogate modellers, it provides a reference set of cutting-edge emulators to quickly benchmark new models against. The package includes classic emulators such as Radial Basis Functions and Gaussian Processes, established ML models like Gradient Boosting and Support Vector Machines, as well as experimental deep learning emulators such as [Conditional Neural Processes](https://yanndubs.github.io/Neural-Process-Family/text/Intro.html) [@garnelo_conditional_2018]. AutoEmulate is built to be extensible. Emulators follow the well established [scikit-learn estimator template](https://scikit-learn.org/1.5/developers/develop.html#rolling-your-own-estimator) and deep learning models written in PyTorch [@paszke_pytorch_2019] are supported with little overhead through a skorch [@tietz_skorch_2017] interface. AutoEmulate fills a gap in the current landscape of surrogate modeling tools as it’s both highly accessible for newcomers while providing cutting edge-methods for experienced surrogate modelers. In contrast, existing libraries either focus on lower level implementations of specific models, like GPflow [@matthews_gpflow_2017] and GPytorch [@gardner_gpytorch_2018], provide multiple emulators but require to manually pre-process data, compare emulators and optimise parameters like SMT in Python [@bouhlel_smt_2019] or Surrogates.jl in Julia [@gorissen_surrogates.jl_2010].

# Pipeline

The minimal input for AutoEmulate are X, y, where X is a 2D array (e.g. numpy-array, Pandas DataFrame) containing one simulation parameter per column and their values in rows, and y is an array containing the corresponding simulation outputs, where y can be either single or multi-output. After a dataset X, y has been constructed by evaluating the original simulation, we can create an emulator with AutoEmulate in just three lines of code:

```python
from autoemulate.compare import AutoEmulate

# creating an emulator
ae = AutoEmulate()
ae.setup(X, y)                    # allows to customise pipeline 
emulator = ae.compare()           # compares emulators & returns 
```

Under the hood, AutoEmulate runs a complete ML pipeline. It splits the data into training and test sets, standardises inputs, fits a set of user-specified emulators, compares them using cross-validation and optionally optimises hyperparameters using pre-defined search spaces. It then returns the emulator with the highest average cross-validation R^2 score. The results can then easily be summarised and visualised.

```python
# cross-validation results
ae.summarise_cv()                 # cv metrics for each model
ae.plot_cv()                      # visualise best cv fold per model
```

After choosing an emulator based on its cross-validation performance, it can be evaluated on the test set, which by default is 20% of the original dataset. If the test-set performance is acceptable, the emulator can be refitted on the combined training and test data before applying it.

```python
# evaluating the emulator
ae.evaluate(emulator)             # test set scores
emulator = ae.refit(emulator)     # refit using full data
```

The emulator can now be used as an efficient replacement for the original simulation by generating tens of thousands of new data points in milliseconds using predict(). We’ve also implemented global sensitivity analysis, a common use-case for emulators, which decomposes the variance in the output(s) into the contributions of the various simulation parameters and their interactions.

```python
# application
emulator.predict(X)               # generate new samples
ae.sensitivity_analysis(emulator) # global SA with Sobol indices
```

# Acknowledgements

We thank the Turing Research and Innovation Cluster in Digital Twins for supporting this work. We also thank Keith Worden, Ieva Kazlauskaite, Rosie Williams, Robert Arthern, Peter Yatsyshin, Christopher Burr, James Byrne for discussions and Marina Strocci, Zack Xuereb Conti, Richard Wilkinson for also providing datasets.

# References