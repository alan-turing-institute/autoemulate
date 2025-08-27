---
title: "AutoEmulate: A python package. . .  "
tags: 
    - Python
    - Surrogate Modelling
    - Emulation
    - Simulation 
    - Machine Learning 
    - Pytorch
authors:
    - name: Radka 
    orcid: 
    affiliation: 1
    corresponding: true
    - name: Sam Greenbury 
    orcid: 
    affilitation: 1
    equal-contrib: true 
affiliations:
    - name: Alan Turing Institute
    index: 1
date: 22 August 2025
bibliography: paper.bib
---

# Summary

Computational simulations lie at the heart of modern science and engineering, but they are often slow and computationally costly. This poses a significant bottleneck. A common solution is to use emulators, fast and cheap models trained to approximate the simulator, but constructing these requires substantial expertise. AutoEmulate is a low-code Python package for emulation workflows, making it easy to replace simulations with fast, accurate emulators. In version 1.0, AutoEmulate has been fully refactored to use PyTorch as a backend, enabling GPU acceleration, automatic differentiation, and seamless integration with the broader PyTorch ecosystem. The toolkit has also been extended with new uncertainty aware emulator models and support for common emulation tasks like model calibration and active learning, enabling end-to-end emulation workflows.

# Statement of need

<!-- AutoEmulate either has the task or makes it easy to use other tools for it... mention UQ -->

<!-- as simulation models become more complex, so do the tasks that we require of them such as sensitivity analysis or model calibration -->

To understand complex real-world systems, researchers and engineers often construct computer simulations. These can be computationally expensive and take minutes, hours or even days to run. A solution to this bottleneck is to approximate simulations with emulators, which can be orders of magnitudes faster. Emulators are key in enabling complex downstream tasks that require generating predictions for a high number if inputs. Thse include sensitivity analysis, quantifying how much each input parameter affects the output, and model calibration, determining which input values are most likely to have generated real-world observations.

Emulation requires significant expertise in methods like machine learning and uncertainty quantification as well as familiarity with a broad and evolving ecosystem of tools for model training and the accompanying downstream tasks. This creates a barrier to entry for domain researchers whose focus is on the underlying scientific problem. 

AutoEmulate lowers the barrier to entry by automating the entire emulator construction process (training, evaluation, model selection, and hyperparameter tuning). This makes emulation accessible to non-specialists while also offering a reference set of cutting-edge emulators, from classical approaches (e.g. Gaussian Processes) to modern deep learning methods, for benchmarking by experienced users. 

Additionally, AutoEmulate provides simple interfaces for common emulation tasks, like sensitivity analysis and model calibration. Having everything in the same package means that users can easily build complex workflows, such as using sensitivity analysis to reduce the parameter space to a small number of key variables before calibrating this subset of parameters against real world data. 

AutoEmulate also supports direct integration of custom simulators. This enables advanced use cases such as active learning, in which the tool adaptively selects informative simulations to improve emulator performance at minimal computational cost. 

AutoEmulate was originally built on scikit-learn, which is well suited for traditional machine learning but less flexible for complex workflows. Version 1.0 introduces a PyTorch backend that provides GPU acceleration for faster training and inference and automatic differentiation via PyTorch’s autograd system. It also makes AutoEmulate easy to integrate with other PyTorch-based tools. For example, the PyTorch refactor enables fast Bayesian calibration using gradient-based inference methods such as Hamiltonian Monte Carlo exposed through Pyro.

AutoEmulate fills a gap in the current landscape of emulation tools as it is both accessible to newcomers and powerful enough for advanced users. It also uniquely combines emulator training with support for a wide range of downstream tasks such as sensitivity analysis, model calibration adn active learning.

# References