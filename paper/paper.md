---
title: "AutoEmulate v1.0: a PyTorch tool for end-to-end emulation workflows"
tags: 
    - Python
    - Surrogate Modelling
    - Emulation
    - Simulation 
    - Machine Learning 
    - Pytorch
    - Gaussian Processes
    - Sensitivity analysis
    - Model calibration
    - Uncertainty quantification
    - Active learning
authors:
    - name: Radka Jersakova
      orcid: 0000-0001-6846-7158
      affiliation: 1
      corresponding: true
    - name: Sam F. Greenbury
      orcid: 0000-0003-4452-2006
      affiliation: 1
      equal-contrib: true
    - name: Ed Chalstrey
      orcid: 0000-0003-2560-1294
      affiliation: 1
    - name: Edwin Brown
      orcid: 0009-0004-1124-469X
      affiliation: 2
    - name: Marjan Famili
      orcid: 
      affiliation: 1
    - name: Chris Sprague
      orcid: 
      affiliation: 1
    - name: Paolo Conti
      orcid: 
      affiliation: 1
    - name: Camila Rangel Smith
      orcid: 
      affiliation: 1
    - name: Martin A. Stoffel
      orcid: 0000-0003-4030-3543
      affiliation: 1
    - name: Bryan M. Li
      orcid: 0000-0003-3144-4838
      affiliation: 1, 5
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
    - name: Andrew B. Duncan
      orcid: 0000-0001-5762-164X
      affiliation: 3
    - name: Jason McEwen
      orcid:
      affiliation: 1, 4
affiliations:
    - name: The Alan Turing Institute, London, United Kingdom
      index: 1
    - name: University of Sheffield, Sheffield, United Kingdom
      index: 2
    - name: Imperial College London, London, United Kingdom
      index: 3
    - name: University College London, London, United Kingdom
      index: 4
    - name: University of Edinburgh, Edinburgh, United Kingdom
      index: 5
date: 26 August 2025
bibliography: paper.bib
---

# Summary

Computational simulations lie at the heart of modern science and engineering, but they are often slow and computationally costly. This poses a significant bottleneck. A common solution is to use emulators, fast and cheap models trained to approximate the simulator, but constructing these requires substantial expertise. AutoEmulate is a low-code Python package for emulation workflows, making it easy to replace simulations with fast, accurate emulators. In version 1.0, AutoEmulate has been fully refactored to use PyTorch as a backend, enabling GPU acceleration, automatic differentiation, and seamless integration with the broader PyTorch ecosystem. The toolkit has also been extended with new uncertainty aware emulator models and support for common emulation tasks like model calibration and active learning, enabling end-to-end emulation workflows.

# Statement of need

To understand complex real-world systems, researchers and engineers often construct computer simulations. These can be computationally expensive and take minutes, hours or even days to run. A solution to this bottleneck is to approximate simulations with emulators, which can be orders of magnitudes faster. Emulators are key in enabling complex downstream tasks that require generating predictions for a high number if inputs. Thse include sensitivity analysis, quantifying how much each input parameter affects the output, and model calibration, determining which input values are most likely to have generated real-world observations.

Emulation requires significant expertise in methods like machine learning and uncertainty quantification as well as familiarity with a broad and evolving ecosystem of tools for model training and the accompanying downstream tasks. This creates a barrier to entry for domain researchers whose focus is on the underlying scientific problem. 

AutoEmulate lowers the barrier to entry by automating the entire emulator construction process (training, evaluation, model selection, and hyperparameter tuning). This makes emulation accessible to non-specialists while also offering a reference set of cutting-edge emulators, from classical approaches (e.g. Gaussian Processes) to modern deep learning methods, for benchmarking by experienced users. 

Additionally, AutoEmulate provides simple interfaces for common emulation tasks, like sensitivity analysis and model calibration. Having everything in the same package means that users can easily build complex workflows, such as using sensitivity analysis to reduce the parameter space to a small number of key variables before calibrating this subset of parameters against real world data. 

AutoEmulate also supports direct integration of custom simulators. This enables advanced use cases such as active learning, in which the tool adaptively selects informative simulations to improve emulator performance at minimal computational cost. 

AutoEmulate was originally built on scikit-learn, which is well suited for traditional machine learning but less flexible for complex workflows. Version 1.0 introduces a PyTorch backend that provides GPU acceleration for faster training and inference and automatic differentiation via PyTorch’s autograd system. It also makes AutoEmulate easy to integrate with other PyTorch-based tools. For example, the PyTorch refactor enables fast Bayesian calibration using gradient-based inference methods such as Hamiltonian Monte Carlo exposed through Pyro.

AutoEmulate fills a gap in the current landscape of emulation tools as it is both accessible to newcomers and powerful enough for advanced users. It also uniquely combines emulator training with support for a wide range of downstream tasks such as sensitivity analysis, model calibration adn active learning.

# References