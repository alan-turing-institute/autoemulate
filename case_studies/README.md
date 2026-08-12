# Case Studies

This directory contains example case studies demonstrating the use of the AutoEmulate framework for various scientific and engineering problems.

## Overview

Each subdirectory provides data, scripts, and documentation relevant to a specific application area.

- **patient_calibration/**: Demonstrates cardiovascular model calibration using the Naghavi Model from [ModularCirc](https://github.com/alan-turing-institute/ModularCirc). Shows end-to-end workflow including sensitivity analysis, history matching, and Bayesian calibration for blood pressure dynamics in the left ventricle.
- **model_comparison/**: A toy problem case study demonstrating how to perform surrogate calibration and compute Bayesian Evidence. It uses Gaussian Process emulators to quantitatively compare SIR and SEIR epidemic models and introduces the concept of the Bayes Factor to guard against overfitting.

## External examples

- **Hydrogen transport**: The [FESTIM](https://festim-workshop.readthedocs.io/en/latest/intro.html) simulator documentation includes examples of using AutoEmulate for [surrogate modelling](https://festim-workshop.readthedocs.io/en/latest/content/applications/surrogate.html) and [active learning](https://festim-workshop.readthedocs.io/en/latest/content/applications/active_learning.html).

