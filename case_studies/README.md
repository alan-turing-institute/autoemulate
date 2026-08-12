# Case Studies

This directory contains example case studies demonstrating the use of the AutoEmulate framework for various scientific and engineering problems. Each subdirectory provides data, scripts, and documentation relevant to a specific application area.

## Overview
- **patient_calibration/**: Demonstrates cardiovascular model calibration using the Naghavi Model from ModularCirc. Shows end-to-end workflow including sensitivity analysis, history matching, and Bayesian calibration for blood pressure dynamics in the left ventricle.

Additional case studies may be added in the future to showcase more applications and methodologies.

For details on each case study, refer to the README or documentation within the respective subdirectory.

### 📊 Bayesian Model Comparison
**Location:** [`model_comparison/`](model_comparison/)
**Description:** A toy problem case study demonstrating how to perform surrogate calibration and compute Bayesian Evidence. It uses Gaussian Process emulators to quantitatively compare SIR and SEIR epidemic models and introduces the concept of the Bayes Factor to guard against overfitting.
