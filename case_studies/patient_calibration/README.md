# Patient-Specific Calibration in Cardiovascular Modeling

This case study demonstrates the integration of a cardiovascular simulator, the Naghavi Model from the [ModularCirc](https://github.com/alan-turing-institute/ModularCirc) package, into an end-to-end `AutoEmulate` workflow. The goal is to showcase how `AutoEmulate` can be used to calibrate and validate physiological models, specifically focusing on blood pressure dynamics measured in the left ventricle.

The Naghavi Model is a modular cardiovascular simulator that provides a detailed representation of blood pressure regulation mechanisms. By using `AutoEmulate`, we can perform Bayesian calibration of the model parameters, quantify uncertainties, and generate posterior predictive distributions for the simulated outputs.

## Workflow Overview

The workflow includes the following steps:
1. **Model Integration**: The Naghavi Model is integrated into the `AutoEmulate` framework.
2. **Sensitivity Analysis**: Sensitivity analysis is performed to identify the most influential parameters affecting the model outputs.
3. **History Matching**: History matching is conducted to rule out implausible regions of the parameter space given clinical patient data.
4. **Bayesian Calibration**: Bayesian calibration is performed using Markov Chain Monte Carlo (MCMC) methods to estimate the model parameters.
5. **Uncertainty Quantification**: The posterior distributions of the parameters are used to quantify uncertainties.

This case study provides a practical example of how `AutoEmulate` can be applied to complex physiological models, enabling robust parameter estimation and model validation.

## How to Run the Case Study

Follow the steps below to run the case study:

1. Clone the repository:

    ```bash
    git clone https://github.com/alan-turing-institute/autoemulate.git
    cd case_studies/patient_calibration
    ```

2. Install dependencies:

    ```bash
    pip install -e .
    ```

3. Run the Jupyter Notebook to explore the case study interactively:

    ```bash
    jupyter lab
    ```

4. Alternatively, you can run the equivalent Python script for this case study:

    ```bash
    python blood_pressure_case_study.py
    ```

## Files in This Case Study

- **`patient_calibration_case_study.ipynb`**: A Jupyter Notebook that provides an interactive walkthrough of the case study.
- **`patient_calibration_workflow.py`**: A Python script containing the same workflow as the notebook, for users who prefer running the case study programmatically.
- **`README.md`**: This file, which provides an overview of the case study and instructions for running it.

## Dependencies

This case study requires the following dependencies:
- Python 3.8+
- `AutoEmulate`
- `ModularCirc`
- `JupyterLab`

Ensure all dependencies are installed by running the command in Step 2 above.

## Additional Information

For more details about the `AutoEmulate` framework and its capabilities, visit the [official repository](https://github.com/alan-turing-institute/autoemulate).

For more information about the Naghavi Model and the `ModularCirc` package, visit the [ModularCirc repository](https://github.com/alan-turing-institute/ModularCirc).