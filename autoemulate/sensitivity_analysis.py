import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli


def perform_sobol_analysis(model, problem, N=1000):
    """
    Perform Sobol sensitivity analysis on a fitted emulator.

    Parameters:
    -----------
    model : fitted emulator model
        The emulator model to analyze.
    problem : dict
        The problem definition, including 'num_vars', 'names', and 'bounds'.
    N : int, optional
        The number of samples to generate (default is 1000).

    Returns:
    --------
    dict
        A dictionary containing the Sobol indices.
    """
    # Generate samples
    param_values = saltelli.sample(problem, N)

    # Evaluate model
    Y = model.predict(param_values)

    # Perform analysis
    Si = sobol.analyze(problem, Y)

    return Si


def plot_sensitivity_indices(Si, problem):
    """
    Plot the Sobol sensitivity indices.

    Parameters:
    -----------
    Si : dict
        The Sobol indices returned by perform_sobol_analysis.
    problem : dict
        The problem definition used in the analysis.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    indices = Si["S1"]
    names = problem["names"]

    ax.bar(names, indices)
    ax.set_ylabel("First-order Sobol index")
    ax.set_title("Sensitivity Analysis Results")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
