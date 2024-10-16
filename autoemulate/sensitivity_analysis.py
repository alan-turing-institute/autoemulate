import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample


def sensitivity_analysis(model, problem, N=1000, as_df=False):
    Si = sobol_analysis(model, problem, N)

    if as_df:
        return sobol_results_to_df(Si)
    else:
        return Si


def sobol_analysis(model, problem, N=1024):
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
    # samples
    param_values = sample(problem, N)

    # evaluate
    Y = model.predict(param_values)

    # multiple outputs
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    num_outputs = Y.shape[1]
    output_names = [f"y{i}" for i in range(num_outputs)]

    results = {}
    for i in range(num_outputs):
        Si = analyze(problem, Y[:, i])
        results[output_names[i]] = Si

    return results


def sobol_results_to_df(results):
    """
    Convert Sobol results to a pandas DataFrame.

    Parameters:
    -----------
    results : dict
        The Sobol indices returned by sobol_analysis.

    Returns:
    --------
    pd.DataFrame
    """
    rows = []
    for output, indices in results.items():
        for index_type in ["S1", "ST", "S2"]:
            values = indices.get(index_type)
            conf_values = indices.get(f"{index_type}_conf")
            if values is None or conf_values is None:
                continue

            if index_type in ["S1", "ST"]:
                rows.extend(
                    {
                        "output": output,
                        "parameter": f"X{i+1}",
                        "index": index_type,
                        "value": value,
                        "confidence": conf,
                    }
                    for i, (value, conf) in enumerate(zip(values, conf_values))
                )

            elif index_type == "S2":
                n = values.shape[0]
                rows.extend(
                    {
                        "output": output,
                        "parameter": f"X{i+1}-X{j+1}",
                        "index": index_type,
                        "value": values[i, j],
                        "confidence": conf_values[i, j],
                    }
                    for i in range(n)
                    for j in range(i + 1, n)
                    if not np.isnan(values[i, j])
                )

    return pd.DataFrame(rows)


# def plot_sensitivity_indices(Si, problem):
#     """
#     Plot the Sobol sensitivity indices.

#     Parameters:
#     -----------
#     Si : dict
#         The Sobol indices returned by perform_sobol_analysis.
#     problem : dict
#         The problem definition used in the analysis.
#     """
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(10, 6))

#     indices = Si["S1"]
#     names = problem["names"]

#     ax.bar(names, indices)
#     ax.set_ylabel("First-order Sobol index")
#     ax.set_title("Sensitivity Analysis Results")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
