import numpy as np
import pandas as pd
import plotnine as p9
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
    output_names = [f"y{i+1}" for i in range(num_outputs)]

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
        A DataFrame with columns: 'output', 'parameter', 'index', 'value', 'confidence'.
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


def plot_sensitivity_analysis(results, type="bar"):
    """
    Plot the sensitivity analysis results.

    Parameters:
    -----------
    results : pd.DataFrame
        The results from sobol_results_to_df.
    """

    if not isinstance(results, pd.DataFrame):
        results = sobol_results_to_df(results)

    if type == "bar":
        # filter S1 and ST
        results = results[results["index"].isin(["S1", "ST"])]

        p = (
            p9.ggplot(results, p9.aes(x="parameter", y="value", fill="index"))
            + p9.geom_bar(stat="identity", position="dodge")
            + p9.facet_wrap("~output")
            + p9.theme_538()
            + p9.scale_fill_manual(values=["#5386E4", "#4C4B63"])
            + p9.labs(y="Sobol Index")
            + p9.geom_errorbar(
                p9.aes(ymin="value-confidence/2", ymax="value+confidence/2"),
                position=p9.position_dodge(width=0.9),
                width=0.25,
            )
            + p9.ggtitle(
                "Sensitivity Analysis: First-Order (S1) and \n Total-Order (ST) Indices and 95% CI"
            )
            + p9.theme(plot_title=p9.element_text(hjust=0.5))  # Center the title
        )
    elif type == "heatmap":
        results = results[results["index"].isin(["S2"])]
        results[["param1", "param2"]] = results["parameter"].str.split("-", expand=True)
        p = (
            p9.ggplot(results, p9.aes("param1", "param2", fill="value"))
            + p9.geom_tile()
            + p9.scale_fill_gradient(low="#33658A", high="#9B1D20", limits=(0, 1))
            + p9.facet_wrap("~output")
            + p9.theme_classic()
        )
    return p
