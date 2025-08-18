import logging

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample
from SALib.util import ResultDict

from autoemulate.core.plotting import display_figure
from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.data.utils import ConversionMixin
from autoemulate.emulators.base import Emulator

logger = logging.getLogger("autoemulate")


class SensitivityAnalysis(ConversionMixin):
    """Global sensitivity analysis."""

    def __init__(
        self,
        emulator: Emulator,
        x: TensorLike | None = None,
        problem: dict | None = None,
    ):
        """
        Initialize the SensitivityAnalysis.

        Parameters
        ----------
        emulator: Emulator
            Fitted emulator.
        x: InputLike | None
            Simulator input parameter values.
        problem: dict | None
            The problem definition dictionary. If None, the problem is generated
            from x using minimum and maximum values of the features as bounds.
            The dictionary should contain:
                - 'num_vars': Number of input variables (int)
                - 'names': List of variable names (list of str)
                - 'bounds': List of [min, max] bounds for each variable (list of lists)
                - 'output_names': Optional list of output names (list of str)

            Example::
                problem = {
                    "num_vars": 2,
                    "names": ["x1", "x2"],
                    "bounds": [[0, 1], [0, 1]],
                    "output_names": ["y1", "y2"],  # optional
                }
        """
        if problem is not None:
            problem = self._check_problem(problem)
        elif x is not None:
            problem = self._generate_problem(x)
        else:
            msg = "Either problem or x must be provided."
            raise ValueError(msg)

        self.emulator = emulator
        self.problem = problem

    @staticmethod
    def _check_problem(problem: dict) -> dict:
        """Check that the problem definition is valid."""
        if not isinstance(problem, dict):
            msg = "problem must be a dictionary."
            raise ValueError(msg)

        if "num_vars" not in problem:
            msg = "problem must contain 'num_vars'."
            raise ValueError(msg)
        if "names" not in problem:
            msg = "problem must contain 'names'."
            raise ValueError(msg)
        if "bounds" not in problem:
            msg = "problem must contain 'bounds'."
            raise ValueError(msg)

        if len(problem["names"]) != problem["num_vars"]:
            msg = "Length of 'names' must match 'num_vars'."
            raise ValueError(msg)
        if len(problem["bounds"]) != problem["num_vars"]:
            msg = "Length of 'bounds' must match 'num_vars'."
            raise ValueError(msg)

        return problem

    @staticmethod
    def _generate_problem(x: TensorLike) -> dict:
        """
        Generate a problem definition from a design matrix.

        Parameters
        ----------
        x: TensorLike
            Simulator input parameter values [n_samples, n_parameters].

        Returns
        -------
        dict
            Dictionary with keys ["num_vars", "names", "bounds"]
        """
        if x.ndim == 1:
            msg = "x must be a 2D array."
            raise ValueError(msg)

        return {
            "num_vars": x.shape[1],
            "names": [f"x{i + 1}" for i in range(x.shape[1])],
            "bounds": [
                [x[:, i].min().item(), x[:, i].max().item()] for i in range(x.shape[1])
            ],
        }

    def _sample(self, method: str, N: int) -> NumpyLike:
        if method == "sobol":
            # Saltelli sampling
            return sobol_sample(self.problem, N)
        if method == "morris":
            # vanilla Morris (1991) sampling
            return morris_sample(self.problem, N)
        msg = f"Unknown method: {method}. Must be 'sobol' or 'morris'."
        raise ValueError(msg)

    def _predict(self, param_samples: NumpyLike) -> NumpyLike:
        """
        Make predictions with emulator for parameter samples.

        Parameters
        ----------
        param_samples: NumpyLike
            Array of parameter samples.

        Returns
        -------
        NumpyLike
            Array of emulator predictions.
        """
        param_tensor = self._convert_to_tensors(param_samples)
        assert isinstance(param_tensor, TensorLike)
        y_pred = self.emulator.predict_mean(param_tensor)
        y_pred_np, _ = self._convert_to_numpy(y_pred)
        return y_pred_np

    def _get_output_names(self, num_outputs: int) -> list[str]:
        """Get output names from the problem definition or generate default names."""
        # check if output_names is given
        if "output_names" not in self.problem:
            output_names = [f"y{i + 1}" for i in range(num_outputs)]
        elif isinstance(self.problem["output_names"], list):
            output_names = self.problem["output_names"]
        else:
            msg = "'output_names' must be a list of strings."
            raise ValueError(msg)

        return output_names

    def run(
        self,
        method: str = "sobol",
        n_samples: int = 1024,
        conf_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Perform global sensitivity analysis on a fitted emulator.

        Parameters
        ----------
        method: str
            The sensitivity analysis method to perform, one of ["sobol", "morris"].
        n_samples: int
            Number of samples to generate for the analysis. Higher values give more
            accurate results but increase computation time. Default is 1024.
        conf_level: float
            Confidence level (between 0 and 1) for calculating confidence intervals
            of the Sobol sensitivity indices. Default is 0.95 (95% confidence).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
                - 'parameter': Input parameter name
                - 'output': Output variable name
            If using Sobol, the columns include:
                - 'index': S1, S2 or ST (first, second, and total order sensitivity)
                - 'confidence': confidence intervals for each index
            If using Morris, the columns include:
                - 'mu': mean of the distribution of elementary effects
                - 'mu_star': mean of the distribution of absolute value
                - 'sigma': standard deviation of the distribution, used as indication of
                    interactions between parameters
                - 'mu_star_conf: boostrapped confidence interval

        Notes
        -----
        The Sobol method requires N * (2D + 2) model evaluations, where D is the number
        of input parameters. For example, with N=1024 and 5 parameters, this requires
        12,288 evaluations. The Morris method requires far fewer computations.
        """
        logger.debug(
            "Running sensitivity analysis with method=%s, n_samples=%d, conf_level=%s",
            method,
            n_samples,
            conf_level,
        )

        if method not in ["sobol", "morris"]:
            msg = f"Unknown method: {method}. Must be 'sobol' or 'morris'."
            raise ValueError(msg)

        param_samples = self._sample(method, n_samples)
        y = self._predict(param_samples)
        output_names = self._get_output_names(y.shape[1])

        results = {}
        for i, name in enumerate(output_names):
            if method == "sobol":
                Si = sobol_analyze(self.problem, y[:, i], conf_level=conf_level)
            elif method == "morris":
                Si = morris_analyze(
                    self.problem, param_samples, y[:, i], conf_level=conf_level
                )
            else:
                msg = f"Unknown method: {method}. Must be 'sobol' or 'morris'."
                raise ValueError(msg)
            results[name] = Si

        if method == "sobol":
            return _sobol_results_to_df(results)
        return _morris_results_to_df(results, self.problem)

    @staticmethod
    def plot_sobol(
        results: pd.DataFrame,
        index: str = "S1",
        ncols: int | None = None,
        figsize: tuple | None = None,
        fname: str | None = None,
    ):
        """
        Plot Sobol sensitivity analysis results. Optionally save to file.

        Parameters
        ----------
        results: pd.DataFrame
            The results from sobol_results_to_df.
        index: str
            The type of sensitivity index to plot.
            - "S1": first-order indices
            - "S2": second-order/interaction indices
            - "ST": total-order indices
            Defaults to "S1".
        ncols: int | None
            The number of columns in the plot. Defaults to 3 if there are 3 or
            more outputs, otherwise the number of outputs. Defaults to None.
        figsize: tuple | None
            Figure size as (width, height) in inches. If None, set automatically.
            Defaults to None.
        fname: str | None
            If provided, saves the figure to this file path. Defaults to None.
        """
        fig = _plot_sobol_analysis(results, index, ncols, figsize)
        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

    @staticmethod
    def plot_morris(
        results: pd.DataFrame,
        param_groups: dict[str, list[str]] | None = None,
        ncols: int | None = None,
        figsize: tuple | None = None,
        fname: str | None = None,
    ):
        """
        Plot Morris analysis results. Optionally save to file.

        Parameters
        ----------
        results: pd.DataFrame
            The results from sobol_results_to_df.
        param_groups: dic[str, list[str]] | None
            Optional parameter groupings used to give all the same plot color
            of the form ({<group name> : [param1, ...], }).
        ncols: int, optional
            The number of columns in the plot. Defaults to 3 if there are 3 or
            more outputs, otherwise the number of outputs.
        figsize: tuple, optional
            Figure size as (width, height) in inches.If None, set calculated.
        fname: str | None
            If provided, saves the figure to this file path. Defaults to None.
        """
        fig = _plot_morris_analysis(results, param_groups, ncols, figsize)
        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

    @staticmethod
    def top_n_sobol_params(
        sa_results_df: pd.DataFrame, top_n: int, sa_index: str = "ST"
    ) -> list:
        """
        Return `top_n` most important parameters given Sobol sensitivity analysis.

        In case of multiple outputs, averages over them to rank the parameters.

        Parameters
        ----------
        sa_results_df: pd.DataFrame
            Dataframe results by `SensitivityAnalysis().run()`
        top_n: int
            Number of parameters to return.
        sa_index: str
            Which sensitivity index to rank the parameters by. One of ["S1", "S2", "ST].

        Returns
        -------
        list[str]
            List of `top_n` parameter names.
        """
        if not all(
            col in sa_results_df.columns for col in ["index", "parameter", "value"]
        ):
            msg = (
                "sa_results_df is missing required columns: 'index', 'parameter',"
                "or 'value'"
            )
            raise ValueError(msg)

        st_results = sa_results_df[sa_results_df["index"] == sa_index]

        return (
            st_results.groupby("parameter")["value"]  # pyright: ignore[reportCallIssue]
            # each parameter is evalued against each output
            # to rank parameters, average over how sensitive all outputs are to it
            .mean()
            .nlargest(top_n)
            .index.tolist()
        )


def _sobol_results_to_df(results: dict[str, ResultDict]) -> pd.DataFrame:
    """
    Convert Sobol results to a (long-format) pandas DataFrame.

    Parameters
    ----------
    results: dict
        The Sobol indices returned by sobol_analysis.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: 'output', 'parameter', 'index', 'value', 'confidence'.
    """
    rename_dict = {
        "variable": "index",
        "S1": "value",
        "S1_conf": "confidence",
        "ST": "value",
        "ST_conf": "confidence",
        "S2": "value",
        "S2_conf": "confidence",
    }
    rows = []
    for output, result in results.items():
        # https://salib.readthedocs.io/en/latest/api/SALib.analyze.html#SALib.analyze.sobol.to_df
        # from SALib docs: `to_df()` returns indices in order of [total, first, second]
        st, s1, s2 = result.to_df()
        s1 = (
            s1.reset_index()
            .rename(columns={"index": "parameter"})
            .rename(columns=rename_dict)
        )
        s1["index"] = "S1"
        st = (
            st.reset_index()
            .rename(columns={"index": "parameter"})
            .rename(columns=rename_dict)
        )
        st["index"] = "ST"
        s2 = (
            s2.reset_index()
            .rename(columns={"index": "parameter"})
            .rename(columns=rename_dict)
        )
        s2["index"] = "S2"

        df = pd.concat([s1, st, s2])
        df["output"] = output
        rows.append(df[["output", "parameter", "index", "value", "confidence"]])

    return pd.concat(rows)


def _validate_input(results: pd.DataFrame, index: str):
    if not isinstance(results, pd.DataFrame):
        results = _sobol_results_to_df(results)
        # we only want to plot one index type at a time
    valid_indices = ["S1", "S2", "ST"]
    if index not in valid_indices:
        raise ValueError(
            f"Invalid index type: {index}. Must be one of {valid_indices}."
        )
    return results[results["index"].isin([index])]


def _calculate_layout(n_outputs: int, ncols: int | None = None):
    """Calculate plot layout (n rows, n cols)."""
    if ncols is None:
        ncols = 3 if n_outputs >= 3 else n_outputs
    nrows = int(np.ceil(n_outputs / ncols))
    return nrows, ncols


def _create_bar_plot(ax: Axes, output_data: pd.DataFrame, output_name: str):
    """Create a bar plot for a single output."""
    bar_color = "#4C4B63"
    x_pos = np.arange(len(output_data))
    conf = output_data["confidence"].values
    assert isinstance(conf, NumpyLike)
    yerr = conf / 2

    _ = ax.bar(
        x_pos,
        output_data["value"],
        color=bar_color,
        yerr=yerr,
        capsize=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(output_data["parameter"], rotation=45, ha="right")
    ax.set_ylabel("Sobol Index")
    ax.set_title(f"Output: {output_name}")


def _plot_sobol_analysis(
    results: pd.DataFrame,
    index: str = "S1",
    ncols: int | None = None,
    figsize: tuple | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot the sobol sensitivity analysis results.

    Parameters
    ----------
    results: pd.DataFrame
        The results from sobol_results_to_df.
    index: str, default "S1"
        The type of sensitivity index to plot.
        - "S1": first-order indices
        - "S2": second-order/interaction indices
        - "ST": total-order indices
    ncols: int | None
        The number of columns in the plot. If None, sets ncols to 3 if there are 3 or
        more outputs, otherwise the number of outputs. Defaults to None.
    figsize: tuple | None
        Figure size as (width, height) in inches.If None, automatically calculated.
        Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure containing the Sobol plots.
    """
    with plt.style.context("fast"):
        # prepare data
        results = _validate_input(results, index)  # pyright: ignore[reportAssignmentType]
        unique_outputs = results["output"].unique()
        n_outputs = len(unique_outputs)

        # layout
        nrows, ncols = _calculate_layout(n_outputs, ncols)
        figsize = figsize or (4.5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        elif n_outputs == 1:
            axes = [axes]

        for ax, output in zip(axes, unique_outputs, strict=False):
            output_data = results[results["output"] == output]
            assert isinstance(output_data, pd.DataFrame)
            _create_bar_plot(ax, output_data, output)

        # remove any empty subplots
        for idx in range(len(unique_outputs), len(axes)):
            fig.delaxes(axes[idx])

        index_names = {
            "S1": "First-Order",
            "S2": "Second-order/Interaction",
            "ST": "Total-Order",
        }

        # title
        fig.suptitle(
            f"{index_names[index]} indices and 95% CI",
            fontsize=14,
        )

        plt.tight_layout()

    return fig


def _morris_results_to_df(
    results: dict[str, ResultDict], problem: dict
) -> pd.DataFrame:
    """
    Convert Morris results to a (long-format) pandas DataFrame.

    Parameters
    ----------
    results: dict
        The Morris indices returned by morris_analysis.
    problem: dict
        The problem definition, including 'names'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: 'output', 'parameter', 'mu', 'mu_star', 'sigma',
        'mu_star_conf'.
    """
    rows = []
    parameter_names = problem["names"]

    for output, result in results.items():
        df_data = {
            "output": [output] * len(parameter_names),
            "parameter": parameter_names,
            "mu": result["mu"],
            "mu_star": result["mu_star"],
            "sigma": result["sigma"],
            "mu_star_conf": result["mu_star_conf"],
        }

        df = pd.DataFrame(df_data)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def _plot_morris_analysis(
    results: pd.DataFrame,
    param_groups: dict[str, list[str]] | None = None,
    ncols: int | None = None,
    figsize: tuple | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot the Morris sensitivity analysis results.

    Parameters
    ----------
    results: pd.DataFrame
        The results from morris_results_to_df.
    param_groups: dict, optional
        Dictionary mapping parameter names to groups for coloring.
    ncols: int, optional
        The number of columns in the plot. Defaults to 3 if there are 3 or more outputs,
        otherwise the number of outputs.
    figsize: tuple, optional
        Figure size as (width, height) in inches. If None, automatically calculated.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure containing the Morris plots.
    """
    with plt.style.context("fast"):
        unique_outputs = results["output"].unique()
        n_outputs = len(unique_outputs)

        # layout - add space for legend
        nrows, ncols = _calculate_layout(n_outputs, ncols)
        figsize = figsize or (4.5 * ncols + 2, 4 * nrows)  # Extra width for legend

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        elif n_outputs == 1:
            axes = [axes]

        # Create color mappings once for all plots
        colors = [
            "#4C4B63",
            "#E63946",
            "#F77F00",
            "#FCBF49",
            "#06D6A0",
            "#118AB2",
            "#073B4C",
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
        ]

        if param_groups is None:
            # Different color for each parameter
            all_params = results["parameter"].unique()
            param_colors = {
                param: colors[i % len(colors)] for i, param in enumerate(all_params)
            }
            legend_items = [(param, param_colors[param]) for param in all_params]
            legend_title = "Parameters"
        else:
            # Color by parameter groups
            unique_groups = list(set(param_groups.values()))
            param_colors = {
                group: colors[i % len(colors)] for i, group in enumerate(unique_groups)
            }
            legend_items = [(group, param_colors[group]) for group in unique_groups]
            legend_title = "Parameter Groups"

        # Plot each output
        for ax, output in zip(axes, unique_outputs, strict=False):
            output_data = results[results["output"] == output]
            assert isinstance(output_data, pd.DataFrame)
            _create_morris_plot(
                ax,
                output_data,
                output,
                param_groups,
                param_colors,
            )

        # remove any empty subplots
        for idx in range(len(unique_outputs), len(axes)):
            fig.delaxes(axes[idx])

        # Create single legend on the right side
        legend_handles = []
        legend_labels = []
        for label, color in legend_items:
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                alpha=0.7,
                linewidth=0,
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        # Add legend to the right of the plots
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center right",
            bbox_to_anchor=(0.98, 0.5),
            title=legend_title,
            framealpha=0.9,
            fontsize=10,
        )

        # title
        fig.suptitle(
            r"Morris Sensitivity Analysis ($\mu^*$ vs $\sigma$)",
            fontsize=14,
        )

        plt.tight_layout()

    return fig


def _create_morris_plot(
    ax: Axes,
    output_data: pd.DataFrame,
    output_name: str,
    param_groups: dict | None = None,
    color_mapping: dict | None = None,
):
    """Create a Morris plot (mu_star vs sigma) for a single output."""
    # Default colors - expanded palette for more variety
    colors = [
        "#4C4B63",
        "#E63946",
        "#F77F00",
        "#FCBF49",
        "#06D6A0",
        "#118AB2",
        "#073B4C",
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
    ]

    # Use provided color mapping or create default
    if color_mapping is None:
        if param_groups is None:
            # Different color for each parameter when no groups specified
            unique_params = output_data["parameter"].unique()
            param_colors = {
                param: colors[i % len(colors)] for i, param in enumerate(unique_params)
            }
            color_mapping = param_colors
        else:
            # Color by parameter groups
            unique_groups = list(set(param_groups.values()))
            group_colors = {
                group: colors[i % len(colors)] for i, group in enumerate(unique_groups)
            }
            color_mapping = group_colors

    # Plot points without labels for legend (legend is handled at figure level)
    for _, row in output_data.iterrows():
        param_name = row["parameter"]

        if param_groups is None:
            color = color_mapping[param_name]
        else:
            group = param_groups.get(param_name, "default")
            color = color_mapping.get(group, colors[0])

        ax.scatter(row["sigma"], row["mu_star"], color=color, alpha=0.7, s=60)

    # Add parameter labels with matching colors
    for _, row in output_data.iterrows():
        param_name = row["parameter"]
        assert isinstance(param_name, str)

        if param_groups is None:
            # Use same color as the dot (individual parameter color)
            label_color = color_mapping[param_name]
        else:
            # Use group color
            group = param_groups.get(param_name, "default")
            label_color = color_mapping.get(group, colors[0])

        x_coord, y_coord = row["sigma"], row["mu_star"]
        assert isinstance(x_coord, float)
        assert isinstance(y_coord, float)
        ax.annotate(
            param_name,
            (x_coord, y_coord),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
            color=label_color,
            fontweight="bold",
        )

    ax.set_xlabel("σ (Standard Deviation)")  # noqa: RUF001
    ax.set_ylabel("μ* (Modified Mean)")
    ax.set_title(f"Output: {output_name}")
    ax.grid(True, alpha=0.3)
