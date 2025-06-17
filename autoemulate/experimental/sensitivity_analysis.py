from typing import Optional

from autoemulate.sensitivity_analysis import _sensitivity_analysis, _plot_morris_analysis, _plot_sobol_analysis
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import InputLike

class SensitivityAnalysis:
    """
    Global sensitivity analysis.
    """
    
    def __init__(
        self,
        model: Emulator,
        X: InputLike,
        method: str = "sobol",
        problem: Optional[dict] = None,
        N: int = 1024,
        conf_level: float = 0.95,
        as_df: bool = True,
    ):
        """
        Perform global sensitivity analysis on a fitted emulator.

        Parameters
        ----------
        model : Emulator
            Fitted emulator.
        X : InputLike
            Simulator input parameter values.
        method: str
            The sensitivity analysis method to perform, one of ["sobol", "morris"].
        problem : dict | None
            The problem definition dictionary. If None, the problem is generated from X using
            minimum and maximum values of the features as bounds. The dictionary should contain:

            - 'num_vars': Number of input variables (int)
            - 'names': List of variable names (list of str)
            - 'bounds': List of [min, max] bounds for each variable (list of lists)
            - 'output_names': Optional list of output names (list of str)

            Example::

                problem = {
                    "num_vars": 2,
                    "names": ["x1", "x2"],
                    "bounds": [[0, 1], [0, 1]],
                    "output_names": ["y1", "y2"]  # optional
                }
        N : int 
            Number of samples to generate for the analysis. Higher values give more accurate
            results but increase computation time. Default is 1024.
        conf_level : float
            Confidence level (between 0 and 1) for calculating confidence intervals of the
            sensitivity indices. Default is 0.95 (95% confidence).
        as_df : bool 
            If True, returns results as a long-format pandas DataFrame with columns for
            parameters, sensitivity indices, and confidence intervals. If False, returns
            the raw SALib results dictionary. Default is True.

        Returns
        -------
        pandas.DataFrame or dict
            If as_df=True (default), returns a DataFrame with columns:
                - 'parameter': Input parameter name
                - 'output': Output variable name
                - 'S1', 'S2', 'ST': First, second, and total order sensitivity indices
                - 'S1_conf', 'S2_conf', 'ST_conf': Confidence intervals for each index
            If as_df=False, returns the raw SALib results dictionary.

        Notes
        -----
        The Sobol method requires N * (2D + 2) model evaluations, where D is the number of input
        parameters. For example, with N=1024 and 5 parameters, this requires 12,288 evaluations.
        The Morris method requires fewer computations. 
        """
        self.method = method
        if method not in ["sobol", "morris"]:
            raise ValueError(f"Unknown method: {method}. Must be 'sobol' or 'morris'.")

        df_results = _sensitivity_analysis(
            model=model,
            method=method,
            problem=problem,
            X=X,
            N=N,
            conf_level=conf_level,
            as_df=as_df,
        )

        return df_results

    def plot_sensitivity_analysis(
        self, results, index="S1", param_groups=None, n_cols=None, figsize=None
    ):
        """
        Plot the sensitivity analysis results.

        Parameters:
        -----------
        results : pd.DataFrame
            The results from sobol_results_to_df.
        index : str, default "S1"
            Used only in Sobol analysis plot.
            The type of sensitivity index to plot.
            - "S1": first-order indices
            - "S2": second-order/interaction indices
            - "ST": total-order indices
        param_groups : dic[str, list[str]] | None
            Used only in Morris analysis plot.
            Optional parameter groupings of the form ({<group name> : [param1, ...], }) 
            used to give all the same plot color.
        n_cols : int, optional
            The number of columns in the plot. Defaults to 3 if there are 3 or more outputs,
            otherwise the number of outputs.
        figsize : tuple, optional
            Figure size as (width, height) in inches.If None, automatically calculated.

        """
        if self.method == "sobol":
            return _plot_sobol_analysis(results, index, n_cols, figsize)
        elif self.method == "morris":
            return _plot_morris_analysis(results, param_groups, n_cols, figsize)
)