import pandas as pd
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample

from autoemulate.experimental.data.utils import ConversionMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DistributionLike, NumpyLike, TensorLike

# NOTE: we still use these functions from main
# should we just move them to experimental as well?
from autoemulate.sensitivity_analysis import (
    _morris_results_to_df,
    _plot_morris_analysis,
    _plot_sobol_analysis,
    _sobol_results_to_df,
)


class SensitivityAnalysis(ConversionMixin):
    """
    Global sensitivity analysis.
    """

    def __init__(
        self,
        emulator: Emulator,
        x: TensorLike | None = None,
        problem: dict | None = None,
    ):
        """
        Parameters
        ----------
        emulator : Emulator
            Fitted emulator.
        x : InputLike | None
            Simulator input parameter values.
        problem : dict | None
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
        """
        Check that the problem definition is valid.
        """
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
        x : TensorLike
            Simulator input parameter values [n_samples, n_parameters].
        """
        if x.ndim == 1:
            msg = "x must be a 2D array."
            raise ValueError(msg)

        return {
            "num_vars": x.shape[1],
            "names": [f"X{i + 1}" for i in range(x.shape[1])],
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
        Make predictions with emulator for N input samples.
        """

        param_tensor = self._convert_to_tensors(param_samples)
        assert isinstance(param_tensor, TensorLike)
        y_pred = self.emulator.predict(param_tensor)

        # handle types, convert to numpy
        if isinstance(y_pred, TensorLike):
            y_pred_np, _ = self._convert_to_numpy(y_pred)
        elif isinstance(y_pred, DistributionLike):
            y_pred_np, _ = self._convert_to_numpy(y_pred.mean)
        else:
            msg = "Emulator has to return Tensor or Distribution"
            raise ValueError(msg)

        return y_pred_np

    def _get_output_names(self, num_outputs: int) -> list[str]:
        """
        Get the output names from the problem definition or generate default names.
        """
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
        n_samples : int
            Number of samples to generate for the analysis. Higher values give more
            accurate results but increase computation time. Default is 1024.
        conf_level : float
            Confidence level (between 0 and 1) for calculating confidence intervals
            of the Sobol sensitivity indices. Default is 0.95 (95% confidence). This
            is not used in Morris sensitivity analysis.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
                - 'parameter': Input parameter name
                - 'output': Output variable name
                - 'S1', 'S2', 'ST': First, second, and total order sensitivity indices
                - 'S1_conf', 'S2_conf', 'ST_conf': Confidence intervals for each index

        Notes
        -----
        The Sobol method requires N * (2D + 2) model evaluations, where D is the number
        of input parameters. For example, with N=1024 and 5 parameters, this requires
        12,288 evaluations. The Morris method requires far fewer computations.
        """
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
                Si = morris_analyze(self.problem, param_samples, y[:, i])
            results[name] = Si  # type: ignore PGH003

        if method == "sobol":
            return _sobol_results_to_df(results)
        return _morris_results_to_df(results, self.problem)

    @staticmethod
    def plot_sobol(results, index="S1", n_cols=None, figsize=None):
        """
        Plot Sobol sensitivity analysis results.

        Parameters:
        -----------
        results : pd.DataFrame
            The results from sobol_results_to_df.
        index : str, default "S1"
            The type of sensitivity index to plot.
            - "S1": first-order indices
            - "S2": second-order/interaction indices
            - "ST": total-order indices
        n_cols : int, optional
            The number of columns in the plot. Defaults to 3 if there are 3 or
            more outputs, otherwise the number of outputs.
        figsize : tuple, optional
            Figure size as (width, height) in inches. If None, set automatically.
        """
        return _plot_sobol_analysis(results, index, n_cols, figsize)

    @staticmethod
    def plot_morris(results, param_groups=None, n_cols=None, figsize=None):
        """
        Plot Morris analysis results.

        Parameters:
        -----------
        results : pd.DataFrame
            The results from sobol_results_to_df.
        param_groups : dic[str, list[str]] | None
            Optional parameter groupings used to give all the same plot color
            of the form ({<group name> : [param1, ...], }).
        n_cols : int, optional
            The number of columns in the plot. Defaults to 3 if there are 3 or
            more outputs, otherwise the number of outputs.
        figsize : tuple, optional
            Figure size as (width, height) in inches.If None, set calculated.
        """
        return _plot_morris_analysis(results, param_groups, n_cols, figsize)
