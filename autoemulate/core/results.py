import logging

import pandas as pd

from autoemulate.core.metrics import Metric, get_metric
from autoemulate.core.types import ModelParams
from autoemulate.emulators.transformed.base import TransformedEmulator

logger = logging.getLogger("autoemulate")


class Result:
    """Represents a single result of an emulator evaluation."""

    def __init__(
        self,
        id: int,
        model_name: str,
        model: TransformedEmulator,
        params: ModelParams,
        test_metrics: dict[Metric, tuple[float, float]],
        train_metrics: dict[Metric, tuple[float, float]],
    ):
        """Initialize a Result object.

        Parameters
        ----------
        id: int
            Unique identifier for the result.
        model_name: str
            Name of the model used in the evaluation.
        model: TransformedEmulator
            The emulator model used for predictions.
        params: ModelParams
            Parameters used for the model.
        test_metrics: dict[str, tuple[float, float]]
            Dictionary of metrics on the test set. Each key is a metric name and
            each value is a tuple of (mean, std).
        train_metrics: dict[str, tuple[float, float]]
            Dictionary of metrics on the training set. Each key is a metric name and
            each value is a tuple of (mean, std).

        """
        self.id = id
        self.model_name = model_name
        self.model = model
        self.x_transforms = model.x_transforms
        self.y_transforms = model.y_transforms
        self.params = params
        self.test_metrics = test_metrics
        self.train_metrics = train_metrics

    def metadata_df(self) -> pd.DataFrame:
        """
        Return the Result object as a dataframe, without the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ['id', 'model_name', 'x_transforms', 'y_transforms', 'params']
            plus columns for each metric in test_metrics and train_metrics.
        """

        # Serialize the params dictionary to a string representation
        def serialize_params(params):
            out = {}
            for k, v in params.items():
                if callable(v) or isinstance(v, type):
                    out[k] = v.__name__
                else:
                    out[k] = v
            return out

        data = {
            "id": [self.id],
            "model_name": [self.model_name],
            "x_transforms": [self.x_transforms],
            "y_transforms": [self.y_transforms],
            "params": str(serialize_params(self.params)),
        }

        # Add test metrics
        for metric_name, (mean, std) in self.test_metrics.items():
            data[f"{metric_name}_test"] = [mean]
            data[f"{metric_name}_test_std"] = [std]

        # Add train metrics
        for metric_name, (mean, std) in self.train_metrics.items():
            data[f"{metric_name}_train"] = [mean]
            data[f"{metric_name}_train_std"] = [std]

        return pd.DataFrame(data)


class Results:
    """Container for multiple Result objects."""

    def __init__(
        self,
        results: list[Result] | None = None,
    ):
        """Initialize a Results object.

        Parameters
        ----------
        results: list[Result] | None
            A list of Result objects. If None, an empty list is created.
        """
        if results is None:
            results = []
        self.results = results
        self._id_to_result = {result.id: result for result in self.results}

    def _update_index(self):
        self._id_to_result = {result.id: result for result in self.results}

    def add_result(self, result: Result):
        """Add a Result object to the collection."""
        self.results.append(result)
        self._update_index()

    def summarize(self) -> pd.DataFrame:
        """
        Summarize the results in a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                ['model_name', 'x_transforms', 'y_transforms', 'params']
                plus columns for all metrics in the results.
        """
        if not self.results:
            return pd.DataFrame()

        data = {
            "model_name": [result.model_name for result in self.results],
            "x_transforms": [result.x_transforms for result in self.results],
            "y_transforms": [result.y_transforms for result in self.results],
            "params": [result.params for result in self.results],
        }

        # Collect all unique metrics from all results
        all_test_metrics = set()
        all_train_metrics = set()
        for result in self.results:
            all_test_metrics.update(result.test_metrics.keys())
            all_train_metrics.update(result.train_metrics.keys())

        # Add test metrics columns
        for metric in sorted(all_test_metrics):
            data[f"{metric}_test"] = [
                result.test_metrics.get(metric, (float("nan"), float("nan")))[0]
                for result in self.results
            ]
            data[f"{metric}_test_std"] = [
                result.test_metrics.get(metric, (float("nan"), float("nan")))[1]
                for result in self.results
            ]

        # Add train metrics columns
        for metric in sorted(all_train_metrics):
            data[f"{metric}_train"] = [
                result.train_metrics.get(metric, (float("nan"), float("nan")))[0]
                for result in self.results
            ]
            data[f"{metric}_train_std"] = [
                result.train_metrics.get(metric, (float("nan"), float("nan")))[1]
                for result in self.results
            ]

        df = pd.DataFrame(data)
        # Sort by r2_test if available, otherwise by the first available test metric
        sort_by = "r2_test" if "r2_test" in df.columns else str(df.columns[4])
        return df.sort_values(by=sort_by, ascending=False)

    summarise = summarize

    def best_result(self, metric: str | Metric | None = None) -> Result:
        """
        Get the model with the best result based on the given metric.

        Parameters
        ----------
        metric: str | Metric | None
            The name of the metric to use for comparison. If None, uses the first
            available metric found in the results. The metric should exist in the
            test_metrics of the results.
        metric_maximize: bool | None
            Whether higher values are better for the metric. If None, defaults to True
            (assumes higher is better). Set to False for metrics like RMSE or MAE where
            lower is better.

        Returns
        -------
        Result
            The result with the best score for the specified metric.
        """
        if not self.results:
            msg = "No results available. Please run AutoEmulate.compare() first."
            raise ValueError(msg)

        # If metric_name is None, use the first available metric
        if metric is None:
            # Collect all available metrics
            available_metrics = set()
            for result in self.results:
                available_metrics.update(result.test_metrics.keys())

            if not available_metrics:
                msg = "No metrics available in results."
                raise ValueError(msg)

            # Use the first metric
            metric_selected = next(iter(available_metrics))
            logger.info("Using metric '%s' to determine best result.", metric_selected)
        else:
            # Check if the specified metric exists in at least one result
            if not any(metric in result.test_metrics for result in self.results):
                available_metrics = set()
                for result in self.results:
                    available_metrics.update(result.test_metrics.keys())
                msg = (
                    f"Metric '{metric}' not found in any results. "
                    f"Available metrics: {sorted(available_metrics)}"
                )
                raise ValueError(msg)
            metric_selected = get_metric(metric)
            logger.info("Using metric '%s' to determine best result.", metric_selected)

        # Select best result based on whether we're maximizing or minimizing
        if metric_selected.maximize:
            return max(
                self.results,
                key=lambda r: r.test_metrics.get(metric_selected, (float("-inf"), 0))[
                    0
                ],
            )
        return min(
            self.results,
            key=lambda r: r.test_metrics.get(metric_selected, (float("inf"), 0))[0],
        )

    def get_result(self, result_id: int) -> Result:
        """
        Get a result by its ID.

        Parameters
        ----------
        result_id: int
            The ID of the model to retrieve.

        Returns
        -------
        Result
            The result with the specified ID.
        """
        try:
            return self._id_to_result[result_id]
        except KeyError as err:
            raise ValueError(f"No result found with ID: {result_id!s}") from err
