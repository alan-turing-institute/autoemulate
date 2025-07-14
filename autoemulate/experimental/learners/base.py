from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isabstract

import torch
from anytree import Node, RenderTree
from torch.distributions import MultivariateNormal
from torcheval.metrics import MeanSquaredError, R2Score

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.logging_config import get_configured_logger
from autoemulate.experimental.simulations.base import Simulator

from ..types import GaussianLike, TensorLike


@dataclass(kw_only=True)
class Learner(ValidationMixin, ABC):
    """
    Learner class that combines a simulator and an emulator for active learning.

    The learner uses a simulator to generate ground-truth outputs and an emulator to
    approximate the simulator. Training data is stored and used to update the emulator.

    Parameters
    ----------
    simulator: Simulator
        Simulator instance used to generate ground truth outputs.
    emulator: Emulator
        Emulator instance used to approximate the simulator.
    x_train: TensorLike
        Initial training input tensor.
    y_train: TensorLike
        Initial training output tensor.
    """

    simulator: Simulator
    emulator: Emulator
    x_train: TensorLike
    y_train: TensorLike
    log_level: str = "progress_bar"
    in_dim: int = field(init=False)
    out_dim: int = field(init=False)

    def __post_init__(self):
        """
        Initialize the learner with training data and fit the emulator.
        """
        log_level = getattr(self, "log_level", "progress_bar")
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.logger.info("Initializing Learner with training data.")
        self.emulator.fit(self.x_train, self.y_train)
        self.logger.info("Emulator fitted with initial training data.")
        self.in_dim = self.x_train.shape[1]
        self.out_dim = self.y_train.shape[1]

    @classmethod
    def registry(cls) -> dict:
        """
        Recursively builds a dictionary mapping class names to class objects,
        starting from cls.

        Returns
        -------
        dict
            A dictionary where keys are class names and values are the class objects.
        """
        d = {cls.__name__: cls}
        for sub in cls.__subclasses__():
            d.update(sub.registry())
        return d

    @classmethod
    def hierarchy(cls) -> list[tuple[str, str]]:
        """
        Recursively collects the inheritance relationships (as ordered pairs)
        starting from cls.

        Returns
        -------
        list[tuple[str, str]]
            A list of tuples where each tuple is (parent_class_name, child_class_name).
        """
        edges = []
        for base in cls.__subclasses__():
            edges.append((cls.__name__, base.__name__))
            edges.extend(base.hierarchy())
        return edges

    @classmethod
    def plot_hierarchy(cls):
        """
        Builds and prints an ASCII tree of the class hierarchy starting from cls.

        Each class name is annotated with a marker indicating whether it is abstract:
          - [Abstract] for classes with one or more abstract methods.
          - [Concrete] for classes that are fully implemented.

        The method uses the anytree library to construct and render the tree.
        """
        pairs = cls.hierarchy()
        nodes = {name: Node(name) for pair in pairs for name in pair}
        for parent, child in pairs:
            nodes[child].parent = nodes[parent]
        root = nodes[cls.__name__]
        registry = cls.registry()
        for pre, _, node in RenderTree(root):
            class_obj = registry.get(node.name)
            mark = "[Abstract]" if class_obj and isabstract(class_obj) else "[Concrete]"
            print(f"{pre}{node.name} {mark}")


@dataclass(kw_only=True)
class Active(Learner):
    def __post_init__(self):
        super().__post_init__()
        self.metrics = {
            k: []
            for k in ["mse", "r2", "rate", "logdet", "trace", "max_eigval", "n_queries"]
        }
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.n_queries = 0

    def fit(self, *args):
        # Query simulator and fit emulator
        x, output, extra = self.query(*args)
        if isinstance(output, TensorLike):
            y_pred = output
        elif isinstance(output, GaussianLike):
            assert output.variance.ndim == 2
            y_pred, _ = output.mean, output.variance
        elif isinstance(output, GaussianLike):
            y_pred, _ = output.loc, None
        else:
            msg = (
                f"Output must be either `Tensor` or `MultivariateNormal` but got "
                f"{type(output)}"
            )
            raise TypeError(msg)

        if x is not None:
            # If x is not, we skip the point (typically for Stream learners)
            self.logger.info("Appending new training data and refitting emulator.")
            y_true = self.simulator.forward(x)
            self.x_train = torch.cat([self.x_train, x])
            self.y_train = torch.cat([self.y_train, y_true])
            self.emulator.fit(self.x_train, self.y_train)
            self.mse.update(y_pred, y_true)
            self.r2.update(y_pred, y_true)
            self.n_queries += 1
            self.logger.info("Training data updated. Total queries: %s", self.n_queries)

        # Only compute once we have ≥2 labeled points
        if self.n_queries >= 2:
            mse_val = self.mse.compute().item()
            r2_val = self.r2.compute().item()
        else:
            mse_val = float("nan")
            r2_val = float("nan")

        self.metrics["mse"].append(mse_val)
        self.metrics["r2"].append(r2_val)
        self.metrics["rate"].append(self.n_queries / (len(self.metrics["rate"]) + 1))
        self.metrics["n_queries"].append(self.n_queries)
        self.logger.info("Metrics updated: MSE=%s, R2=%s", mse_val, r2_val)

        # If Gaussian output
        # TODO: check generality for other GPs (e.g. with full covariance)
        if isinstance(output, MultivariateNormal):
            assert isinstance(output.variance, TensorLike)
            assert output.variance.ndim == 2
            assert output.variance.shape[1] == self.out_dim
            # For Multivariate Normal, the variance property gives the correct value
            # required here with shape: (batch, out_dim)
            covariance = output.variance
            self.metrics["trace"].append(self.trace(covariance, self.out_dim).item())
            self.metrics["logdet"].append(self.logdet(covariance, self.out_dim).item())
            self.metrics["max_eigval"].append(self.max_eigval(covariance).item())
            self.logger.info("Gaussian output metrics updated.")

        # extra per-strategy metrics
        for k, v in extra.items():
            self.metrics.setdefault(k, []).append(v)
            self.logger.info("Extra metric '%s' updated: %s", k, v)

    @property
    def summary(self):
        """
        Compute summary metrics for the active learner based on recorded learning
        histories.

        This property converts the history of MSE and cumulative query counts into float
        tensors, and then computes two types of summary metrics:

        1. **Per-Query Ratios:** For each metric in ("mse", "r2", "trace", "logdet",
           "max_eigval"), the ratio is defined as the last recorded value of the metric
           divided by the last recorded cumulative number of queries. If no queries have
           been made (i.e. the last query count is 0), NaN is returned for all per-query
           ratios.

        2. **Area Under the MSE Curve (AUC):** The area is computed via trapezoidal
           integration of the MSE values over the cumulative number of queries, but only
           if there are at least 2 valid MSE entries. Otherwise, the AUC is set to NaN.

        Returns
        -------
        dict
            A dictionary containing:
                - "<metric>_per_query": The per-query ratio for each metric in ("mse",
                  "r2", "trace", "logdet", "max_eigval").
                - "auc_mse": The area under the MSE curve computed with respect to the
                  cumulative number of queries.
        """
        # pull histories into float tensors
        mse = torch.tensor(self.metrics["mse"], dtype=torch.float32)
        q = torch.tensor(self.metrics["n_queries"], dtype=torch.float32)

        # build per-query ratios safely (avoid zero division)
        d = {}
        if q[-1] > 0:
            for k in ("mse", "r2", "trace", "logdet", "max_eigval"):
                d[f"{k}_per_query"] = (
                    self.metrics[k][-1] / self.metrics["n_queries"][-1]
                )
        else:
            for k in ("mse", "r2", "trace", "logdet", "max_eigval"):
                d[f"{k}_per_query"] = float("nan")

        # mask out nan entries for MSE
        valid_mse = ~torch.isnan(mse)

        # compute AUC only if ≥2 valid points
        d["auc_mse"] = (
            torch.trapz(mse[valid_mse], q[valid_mse]).item()
            if valid_mse.sum() >= 2
            else float("nan")
        )
        return d

    @abstractmethod
    def query(
        self, x: TensorLike | None = None
    ) -> tuple[TensorLike | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Abstract method to query new samples.

        Parameters
        ----------
        *arg: TensorLike or None
            Optional input samples.

        Returns
        -------
        tuple[TensorLike or None, TensorLike, TensorLike, Dict[str, list[Any]]]
            A tuple containing:
            - The queried samples (or None if no query is made),
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary of additional metrics.
        """
