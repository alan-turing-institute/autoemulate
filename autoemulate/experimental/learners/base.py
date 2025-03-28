from dataclasses import dataclass, field, InitVar
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, List, Any
import torch
from torcheval.metrics import MeanSquaredError, R2Score
from anytree import Node, RenderTree
from inspect import isabstract


@dataclass(kw_only=True)
class Base(ABC):
    """
    Base class for active learning simulation and emulation.

    Provides utility methods for tensor validation and design criteria computations.
    """

    @staticmethod
    def check_vector(X: torch.Tensor) -> torch.Tensor:
        """
        Validate that the input is a 1D torch.Tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor to validate.

        Returns
        -------
        torch.Tensor
            Validated 1D tensor.

        Raises
        ------
        ValueError
            If X is not a torch.Tensor or is not 1-dimensional.
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(X)}")
        elif X.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got {X.ndim}D")
        else:
            return X

    @staticmethod
    def check_matrix(X: torch.Tensor) -> torch.Tensor:
        """
        Validate that the input is a 2D torch.Tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor to validate.

        Returns
        -------
        torch.Tensor
            Validated 2D tensor.

        Raises
        ------
        ValueError
            If X is not a torch.Tensor or is not 2-dimensional.
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(X)}")
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {X.ndim}D")
        else:
            return X

    @staticmethod
    def check_pair(
        X: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate that two tensors have the same number of rows.

        Parameters
        ----------
        X : torch.Tensor
            First tensor.
        Y : torch.Tensor
            Second tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The validated pair of tensors.

        Raises
        ------
        ValueError
            If X and Y do not have the same number of rows.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        else:
            return X, Y

    @staticmethod
    def check_covariance(Y: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Validate and return the covariance matrix.

        Parameters
        ----------
        Y : torch.Tensor
            Output tensor.
        Sigma : torch.Tensor
            Covariance matrix, which may be full, diagonal, or a scalar per sample.

        Returns
        -------
        torch.Tensor
            Validated covariance matrix.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape relative to Y.
        """
        if Sigma.shape == (Y.shape[0], Y.shape[1], Y.shape[1]):
            return Sigma
        elif Sigma.shape == (Y.shape[0], Y.shape[1]):
            return Sigma
        elif Sigma.shape == (Y.shape[0],):
            return Sigma
        else:
            raise ValueError("Invalid covariance matrix shape")

    @staticmethod
    def trace(Sigma: torch.Tensor, d: int) -> torch.Tensor:
        """
        Compute the trace of the covariance matrix (A-optimal design criterion).

        Parameters
        ----------
        Sigma : torch.Tensor
            Covariance matrix (full, diagonal, or scalar).
        d : int
            Dimension of the output.

        Returns
        -------
        torch.Tensor
            The computed trace value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (d, d):
            return torch.diagonal(Sigma, dim1=1, dim2=2).sum(dim=1).mean()
        elif Sigma.dim() == 2 and Sigma.shape[1] == d:
            return Sigma.sum(dim=1).mean()
        elif Sigma.dim() == 1:
            return d * Sigma.mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def logdet(Sigma: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute the log-determinant of the covariance matrix (D-optimal design criterion).

        Parameters
        ----------
        Sigma : torch.Tensor
            Covariance matrix (full, diagonal, or scalar).
        dim : int
            Dimension of the output.

        Returns
        -------
        torch.Tensor
            The computed log-determinant value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if len(Sigma.shape) == 3 and Sigma.shape[1:] == (dim, dim):
            return torch.logdet(Sigma).mean()
        elif len(Sigma.shape) == 2 and Sigma.shape[1] == dim:
            return torch.sum(torch.log(Sigma), dim=1).mean()
        elif len(Sigma.shape) == 1:
            return dim * torch.log(Sigma).mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def max_eigval(Sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum eigenvalue of the covariance matrix (E-optimal design criterion).

        Parameters
        ----------
        Sigma : torch.Tensor
            Covariance matrix (full, diagonal, or scalar).

        Returns
        -------
        torch.Tensor
            The average maximum eigenvalue.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (Sigma.shape[1], Sigma.shape[1]):
            eigvals = torch.linalg.eigvalsh(Sigma)
            return eigvals[:, -1].mean()  # Eigenvalues are sorted in ascending order
        elif Sigma.dim() == 2:
            return Sigma.max(dim=1).values.mean()
        elif Sigma.dim() == 1:
            return Sigma.mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")


@dataclass(kw_only=True)
class Simulator(Base):
    """
    Simulator abstract class for generating outputs from inputs.

    This class defines the interface for a simulator that produces samples based on input X.

    Parameters
    ----------
    (No additional parameters)
    """

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from input data using the simulator.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Simulated output tensor.
        """
        Y = self.check_matrix(self.sample_forward(self.check_matrix(X)))
        X, Y = self.check_pair(X, Y)
        return Y

    @abstractmethod
    def sample_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to perform the forward simulation.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Simulated output tensor.
        """
        pass


@dataclass(kw_only=True)
class Emulator(Base):
    """
    Emulator abstract class for approximating simulator outputs along with uncertainty.

    Provides an interface for fitting an emulator model to training data and generating predictions with
    associated covariance.

    Parameters
    ----------
    (No additional parameters)
    """

    def sample(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate emulator predictions and covariance estimates for given inputs.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the predicted outputs and covariance estimates.
        """
        X = self.check_matrix(X)
        Y, Sigma = self.sample_forward(X)
        Y = self.check_matrix(Y)
        X, Y = self.check_pair(X, Y)
        Sigma = self.check_covariance(Y, Sigma)
        return Y, Sigma

    def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Fit the emulator model using the training data.

        Parameters
        ----------
        X_train : torch.Tensor
            Training input tensor.
        Y_train : torch.Tensor
            Training output tensor.
        """
        self.check_matrix(X_train)
        self.check_matrix(Y_train)
        self.check_pair(X_train, Y_train)
        self.fit_forward(X_train, Y_train)

    @abstractmethod
    def fit_forward(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Abstract method to fit the emulator model using training data.

        Parameters
        ----------
        X_train : torch.Tensor
            Training input tensor.
        Y_train : torch.Tensor
            Training output tensor.
        """
        pass

    @abstractmethod
    def sample_forward(self, X: torch.Tensor):
        """
        Abstract method to generate predictions and covariance estimates.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted outputs and covariance estimates.
        """
        pass


@dataclass(kw_only=True)
class Learner(Base):
    """
    Learner class that combines a simulator and an emulator for active learning.

    The learner uses a simulator to generate ground-truth outputs and an emulator to approximate
    the simulator. Training data is stored and used to update the emulator.

    Parameters
    ----------
    simulator : Simulator
        Simulator instance used to generate ground truth outputs.
    emulator : Emulator
        Emulator instance used to approximate the simulator.
    X_train : torch.Tensor
        Initial training input tensor.
    Y_train : torch.Tensor
        Initial training output tensor.
    """

    simulator: Simulator
    emulator: Emulator
    in_dim: int = field(init=False)
    out_dim: int = field(init=False)
    X_train: InitVar[torch.Tensor]
    Y_train: InitVar[torch.Tensor]

    def __post_init__(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Initialize the learner with training data and fit the emulator.

        Parameters
        ----------
        X_train : torch.Tensor
            Initial training input tensor.
        Y_train : torch.Tensor
            Initial training output tensor.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        if X_train is not None and Y_train is not None:
            self.emulator.fit(X_train, Y_train)
            self.in_dim = self.X_train.shape[1]
            self.out_dim = self.Y_train.shape[1]
        else:
            self.in_dim = None
            self.out_dim = None

    @classmethod
    def registry(cls) -> dict:
        """
        Recursively builds a dictionary mapping class names to class objects,
        starting from cls.

        Returns:
            A dictionary where keys are class names and values are the class objects.
        """
        d = {cls.__name__: cls}
        for sub in cls.__subclasses__():
            d.update(sub.registry())
        return d

    @classmethod
    def hierarchy(cls) -> List[Tuple[str, str]]:
        """
        Recursively collects the inheritance relationships (as ordered pairs)
        starting from cls.

        Returns:
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
    def __post_init__(self, X_train, Y_train):
        super().__post_init__(X_train, Y_train)
        self.metrics = {
            k: list()
            for k in ["mse", "r2", "rate", "logdet", "trace", "max_eigval", "n_queries"]
        }
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.n_queries = 0

    def fit(self, *args):
        # Query simulator and fit emulator
        X, Y_pred, Sigma, extra = self.query(*args)
        if X is not None:
            Y_true = self.simulator.sample(X)
            self.X_train = torch.cat([self.X_train, X])
            self.Y_train = torch.cat([self.Y_train, Y_true])
            self.emulator.fit(self.X_train, self.Y_train)
            self.mse.update(Y_pred, Y_true)
            self.r2.update(Y_pred, Y_true)
            self.n_queries += 1

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
        self.metrics["trace"].append(self.trace(Sigma, self.out_dim).item())
        self.metrics["logdet"].append(self.logdet(Sigma, self.out_dim).item())
        self.metrics["max_eigval"].append(self.max_eigval(Sigma).item())

        # extra per‑strategy metrics
        for k, v in extra.items():
            self.metrics.setdefault(k, []).append(v)

    @property
    def summary(self):
        """
        Compute summary metrics for the active learner based on recorded learning histories.

        This property converts the history of MSE and cumulative query counts into float tensors,
        and then computes two types of summary metrics:

        1. **Per-Query Ratios:** For each metric in ("mse", "r2", "trace", "logdet", "max_eigval"),
           the ratio is defined as the last recorded value of the metric divided by the last recorded
           cumulative number of queries. If no queries have been made (i.e. the last query count is 0),
           NaN is returned for all per-query ratios.

        2. **Area Under the MSE Curve (AUC):** The area is computed via trapezoidal integration of the MSE
           values over the cumulative number of queries, but only if there are at least 2 valid MSE entries.
           Otherwise, the AUC is set to NaN.

        Returns
        -------
        dict
            A dictionary containing:
                - "<metric>_per_query": The per-query ratio for each metric in ("mse", "r2", "trace",
                  "logdet", "max_eigval").
                - "auc_mse": The area under the MSE curve computed with respect to the cumulative number
                  of queries.
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
        self, *arg: Union[torch.Tensor, None]
    ) -> Tuple[
        Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict[str, List[Any]]
    ]:
        """
        Abstract method to query new samples.

        Parameters
        ----------
        *arg : torch.Tensor or None
            Optional input samples.

        Returns
        -------
        Tuple[torch.Tensor or None, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples (or None if no query is made),
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary of additional metrics.
        """
        pass
