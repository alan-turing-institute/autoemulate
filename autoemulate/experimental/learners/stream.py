from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from ..types import GaussianLike, TensorLike
from .base import Active


@dataclass(kw_only=True)
class Stream(Active):
    """
    Active learning strategy for streaming data.

    Parameters
    ----------
    (Inherits parameters from Active)
    """

    show_progress: bool = field(default=True)

    @abstractmethod
    def query(
        self, X: TensorLike | None = None
    ) -> tuple[TensorLike | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Abstract method to query new samples from a stream.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.

        Returns
        -------
        Tuple[torch.Tensor or None, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples (or None if no query is made),
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary of additional metrics.
        """

    def fit_samples(self, X: torch.Tensor):
        """
        Fit the active learner using a stream of samples.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.
        """
        X = self.check_matrix(X)
        for x in (
            pb := tqdm(
                X,
                desc=self.__class__.__name__,
                leave=True,
                disable=not self.show_progress,
            )
        ):
            self.fit(x.reshape(1, -1))
            pb.set_postfix(
                ordered_dict={key: val[-1] for key, val in self.metrics.items()}
            )

    def fit_batches(self, X: torch.Tensor, batch_size: int):
        """
        Fit the active learner using batches of samples.

        This method automatically splits X into batches of the specified size and then
        sequentially fits each batch.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.
        batch_size : int
            Number of samples per batch.
        """
        X = self.check_matrix(X)
        for i in (
            pb := tqdm(
                range(0, X.shape[0], batch_size),
                desc=f"{self.__class__.__name__} (batches)",
                disable=not self.show_progress,
            )
        ):
            batch = X[i : i + batch_size]
            self.fit(batch)
            pb.set_postfix(ordered_dict={k: v[-1] for k, v in self.metrics.items()})


@dataclass(kw_only=True)
class Random(Stream):
    """
    Random active learning strategy that queries samples based on a fixed probability.

    Parameters
    ----------
    p_query : float
        Query probability for selecting a sample.
    """

    p_query: float

    def query(
        self, X: TensorLike | None = None
    ) -> tuple[torch.Tensor | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Query new samples randomly based on a fixed probability.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.

        Returns
        -------
        Tuple[torch.Tensor or None, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples (or None if the random condition is not met),
            - The predicted outputs,
            - The covariance estimates,
            - An empty dictionary of additional metrics.
        """
        assert isinstance(X, TensorLike)
        # TODO: move handling to check method in base class
        output = self.emulator.predict(X)
        assert isinstance(output, TensorLike | GaussianLike)
        # assert isinstance(output, TensorLike | DistributionLike)
        X = X if np.random.rand() < self.p_query else None
        return X, output, {}


@dataclass(kw_only=True)
class Threshold(Stream):
    """
    Threshold-based active learning strategy that queries samples based on a score
    threshold.

    Parameters
    ----------
    threshold : float
        Threshold value for querying.
    """

    threshold: float

    def __post_init__(self):
        """
        Initialize the threshold-based learner and update metrics.

        Parameters
        ----------
        X_train : torch.Tensor
            Training input tensor.
        Y_train : torch.Tensor
            Training output tensor.
        """
        super().__post_init__()
        self.metrics.update({"score": []})

    @abstractmethod
    def score(
        self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstract method to compute a score for querying based on the input sample.

        Parameters
        ----------
        X : torch.Tensor
            Input sample tensor.
        Y : torch.Tensor
            Predicted output tensor.
        Sigma : torch.Tensor
            Covariance estimates tensor.

        Returns
        -------
        torch.Tensor
            Computed score for the input sample.
        """

    def query(
        self, X: TensorLike | None = None
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | GaussianLike,
        dict[str, float],
    ]:
        """
        Query new samples based on whether the computed score exceeds a threshold.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.

        Returns
        -------
        Tuple[torch.Tensor or None, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples (or None if the score does not exceed the threshold),
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary with the computed score.
        """
        # TODO: move handling to check method in base class
        assert isinstance(X, torch.Tensor)
        output = self.emulator.predict(X)
        assert isinstance(output, GaussianLike)
        assert isinstance(output.variance, torch.Tensor)
        score = self.score(X, output.mean, output.variance)
        X = X if score > self.threshold else None
        return X, output, {"score": score.item()}


@dataclass(kw_only=True)
class Input(Threshold):
    """
    Active learning strategy based on input space criteria.

    Parameters
    ----------
    (Inherits parameters from Threshold)
    """


@dataclass(kw_only=True)
class Distance(Input):
    """
    Active learning strategy that scores samples based on distance metrics.

    Queries samples based on the average minimum distance to the training set.

    Parameters
    ----------
    (Inherits parameters from Input)
    """

    def score(
        self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the average minimum distance from the input samples to the training
        data.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.

        Returns
        -------
        float
            The average minimum distance.
        """
        _, _, _ = X, Y, Sigma  # Unused variables
        distances = torch.cdist(X, self.X_train)
        min_dists, _ = distances.min(dim=1)
        return min_dists.mean()


@dataclass(kw_only=True)
class Output(Threshold):
    """
    Active learning strategy based on output space criteria.

    Parameters
    ----------
    (Inherits parameters from Threshold)
    """


@dataclass(kw_only=True)
class A_Optimal(Output):
    """
    Active learning strategy using the A-optimal design criterion.

    Uses the trace of the covariance matrix as the scoring function.

    Parameters
    ----------
    (Inherits parameters from Output)
    """

    def score(
        self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the score using the trace of the covariance matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.
        Y : torch.Tensor
            Predicted outputs (not used).
        Sigma : torch.Tensor
            Covariance estimates (not used).

        Returns
        -------
        torch.Tensor
            Score based on the trace of the covariance matrix.
        """
        _, _ = X, Y  # Unused variables
        return self.trace(Sigma, self.out_dim)


@dataclass(kw_only=True)
class D_Optimal(Output):
    """
    Active learning strategy using the D-optimal design criterion.

    Uses the log-determinant of the covariance matrix as the scoring function.

    Parameters
    ----------
    (Inherits parameters from Output)
    """

    def score(
        self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the score using the log-determinant of the covariance matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.
        Y : torch.Tensor
            Predicted outputs.
        Sigma : torch.Tensor
            Covariance estimates.

        Returns
        -------
        torch.Tensor
            Score based on the log-determinant of the covariance matrix.
        """
        _, _ = X, Y  # Unused variables
        return self.logdet(Sigma, self.out_dim)


@dataclass(kw_only=True)
class E_Optimal(Output):
    """
    Active learning strategy using the E-optimal design criterion.

    Uses the maximum eigenvalue of the covariance matrix as the scoring function.

    Parameters
    ----------
    (Inherits parameters from Output)
    """

    def score(
        self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the score using the maximum eigenvalue of the covariance matrix.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.
        Y : torch.Tensor
            Predicted outputs.
        Sigma : torch.Tensor
            Covariance estimates.

        Returns
        -------
        torch.Tensor
            Score based on the maximum eigenvalue of the covariance matrix.
        """
        _, _ = X, Y  # Unused variables
        return self.max_eigval(Sigma)


@dataclass(kw_only=True)
class Adaptive(Threshold):
    """
    Adaptive active learning strategy that adjusts the query threshold using PID
    control.

    The adaptive mechanism modifies the threshold based on proportional, integral, and
    derivative components. Errors can be computed using a sliding window to reduce
    dependency on initial errors.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.
    key : str
        Key used to reference a metric.
    target : float
        Target value for the metric.
    min_threshold : float or None
        Minimum allowable threshold.
    max_threshold : float or None
        Maximum allowable threshold.
    window_size : int or None, optional
        Size of the sliding window to use for error computation. If None, use the full
        error history.
    """

    Kp: float
    Ki: float
    Kd: float
    key: str
    target: float
    min_threshold: float | None
    max_threshold: float | None
    window_size: int | None

    def __post_init__(self):
        """
        Initialize the adaptive learner and update metrics.

        Parameters
        ----------
        X_train : torch.Tensor
            Training input tensor.
        Y_train : torch.Tensor
            Training output tensor.
        """
        super().__post_init__()
        self.metrics.update(
            {
                "threshold": [],
                "error_prop": [],
                "error_int": [],
                "error_deriv": [],
            }
        )

    def query(
        self, X: TensorLike | None = None
    ) -> tuple[TensorLike | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Query new samples and adapt the threshold using a PID controller with a sliding
        window for error computation.

        Parameters
        ----------
        X : torch.Tensor
            Stream of input samples.

        Returns
        -------
        Tuple[torch.Tensor or None, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples,
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary with the PID control metrics.
        """
        # Call the parent query (e.g., from Threshold) to get initial values and
        # metrics.
        X, output, metrics = super().query(X)

        # Retrieve the error history for the specified metric.
        error_list = self.metrics.get(self.key, [])
        if error_list:
            if self.window_size is not None:
                n = min(len(error_list), self.window_size)
                errors = torch.tensor(error_list[-n:]) - self.target
            else:
                errors = torch.tensor(error_list) - self.target
        else:
            errors = torch.tensor([0.0])

        # Compute PID components.
        ep = errors[-1].item() if len(errors) >= 1 else 0.0
        ei = errors.sum().item() if len(errors) >= 1 else 0.0
        ed = errors[-1].item() - errors[-2].item() if len(errors) >= 2 else 0.0

        # Update the threshold using the PID control law and enforce bounds.
        self.threshold += self.Kp * ep + self.Ki * ei + self.Kd * ed
        if self.min_threshold is not None:
            self.threshold = max(self.threshold, self.min_threshold)
        if self.max_threshold is not None:
            self.threshold = min(self.threshold, self.max_threshold)

        # Update the metrics dictionary with the PID control terms.
        metrics.update(
            {
                "threshold": self.threshold,
                "error_prop": ep,
                "error_int": ei,
                "error_deriv": ed,
            }
        )
        return X, output, metrics


@dataclass(kw_only=True)
class Adaptive_Distance(Adaptive, Distance):
    """
    Adaptive input distance-based active learning strategy using PID control.

    Parameters
    ----------
    (Inherits parameters from Adaptive and Distance)
    """


@dataclass(kw_only=True)
class Adaptive_A_Optimal(Adaptive, A_Optimal):
    """
    Adaptive A-optimal active learning strategy using PID control.

    Parameters
    ----------
    (Inherits parameters from Adaptive and A_Optimal)
    """

    def __post_init__(self):
        if self.min_threshold is not None and self.min_threshold < 0.0:
            msg = (
                f"Minimum threashold ({self.min_threshold}) must be greater than or "
                "equal to 0 since it uses trace of a positive semi-definite matrix."
            )
            raise ValueError(msg)

        if self.min_threshold is None:
            self.min_threshold = 0.0

        A_Optimal.__post_init__(self)


@dataclass(kw_only=True)
class Adaptive_D_Optimal(Adaptive, D_Optimal):
    """
    Adaptive D-optimal active learning strategy using PID control.

    Parameters
    ----------
    (Inherits parameters from Adaptive and D_Optimal)
    """


@dataclass(kw_only=True)
class Adaptive_E_Optimal(Adaptive, E_Optimal):
    """
    Adaptive E-optimal active learning strategy using PID control.

    Parameters
    ----------
    (Inherits parameters from Adaptive and E_Optimal)
    """

    def __post_init__(self):
        if self.min_threshold is not None and self.min_threshold < 0.0:
            msg = (
                f"Minimum threshold ({self.min_threshold}) must be greater than or "
                "equal to 0 since it uses max eigenvalue of a positive semi-definite "
                "matrix."
            )
            raise ValueError(msg)

        if self.min_threshold is None:
            self.min_threshold = 0.0

        E_Optimal.__post_init__(self)
