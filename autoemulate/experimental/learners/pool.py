from abc import abstractmethod
from dataclasses import dataclass

from ..types import GaussianLike, TensorLike
from .base import Active


@dataclass(kw_only=True)
class Pool(Active):
    """
    Active learning strategy that queries from a pool of unlabeled samples.

    Parameters
    ----------
    (Inherits parameters from Active)
    """

    @abstractmethod
    def query(
        self, X: TensorLike | None = None
    ) -> tuple[TensorLike | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Abstract method to query new samples from a given pool.

        Parameters
        ----------
        X : torch.Tensor
            Pool of input samples.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples,
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary of additional metrics.
        """
