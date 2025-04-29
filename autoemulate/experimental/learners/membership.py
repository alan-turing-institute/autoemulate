from abc import abstractmethod
from dataclasses import dataclass

from ..types import GaussianLike, TensorLike
from .base import Active


@dataclass(kw_only=True)
class Membership(Active):
    """
    Active learning strategy based on membership queries.

    Parameters
    ----------
    (Inherits parameters from Active)
    """

    @abstractmethod
    def query(
        self, X: TensorLike | None = None
    ) -> tuple[TensorLike | None, TensorLike | GaussianLike, dict[str, float]]:
        """
        Abstract method to query new samples using a membership strategy.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]
            A tuple containing:
            - The queried samples,
            - The predicted outputs,
            - The covariance estimates,
            - A dictionary of additional metrics.
        """
