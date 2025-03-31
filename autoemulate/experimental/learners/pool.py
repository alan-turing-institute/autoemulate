from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from inspect import isabstract
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from anytree import Node, RenderTree
from torcheval.metrics import MeanSquaredError, R2Score
from tqdm import tqdm

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
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[Any]]]:
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
        pass
