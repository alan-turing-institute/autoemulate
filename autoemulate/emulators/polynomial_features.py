import itertools

import torch

from autoemulate.core.types import TensorLike


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    """

    def __init__(self, n_features: int, degree: int = 2, include_bias: bool = True):
        """
        Initialize polynomial feature generator.

        Parameters
        ----------
        n_features: int
            The number of features in the input data.
        degree: int
            The maximum degree of the generated polynomial features. Defaults to 2.
        include_bias: bool
            If true (default), include a bias column (i.e., a column of ones).
        """
        self.n_features = n_features
        self.degree = degree
        self.include_bias = include_bias
        self._powers = self._compute_powers(n_features)

    def transform(self, x: TensorLike) -> TensorLike:
        """
        Generate polynomial and interaction features.

        Parameters
        ----------
        x: TensorLike
            The input data to transform of shape (n_samples, n_features)

        Returns
        -------
        TensorLike
            Features generated from the inputs.
        """
        # start with the bias term if needed
        x_expanded = (
            [torch.ones(x.shape[0], device=x.device, dtype=x.dtype)]
            if self.include_bias
            else []
        )

        # each term in _powers corresponds to one output feature (e.g., x1^2*x2)
        for powers in self._powers:
            term = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            for i, p in enumerate(powers):
                if p != 0:
                    term = term * x[:, i] ** p
            x_expanded.append(term)

        return torch.stack(x_expanded, dim=1)

    def _compute_powers(self, n_features: int) -> list[tuple[int, ...]]:
        """
        Compute the powers for each polynomial feature.

        For n_features=2 and self.degree=2, the output combines:
            degree 1: [(1, 0), (0, 1)]
            degree 2: [(2, 0), (1, 1), (0, 2)]
        resulting in:
            [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        which corresponds to:
            x1, x2, x1^2, x1*x2, x2^2

        Parameters
        ----------
        n_features: int
            The number of features in the input data.

        Returns
        -------
        list[tuple[int, ...]]
            A list of tuples representing the powers for each polynomial feature.
        """
        combs = []
        start = 0 if self.include_bias else 1
        for d in range(start, self.degree + 1):
            for comb in self._combinations(n_features, d):
                combs.append(comb)
        return combs

    def _combinations(self, n_features: int, degree: int) -> list[tuple[int, ...]]:
        """
        Return all combinations of feature powers for a given degree.

        For n_features=2 and degree=3, the combinations are:
            [(3, 0), (2, 1), (1, 2), (0, 3)]
        which corresponds to:
            x1^3, x1^2*x2, x1*x2^2, x2^3

        Parameters
        ----------
        n_features: int
            The number of features in the input data.
        degree: int
            The degree of the polynomial features.

        Returns
        -------
        list[tuple[int, ...]]
            A list of tuples representing the combinations of feature powers.
        """
        return [
            tuple(sum(1 for i in comb if i == j) for j in range(n_features))
            for comb in itertools.combinations_with_replacement(
                range(n_features), degree
            )
        ]
