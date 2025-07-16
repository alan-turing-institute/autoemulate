import itertools

import torch

from autoemulate.experimental.types import TensorLike


class PolynomialFeatures:
    """
    This class is used to map an existing feature set to a polynomial feature set.

    Parameters
    --------
    degree: int
        The degree of the polynomial for which we are defining
        the mapping.
    input_size: int
        The number of features to be mapped.

    Examples
    -------
    Initialize the class to map the feature `X` (`n1` samples x `n2` features):

    >>> pf = PolynomialFeatures(degree=2, input_size=X.shape[1])

    Fit the instance in order to predefine the features that need to be multiplied to
    create the new features.

    >>> pf.fit()

    Generate the new polynomial feature set:

    >>> X_deg_2 = pf.transform(X)
    """

    def __init__(self, degree: int, input_size: int):
        assert degree > 0, "`degree` input must be greater than 0."
        assert input_size > 0, (
            "`input_size`, which defines the number of features, for has to be "
            "greater than 0"
        )
        self.degree = degree
        self.indices = None
        self.input_size = input_size

    def fit(self):
        x = list(range(self.input_size))

        d = self.degree
        L = []
        while d > 1:
            combinations = [list(p) for p in itertools.product(x, repeat=d)]
            for li in combinations:
                li.sort()
            L += list(map(list, torch.unique(torch.tensor(combinations), dim=0)))
            d -= 1
        L += [[i] for i in x]

        Ls = []
        for d in range(1, self.degree + 1):
            ld = []
            for combination in L:
                if len(combination) == d:
                    ld.append(combination)
            ld.sort()
            Ls += ld
        self.indices = Ls

    def transform(self, x: TensorLike):
        """Generate the polynomial feature set."""
        if not self.indices:
            msg = "self.indices is set to None. Did you forget to call 'fit'?"
            raise ValueError(msg)

        return torch.stack(
            [torch.prod(x[..., i], dim=-1) for i in self.indices], dim=-1
        )
