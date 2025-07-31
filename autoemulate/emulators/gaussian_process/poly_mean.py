import gpytorch
import torch

from autoemulate.core.types import TensorLike

from .polynomial_features import PolynomialFeatures


class PolyMean(gpytorch.means.Mean):
    """
    A general polynomial mean module for Gaussian Process emulators.

    Parameters
    ----------
    degree: int
        The degree of the polynomial for which we are defining
        the mapping.
    input_size: int
        The number of features to be mapped.
    batch_shape: int | None
        Optional batch dimension.
    bias: bool
        Flag for including a bias in the definition of the polynomial.
        If set to `False` polynomial includes weights only.
    """

    def __init__(
        self,
        degree: int,
        input_size: int,
        batch_shape: torch.Size | None = None,
        bias=True,
    ):
        super().__init__()
        self.degree = degree
        self.input_size = input_size

        if batch_shape is None:
            batch_shape = torch.Size()

        self.poly = PolynomialFeatures(self.degree, self.input_size)
        self.poly.fit()

        assert self.poly.indices is not None
        n_weights = len(self.poly.indices)
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(torch.randn(*batch_shape, n_weights, 1)),
        )

        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
            )
        else:
            self.bias = None

    def forward(self, x: TensorLike):
        """Forward pass through the polynomial mean module."""
        x_ = self.poly.transform(x)
        assert isinstance(self.weights, TensorLike)
        res = x_.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def __repr__(self):
        """Return the string representation of the PolyMean module."""
        return f"PolyMean(degree={self.degree}, input_size={self.input_size})"
