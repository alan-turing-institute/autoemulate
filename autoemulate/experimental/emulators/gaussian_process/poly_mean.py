import gpytorch
import torch

from .polynomial_features import PolynomialFeatures


class PolyMean(gpytorch.means.Mean):
    """A geneneral polynomial mean module to be used to construct
    `guassian_process_torch` emulators.

    Parameters
    --------
    degree : int
        The degree of the polynomial for which we are defining
        the mapping.
    input_size : int
        The number of features to be mapped.
    barch_shape : int
    bias : bool
        Flag for including a bias in the defnition of the polymial.
        If set to `False` polynomial includes weights only.
    """

    def __init__(self, degree, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.degree = degree
        self.input_size = input_size

        self.poly = PolynomialFeatures(self.degree, self.input_size)
        self.poly.fit()

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

    def forward(self, x):
        x_ = self.poly.transform(x)
        res = x_.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def __repr__(self):
        return f"Polymean(degree={self.degree}, input_size={self.input_size})"
