import gpytorch
import torch

from .polynomial_features import PolynomialFeatures

class PolyMean(gpytorch.means.Mean):
    def __init__(self, degree, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.degree = degree
        self.input_size = input_size

        poly = PolynomialFeatures(self.degree)
        poly.fit(self.input_size)

        n_weights = len(poly.indices)
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
        poly = PolynomialFeatures(self.degree)
        poly.fit(self.input_size)
        x_ = poly.transform(x)
        res = x_.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res
    
    def __repr__(self):
        return f'Polymean(degree={self.degree}, input_size={self.input_size})'
