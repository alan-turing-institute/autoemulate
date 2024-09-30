import gpytorch
import pytest
import torch

from autoemulate.emulators.gaussian_process_utils.poly_mean import PolyMean
from autoemulate.emulators.gaussian_process_utils.polynomial_features import (
    PolynomialFeatures,
)


# -------------------test PolyMean--------------------------------
@pytest.fixture
def poly_mean():
    return PolyMean(degree=2, input_size=3, bias=True)


def test_poly_mean_initialization(poly_mean):
    assert isinstance(poly_mean, gpytorch.means.Mean)
    assert poly_mean.degree == 2
    assert poly_mean.input_size == 3
    assert poly_mean.bias is not None


def test_poly_mean_forward(poly_mean):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = poly_mean(x)
    assert output.shape == torch.Size([2])


def test_poly_mean_no_bias():
    poly_mean_no_bias = PolyMean(degree=2, input_size=3, bias=False)
    assert poly_mean_no_bias.bias is None


def test_poly_mean_repr(poly_mean):
    assert repr(poly_mean) == "Polymean(degree=2, input_size=3)"


def test_poly_mean_parameter_shapes(poly_mean):
    assert (
        poly_mean.weights.shape[-2] == 9
    )  # Number of polynomial features for degree 2 and input_size 3
    assert poly_mean.bias.shape == torch.Size([1])


def test_poly_mean_forward_batch():
    batch_poly_mean = PolyMean(degree=2, input_size=3, batch_shape=torch.Size([2]))
    print(batch_poly_mean.weights.shape)
    print(batch_poly_mean.bias.shape)
    x = torch.randn(4, 3)  # 4 samples, batch size 2, 3 features
    output = batch_poly_mean(x)
    assert output.shape == torch.Size([2, 4])  # batch size 2, 4 samples


def test_poly_mean_gradients():
    poly_mean = PolyMean(degree=2, input_size=3, bias=True)
    x = torch.randn(5, 3, requires_grad=True)
    output = poly_mean(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert poly_mean.weights.grad is not None
    assert poly_mean.bias.grad is not None


# ------------------------test PolynomialFeatures--------------------------------


def test_initialization():
    pf = PolynomialFeatures(degree=2, input_size=3)
    assert pf.degree == 2
    assert pf.input_size == 3
    assert pf.indices is None


def test_fit():
    pf = PolynomialFeatures(degree=2, input_size=2)
    pf.fit()
    expected_indices = [[0], [1], [0, 0], [0, 1], [1, 1]]
    assert pf.indices == expected_indices


def test_transform():
    pf = PolynomialFeatures(degree=2, input_size=2)
    pf.fit()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = pf.transform(x)
    expected = torch.tensor([[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]])
    assert torch.allclose(result, expected)


def test_higher_degree():
    pf = PolynomialFeatures(degree=3, input_size=2)
    pf.fit()
    x = torch.tensor([[2.0, 3.0], [3.0, 4.0]])
    result = pf.transform(x)
    print(f"this is the result: {result}")
    expected = torch.tensor(
        [
            [2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0],
            [3.0, 4.0, 9.0, 12.0, 16.0, 27.0, 36.0, 48.0, 64.0],
        ]
    )
    assert torch.allclose(result, expected)


def test_transform_without_fit():
    pf = PolynomialFeatures(degree=2, input_size=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Did you forget to call 'fit'?"):
        pf.transform(x)


def test_invalid_inputs():
    with pytest.raises(AssertionError, match="`degree` input must be greater than 0."):
        PolynomialFeatures(degree=0, input_size=2)

    with pytest.raises(
        AssertionError,
        match="`input_size`, which defines the number of features, for has to be greate than 0",
    ):
        PolynomialFeatures(degree=2, input_size=0)
