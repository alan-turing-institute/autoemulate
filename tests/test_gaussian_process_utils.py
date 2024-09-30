import gpytorch
import pytest
import torch

from autoemulate.emulators.gaussian_process_utils.poly_mean import PolyMean


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
