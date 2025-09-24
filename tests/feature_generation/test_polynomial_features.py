import torch
from autoemulate.feature_generation.polynomial_features import PolynomialFeatures


def test_polynomial_features():
    poly = PolynomialFeatures(n_features=2, degree=2, include_bias=True)
    x = torch.tensor([[0, 1], [2, 3], [4, 5]])
    x_poly = poly.transform(x)

    # terms = bias, x1, x2, x1^2, x1*x2, x2^2
    expected_output = torch.tensor(
        [[1, 0, 1, 0, 0, 1], [1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]]
    )
    assert torch.allclose(x_poly, expected_output)

    poly_no_bias = PolynomialFeatures(n_features=2, degree=2, include_bias=False)
    x_poly_no_bias = poly_no_bias.transform(x)
    expected_output_no_bias = torch.tensor(
        [[0, 1, 0, 0, 1], [2, 3, 4, 6, 9], [4, 5, 16, 20, 25]]
    )
    assert torch.allclose(x_poly_no_bias, expected_output_no_bias)

    poly_degree_3 = PolynomialFeatures(n_features=2, degree=3, include_bias=True)
    x_poly_degree_3 = poly_degree_3.transform(x)

    # terms = bias, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
    expected_output_degree_3 = torch.tensor(
        [
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
            [1, 4, 5, 16, 20, 25, 64, 80, 100, 125],
        ]
    )
    assert torch.allclose(x_poly_degree_3, expected_output_degree_3)

    poly_n_features_3 = PolynomialFeatures(n_features=3, degree=2, include_bias=True)
    x3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x_poly_n_features_3 = poly_n_features_3.transform(x3)

    # terms = bias, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2
    expected_output_n_features_3 = torch.tensor(
        [
            [1, 1, 2, 3, 1, 2, 3, 4, 6, 9],
            [1, 4, 5, 6, 16, 20, 24, 25, 30, 36],
        ]
    )
    assert torch.allclose(x_poly_n_features_3, expected_output_n_features_3)
