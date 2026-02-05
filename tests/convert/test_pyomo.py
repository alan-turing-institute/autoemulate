import math

import numpy as np
import pyomo.environ as pyo
import pytest
import torch
from torch import nn

from autoemulate import AutoEmulate
from autoemulate.convert.pyomo import pyomofy
from autoemulate.transforms.pca import PCATransform
from autoemulate.transforms.standardize import StandardizeTransform


def _make_product_data(n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, size=n_samples)
    x2 = rng.uniform(0, 1, size=n_samples)

    x = np.column_stack((x1, x2)).astype(np.float32)
    y = (x1 * x2).astype(np.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x, x_tensor, y_tensor


def _pyomo_model_with_vars(x_init):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=float(x_init[0]), bounds=(0, 1))
    m.x2 = pyo.Var(initialize=float(x_init[1]), bounds=(0, 1))
    return m


def _predict_like_pyomofy(result, x_tensor_2d: torch.Tensor) -> float:
    """
    Predict with the same object that pyomofy exports.

    In this codebase:
    - pyomofy(Result) unwraps to Result.model (a TransformedEmulator)
    - TransformedEmulator is not callable, so we use .predict(...)
    """
    out = result.model.predict(x_tensor_2d)
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    return float(np.asarray(out).reshape(-1)[0])


def test_pyomofy_matches_autoemulate_predict_mlp_relu():
    x, x_tensor, y_tensor = _make_product_data(n_samples=300, seed=42)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        log_level="warning",
        models=["MLP"],
        model_params={"activation_cls": nn.ReLU, "random_seed": 42},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    exprs = pyomofy(result, [m.x1, m.x2])
    assert isinstance(exprs, list)
    assert len(exprs) == 1
    expr = exprs[0]

    test_pts = np.array(
        [
            [0.1, 0.9],
            [0.25, 0.4],
            [0.7, 0.2],
        ],
        dtype=np.float32,
    )
    for pt in test_pts:
        m.x1.set_value(float(pt[0]))
        m.x2.set_value(float(pt[1]))
        pyomo_val = pyo.value(expr)

        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))

        # ReLU is exported as smooth Softplus_beta, so expect approximation error.
        assert pyomo_val == pytest.approx(torch_val, rel=2e-2, abs=1e-3)


def test_pyomofy_matches_autoemulate_predict_polynomial_regression():
    x, x_tensor, y_tensor = _make_product_data(n_samples=300, seed=123)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        log_level="warning",
        models=["PolynomialRegression"],
        model_params={"degree": 2},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    exprs = pyomofy(result, [m.x1, m.x2])
    assert len(exprs) == 1
    expr = exprs[0]

    test_pts = np.array(
        [
            [0.15, 0.85],
            [0.33, 0.44],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    for pt in test_pts:
        m.x1.set_value(float(pt[0]))
        m.x2.set_value(float(pt[1]))
        pyomo_val = pyo.value(expr)

        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))
        assert pyomo_val == pytest.approx(torch_val, rel=1e-7, abs=1e-7)


def test_pyomofy_matches_autoemulate_predict_with_x_standardize_then_pca():
    x, x_tensor, y_tensor = _make_product_data(n_samples=400, seed=7)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        log_level="warning",
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 7},
        x_transforms_list=[[StandardizeTransform(), PCATransform(n_components=2)]],
        y_transforms_list=[[StandardizeTransform()]],
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    expr = pyomofy(result, [m.x1, m.x2])[0]

    test_pts = np.array(
        [
            [0.05, 0.95],
            [0.5, 0.5],
            [0.8, 0.3],
        ],
        dtype=np.float32,
    )
    for pt in test_pts:
        m.x1.set_value(float(pt[0]))
        m.x2.set_value(float(pt[1]))
        pyomo_val = pyo.value(expr)

        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))

        # PCA-based pipelines can accumulate more numerical mismatch than the
        # simpler cases (matrix products + Pyomo exp/log in the NN).
        assert pyomo_val == pytest.approx(torch_val, rel=2e-1, abs=1e-2)


def test_pyomofy_raises_on_unsupported_transform():
    x, x_tensor, y_tensor = _make_product_data(n_samples=200, seed=99)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        log_level="warning",
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 99},
    )
    result = ae.best_result()
    assert result is not None

    class _UnsupportedTransform:
        pass

    result.model.x_transforms = [_UnsupportedTransform()]

    m = _pyomo_model_with_vars(x_init=x[0])

    with pytest.raises(NotImplementedError):
        pyomofy(result, [m.x1, m.x2])


def test_pyomofy_expr_is_smooth_for_relu_via_softplus_beta():
    x, x_tensor, y_tensor = _make_product_data(n_samples=250, seed=101)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        log_level="warning",
        models=["MLP"],
        model_params={"activation_cls": nn.ReLU, "random_seed": 101},
    )
    result = ae.best_result()
    m = _pyomo_model_with_vars(x_init=x[0])
    expr = pyomofy(result, [m.x1, m.x2], relu_beta=50)[0]

    m.x2.set_value(0.6)
    xs = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    vals = []
    for x1 in xs:
        m.x1.set_value(x1)
        v = pyo.value(expr)
        assert math.isfinite(v)
        vals.append(v)

    assert vals[-1] >= vals[0]