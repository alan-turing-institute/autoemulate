import math

import numpy as np
import pyomo.environ as pyo
import pytest  # type: ignore[import]
import torch
from autoemulate import AutoEmulate
from autoemulate.convert.pyomo import pyomofy
from autoemulate.emulators.nn.mlp import MLP
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms.pca import PCATransform
from autoemulate.transforms.standardize import StandardizeTransform
from torch import nn


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
    m.x1 = pyo.Var(  # type: ignore[assignment]
        initialize=float(x_init[0]), bounds=(0, 1)
    )
    m.x2 = pyo.Var(  # type: ignore[assignment]
        initialize=float(x_init[1]), bounds=(0, 1)
    )
    return m


def _predict_like_pyomofy(result, x_tensor_2d: torch.Tensor) -> float:
    """Predict with the same object that pyomofy exports.

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
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.ReLU, "random_seed": 42},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    exprs = pyomofy(result, [x1_var, x2_var])
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
        x1_var.set_value(float(pt[0]))  # type: ignore[union-attr]
        x2_var.set_value(float(pt[1]))  # type: ignore[union-attr]
        pyomo_val = pyo.value(expr)

        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))

        # ReLU is exported as smooth Softplus_beta, so expect approximation error.
        assert pyomo_val == pytest.approx(torch_val, rel=2e-2, abs=1e-3)


@pytest.mark.parametrize(
    "activation_cls",
    [nn.SiLU, nn.Tanh, nn.Sigmoid, nn.Softplus],
)
def test_pyomofy_matches_autoemulate_predict_mlp_smooth_activations(activation_cls):
    x, x_tensor, y_tensor = _make_product_data(n_samples=200, seed=42)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": activation_cls, "random_seed": 42},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    expr = pyomofy(result.model, [x1_var, x2_var])[0]

    for pt in np.array([[0.2, 0.8], [0.6, 0.3]], dtype=np.float32):
        x1_var.set_value(float(pt[0]))  # type: ignore[union-attr]
        x2_var.set_value(float(pt[1]))  # type: ignore[union-attr]
        pyomo_val = pyo.value(expr)
        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))
        assert pyomo_val == pytest.approx(torch_val, rel=1e-5, abs=1e-5)


def test_pyomofy_matches_autoemulate_predict_polynomial_regression():
    x, x_tensor, y_tensor = _make_product_data(n_samples=300, seed=123)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["PolynomialRegression"],
        model_params={"degree": 2},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    exprs = pyomofy(result, [x1_var, x2_var])
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
        x1_var.set_value(float(pt[0]))  # type: ignore[union-attr]
        x2_var.set_value(float(pt[1]))  # type: ignore[union-attr]
        pyomo_val = pyo.value(expr)

        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))
        assert pyomo_val == pytest.approx(torch_val, rel=1e-7, abs=1e-7)


def test_pyomofy_supports_indexed_var_and_y_pca_transform():
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 1, size=(200, 2)).astype(np.float32)
    y = np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1], 2.0 * x[:, 0])).astype(
        np.float32
    )
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["PolynomialRegression"],
        model_params={"degree": 2},
        x_transforms_list=[[StandardizeTransform()]],
        y_transforms_list=[[StandardizeTransform(), PCATransform(n_components=2)]],
    )
    result = ae.best_result()
    assert result is not None

    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1], bounds=(0, 1))  # type: ignore[assignment]
    exprs = pyomofy(result, m.x)
    assert len(exprs) == 3

    pt = np.array([0.3, 0.7], dtype=np.float32)
    m.x[0].set_value(float(pt[0]))  # type: ignore[index]
    m.x[1].set_value(float(pt[1]))  # type: ignore[index]

    out = result.model.predict(torch.tensor(pt.reshape(1, -1)))
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    torch_vals = np.asarray(out).reshape(-1)

    for expr, expected in zip(exprs, torch_vals, strict=True):
        assert pyo.value(expr) == pytest.approx(float(expected), rel=1e-5, abs=1e-5)


def test_pyomofy_matches_autoemulate_predict_with_x_standardize_then_pca():
    x, x_tensor, y_tensor = _make_product_data(n_samples=400, seed=7)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 7},
        x_transforms_list=[[StandardizeTransform(), PCATransform(n_components=2)]],
        y_transforms_list=[[StandardizeTransform()]],
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    expr = pyomofy(result, [x1_var, x2_var])[0]

    test_pts = np.array(
        [
            [0.05, 0.95],
            [0.5, 0.5],
            [0.8, 0.3],
        ],
        dtype=np.float32,
    )
    for pt in test_pts:
        x1_var.set_value(float(pt[0]))  # type: ignore[union-attr]
        x2_var.set_value(float(pt[1]))  # type: ignore[union-attr]
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
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 99},
    )
    result = ae.best_result()
    assert result is not None

    class _UnsupportedTransform:
        pass

    # Use setattr to avoid type checker issues with assignment
    model = result.model
    model.x_transforms = [_UnsupportedTransform()]  # type: ignore[assignment]

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]

    with pytest.raises(NotImplementedError):
        pyomofy(result, [x1_var, x2_var])


def test_pyomofy_raises_on_unsupported_inner_emulator_transform():
    """Verify unsupported transforms on the inner emulator are rejected upfront."""
    x, x_tensor, y_tensor = _make_product_data(n_samples=200, seed=99)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 99},
    )
    result = ae.best_result()
    assert result is not None

    class _UnsupportedTransform:
        pass

    # Inject an unsupported transform at the inner emulator level
    emulator = result.model.model
    emulator.x_transform = _UnsupportedTransform()  # type: ignore[assignment]

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]

    with pytest.raises(NotImplementedError):
        pyomofy(result, [x1_var, x2_var])


def test_pyomofy_validation_errors():
    x, x_tensor, y_tensor = _make_product_data(n_samples=100, seed=11)
    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]

    # Invalid top-level model type
    with pytest.raises(TypeError, match="TransformedEmulator or Result"):
        pyomofy("invalid_model", [x1_var, x2_var])  # type: ignore[arg-type]

    # Unfitted model inside TransformedEmulator
    unfitted_mlp = MLP(x_tensor, y_tensor.unsqueeze(1))
    unfitted_mlp.is_fitted_ = False
    unfitted_wrapped = TransformedEmulator(
        x_tensor,
        y_tensor.unsqueeze(1),
        model=MLP,
        x_transforms=[],
        y_transforms=[],
    )
    unfitted_wrapped.model = unfitted_mlp
    with pytest.raises(ValueError, match="Model must be fitted"):
        pyomofy(unfitted_wrapped, [x1_var, x2_var])

    # Fitted model for remaining validation checks
    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 11},
    )
    result = ae.best_result()
    assert result is not None

    # Non-positive relu_beta
    with pytest.raises(ValueError, match="relu_beta must be positive"):
        pyomofy(result, [x1_var, x2_var], relu_beta=0)

    # Empty pyomo_vars
    with pytest.raises(ValueError, match="at least one Pyomo variable"):
        pyomofy(result, [])

    # Dimension mismatch (only 1 variable passed for 2D model)
    with pytest.raises(ValueError, match="dimension mismatch"):
        pyomofy(result, [x1_var])

    # Unsupported activation function in MLP
    assert isinstance(result.model.model, MLP)
    result.model.model.nn.append(nn.LeakyReLU())
    with pytest.raises(TypeError, match="Unsupported activation function"):
        pyomofy(result, [x1_var, x2_var])


def test_pyomofy_matches_emulator_with_inner_transforms_only():
    """Verify correctness when inner emulator transforms are non-trivial.

    We deliberately bypass the outer TransformedEmulator standardization by passing
    x_transforms_list=[[]] and y_transforms_list=[[]], so that the MLP's own
    x_transform/y_transform (standardize_x=True, standardize_y=True by default) are
    the only active transforms. In this case pyomofy must apply them explicitly,
    otherwise the Pyomo expression would be evaluated on the raw un-scaled inputs and
    produce wrong outputs.
    """
    x, x_tensor, y_tensor = _make_product_data(n_samples=300, seed=55)

    # No outer transforms — inner MLP transforms will now have non-trivial mean/std
    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.Sigmoid, "random_seed": 55},
        x_transforms_list=[[]],
        y_transforms_list=[[]],
    )
    result = ae.best_result()
    assert result is not None

    emulator = result.model.model
    # Confirm the inner transforms are non-trivial (mean not ≈ 0 or std not ≈ 1)
    if (
        emulator.x_transform is not None
        and emulator.x_transform.mean is not None
        and emulator.x_transform.std is not None
    ):
        inner_mean = emulator.x_transform.mean.abs().max().item()
        inner_std = (emulator.x_transform.std - 1).abs().max().item()
        assert inner_mean > 1e-3 or inner_std > 1e-3, (
            "Expected non-trivial inner x_transform when no outer transforms are used"
        )

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    expr = pyomofy(result, [x1_var, x2_var])[0]

    test_pts = np.array(
        [[0.1, 0.9], [0.4, 0.4], [0.8, 0.2]],
        dtype=np.float32,
    )
    for pt in test_pts:
        x1_var.set_value(float(pt[0]))  # type: ignore[union-attr]
        x2_var.set_value(float(pt[1]))  # type: ignore[union-attr]
        pyomo_val = pyo.value(expr)
        torch_val = _predict_like_pyomofy(result, torch.tensor(pt.reshape(1, -1)))
        assert pyomo_val == pytest.approx(torch_val, rel=1e-5, abs=1e-5)


def test_pyomofy_expr_is_smooth_for_relu_via_softplus_beta():
    x, x_tensor, y_tensor = _make_product_data(n_samples=250, seed=101)

    ae = AutoEmulate(
        x_tensor,
        y_tensor,
        show_progress_bar=False,
        models=["MLP"],
        model_params={"activation_cls": nn.ReLU, "random_seed": 101},
    )
    result = ae.best_result()
    assert result is not None

    m = _pyomo_model_with_vars(x_init=x[0])
    x1_var = m.x1  # type: ignore[assignment]
    x2_var = m.x2  # type: ignore[assignment]
    expr = pyomofy(result, [x1_var, x2_var], relu_beta=50)[0]

    x2_var.set_value(0.6)  # type: ignore[union-attr]
    xs = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    vals = []
    for x1 in xs:
        x1_var.set_value(x1)  # type: ignore[union-attr]
        v = pyo.value(expr)
        assert math.isfinite(v)  # type: ignore[arg-type]
        vals.append(v)

    assert vals[-1] >= vals[0]
