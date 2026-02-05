import pyomo.environ as pyo
from torch import nn

from autoemulate.core.results import Result
from autoemulate.emulators.nn.mlp import MLP
from autoemulate.emulators.polynomials import PolynomialRegression
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms.pca import PCATransform
from autoemulate.transforms.standardize import StandardizeTransform

_SUPPORTED_TRANSFORMS = (StandardizeTransform, PCATransform)

# Default beta for the Softplus approximation when the model was trained with ReLU.
# As beta -> inf, Softplus_beta -> ReLU. beta=50 is sharp enough to closely track
# ReLU while remaining smooth and differentiable everywhere.
_DEFAULT_RELU_BETA = 50


def pyomofy(model: TransformedEmulator | Result, pyomo_vars, relu_beta: int = _DEFAULT_RELU_BETA):
    """Convert a fitted emulator into Pyomo expressions.

    Parameters
    ----------
    model : TransformedEmulator | Result
        A fitted TransformedEmulator or Result wrapping one.
    pyomo_vars : list or pyomo.core.base.var.Var or pyomo.core.base.var.IndexedVar
        Input decision variables in the same order used for training.
    relu_beta : int
        Sharpness parameter for the Softplus approximation used when the model
        was trained with ReLU. Softplus_beta(x) = (1/beta) * log(1 + exp(beta*x)).
        As beta increases the approximation gets closer to ReLU but less smooth.
        Defaults to 50.

    Returns
    -------
    list
        Pyomo expressions, one per output dimension.

    Raises
    ------
    TypeError
        If model is not a TransformedEmulator or Result.
    NotImplementedError
        If model type is not PolynomialRegression or MLP.
    ValueError
        If the model has not been fitted.
    """
    if isinstance(model, Result):
        model = model.model
    if not isinstance(model, TransformedEmulator):
        raise TypeError("Input model must be a TransformedEmulator or Result instance.")

    # The actual emulator lives inside TransformedEmulator.model
    emulator = model.model
    if not isinstance(emulator, (PolynomialRegression, MLP)):
        raise NotImplementedError(
            "Currently, only PolynomialRegression and MLP models are supported for Pyomo conversion."
        )
    if not emulator.is_fitted_:
        raise ValueError("Model must be fitted before conversion to Pyomo.")

    # Flatten possible IndexedVar or VarSet to an ordered list
    if hasattr(pyomo_vars, "component_data_objects"):
        var_list = [v for v in pyomo_vars.component_data_objects(sort=True)]
    else:
        var_list = list(pyomo_vars)

    # --- Validate all transforms upfront before doing any work ---
    _validate_transforms(model.x_transforms)
    _validate_transforms(model.y_transforms)

    # --- Apply x transforms in forward order ---
    activations = _apply_x_transforms(model.x_transforms, var_list)

    # --- Emulator-specific forward pass ---
    if isinstance(emulator, PolynomialRegression):
        activations = _polynomial_forward(emulator, activations)
    elif isinstance(emulator, MLP):
        activations = _mlp_forward(emulator, activations, relu_beta)

    # --- Apply y transforms in inverse order ---
    return _apply_y_inverse_transforms(model.y_transforms, activations)


# ---------------------------------------------------------------------------
# Transform validation and application
# ---------------------------------------------------------------------------


def _validate_transforms(transforms: list):
    """Check every transform in the list is supported. Fail fast before any work."""
    for t in transforms:
        if not isinstance(t, _SUPPORTED_TRANSFORMS):
            supported = ", ".join(cls.__name__ for cls in _SUPPORTED_TRANSFORMS)
            msg = (
                f"Unsupported transform: '{type(t).__name__}'. "
                f"Only {supported} are supported for Pyomo conversion."
            )
            raise NotImplementedError(msg)


def _apply_x_transforms(transforms: list, activations: list) -> list:
    """Apply x transforms in forward order as Pyomo expressions."""
    for t in transforms:
        if isinstance(t, StandardizeTransform):
            activations = _standardize_forward(t, activations)
        elif isinstance(t, PCATransform):
            activations = _pca_forward(t, activations)
    return activations


def _apply_y_inverse_transforms(transforms: list, activations: list) -> list:
    """Apply y transforms in inverse order as Pyomo expressions.

    Mirrors TransformedEmulator._inv_transform_y_gaussian:
        for transform in self.y_transforms[::-1]: ...
    """
    for t in reversed(transforms):
        if isinstance(t, StandardizeTransform):
            activations = _standardize_inverse(t, activations)
        elif isinstance(t, PCATransform):
            activations = _pca_inverse(t, activations)
    return activations


# ---------------------------------------------------------------------------
# StandardizeTransform: forward and inverse
# ---------------------------------------------------------------------------


def _standardize_forward(transform: StandardizeTransform, activations: list) -> list:
    """(x - mean) / std"""
    mean = transform.mean.detach().cpu().numpy().flatten().tolist()
    std = transform.std.detach().cpu().numpy().flatten().tolist()

    if len(mean) == 1 and len(activations) > 1:
        mean = mean * len(activations)
        std = std * len(activations)

    return [(activations[i] - mean[i]) / std[i] for i in range(len(activations))]


def _standardize_inverse(transform: StandardizeTransform, activations: list) -> list:
    """y * std + mean"""
    mean = transform.mean.detach().cpu().numpy().flatten().tolist()
    std = transform.std.detach().cpu().numpy().flatten().tolist()

    if len(mean) == 1 and len(activations) > 1:
        mean = mean * len(activations)
        std = std * len(activations)

    return [activations[i] * std[i] + mean[i] for i in range(len(activations))]


# ---------------------------------------------------------------------------
# PCATransform: forward and inverse
# ---------------------------------------------------------------------------


def _pca_forward(transform: PCATransform, activations: list) -> list:
    """(x - mean) @ components  ->  dot product per component."""
    mean = transform.mean.detach().cpu().numpy().flatten().tolist()
    # components shape: (n_features, n_components)
    components = transform.components.detach().cpu().numpy()

    # Center first
    centered = [activations[i] - mean[i] for i in range(len(activations))]

    # Each output component j = sum_i( centered[i] * components[i, j] )
    n_components = components.shape[1]
    return [
        sum(float(components[i, j]) * centered[i] for i in range(len(centered)))
        for j in range(n_components)
    ]


def _pca_inverse(transform: PCATransform, activations: list) -> list:
    """y @ components.T + mean  ->  dot product per original feature."""
    mean = transform.mean.detach().cpu().numpy().flatten().tolist()
    # components shape: (n_features, n_components)
    components = transform.components.detach().cpu().numpy()

    n_features = components.shape[0]
    # Each output feature i = sum_j( activations[j] * components[i, j] ) + mean[i]
    return [
        sum(float(components[i, j]) * activations[j] for j in range(len(activations))) + mean[i]
        for i in range(n_features)
    ]


# ---------------------------------------------------------------------------
# PolynomialRegression forward pass
# ---------------------------------------------------------------------------


def _polynomial_forward(model: PolynomialRegression, activations: list) -> list:
    """Expand polynomial features and apply the linear layer in Pyomo expressions.

    Mirrors PolynomialRegression.forward():
        x_poly = self.poly(x)
        return self.linear(x_poly)
    """
    # 1. Expand polynomial features using the stored powers matrix
    #    model.poly._powers has shape (n_output_features, n_input_features)
    #    Each row defines one monomial: x0^p0 * x1^p1 * ...
    powers = model.poly._powers.detach().cpu().numpy()
    poly_features = []
    for row in powers:
        term = 1
        for feat_idx, power in enumerate(row):
            if power > 0:
                term = term * activations[feat_idx] ** int(power)
        poly_features.append(term)

    # 2. Single linear layer (no bias) on the expanded features
    weight = model.linear.weight.detach().cpu().numpy()  # shape: (n_outputs, n_poly_features)
    out_exprs = []
    for j in range(weight.shape[0]):
        lin = sum(float(weight[j, k]) * poly_features[k] for k in range(len(poly_features)))
        out_exprs.append(lin)

    return out_exprs


# ---------------------------------------------------------------------------
# MLP forward pass
# ---------------------------------------------------------------------------


def _silu_expr(val):
    """SiLU / Swish: x * sigmoid(x)"""
    return val / (1 + pyo.exp(-val))


def _sigmoid_expr(val):
    """Standard Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + pyo.exp(-val))


def _softplus_expr(val):
    """Softplus: log(1 + exp(x))  — standard form, beta=1"""
    return pyo.log(1 + pyo.exp(val))


def _softplus_beta_expr(val, beta: float):
    """Softplus with tunable sharpness: (1/beta) * log(1 + exp(beta * x))

    As beta -> inf this converges to ReLU. Used as a drop-in replacement
    when the model was trained with ReLU.
    """
    return (1.0 / beta) * pyo.log(1 + pyo.exp(beta * val))


# Activation map for natively supported activations (no extra args needed).
# ReLU is handled separately in _mlp_forward via Softplus_beta approximation.
_ACTIVATION_MAP = {
    nn.SiLU: _silu_expr,
    nn.Tanh: pyo.tanh,
    nn.Sigmoid: _sigmoid_expr,
    nn.Softplus: _softplus_expr,
}

# Full supported list: includes ReLU (auto-approximated) + everything in _ACTIVATION_MAP
_SUPPORTED_ACTIVATIONS = (*_ACTIVATION_MAP.keys(), nn.ReLU)


def _mlp_forward(model: MLP, activations: list, relu_beta: float) -> list:
    """Unroll the MLP layers into Pyomo expressions.

    Mirrors MLP.forward():
        return self.nn(x)

    ReLU layers are automatically replaced by Softplus_beta using the provided
    relu_beta sharpness parameter.
    """
    for module in model.nn:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None

            out_exprs = []
            for j in range(weight.shape[0]):
                lin = sum(float(weight[j, k]) * activations[k] for k in range(len(activations)))
                if bias is not None:
                    lin += float(bias[j])
                out_exprs.append(lin)
            activations = out_exprs

        elif isinstance(module, nn.Dropout):
            # Dropout is only active during training — skip at inference
            continue

        elif isinstance(module, nn.ReLU):
            # ReLU is not smooth — replace with Softplus_beta approximation
            activations = [_softplus_beta_expr(a, relu_beta) for a in activations]

        elif type(module) in _ACTIVATION_MAP:
            act_fn = _ACTIVATION_MAP[type(module)]
            activations = [act_fn(a) for a in activations]

        else:
            supported = ", ".join(act.__name__ for act in _SUPPORTED_ACTIVATIONS)
            msg = (
                f"Unsupported activation function: '{type(module).__name__}'\n"
                f"Supported activations: {supported}\n"
            )
            raise TypeError(msg)

    return activations