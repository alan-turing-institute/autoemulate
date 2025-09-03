import torch
from autoemulate.emulators import GaussianProcess
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms import PCATransform, StandardizeTransform, VAETransform


def _toy_data(y_dim=4, x_dim=5, n=40):
    torch.manual_seed(0)
    X = torch.randn(n, x_dim)
    W = torch.randn(x_dim, y_dim) * 0.5
    Y = X @ W + 0.1 * torch.randn(n, y_dim)
    return X, Y


def test_affine_detection_flags_linear_transforms():
    X, Y = _toy_data(y_dim=6)

    # Standardize only (affine), then PCA (affine linear but non-bijective)
    y_transforms = [StandardizeTransform(), PCATransform(n_components=3)]

    em = TransformedEmulator(
        X,
        Y,
        x_transforms=[StandardizeTransform()],
        y_transforms=y_transforms,
        model=GaussianProcess,
        output_from_samples=False,
        full_covariance=False,
    )

    # Internal flag should mark all y transforms as affine
    assert em.linear_y_transforms is True


def test_predict_mean_uses_fast_linear_path(monkeypatch):
    X, Y = _toy_data(y_dim=3)

    calls = {"delta_mean_only": 0}

    # Wrap the exact symbol imported in TransformedEmulator to count calls
    import autoemulate.emulators.transformed.base as te_base

    orig = te_base.delta_method_mean_only

    def wrapped(*args, **kwargs):
        calls["delta_mean_only"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(te_base, "delta_method_mean_only", wrapped)

    em = TransformedEmulator(
        X,
        Y,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), PCATransform(n_components=2)],
        model=GaussianProcess,
        output_from_samples=False,
        full_covariance=False,
    )
    em.fit(X, Y)
    _ = em.predict_mean(X[:5])

    # For affine y_transforms, predict_mean should use the fast tensor inversion path
    # and not call delta_method_mean_only.
    assert calls["delta_mean_only"] == 0


def test_vae_marked_nonlinear_and_mean_uses_delta(monkeypatch):
    X, Y = _toy_data(y_dim=6)

    calls = {"delta_mean_only": 0}

    # Wrap the exact symbol imported in TransformedEmulator to count calls
    import autoemulate.emulators.transformed.base as te_base

    orig = te_base.delta_method_mean_only

    def wrapped(*args, **kwargs):
        calls["delta_mean_only"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(te_base, "delta_method_mean_only", wrapped)

    # Use a very small VAE for speed
    vae = VAETransform(
        latent_dim=2,
        hidden_layers=[8],
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=0,
        verbose=False,
    )

    em = TransformedEmulator(
        X,
        Y,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), vae],
        model=GaussianProcess,
        output_from_samples=False,
        full_covariance=False,
    )

    # VAE is nonlinear, so affine flag should be False
    assert em.linear_y_transforms is False

    em.fit(X, Y)
    _ = em.predict_mean(X[:5])

    # For nonlinear y_transforms, predict_mean should invoke delta_method_mean_only
    assert calls["delta_mean_only"] > 0
