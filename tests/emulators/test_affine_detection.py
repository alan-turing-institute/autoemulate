import torch
from autoemulate.emulators import GaussianProcessRBF
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms import PCATransform, StandardizeTransform, VAETransform


def _toy_data(y_dim=4, x_dim=5, n=40):
    torch.manual_seed(0)
    x = torch.randn(n, x_dim)
    w = torch.randn(x_dim, y_dim) * 0.5
    y = x @ w + 0.1 * torch.randn(n, y_dim)
    return x, y


def test_affine_detection_flags_linear_transforms():
    x, y = _toy_data(y_dim=6)

    # Standardize only (affine), then PCA (affine linear but non-bijective)
    y_transforms = [StandardizeTransform(), PCATransform(n_components=3)]

    em = TransformedEmulator(
        x,
        y,
        x_transforms=[StandardizeTransform()],
        y_transforms=y_transforms,
        model=GaussianProcessRBF,
        output_from_samples=False,
        full_covariance=False,
    )

    # Internal flag should mark all y transforms as affine
    assert em.all_y_transforms_affine is True


def test_predict_mean_uses_fast_linear_path(monkeypatch):
    x, y = _toy_data(y_dim=3)

    calls = {"delta_mean_only": 0}

    # Wrap the exact symbol imported in TransformedEmulator to count calls
    import autoemulate.emulators.transformed.base as te_base  # noqa: PLC0415

    orig = te_base.delta_method_mean_only

    def wrapped(*args, **kwargs):
        calls["delta_mean_only"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(te_base, "delta_method_mean_only", wrapped)

    em = TransformedEmulator(
        x,
        y,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), PCATransform(n_components=2)],
        model=GaussianProcessRBF,
        output_from_samples=False,
        full_covariance=False,
    )
    em.fit(x, y)
    _ = em.predict_mean(x[:5])

    # For affine y_transforms, predict_mean should use the fast tensor inversion path
    # and not call delta_method_mean_only.
    assert calls["delta_mean_only"] == 0


def test_vae_marked_nonlinear_and_mean_uses_delta(monkeypatch):
    x, y = _toy_data(y_dim=6)

    calls = {"delta_mean_only": 0}

    # Wrap the exact symbol imported in TransformedEmulator to count calls
    import autoemulate.emulators.transformed.base as te_base  # noqa: PLC0415

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
        x,
        y,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), vae],
        model=GaussianProcessRBF,
        output_from_samples=False,
        full_covariance=False,
    )

    # VAE is nonlinear, so affine flag should be False
    assert em.all_y_transforms_affine is False

    em.fit(x, y)
    _ = em.predict_mean(x[:5])

    # For nonlinear y_transforms, predict_mean should invoke delta_method_mean_only
    assert calls["delta_mean_only"] > 0
