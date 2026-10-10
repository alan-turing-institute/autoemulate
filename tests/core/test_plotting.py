import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from autoemulate.core import plotting
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.emulators.polynomials import PolynomialRegression
from autoemulate.emulators.random_forest import RandomForest


def test_display_figure_jupyter(monkeypatch):
    # Simulate Jupyter environment
    class DummyShell:
        __name__ = "ZMQInteractiveShell"

    # Simulate the get_ipython function to return a Jupyter shell
    monkeypatch.setattr(plotting, "get_ipython", lambda: DummyShell())
    fig = plt.figure()
    result = plotting.display_figure(fig)
    assert result is fig


def test_display_figure_terminal(monkeypatch):
    # Simulate a non-Jupyter environment
    # Here we just return None to simulate a terminal environment
    monkeypatch.setattr(plotting, "get_ipython", lambda: None)
    fig = plt.figure()
    # Should not raise
    result = plotting.display_figure(fig)
    assert result is fig


def test_plot_xy():
    X = np.linspace(0, 5, 10).reshape(-1, 1)
    y = X.flatten()
    y_pred = y * 1.1
    y_variance = np.abs(y * 0.1)

    # plot without error bars
    fig, ax = plt.subplots()
    plotting.plot_xy(
        X, y, y_pred, None, ax=ax, input_label="1", output_label="2", r2_score=0.5
    )
    # test for error bars
    assert len(ax.containers) == 0
    # test for scatter points
    assert len(ax.collections) > 0

    # plot with error bars
    fig, ax = plt.subplots()
    plotting.plot_xy(
        X, y, y_pred, y_variance, ax=ax, input_label="1", output_label="2", r2_score=0.5
    )
    assert len(ax.containers) > 0
    assert len(ax.collections) > 0


def test_plot_xy_bars_interval_width():
    X = np.linspace(0, 5, 10)
    y = X.copy()
    y_pred = np.full(10, 2.5)
    y_variance = np.full(10, 4.0)
    y_std = np.sqrt(4.0)

    fig, ax = plt.subplots()
    plotting.plot_xy(X, y, y_pred, y_variance, ax=ax, r2_score=0.9)

    _, _, barlinecols = ax.containers[0]
    seg = barlinecols[0].get_segments()[0]
    half_width = abs(seg[1][1] - seg[0][1]) / 2
    assert np.isclose(half_width, plotting.PREDICTION_INTERVAL_Z * y_std)


def test_plot_xy_fill_interval_width():
    X = np.linspace(0, 5, 10)
    y = X.copy()
    y_pred = np.full(10, 2.5)  # constant so band is flat
    y_variance = np.full(10, 4.0)
    y_std = np.sqrt(4.0)

    fig, ax = plt.subplots()
    plotting.plot_xy(X, y, y_pred, y_variance, ax=ax, r2_score=0.9, error_style="fill")

    verts = np.asarray(ax.collections[0].get_paths()[0].vertices)
    band_width = verts[:, 1].max() - verts[:, 1].min()
    assert np.isclose(band_width, 2 * plotting.PREDICTION_INTERVAL_Z * y_std)


@pytest.mark.parametrize(
    ("n_plots", "n_cols", "expected"),
    [
        (1, 3, (1, 1)),
        (2, 3, (1, 2)),
        (4, 3, (2, 3)),
        (7, 3, (3, 3)),
        (5, 2, (3, 2)),
    ],
)
def test_calculate_subplot_layout(n_plots, n_cols, expected):
    result = plotting.calculate_subplot_layout(n_plots, n_cols)
    assert result == expected


@pytest.mark.parametrize(
    ("model_class", "should_raise", "title"),
    [
        (PolynomialRegression, False, "Training Curve"),
        (RandomForest, True, "My Loss Plot"),
        (PolynomialRegression, False, None),
    ],
)
def test_plot_loss(model_class, should_raise, title):
    np.random.seed(42)
    x = np.random.rand(20, 2)
    y = (x[:, 0] + 2 * x[:, 1] > 1).astype(int)

    model = model_class(x, y)
    model.fit(x, y)

    if should_raise:
        with pytest.raises(AttributeError):
            fig, ax = plotting.plot_loss(model=model, title=title)
        return

    fig, ax = plotting.plot_loss(model=model, title=title)

    if title is not None:
        assert ax.get_title() == title

    assert ax.get_xlabel() == "Epochs"
    assert ax.get_ylabel() == "Train Loss"

    epochs = np.arange(1, len(model.loss_history) + 1)
    line_x, line_y = ax.get_lines()[0].get_data()

    assert np.allclose(line_x, epochs)
    assert np.allclose(line_y, model.loss_history)


def test_create_and_plot_slice_shows_interval_width_not_variance():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(30, 2))
    y = (np.sin(x[:, 0] * 3) + x[:, 1] ** 2).reshape(-1, 1)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = GaussianProcess(x_t, y_t)
    model.fit(x_t, y_t)

    fig, axs = plotting.create_and_plot_slice(
        model,
        {"a": (-1.0, 1.0), "b": (-1.0, 1.0)},
        param_pair=(0, 1),
        n_points=10,
    )

    ax_right = axs[0, 1]
    assert "95%" in ax_right.get_title()
    assert "variance" not in ax_right.get_title().lower()

    plotted = np.asarray(ax_right.get_images()[0].get_array())
    assert np.all(plotted >= 0)

    # recompute expected 95% interval width directly from the raw variance surface,
    # reshaped and transposed the same way _plot_2d_slice_with_fixed_params does
    # before imshow, so this also catches a wrong axis order, not just wrong values.
    _, var, _ = plotting.mean_and_var_surface(
        model, {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}, variables=["a", "b"], n_points=10
    )
    assert var is not None
    expected = 2 * plotting.PREDICTION_INTERVAL_Z * var.clamp(min=0).sqrt()
    expected_2d = expected.reshape(10, 10).T
    assert np.allclose(plotted, expected_2d.numpy())
