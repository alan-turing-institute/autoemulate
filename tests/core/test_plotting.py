import matplotlib.pyplot as plt
import numpy as np
import pytest
from autoemulate.core import plotting


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


from autoemulate.emulators.polynomials import PolynomialRegression
from autoemulate.emulators.random_forest import RandomForest


@pytest.mark.parametrize(
    "model_class,should_raise",
    [
        (PolynomialRegression, False),
        (RandomForest, True),
    ],
)
def test_plot_loss(model_class, should_raise):
    np.random.seed(42)
    x = np.random.rand(20, 2)
    y = (x[:, 0] + 2 * x[:, 1] > 1).astype(int)

    model = model_class(x, y)
    model.fit(x, y)

    if should_raise:
        with pytest.raises(AttributeError):
            plotting.plot_loss(model=model, title="Train Loss")
    else:
        fig, ax = plotting.plot_loss(model=model, title="Train Loss")
        assert ax.get_title() == "Train Loss"
        assert ax.get_xlabel() == "Epochs"
        assert ax.get_ylabel() == "Train Loss"
        assert len(model.loss_history) > 0
        assert np.allclose(
            ax.get_lines()[0].get_data(),
            (range(1, len(model.loss_history) + 1), model.loss_history),
        )
