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
    fig, ax = plt.subplots()
    plotting.plot_xy(
        X, y, y_pred, None, ax=ax, input_index=1, output_index=2, r2_score=0.5
    )
    assert len(ax.lines) > 0
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
