import matplotlib.pyplot as plt
import numpy as np
import pytest
from autoemulate.core import plotting
from autoemulate.emulators.polynomials import PolynomialRegression
from autoemulate.emulators.random_forest import RandomForest

def test_plot_xy():
    X = np.linspace(0, 5, 10).reshape(-1, 1)
    y = X.flatten()
    y_pred = y * 1.1
    y_variance = np.abs(y * 0.1)

    metrics = {"R2": 0.5, "RMSE": 0.1}

    fig, ax = plt.subplots()
    plotting.plot_xy(
        X, y, y_pred, None, ax=ax, input_label="1", output_label="2", metrics=metrics
    )

    assert len(ax.containers) == 0
    assert len(ax.collections) > 0
    assert "R2" in [text.get_text() for text in ax.texts]
    assert "RMSE" in [text.get_text() for text in ax.texts]
    plt.close(fig)

    fig, ax = plt.subplots()
    plotting.plot_xy(
        X,
        y,
        y_pred,
        y_variance,
        ax=ax,
        input_label="1",
        output_label="2",
        metrics=metrics,
    )

    assert len(ax.containers) > 0
    assert len(ax.collections) > 0
    assert "R2" in [text.get_text() for text in ax.texts]
    assert "RMSE" in [text.get_text() for text in ax.texts]
    plt.close(fig)
    
def test_plot_xy_without_metrics():
    X = np.linspace(0, 5, 10).reshape(-1, 1)
    y = X.flatten()
    y_pred = y * 1.1

    fig, ax = plt.subplots()

    plotting.plot_xy(
        X, y, y_pred, None, ax=ax, input_label="1", output_label="2", metrics=None
    )

    assert len(ax.collections) > 0
    plt.close(fig)

def test_plot_xy_without_metrics():
    X = np.linspace(0, 5, 10).reshape(-1, 1)
    y = X.flatten()
    y_pred = y * 1.1

    fig, ax = plt.subplots()

    plotting.plot_xy(
        X,
        y,
        y_pred,
        None,
        ax=ax,
        input_label="1",
        output_label="2",
        metrics=None,
    )

    assert len(ax.collections) > 0
    plt.close(fig)

def test_plot_xy_accepts_single_metric():
    X = np.linspace(0,5,10).reshape(-1,1)
    y = X.flatten()
    y_pred = y*1.1

    fig, ax = plt.subplots()

    plotting.plot_xy(
        X,
        y,
        y_pred,
        None,
        ax=ax,
        input_label="1",
        output_label="2",
        metrics = {"MAE":0.25}
    )

    text_values = [text.get_text() for text in ax.texts]

    assert "MAE" in text_values
    assert len(ax.collections) > 0

    plt.close(fig)

def test_plot_xy_accepts_multiple_metrics():
    X = np.linspace(0,5,10).reshape(-1,1)
    y = X.flatten()
    y_pred = y * 1.1

    fig, ax = plt.subplots()

    plotting.plot_xy(
        X,
        y,
        y_pred,
        None,
        ax=ax,
        input_label = "1",
        output_label= "2",
        metrics = {"R2":0.5, "RMSE":0.1,"MAE":0.2}
    )

    text_values = [text.get_text() for text in ax.texts]

    assert "R2" in text_values
    assert "RMSE" in text_values
    assert "MAE" in text_values
    assert len(ax.collections) > 0

    plt.close(fig)