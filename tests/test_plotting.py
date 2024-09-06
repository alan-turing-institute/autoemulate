import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.plotting import _plot_model
from autoemulate.plotting import _plot_results
from autoemulate.plotting import _plot_single_fold
from autoemulate.plotting import _predict_with_optional_std
from autoemulate.plotting import _validate_inputs
from autoemulate.plotting import check_multioutput


@pytest.fixture
def ae_single_output():
    X, y = make_regression(n_samples=50, n_features=2, noise=0.5, random_state=42)
    em = AutoEmulate()
    em.setup(X, y, model_subset=["gpt", "rbf"])
    em.compare()
    return em


@pytest.fixture
def ae_multi_output():
    X, y = make_regression(
        n_samples=50, n_features=2, n_targets=2, noise=0.5, random_state=42
    )
    em = AutoEmulate()
    em.setup(X, y, model_subset=["gpt", "rbf"])
    em.compare()
    return em


# ------------------------------ test validate_inputs ------------------------------
def test_validate_inputs_with_empty_cv_results():
    cv_results = {}
    model_name = "model1"
    try:
        _validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Run .compare() first."


def test_validate_inputs_with_invalid_model_name():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model2"
    try:
        _validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert (
            str(e)
            == "Model model2 not found. Available models are: dict_keys(['model1'])"
        )


def test_validate_inputs_with_valid_inputs():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model1"
    _validate_inputs(
        cv_results, model_name
    )  # No exception should be raisedfrom autoemulate.plotting import check_multioutput


# ------------------------------ test check_multioutput ------------------------------
def test_check_multioutput_with_single_output():
    y = np.array([1, 2, 3, 4, 5])
    output_index = 0
    try:
        check_multioutput(y, output_index)
    except ValueError as e:
        assert False, f"Unexpected ValueError: {str(e)}"


def test_check_multioutput_with_multioutput():
    y = np.array([[1, 2, 3], [4, 5, 6]])
    output_index = 1
    try:
        check_multioutput(y, output_index)
    except ValueError as e:
        assert False, f"Unexpected ValueError: {str(e)}"


def test_check_multioutput_with_invalid_output_index():
    y = np.array([[1, 2, 3], [4, 5, 6]])
    output_index = 3
    try:
        check_multioutput(y, output_index)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert (
            str(e)
            == "Output index 3 is out of range. The index should be between 0 and 2."
        )


# ------------------------------ test _predict_with_optional_std --------------------
def test_predict_with_optional_std(ae_single_output):
    # test whether the function correctly returns None for rbf's std
    rbf = ae_single_output.get_model(name="rbf")
    X = ae_single_output.X
    y_pred, y_std = _predict_with_optional_std(rbf, X)
    assert y_pred.shape == (X.shape[0],)
    assert y_std is None

    # test whether the function correctly returns the std for gpt
    gpt = ae_single_output.get_model(name="gpt")
    y_pred, y_std = _predict_with_optional_std(gpt, X)
    assert y_pred.shape == (X.shape[0],)
    assert y_std.shape == (X.shape[0],)
    assert np.all(y_std >= 0)


# ------------------------------ test plot_single_fold ------------------------------
def test_plot_single_fold_with_single_output():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a mock cv_results dictionary
    cv_results = {
        "model1": {
            "indices": {"test": [[0, 1, 2, 3, 4]]},
            "estimator": [model],
            "test_r2": [0.9],
        }
    }

    # Create a mock axes object
    fig, ax = plt.subplots()

    # Call the plot_single_fold function
    _plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        plot="standard",
        annotation="Test",
        output_index=0,
    )

    # Assert that the plot is displayed correctly
    assert ax.get_title() == "model1 - Test: 0"
    # assert ax.texts[0].get_text() == "$R^2$ = 0.900"


def test_plot_single_fold_with_multioutput():
    # Generate synthetic data
    X, y = make_regression(
        n_samples=100, n_features=1, n_targets=2, noise=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a mock cv_results dictionary
    cv_results = {
        "model1": {
            "indices": {"test": [[0, 1, 2, 3, 4]]},
            "estimator": [model],
            "test_r2": [0.9],
        }
    }

    # Create a mock axes object
    fig, ax = plt.subplots()

    # Call the plot_single_fold function
    _plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        plot="residual",
        annotation="Test",
        output_index=1,
    )

    # Assert that the plot is displayed correctly
    assert ax.get_title() == "model1 - Test: 0"
    # assert ax.texts[0].get_text() == "$R^2$ = 0.900"


# ------------------------------ test cv plotting full ------------------------------
# def test_cv_plotting_full_single_output(ae_single_output):
#     ae_single_output.plot_results()


def test_cv_plotting_full_single_output(ae_single_output, monkeypatch):
    # Mock plt.show to do nothing
    monkeypatch.setattr(plt, "show", lambda: None)

    cv_results = ae_single_output.cv_results
    X = ae_single_output.X
    y = ae_single_output.y

    fig = _plot_results(cv_results, X, y)
    print(fig.axes)
    assert isinstance(fig, plt.Figure)


def test__plot_model_Xy():
    X, y = make_regression(n_samples=30, n_features=1, n_targets=1)
    model = RadialBasisFunctions()
    model.fit(X, y)
    # assert that plot_model runs
    # _plot_model(model, X, y, plot="Xy")
