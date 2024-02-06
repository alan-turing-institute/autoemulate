import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from autoemulate.plotting import check_multioutput
from autoemulate.plotting import plot_best_fold_per_model
from autoemulate.plotting import plot_single_fold
from autoemulate.plotting import validate_inputs


# ------------------------------ test validate_inputs ------------------------------
def test_validate_inputs_with_empty_cv_results():
    cv_results = {}
    model_name = "model1"
    try:
        validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Run .compare() first."


def test_validate_inputs_with_invalid_model_name():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model2"
    try:
        validate_inputs(cv_results, model_name)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert (
            str(e)
            == "Model model2 not found. Available models are: dict_keys(['model1'])"
        )


def test_validate_inputs_with_valid_inputs():
    cv_results = {"model1": {"test_r2": 0.8}}
    model_name = "model1"
    validate_inputs(
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
    plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        plot_type="actual_vs_predicted",
        annotation="Test",
        output_index=0,
    )

    # Assert that the plot is displayed correctly
    # (You can add more specific assertions based on your requirements)
    assert ax.get_title() == "model1 - Test: 0"
    assert ax.texts[0].get_text() == "$R^2$ = 0.900"


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
    plot_single_fold(
        cv_results=cv_results,
        X=X_test,
        y=y_test,
        model_name="model1",
        fold_index=0,
        ax=ax,
        plot_type="residual_vs_predicted",
        annotation="Test",
        output_index=1,
    )

    # Assert that the plot is displayed correctly
    # (You can add more specific assertions based on your requirements)
    assert ax.get_title() == "model1 - Test: 0"
    assert ax.texts[0].get_text() == "$R^2$ = 0.900"
