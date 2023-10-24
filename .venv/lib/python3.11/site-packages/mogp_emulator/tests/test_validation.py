import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..validation import mahalanobis, standard_errors, pivoted_errors

targets1 = np.array([0.5, 2.1, 2.8])
targets2 = np.array([2.5, 2.9, 3.5])
mean1 = np.array([1.0, 2.0, 3.0])
mean2 = np.array([2.0, 3.0, 4.0])
err1 = mean1 - targets1
err2 = mean2 - targets2
unc1 = np.array([[0.1, 0.05, 0.02], [0.05, 0.2, 0.01], [0.02, 0.01, 0.15]])


def mock_predict(testing, unc=True, deriv=False, include_nugget=True, full_cov=False):
    mean = mean1
    if full_cov:
        unc = unc1
    else:
        unc = np.diag(unc1)
    return PredictResult(mean=mean, unc=unc, deriv=None)


def mock_predict_mogp(
    testing, unc=True, deriv=False, include_nugget=True, full_cov=False, processes=None
):
    mean = np.vstack([mean1, mean2])
    if full_cov:
        unc = np.stack([unc1] * 2)
    else:
        unc = np.vstack([np.diag(unc1)] * 2)
    return PredictResult(
        mean=mean,
        unc=unc,
        deriv=None,
    )


def mock_stats(*args, **kwds):
    return 2.0, 3.0


@pytest.fixture
def valid_inputs():
    return np.reshape(mean1, (-1, 1))


@pytest.fixture
def valid_targets():
    return targets1


@pytest.fixture
def valid_targets_mogp():
    return np.vstack([targets1, targets2])


def test_standard_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test standard errors for a GP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets)

    idx1 = np.array([1, 2, 0])

    assert_allclose(errors[0], err1[idx1] / np.sqrt(np.diag(unc1)[idx1]))
    assert_equal(errors[1], idx1)


def test_standard_errors_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test standard errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets_mogp)

    idx1 = np.array([1, 2, 0])
    errors_expect = [
        err1[idx1] / np.sqrt(np.diag(unc1))[idx1],
        err2[idx1] / np.sqrt(np.diag(unc1))[idx1],
    ]

    for e, e_expect in zip(errors, errors_expect):
        assert_allclose(e[0], e_expect)
        assert_equal(e[1], idx1)


def test_pivoted_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = pivoted_errors(gp, valid_inputs, valid_targets)

    idx1 = np.array([1, 2, 0])
    A = np.linalg.cholesky(unc1[idx1][:, idx1])
    b = np.linalg.solve(A, err1[idx1])

    assert_allclose(errors[0], b)
    assert_equal(errors[1], idx1)


def test_pivoted_errors_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    errors = pivoted_errors(gp, valid_inputs, valid_targets_mogp)

    idx1 = np.array([1, 2, 0])
    A = np.linalg.cholesky(unc1[idx1][:, idx1])
    b = [np.linalg.solve(A, e[idx1]) for e in [err1, err2]]

    for e, bval in zip(errors, b):
        assert_allclose(e[0], bval)
        assert_equal(e[1], idx1)


def test_mahalanobis_GP(valid_inputs, valid_targets, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)
    monkeypatch.setattr(
        "scipy.stats._distn_infrastructure.rv_generic.stats", mock_stats
    )

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    M = mahalanobis(gp, valid_inputs, valid_targets)

    M_expect = np.dot(err1, np.linalg.solve(unc1, err1))

    assert_allclose(M, M_expect)

    M = mahalanobis(gp, valid_inputs, valid_targets, scaled=True)

    assert_allclose(M, (M_expect - 2.0) / np.sqrt(3.0))


def test_mahalanobis_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)
    monkeypatch.setattr(
        "scipy.stats._distn_infrastructure.rv_generic.stats", mock_stats
    )

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    M = mahalanobis(gp, valid_inputs, valid_targets_mogp)

    M_expect = np.array([np.dot(e, np.linalg.solve(unc1, e)) for e in [err1, err2]])

    assert_allclose(M, M_expect)

    M = mahalanobis(gp, valid_inputs, valid_targets_mogp, scaled=True)

    assert_allclose(M, (M_expect - 2.0) / np.sqrt(3.0))
