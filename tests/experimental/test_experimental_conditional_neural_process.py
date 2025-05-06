import pytest
from autoemulate.experimental.device import check_torch_device_is_available
from autoemulate.experimental.emulators import CNPModule
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike


def test_cnp_module_fit_and_predict(sample_data_y1d, new_data_y1d):
    """
    Tests that the CNPModule can fit to data and make predictions.
    """
    x, y = sample_data_y1d

    cnp = CNPModule(x, y)

    cnp.fit(x, y)
    assert cnp.x_train is not None
    assert cnp.y_train is not None

    x2, _ = new_data_y1d
    y_pred = cnp.predict(x2)
    assert isinstance(y_pred, DistributionLike)

    # Drop the batch dimension from CNP output and final dimension
    assert y_pred.mean.shape == y.unsqueeze(-1).shape
    assert y_pred.variance.shape == y.unsqueeze(-1).shape


def test_cnp_module_fit_and_predict_multi_output(sample_data_y2d, new_data_y2d):
    """
    Tests that the CNPModule can fit to data and make predictions.
    """
    x, y = sample_data_y2d

    cnp = CNPModule(x, y)

    cnp.fit(x, y)
    assert cnp.x_train is not None
    assert cnp.y_train is not None

    x2, _ = new_data_y2d
    y_pred = cnp.predict(x2)
    assert isinstance(y_pred, DistributionLike)

    # Need to add both the final dimension and the batch dimension to
    # match output from cnp
    assert y_pred.mean.shape == y.shape
    assert y_pred.variance.shape == y.shape


def test_cnp_module_predict_fails_with_calling_fit_first(sample_data_y1d):
    """
    Check that calling predict without fitting raises a ValueError
    """
    x, y = sample_data_y1d

    cnp = CNPModule(x, y)

    with pytest.raises(
        ValueError,
        match=r"Model has not been trained. Please call fit\(\) before predict\(\).",
    ):
        cnp.predict(x)


def test_tune_gp(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=20)
    scores, configs = tuner.run(CNPModule)
    assert len(scores) == 20
    assert len(configs) == 20


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_device(sample_data_y2d, new_data_y2d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    cnp = CNPModule(x, y, device=device)
    cnp.fit(x, y)
    y_pred = cnp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)
