import pytest
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.device import check_torch_device_is_available
from autoemulate.experimental.emulators import ALL_EMULATORS


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_compare(sample_data_y2d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, device=device)
    results = ae.compare(2)
    print(results)


def test_compare_user_models(sample_data_y2d, recwarn):
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, models=ALL_EMULATORS)
    results = ae.compare(1)
    print(results)
    assert len(recwarn) == 1
    assert str(recwarn.pop().message) == (
        "Model (<class 'autoemulate.experimental.emulators.lightgbm.Li"
        "ghtGBM'>) is not multioutput but the data is multioutput. Skipping model "
        "(<class 'autoemulate.experimental.emulators.lightgbm.LightGBM'>)..."
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_compare_y1d(sample_data_y1d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    # TODO: add handling when 1D
    y = y.reshape(-1, 1)
    ae = AutoEmulate(x, y)
    results = ae.compare(4)
    print(results)
