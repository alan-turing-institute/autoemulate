from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.emulators import ALL_EMULATORS


def test_compare(sample_data_y2d):
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)
    results = ae.compare(10)
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


def test_compare_y1d(sample_data_y1d):
    x, y = sample_data_y1d
    # TODO: add handling when 1D
    y = y.reshape(-1, 1)
    ae = AutoEmulate(x, y)
    results = ae.compare(10)
    print(results)
