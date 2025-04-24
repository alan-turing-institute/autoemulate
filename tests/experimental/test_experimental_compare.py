from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.emulators.lightgbm import LightGBM


def test_compare(sample_data_y2d):
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, models=[GaussianProcessExact])
    results = ae.compare(10)
    print(results)


def test_compare_y1d(sample_data_y1d):
    x, y = sample_data_y1d
    # TODO: add handling when 1D
    y = y.reshape(-1, 1)
    ae = AutoEmulate(x, y, models=[GaussianProcessExact, LightGBM])
    results = ae.compare(10)
    print(results)
