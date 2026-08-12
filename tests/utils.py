import warnings
from contextlib import contextmanager, nullcontext

from autoemulate.transforms import PCATransform, VAETransform
from linear_operator.utils.warnings import NumericalWarning


@contextmanager
def ignore_expected_transformed_psd_repairs():
    """Ignore expected PSD repairs in transformed full-covariance tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"cov not p\.d\. - .*",
            category=NumericalWarning,
        )
        yield


def transformed_full_covariance_warning_context(
    output_from_samples, full_covariance, y_transforms
):
    """Return a warning context for expected transformed covariance repairs."""
    if (
        full_covariance
        and not output_from_samples
        and y_transforms is not None
        and any(isinstance(t, PCATransform | VAETransform) for t in y_transforms)
    ):
        return ignore_expected_transformed_psd_repairs()
    return nullcontext()
