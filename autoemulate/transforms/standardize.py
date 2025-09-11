import torch
from torch.distributions import Transform, constraints

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import TensorLike
from autoemulate.transforms.base import AutoEmulateTransform


class StandardizeTransform(AutoEmulateTransform):
    """
    Standardize transform for normalizing data.

    This transform is effectively a composition of two AffineTransforms with a
    translation by the mean and a scaling by the inverse of the standard deviation.

    The transform assumes that the input data is a matrix of shape `(n, d)`, where `n`
    is the number of samples and `d` is the number of features.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self):
        # Cache not used as not expected to be beneficial for standardization
        Transform.__init__(self, cache_size=0)

    def fit(self, x: TensorLike):
        """Fit the StandardizeTransform to the data."""
        TorchDeviceMixin.__init__(self, device=x.device)

        self.check_tensor_is_2d(x)
        # Compute statistics without tracking gradients and avoid in-place edits
        with torch.no_grad():
            mean = x.mean(0, keepdim=True)
            std = x.std(0, keepdim=True)
            # Ensure std not zero to avoid division by zero errors since after
            # subtracting the mean the transform would be ~ 0 / 0 which is NaN.
            # See: https://github.com/scikit-learn/scikit-learn/blob/b24c328a304a46369e45de052e7fa695fb072efc/sklearn/preprocessing/_data.py#L92-L124
            eps = 10 * torch.finfo(std.dtype).eps
            std = torch.where(std < eps, torch.ones_like(std), std)
        # Treat stats as constants for downstream autograd
        self.mean = mean
        self.std = std
        self._is_fitted = True

    def _call(self, x):
        self._check_is_fitted()
        return (x - self.mean) / self.std

    def _inverse(self, y):
        self._check_is_fitted()
        return y * self.std + self.mean

    def log_abs_det_jacobian(self, x, y):
        """Compute the log absolute determinant of the Jacobian."""
        _, _ = x, y
        self._check_is_fitted()
        return torch.abs(self.std).log().sum()

    @property
    def _basis_matrix(self) -> TensorLike:
        return torch.eye(self.std.shape[1], device=self.device) * self.std
