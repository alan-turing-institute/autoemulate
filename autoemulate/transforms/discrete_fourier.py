import torch
from torch.distributions import Transform, constraints

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import TensorLike
from autoemulate.transforms.base import AutoEmulateTransform


class DiscreteFourierTransform(AutoEmulateTransform):
    """Discrete Fourier transform for frequency domain representation using matrices.

    This transform represents the DFT as a basis matrix transformation where:
    - Forward: y = x @ A^T (N features → 2*n_components features)
    - Inverse: x_reconstructed = y @ A (2*n_components features → N features)

    The basis matrix A has shape (2*n_components, N) where each pair of rows
    represents the real and imaginary parts of a selected frequency component.
    The output contains real-valued pairs [real_0, imag_0, real_1, imag_1, ...].
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = False

    def __init__(self, n_components: int, cache_size: int = 0):
        """Initialize the Discrete Fourier Transform.

        Parameters
        ----------
        n_components: int
            The number of frequency components to keep.
        cache_size: int, default=0
            Whether to cache previous transform. Set to 0 to disable caching. Set to
            1 to enable caching of the last single value. This might be useful for
            repeated expensive calls with the same input data but is by default
            disabled. See `PyTorch documentation <https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributions/transforms.py#L46-L89>`_
            for more details on caching.
        """
        Transform.__init__(self, cache_size=cache_size)
        self.n_components = n_components
        self.cache_size = cache_size  # Store for serialization

    def fit(self, x: TensorLike):
        """Fit the transform to the input data.

        Parameters
        ----------
        x: TensorLike
            Input data of shape (n_samples, n_features).
        """
        TorchDeviceMixin.__init__(self, device=x.device)
        self.check_tensor_is_2d(x)

        # Get input dimensions
        n_samples, n_features = x.shape

        # Create DFT matrices (real and imaginary parts)
        self._create_dft_matrix(n_features, x.device)

        # Apply DFT to find most important frequency components
        x_freq_real = x @ self.dft_real.T  # Real part
        x_freq_imag = x @ self.dft_imag.T  # Imaginary part

        # Compute power as |real + i*imag|^2 = real^2 + imag^2
        power = (x_freq_real.pow(2) + x_freq_imag.pow(2)).mean(0)

        # Select top n_components frequencies
        _, self.freq_indices = torch.topk(power, self.n_components)

        # Create the basis matrix A with real-valued representation
        # Shape: (2*n_components, n_features) - pairs of [real, imag] for each component
        selected_real = self.dft_real[self.freq_indices, :]
        selected_imag = self.dft_imag[self.freq_indices, :]

        # Interleave real and imaginary parts: [real_0, imag_0, real_1, imag_1, ...]
        self.components = torch.stack([selected_real, selected_imag], dim=1).reshape(
            2 * self.n_components, n_features
        )  # (2*n_components, n_features)

        self._is_fitted = True

    def _create_dft_matrix(self, n: int, device) -> None:
        """Create the DFT matrix for size n with real-valued representation.

        The DFT matrix W has elements W[k,n] = exp(-2πi * k * n / N) / sqrt(N)
        where k is the frequency index and n is the time index.

        Parameters
        ----------
        n : int
            Size of the DFT matrix (number of features).
        device: torch.device
            Device to create tensors on.

        Notes
        -----
        Sets self.dft_real and self.dft_imag as the real and imaginary parts
        of the DFT matrix, each of shape (n, n).
        """
        # Create indices
        k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n, 1)
        n_idx = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)

        # Compute the DFT matrix
        # W[k,n] = exp(-2πi * k * n / N) / sqrt(N)
        angle = -2 * torch.pi * k * n_idx / n
        real_part = torch.cos(angle) / torch.sqrt(torch.tensor(n, dtype=torch.float32))
        imag_part = torch.sin(angle) / torch.sqrt(torch.tensor(n, dtype=torch.float32))

        # Store both real and imaginary parts separately
        # We'll use them to create the real-valued transformation
        self.dft_real = real_part  # (n, n)
        self.dft_imag = imag_part  # (n, n)

    def _call(self, x: TensorLike):
        """Apply the discrete Fourier transform using matrix multiplication.

        This performs: y = x @ A^T where A is the basis matrix.

        Parameters
        ----------
        x: TensorLike
            Input data of shape (n_samples, n_features).

        Returns
        -------
        TensorLike
            Transformed data of shape (n_samples, 2*n_components) where each pair
            of columns represents the real and imaginary parts of a frequency
            component.
        """
        self._check_is_fitted()
        # Transform to frequency domain using selected components
        # Output shape: (n_samples, 2*n_components)
        return x @ self.components.T

    def _inverse(self, y: TensorLike):
        """Apply the inverse discrete Fourier transform using matrix multiplication.

        This performs: x_reconstructed = y @ A where A is the basis matrix.

        Parameters
        ----------
        y: TensorLike
            Tensor of shape (n_samples, 2*n_components) with real/imag pairs.

        Returns
        -------
        TensorLike
            Reconstructed tensor of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        # Reconstruct from frequency domain
        return y @ self.components  # (n_samples, n_features)

    def log_abs_det_jacobian(self, x: TensorLike, y: TensorLike):
        """Log abs det Jacobian not computable for n_components < d as not bijective."""
        _, _ = x, y
        msg = "log det Jacobian not computable for n_components < d as not bijective."
        raise RuntimeError(msg)

    @property
    def _basis_matrix(self) -> TensorLike:
        """Get the basis matrix A.

        Returns
        -------
        TensorLike
            The basis matrix of shape (2*n_components, n_features) containing
            the selected frequency components as real-valued pairs.
        """
        self._check_is_fitted()
        return self.components
