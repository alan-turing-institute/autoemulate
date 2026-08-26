import torch
from torch.distributions import Transform, constraints

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import TensorLike
from autoemulate.transforms.base import AutoEmulateTransform


class DiscreteFourierTransform(AutoEmulateTransform):
    """Discrete Fourier transform for frequency domain dimensionality reduction.

    This transform works by:
    1. Computing FFT of input signals
    2. Selecting the most important frequency components based on power
    3. Zero-padding selected components for inverse transform
    4. Using IFFT for reconstruction

    The forward transform returns real-valued pairs representing the selected
    frequency components: [real_0, imag_0, real_1, imag_1, ...].
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
        self.n_features = n_features

        # Apply DFT to find most important frequency components
        x_fft = torch.fft.fft(x, dim=-1)  # Complex FFT

        # Compute power spectrum averaged across samples
        power = torch.abs(x_fft).pow(2).mean(0)  # (n_features,)

        # Select top n_components frequencies
        _, self.freq_indices = torch.topk(power, self.n_components)

        # Store indices for reconstruction - need to handle real/imaginary pairing
        # For real-valued signals, we need both positive and negative frequency
        # components to ensure real reconstruction
        self.selected_indices = self._get_paired_indices(self.freq_indices, n_features)

        self._is_fitted = True

    def _get_paired_indices(
        self, freq_indices: TensorLike, n_features: int
    ) -> TensorLike:
        """Get frequency indices including their complex conjugate pairs.

        For real-valued signals, we need both positive and negative frequency
        components to ensure real reconstruction after IDFT.

        Parameters
        ----------
        freq_indices: TensorLike
            Selected frequency indices
        n_features: int
            Number of features (length of signal)

        Returns
        -------
        TensorLike
            Paired indices including complex conjugates
        """
        paired_indices = []

        for idx in freq_indices:
            idx_val = idx.item()
            # Add the positive frequency
            paired_indices.append(idx_val)

            # Add the negative frequency (complex conjugate)
            # For DFT, negative frequencies are at n - idx (except for DC and Nyquist)
            if idx_val != 0 and idx_val != n_features // 2:
                neg_idx = n_features - idx_val
                paired_indices.append(neg_idx)

        # Remove duplicates and sort
        paired_indices = sorted(set(paired_indices))
        return torch.tensor(paired_indices, device=freq_indices.device)

    def _call(self, x: TensorLike):
        """Apply the discrete Fourier transform with frequency selection.

        This performs:
        1. FFT of input data
        2. Zero out all but selected frequency components
        3. Return only the selected components as real-valued pairs

        Parameters
        ----------
        x: TensorLike
            Input data of shape (..., n_features).
            Can handle arbitrary batch dimensions.

        Returns
        -------
        TensorLike
            Transformed data of shape (..., 2*n_selected_frequencies) where
            each pair represents real and imaginary parts of selected frequencies.
        """
        self._check_is_fitted()

        # Store original shape for batch dimensions
        original_shape = x.shape
        batch_shape = original_shape[:-1]

        # Flatten batch dimensions for processing
        x_flat = x.reshape(-1, self.n_features)

        # Step 1: Apply FFT to input data
        x_fft = torch.fft.fft(x_flat, dim=-1)  # (n_samples_flat, n_features) complex

        # Step 2: Extract only the selected frequency components
        selected_fft = x_fft[:, self.selected_indices]  # (n_samples_flat, n_selected)

        # Step 3: Convert to real-valued representation
        # Stack real and imaginary parts
        real_parts = selected_fft.real  # (n_samples_flat, n_selected)
        imag_parts = selected_fft.imag  # (n_samples_flat, n_selected)

        # Interleave real and imaginary parts
        stacked = torch.stack(
            [real_parts, imag_parts], dim=-1
        )  # (n_samples_flat, n_selected, 2)
        # (n_samples_flat, 2*n_selected)
        result_flat = stacked.reshape(x_flat.shape[0], -1)

        # Reshape back to original batch dimensions
        output_shape = (*batch_shape, 2 * len(self.selected_indices))
        return result_flat.reshape(output_shape)

    def _inverse(self, y: TensorLike):
        """Apply the inverse discrete Fourier transform.

        This performs:
        1. Reshape real-valued pairs back to complex representation
        2. Reconstruct full frequency domain by zero-padding
        3. Apply IFFT to get back original signal

        Parameters
        ----------
        y: TensorLike
            Tensor of shape (..., 2*n_selected_frequencies) with real/imag pairs.
            Can handle arbitrary batch dimensions.

        Returns
        -------
        TensorLike
            Reconstructed tensor of shape (..., n_features).
        """
        self._check_is_fitted()

        # Store original shape for batch dimensions
        original_shape = y.shape
        batch_shape = original_shape[:-1]
        n_selected = len(self.selected_indices)

        # The input y should have last dimension 2*n_selected
        # where each pair represents (real, imag) for each selected frequency
        expected_features = 2 * n_selected

        if original_shape[-1] != expected_features:
            raise ValueError(
                f"Expected input with {expected_features} features "
                f"(2 * {n_selected} selected frequencies) in last dimension, "
                f"but got shape {original_shape}"
            )

        # Flatten batch dimensions for processing
        y_flat = y.reshape(-1, expected_features)
        n_samples = y_flat.shape[0]

        # Step 1: Reshape back to real and imaginary parts
        y_reshaped = y_flat.reshape(n_samples, n_selected, 2)
        real_parts = y_reshaped[:, :, 0]  # (n_samples, n_selected)
        imag_parts = y_reshaped[:, :, 1]  # (n_samples, n_selected)

        # Step 2: Reconstruct complex representation
        selected_fft = torch.complex(real_parts, imag_parts)  # (n_samples, n_selected)

        # Step 3: Zero-pad to full frequency domain
        full_fft = torch.zeros(
            n_samples, self.n_features, dtype=torch.complex64, device=y.device
        )
        # Expand selected_indices to match batch dimension
        indices = self.selected_indices.unsqueeze(0).expand(n_samples, -1)

        # Use scatter instead of in-place indexing for vmap compatibility
        full_fft = full_fft.scatter(1, indices, selected_fft)

        # Step 4: Apply IFFT to reconstruct signal
        result_flat = torch.fft.ifft(full_fft, dim=-1).real

        # Step 5: Reshape back to original batch dimensions
        output_shape = (*batch_shape, self.n_features)
        return result_flat.reshape(output_shape)

    def log_abs_det_jacobian(self, x: TensorLike, y: TensorLike):
        """Log abs det Jacobian not computable for n_components < d as not bijective."""
        _, _ = x, y
        msg = (
            "log det Jacobian not computable for dimensionality reduction transform. "
            "This transform is not bijective when 2*n_components < n_features."
        )
        raise RuntimeError(msg)

    def forward_shape(self, shape):
        """Compute the forward shape transformation.

        For DFT: (batch_shape, ..., n_features) ->
                 (batch_shape, ..., 2*n_selected_frequencies)
        where n_selected_frequencies depends on the selected frequency components.
        """
        self._check_is_fitted()

        # Use actual selected indices
        n_output_features = 2 * len(self.selected_indices)

        if len(shape) == 0:
            return torch.Size([n_output_features])
        return shape[:-1] + torch.Size([n_output_features])

    def inverse_shape(self, shape):
        """Compute the inverse shape transformation.

        For DFT: (batch_shape, ..., 2*n_selected_frequencies) ->
                 (batch_shape, ..., n_features)
        """
        self._check_is_fitted()
        if len(shape) == 0:
            return torch.Size([self.n_features])
        return shape[:-1] + torch.Size([self.n_features])
