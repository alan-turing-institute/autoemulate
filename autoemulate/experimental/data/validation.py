from dataclasses import dataclass

import torch
from autoemulate.experimental.types import InputLike, OutputLike, TensorLike


@dataclass(kw_only=True)
class Base:
    """
    Base class for active learning simulation and emulation.

    Provides utility methods for tensor validation and design criteria computations.
    """

    @staticmethod
    def _check(x: InputLike, y: InputLike | None):
        # TODO: compare with InputTypeMixin and consider additional implementation
        ...

    @staticmethod
    def _check_output(output: OutputLike):
        # TODO: compare with InputTypeMixin and consider additional implementation
        ...

    @staticmethod
    def check_vector(X: TensorLike) -> TensorLike:
        """
        Validate that the input is a 1D TensorLike.

        Parameters
        ----------
        X : TensorLike
            Input tensor to validate.

        Returns
        -------
        TensorLike
            Validated 1D tensor.

        Raises
        ------
        ValueError
            If X is not a TensorLike or is not 1-dimensional.
        """
        if not isinstance(X, TensorLike):
            raise ValueError(f"Expected TensorLike, got {type(X)}")
        if X.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got {X.ndim}D")
        return X

    @staticmethod
    def check_matrix(X: TensorLike) -> TensorLike:
        """
        Validate that the input is a 2D TensorLike.

        Parameters
        ----------
        X : TensorLike
            Input tensor to validate.

        Returns
        -------
        TensorLike
            Validated 2D tensor.

        Raises
        ------
        ValueError
            If X is not a TensorLike or is not 2-dimensional.
        """
        if not isinstance(X, TensorLike):
            raise ValueError(f"Expected TensorLike, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {X.ndim}D")
        return X

    @staticmethod
    def check_pair(X: TensorLike, Y: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Validate that two tensors have the same number of rows.

        Parameters
        ----------
        X : TensorLike
            First tensor.
        Y : TensorLike
            Second tensor.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            The validated pair of tensors.

        Raises
        ------
        ValueError
            If X and Y do not have the same number of rows.
        """
        if X.shape[0] != Y.shape[0]:
            msg = "X and Y must have the same number of rows"
            raise ValueError(msg)
        return X, Y

    @staticmethod
    def check_covariance(Y: TensorLike, Sigma: TensorLike) -> TensorLike:
        """
        Validate and return the covariance matrix.

        Parameters
        ----------
        Y : TensorLike
            Output tensor.
        Sigma : TensorLike
            Covariance matrix, which may be full, diagonal, or a scalar per sample.

        Returns
        -------
        TensorLike
            Validated covariance matrix.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape relative to Y.
        """
        if (
            Sigma.shape == (Y.shape[0], Y.shape[1], Y.shape[1])
            or Sigma.shape == (Y.shape[0], Y.shape[1])
            or Sigma.shape == (Y.shape[0],)
        ):
            return Sigma
        msg = "Invalid covariance matrix shape"
        raise ValueError(msg)

    @staticmethod
    def trace(Sigma: TensorLike, d: int) -> TensorLike:
        """
        Compute the trace of the covariance matrix (A-optimal design criterion).

        Parameters
        ----------
        Sigma : TensorLike
            Covariance matrix (full, diagonal, or scalar).
        d : int
            Dimension of the output.

        Returns
        -------
        TensorLike
            The computed trace value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (d, d):
            return torch.diagonal(Sigma, dim1=1, dim2=2).sum(dim=1).mean()
        if Sigma.dim() == 2 and Sigma.shape[1] == d:
            return Sigma.sum(dim=1).mean()
        if Sigma.dim() == 1:
            return d * Sigma.mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def logdet(Sigma: TensorLike, dim: int) -> TensorLike:
        """
        Compute the log-determinant of the covariance matrix (D-optimal design
        criterion).

        Parameters
        ----------
        Sigma : TensorLike
            Covariance matrix (full, diagonal, or scalar).
        dim : int
            Dimension of the output.

        Returns
        -------
        TensorLike
            The computed log-determinant value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if len(Sigma.shape) == 3 and Sigma.shape[1:] == (dim, dim):
            return torch.logdet(Sigma).mean()
        if len(Sigma.shape) == 2 and Sigma.shape[1] == dim:
            return torch.sum(torch.log(Sigma), dim=1).mean()
        if len(Sigma.shape) == 1:
            return dim * torch.log(Sigma).mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def max_eigval(Sigma: TensorLike) -> TensorLike:
        """
        Compute the maximum eigenvalue of the covariance matrix (E-optimal design
        criterion).

        Parameters
        ----------
        Sigma : TensorLike
            Covariance matrix (full, diagonal, or scalar).

        Returns
        -------
        TensorLike
            The average maximum eigenvalue.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (Sigma.shape[1], Sigma.shape[1]):
            eigvals = torch.linalg.eigvalsh(Sigma)
            return eigvals[:, -1].mean()  # Eigenvalues are sorted in ascending order
        if Sigma.dim() == 2:
            return Sigma.max(dim=1).values.mean()
        if Sigma.dim() == 1:
            return Sigma.mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")
