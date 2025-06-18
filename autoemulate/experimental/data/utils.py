import numpy as np
import torch
from autoemulate.experimental.types import (
    InputLike,
    OutputLike,
    TensorLike,
    TorchScalarDType,
)
from sklearn.utils.validation import check_X_y
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split


class ConversionMixin:
    """
    Mixin class to convert input data to pytorch Datasets and DataLoaders.
    """

    def _convert_to_dataset(
        self,
        x: InputLike,
        y: InputLike | None = None,
    ) -> Dataset:
        """
        Convert input data to pytorch Dataset.
        """
        # Convert input to Dataset if not already
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(x, torch.Tensor | np.ndarray) and isinstance(
            y, torch.Tensor | np.ndarray
        ):
            dataset = TensorDataset(x, y)
        elif isinstance(x, torch.Tensor | np.ndarray) and y is None:
            dataset = TensorDataset(x)
        elif isinstance(x, Dataset) and y is None:
            dataset = x
        elif isinstance(x, DataLoader) and y is None:
            dataset = x.dataset
        else:
            raise ValueError(
                f"Unsupported type for x ({type(x)}). Must be numpy array or PyTorch "
                "tensor."
            )

        return dataset

    def _convert_to_dataloader(
        self,
        x: InputLike,
        y: InputLike | None = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Convert input data to pytorch DataLoaders.
        """
        if isinstance(x, DataLoader) and y is None:
            dataloader = x
        elif isinstance(x, DataLoader) and y is not None:
            raise ValueError(
                f"Since x is already a DataLoader, expect y to be None, not {type(y)}."
            )
        else:
            dataset = self._convert_to_dataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def _convert_to_tensors(
        self,
        x: InputLike,
        y: InputLike | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Convert InputLike x, y to Tensor or tuple of Tensors.
        """
        dataset = self._convert_to_dataset(x, y)

        # Handle Subset of TensorDataset
        if isinstance(dataset, Subset):
            if isinstance(dataset.dataset, TensorDataset):
                tensors = dataset.dataset.tensors
                indices = dataset.indices

                # Use indexing to get subset tensors
                subset_tensors = tuple(tensor[indices] for tensor in tensors)
                dataset = TensorDataset(*subset_tensors)
            else:
                raise ValueError(
                    f"Subset must wrap a TensorDataset. Found {type(dataset.dataset)}."
                )

        if isinstance(dataset, TensorDataset):
            if len(dataset.tensors) > 2:
                raise ValueError(
                    f"Dataset must have 2 or fewer tensors. Found "
                    f"{len(dataset.tensors)}."
                )
            if len(dataset.tensors) == 2:
                x, y = dataset.tensors
                assert x.ndim == 2
                assert y.ndim in (1, 2)
                # Ensure always 2D tensors
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                return x.to(dtype), y.to(dtype)
            if len(dataset.tensors) == 1:
                (x,) = dataset.tensors
                assert x.ndim == 2
                return x.to(dtype)
            msg = "Number of tensors returned must be greater than zero."
            raise ValueError(msg)
        # Note: this error will never be raised, the same error is raised in
        # _convert_to_dataset
        raise ValueError(
            f"Unsupported type for x ({type(x)}). Must be numpy array or PyTorch "
            "tensor."
        )

    def _convert_to_numpy(
        self,
        x: InputLike,
        y: InputLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert InputLike x, y to tuple of numpy arrays.
        """
        if isinstance(x, np.ndarray) and (y is None or isinstance(y, np.ndarray)):
            return x, y

        result = self._convert_to_tensors(x, y)
        if isinstance(result, tuple):
            x_tensor, y_tensor = result
            x_np, y_np = x_tensor.cpu().numpy(), y_tensor.cpu().numpy()
            if (y_np.ndim == 2 and y_np.shape[1] == 1) or y_np.ndim == 1:
                y_np = y_np.ravel()  # Ensure y is 1-dimensional
                return check_X_y(x_np, y_np, multi_output=False, y_numeric=True)
            return check_X_y(x_np, y_np, multi_output=True, y_numeric=True)

        x_tensor = result
        return x_tensor.cpu().numpy(), None

    def _random_split(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        train_size: float = 0.8,
        test_size: float = 0.2,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Split Dataset into train/test DataLoaders.

        Parameters
        ----------
        dataset: Dataset
            The data to split.
        batch_size: int | None
            The DataLoader batch_size. If None, sets batch_size to lenth of training
            data. Defaults to None.
        """
        if train_size < 0.0 or train_size > 1.0 or test_size < 0.0 or test_size > 1.0:
            raise ValueError(
                f"Train size ({train_size}) and test size ({test_size}) must be "
                "specified as a proportion between 0 and 1"
            )
        if test_size + train_size != 1.0:
            raise ValueError(
                f"Train size ({train_size}) and test size ({test_size}) must sum to 1"
            )
        train, test = tuple(random_split(dataset, [train_size, test_size]))
        if batch_size is None:
            batch_size = len(train)
        train_loader = self._convert_to_dataloader(train, batch_size=batch_size)
        test_loader = self._convert_to_dataloader(test, batch_size=batch_size)
        return train_loader, test_loader

    @staticmethod
    def _normalize(x: TensorLike) -> tuple[TensorLike, TensorLike, TensorLike]:
        x_mean = x.mean(0, keepdim=True)
        x_std = x.std(0, keepdim=True)
        return (x - x_mean) / x_std, x_mean, x_std

    @staticmethod
    def _denormalize(
        x: TensorLike, x_mean: TensorLike, x_std: TensorLike
    ) -> TensorLike:
        return (x * x_std) + x_mean


class ValidationMixin:
    """
    Mixin class for validation methods.
    This class provides static methods for checking the types and shapes of
    input and output data, as well as validating specific tensor shapes.
    """

    @staticmethod
    def _check(x: TensorLike, y: TensorLike | None):
        """
        Check the types and shape are correct for the input data.
        Checks are equivalent to sklearn's check_array.
        """

        if not isinstance(x, TensorLike):
            raise ValueError(f"Expected x to be TensorLike, got {type(x)}")

        if y is not None and not isinstance(y, TensorLike):
            raise ValueError(f"Expected y to be TensorLike, got {type(y)}")

        # Check x
        if not torch.isfinite(x).all():
            msg = "Input tensor x contains non-finite values"
            raise ValueError(msg)
        if x.dtype not in TorchScalarDType:
            msg = (
                f"Input tensor x has unsupported dtype {x.dtype}. "
                "Expected float32, float64, int32, or int64."
            )
            raise ValueError(msg)

        # Check y if not None
        if y is not None:
            if not torch.isfinite(y).all():
                msg = "Input tensor y contains non-finite values"
                raise ValueError(msg)
            if y.dtype not in TorchScalarDType:
                msg = (
                    f"Input tensor y has unsupported dtype {y.dtype}. "
                    "Expected float32, float64, int32, or int64."
                )
                raise ValueError(msg)

        return x, y

    @staticmethod
    def _check_output(output: OutputLike):
        """
        Check the types and shape are correct
        for the output data.
        """
        if not isinstance(output, OutputLike):
            raise ValueError(f"Expected OutputLike, got {type(output)}")

        if isinstance(output, TensorLike) and output.ndim != 2:
            raise ValueError(f"Expected output to be 2D tensor, got {output.ndim}D")

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

    ### Validation methods from old utils.py ###
    ### Leave here in case want to restore/refactor later ###

    # def _ensure_2d(arr):
    #     """Ensure that arr is a 2D."""
    #     if arr.ndim == 1:
    #         arr = arr.reshape(-1, 1)
    #     return arr

    # def _ensure_1d_if_column_vec(arr):
    #     """Ensure that arr is 1D if shape is (n, 1)."""
    #     if arr.ndim == 2 and arr.shape[1] == 1:
    #         arr = arr.ravel()
    #     if arr.ndim > 2 or arr.ndim < 1:
    #         raise ValueError(
    #             f"arr should be 1D or 2D. Found {arr.ndim}D array with shape {arr.shape}"  # noqa: E501
    #         )
    #     return arr

    # def _check_cv(cv):
    #     """Ensure that cross-validation method is valid"""
    #     if cv is None:
    #         msg = "cross_validator cannot be None"
    #         raise ValueError(msg)
    #     if not isinstance(cv, KFold):
    #         msg = (
    #             "cross_validator should be an instance of KFold cross-validation. We do not "  # noqa: E501
    #             "currently support other cross-validation methods."
    #         )
    #         raise ValueError(
    #             msg
    #         )
    #     return cv
