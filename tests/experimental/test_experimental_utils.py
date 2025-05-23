import numpy as np
import pytest
import torch
from autoemulate.experimental.data.utils import (
    ConversionMixin,
    Standardizer,
    ValidationMixin,
)
from autoemulate.experimental.types import NumpyLike
from torch.utils.data import DataLoader, Subset, TensorDataset


class TestConversionMixin:
    """
    Class to test the ConversionMixin class.
    """

    def setup_method(self):
        """
        Define the ConversionMixin instance.
        """
        self.mixin = ConversionMixin()

    def test_convert_to_dataset_numpy(self):
        """
        Test converting numpy arrays to a Dataset.
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        dataset = self.mixin._convert_to_dataset(X, y)

        # TODO: is the distinction between TensorDataset and Dataset important?
        assert isinstance(dataset, TensorDataset)
        assert torch.equal(dataset.tensors[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.equal(dataset.tensors[1], torch.tensor([1.0, 2.0]))

    def test_convert_to_dataset_tensor(self):
        """
        Test converting torch tensors to a Dataset.
        """
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([1.0, 2.0])
        dataset = self.mixin._convert_to_dataset(X, y)

        assert isinstance(dataset, TensorDataset)
        assert torch.equal(dataset.tensors[0], X)
        assert torch.equal(dataset.tensors[1], y)

    def test_convert_to_dataset_invalid(self):
        """
        Test invalid input to _convert_to_dataset.
        """
        X = "invalid input"
        with pytest.raises(ValueError, match="Unsupported type for x"):
            self.mixin._convert_to_dataset(X)  # type: ignore - test for invalid type

    def test_convert_to_tensors(self):
        """
        Test converting input data to tensors.
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        x_tensor, y_tensor = self.mixin._convert_to_tensors(X, y)

        assert isinstance(x_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert torch.equal(x_tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.equal(y_tensor, torch.tensor([[1.0], [2.0]]))

    def test_convert_to_tensors_subset(self):
        """
        Test converting a Subset of a TensorDataset to tensors.
        """
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        dataset = TensorDataset(X, y)
        subset = Subset(dataset, [0, 1])
        x_tensor, y_tensor = self.mixin._convert_to_tensors(subset)

        assert torch.equal(x_tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.equal(y_tensor, torch.tensor([[1.0], [2.0]]))

    # def test_convert_to_tensors_invalid(self):
    #     """
    #     Test invalid input to _convert_to_tensors.
    #     """
    #     # TODO: should this just raise the error from test_convert_to_dataset_invalid
    #     # TODO: can we ensure all warning messages are tested?
    #     X = "invalid input"
    #     msg = f"Unsupported type for dataset ({type(X)}). Must be TensorDataset."
    #     with pytest.raises(ValueError, match=msg):
    #         self.mixin._convert_to_tensors(X)  # type: ignore - test for invalid type

    def test_convert_numpy_array(self):
        """
        Test converting a numpy array to a DataLoader object.
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        dataloader = self.mixin._convert_to_dataloader(
            X, y, batch_size=2, shuffle=False
        )

        assert isinstance(dataloader, DataLoader)
        batches = list(dataloader)
        assert len(batches) == 2
        assert torch.equal(batches[0][0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(batches[0][1], torch.tensor([1.0, 2.0]))

    def test_convert_torch_tensor(self):
        """
        Test converting a torch tensor to a DataLoader object.
        """
        X = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        dataloader = self.mixin._convert_to_dataloader(
            X, y, batch_size=2, shuffle=False
        )

        assert isinstance(dataloader, DataLoader)
        batches = list(dataloader)
        assert len(batches) == 2
        assert torch.equal(batches[0][0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(batches[0][1], torch.tensor([1.0, 2.0]))

    def test_convert_dataloader(self):
        """
        Test converting a DataLoader object to itself.
        """
        X = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        result = self.mixin._convert_to_dataloader(dataloader)
        assert isinstance(result, DataLoader)
        batches = list(result)
        assert len(batches) == 2
        assert torch.equal(batches[0][0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(batches[0][1], torch.tensor([1.0, 2.0]))

    def test_convert_invalid_input(self):
        """
        Test converting an invalid input type.
        """
        X = "invalid input"
        with pytest.raises(ValueError, match="Unsupported type for x."):
            self.mixin._convert_to_dataloader(X)  # type: ignore - test for invalid type

    def test_convert_to_numpy_1d(self, sample_data_y1d):
        x, y = sample_data_y1d
        x_np, y_np = self.mixin._convert_to_numpy(x, y)
        assert isinstance(x_np, NumpyLike)
        assert isinstance(y_np, NumpyLike)
        assert x_np.shape == (20, 5)
        assert y_np.shape == (20,)

    def test_convert_to_numpy_2d(self, sample_data_y2d):
        x, y = sample_data_y2d
        x_np, y_np = self.mixin._convert_to_numpy(x, y)
        assert isinstance(x_np, NumpyLike)
        assert isinstance(y_np, NumpyLike)
        assert x_np.shape == (20, 5)
        assert y_np.shape == (20, 2)

    def test_random_split(self):
        """
        Test splitting a dataset into train and test DataLoaders.
        """
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TensorDataset(X, y)
        train_loader, test_loader = self.mixin._random_split(
            dataset, train_size=0.6, test_size=0.4
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        # Weird this needed to be ignored, there was an error calling len on the dataset
        assert len(train_loader.dataset) == 3  # type: ignore PGH003
        assert len(test_loader.dataset) == 2  # type: ignore PGH003

    def test_normalize(self):
        """
        Test normalizing a tensor.
        """
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalized, mean, std = self.mixin._normalize(X)

        # we use torch.allclose to compare tensors
        assert torch.allclose(mean, torch.tensor([[3.0, 4.0]]))
        assert torch.allclose(std, torch.tensor([[2.0, 2.0]]))
        assert torch.allclose(normalized, (X - mean) / std)

    def test_denormalize(self):
        """
        Test denormalizing a tensor.
        """
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = torch.tensor([[3.0, 4.0]])
        std = torch.tensor([[2.0, 2.0]])
        denormalized = self.mixin._denormalize((X - mean) / std, mean, std)

        assert torch.allclose(denormalized, X)


class TestValidationMixin:
    """
    Class to test the ValidationMixin class.
    """

    def setup_method(self):
        """
        Define the ValidationMixin instance.
        """
        self.mixin = ValidationMixin()

    def test_check_valid(self, tensor_2d, tensor_1d):
        """
        Test _check with valid finite tensors and supported dtypes.
        """
        x = tensor_2d.clone().to(torch.float32)
        y = tensor_1d.clone().to(torch.float64)
        x_checked, y_checked = self.mixin._check(x, y)
        assert torch.equal(x_checked, x)
        assert isinstance(y_checked, torch.Tensor)
        assert torch.equal(y_checked, y)

    def test_check_valid_y_none(self, tensor_2d):
        """
        Test _check with y=None.
        """
        x = tensor_2d.clone().to(torch.float32)
        x_checked, y_checked = self.mixin._check(x, None)
        assert torch.equal(x_checked, x)
        assert y_checked is None

    def test_check_invalid_x_nonfinite(self, tensor_2d):
        """
        Test _check raises if x contains non-finite values.
        """
        x = tensor_2d.clone()
        x[0, 0] = float("nan")
        y = torch.tensor([1.0, 2.0])
        msg = "Input tensor x contains non-finite values"
        with pytest.raises(ValueError, match=msg):
            self.mixin._check(x, y)

    def test_check_invalid_y_nonfinite(self, tensor_2d):
        """
        Test _check raises if y contains non-finite values.
        """
        x = tensor_2d
        y = torch.tensor([1.0, float("inf")])
        msg = "Input tensor y contains non-finite values"
        with pytest.raises(ValueError, match=msg):
            self.mixin._check(x, y)

    def test_check_invalid_x_dtype(self, tensor_2d):
        """
        Test _check raises if x has unsupported dtype.
        """
        x = tensor_2d.clone().to(torch.uint8)
        y = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="Input tensor x has unsupported dtype"):
            self.mixin._check(x, y)

    def test_check_invalid_y_dtype(self, tensor_2d):
        """
        Test _check raises if y has unsupported dtype.
        """
        x = tensor_2d
        y = torch.tensor([1, 2], dtype=torch.uint8)
        with pytest.raises(ValueError, match="Input tensor y has unsupported dtype"):
            self.mixin._check(x, y)

    def test_check_invalid_x_type(self, tensor_1d):
        """
        Test _check raises if x is not a TensorLike.
        """
        x = [1.0, 2.0]  # Not a TensorLike
        y = tensor_1d
        msg = "Expected x to be TensorLike, got <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            self.mixin._check(x, y)  # type: ignore PGH003

    def test_check_invalid_y_type(self, tensor_2d):
        """
        Test _check raises if y is not a TensorLike.
        """
        x = tensor_2d
        y = [1.0, 2.0]  # Not a TensorLike
        msg = "Expected y to be TensorLike, got <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            self.mixin._check(x, y)  # type: ignore PGH003

    def test_check_output(self):
        """
        Test _check_output returns the output unchanged.
        """
        # TODO: add test for _check_output once the method is implemented

    def test_check_vector_valid(self, tensor_1d):
        """
        Test check_vector with a valid 1D tensor.
        """
        result = self.mixin.check_vector(tensor_1d)

        assert torch.equal(result, tensor_1d)

    def test_check_vector_invalid_type(self, np_1d):
        """
        Test check_vector with an invalid input.
        """
        with pytest.raises(
            ValueError, match="Expected TensorLike, got <class 'numpy.ndarray'>"
        ):
            self.mixin.check_vector(np_1d)  # type: ignore PGH003

    def test_check_vector_invalid_dim(self, tensor_2d):
        """
        Test check_vector with wrong dims.
        """
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            self.mixin.check_vector(tensor_2d)

    def test_check_matrix_valid(self, tensor_2d):
        """
        Test check_matrix with a valid 2D tensor.
        """
        result = self.mixin.check_matrix(tensor_2d)

        assert torch.equal(result, tensor_2d)

    def test_check_matrix_invalid_type(self, np_2d):
        """
        Test check_vector with an invalid input.
        """
        with pytest.raises(
            ValueError, match="Expected TensorLike, got <class 'numpy.ndarray'>"
        ):
            self.mixin.check_matrix(np_2d)  # type: ignore PGH003

    def test_check_matrix_invalid_dim(self, tensor_1d):
        """
        Test check_matrix with wrong dims.
        """
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            self.mixin.check_matrix(tensor_1d)

    def test_check_pair_valid(self, tensor_2d, tensor_2d_pair):
        """
        Test check_pair with valid tensors.
        """
        x_checked, y_checked = self.mixin.check_pair(tensor_2d, tensor_2d_pair)

        assert torch.equal(x_checked, tensor_2d)
        assert torch.equal(y_checked, tensor_2d_pair)

    def test_check_pair_invalid(self, tensor_2d, tensor_2d_mismatch):
        """
        Test check_pair with tensors of mismatched rows.
        """
        with pytest.raises(
            ValueError, match="X and Y must have the same number of rows"
        ):
            self.mixin.check_pair(tensor_2d, tensor_2d_mismatch)

    def test_check_covariance_valid(self, sigma_full, tensor_2d, tensor_1d):
        """
        Test check_covariance with valid covariance matrices.

        Explanation:
        - `sigma_full`: Represents a full covariance matrix for each sample in `y`.
          It has a shape of (n_samples, n_features, n_features), where each sample
          has a full covariance matrix.
        - `sigma_diag`: Represents a diagonal covariance matrix for each sample in `y`.
          It has a shape of (n_samples, n_features), where each row corresponds to
          the diagonal elements of the covariance matrix for a sample.
        - `sigma_scalar`: Represents a scalar covariance value for each sample in `y`.
          It has a shape of (n_samples,), where each value corresponds to the same
          scalar covariance for all features of a sample.

        The test ensures that the `check_covariance` method correctly validates and
        returns these different types of covariance matrices without modification.
        """
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Diagonal covariance matrix: shape (n_samples, n_features)
        sigma_diag = tensor_2d

        # Scalar covariance: shape (n_samples,)
        sigma_scalar = tensor_1d

        # Assert that the method returns the same full covariance matrix
        assert torch.equal(self.mixin.check_covariance(y, sigma_full), sigma_full)

        # Assert that the method returns the same diagonal covariance matrix
        assert torch.equal(self.mixin.check_covariance(y, sigma_diag), sigma_diag)

        # Assert that the method returns the same scalar covariance values
        assert torch.equal(self.mixin.check_covariance(y, sigma_scalar), sigma_scalar)

    def test_check_covariance_invalid(self, tensor_2d):
        """
        Test check_covariance with an invalid covariance matrix.
        """
        sigma = torch.tensor([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Invalid covariance matrix shape"):
            self.mixin.check_covariance(tensor_2d, sigma)

    def test_trace(self, sigma_full, tensor_2d, tensor_1d):
        """
        Test trace computation for covariance matrices.
        """
        # TODO: add test for trace

    def test_logdet(self, sigma_full, tensor_2d, tensor_1d):
        """
        Test log-determinant computation for covariance matrices.
        """
        # TODO: add test for logdet

    def test_max_eigval(self, sigma_full, tensor_2d, tensor_1d):
        """
        Test maximum eigenvalue computation for covariance matrices.
        """
        sigma_diag = tensor_2d
        sigma_scalar = tensor_1d

        assert torch.isclose(self.mixin.max_eigval(sigma_full), torch.tensor(1.0))
        assert torch.isclose(self.mixin.max_eigval(sigma_diag), torch.tensor(3.0))
        assert torch.isclose(self.mixin.max_eigval(sigma_scalar), torch.tensor(1.5))


class TestStandardizer:
    def test_init_valid(self):
        """
        Test Standardizer initialization with valid mean and std.
        """
        mean = torch.zeros((2, 2))
        std = torch.ones((2, 2))
        s = Standardizer(mean, std)
        assert torch.equal(s.mean, mean)
        assert torch.equal(s.std, std)

    def test_init_invalid_mean_shape(self):
        """
        Test Standardizer raises ValueError if mean is not 2D.
        """
        mean = torch.zeros((2,))
        std = torch.ones((2, 2))
        with pytest.raises(ValueError, match="mean is expected to be 2D"):
            Standardizer(mean, std)

    def test_init_invalid_std_shape(self):
        """
        Test Standardizer raises ValueError if std is not 2D.
        """
        mean = torch.zeros((2, 2))
        std = torch.ones((2,))
        with pytest.raises(ValueError, match="std is expected to be 2D"):
            Standardizer(mean, std)

    def test_small_std_replaced(self):
        """
        Test that small std values are replaced with 1.0.
        """
        mean = torch.zeros((1, 2))
        std = torch.tensor([[1e-40, 2.0]], dtype=torch.float32)
        s = Standardizer(mean, std)
        assert s.std[0, 0] == 1.0
        assert s.std[0, 1] == 2.0

    def test_preprocess_valid(self):
        """
        Test Standardizer preprocess with valid input.
        """
        mean = torch.tensor([[1.0, 2.0]])
        std = torch.tensor([[2.0, 4.0]])
        s = Standardizer(mean, std)
        x = torch.tensor([[3.0, 6.0]])
        out = s.preprocess(x)
        expected = (x - mean) / std
        assert torch.allclose(out, expected)

    def test_preprocess_invalid_type(self):
        """
        Test Standardizer preprocess raises ValueError if input is not a TensorLike.
        """
        mean = torch.zeros((1, 2))
        std = torch.ones((1, 2))
        s = Standardizer(mean, std)
        msg = "Expected 2D TensorLike, actual type <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            s.preprocess([[1.0, 2.0]])  # type: ignore PGH003

    def test_preprocess_invalid_dim(self):
        """
        Test Standardizer preprocess raises ValueError if input is not 2D.
        """
        mean = torch.zeros((1, 2))
        std = torch.ones((1, 2))
        s = Standardizer(mean, std)
        x = torch.tensor([1.0, 2.0])
        msg = "Expected 2D TensorLike, actual shape dim 1"
        with pytest.raises(ValueError, match=msg):
            s.preprocess(x)
