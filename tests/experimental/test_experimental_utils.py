import numpy as np
import pytest
import torch
from autoemulate.experimental.data.utils import ConversionMixin, ValidationMixin
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

    def test_check_y1d(self, sample_data_y1d):
        """
        Test _check does not alter inputs, y is 1d.
        """
        x, y = sample_data_y1d
        x_checked, y_checked = self.mixin._check(x, y)

        assert torch.equal(x_checked, x)  # type: ignore PGH003
        assert torch.equal(y_checked, y)  # type: ignore PGH003

    def test_check_y2d(self, sample_data_y2d):
        """
        Test _check does not alter inputs, y is 2d.
        """
        x, y = sample_data_y2d
        x_checked, y_checked = self.mixin._check(x, y)

        assert torch.equal(x_checked, x)  # type: ignore PGH003
        assert torch.equal(y_checked, y)  # type: ignore PGH003

    def test_check_vector_valid(self):
        """
        Test check_vector with a valid 1D tensor.
        """
        x = torch.tensor([1.0, 2.0, 3.0])
        result = self.mixin.check_vector(x)

        assert torch.equal(result, x)

    def test_check_vector_invalid_type(self):
        """
        Test check_vector with an invalid input.
        """
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(
            ValueError, match="Expected TensorLike, got <class 'numpy.ndarray'>"
        ):
            self.mixin.check_vector(x)  # type: ignore PGH003

    def test_check_vector_invalid_dim(self):
        """
        Test check_vector with wrong dims.
        """
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            self.mixin.check_vector(x)

    def test_check_matrix_valid(self):
        """
        Test check_matrix with a valid 2D tensor.
        """
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = self.mixin.check_matrix(x)

        assert torch.equal(result, x)

    def test_check_matrix_invalid_type(self):
        """
        Test check_vector with an invalid input.
        """
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(
            ValueError, match="Expected TensorLike, got <class 'numpy.ndarray'>"
        ):
            self.mixin.check_matrix(x)  # type: ignore PGH003

    def test_check_matrix_invalid_dim(self):
        """
        Test check_matrix with wrong dims.
        """
        x = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            self.mixin.check_matrix(x)

    def test_check_pair_valid(self):
        """
        Test check_pair with valid tensors.
        """
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        x_checked, y_checked = self.mixin.check_pair(x, y)

        assert torch.equal(x_checked, x)
        assert torch.equal(y_checked, y)

    def test_check_pair_invalid(self):
        """
        Test check_pair with tensors of mismatched rows.
        """
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0]])
        with pytest.raises(
            ValueError, match="X and Y must have the same number of rows"
        ):
            self.mixin.check_pair(x, y)
