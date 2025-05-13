import numpy as np
import pytest
import torch
from autoemulate.experimental.data.utils import ConversionMixin
from autoemulate.experimental.types import NumpyLike
from torch.utils.data import DataLoader, TensorDataset


class TestConversionMixin:
    """
    Class to test the ConversionMixin class.
    """

    def setup_method(self):
        """
        Define the ConversionMixin instance.
        """
        self.mixin = ConversionMixin()

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
