import numpy as np
import pytest
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from autoemulate.experimental.base import InputTypeMixin
from autoemulate.experimental.base import PyTorchBackend
from autoemulate.experimental.config import FitConfig


@pytest.fixture
def fit_config():
    return FitConfig(
        epochs=10,
        batch_size=2,
        shuffle=False,
        verbose=False,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
    )


class TestInputTypeMixin:
    """
    Class to test the InputTypeMixin class.
    """

    def setup_method(self):
        """
        Define the InputTypeMixin instance.
        """
        self.mixin = InputTypeMixin()

    def test_convert_numpy_array(self):
        """
        Test converting a numpy array to a DataLoader object.
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        dataloader = self.mixin._convert(X, y, batch_size=2, shuffle=False)

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
        dataloader = self.mixin._convert(X, y, batch_size=2, shuffle=False)

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

        result = self.mixin._convert(dataloader)
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
        with pytest.raises(ValueError, match="Unsupported type for X."):
            self.mixin._convert(X)


class TestPyTorchBackend:
    """
    Class to test the PyTorchBackend class.
    """

    class DummyModel(PyTorchBackend):
        """
        A dummy implementation of PyTorchBackend for testing purposes.
        """

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.loss_fn = nn.MSELoss()
            self.optimizer = optim.SGD(self.parameters(), lr=0.01)

        def forward(self, x):
            return self.linear(x)

    def setup_method(self):
        """
        Define the PyTorchBackend instance.
        """
        self.model = self.DummyModel()

    def test_fit(self, fit_config):
        """
        Test the fit method of PyTorchBackend.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[2.0], [4.0], [6.0]])
        loss_history = self.model.fit(x, y, fit_config)

        assert isinstance(loss_history, list)
        assert len(loss_history) == 10
        assert all(isinstance(loss, float) for loss in loss_history)

    def test_predict(self, fit_config):
        """
        Test the predict method of PyTorchBackend.
        """
        x_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([[2.0], [4.0], [6.0]])
        self.model.fit(x_train, y_train, fit_config)

        X_test = torch.tensor([[4.0]])
        y_pred = self.model.predict(X_test)

        assert isinstance(y_pred, torch.Tensor)
        assert y_pred.shape == (1, 1)
        assert y_pred.shape == (1, 1)
