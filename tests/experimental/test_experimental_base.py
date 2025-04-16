import numpy as np
import pytest
import torch
from autoemulate.experimental.data.preprocessors import Standardizer
from autoemulate.experimental.emulators.base import InputTypeMixin, PyTorchBackend
from autoemulate.experimental.tuner import Tuner
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# @pytest.fixture
# def model_config() -> M:
#     return {
#         "epochs": 10,
#         "batch_size": 2,
#         "shuffle": False,
#         "verbose": False,
#         "optimizer": torch.optim.Adam,
#         "criterion": torch.nn.MSELoss,
#     }


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


class TestPyTorchBackend:
    """
    Class to test the PyTorchBackend class.
    """

    class DummyModel(PyTorchBackend):
        """
        A dummy implementation of PyTorchBackend for testing purposes.
        """

        def __init__(self, x=None, y=None, **kwargs):
            super().__init__()
            _, _ = x, y  # unused variables
            self.linear = nn.Linear(1, 1)
            self.loss_fn = nn.MSELoss()
            self.optimizer = optim.SGD(self.parameters(), lr=0.01)
            self.epochs = kwargs.get("epochs", 10)
            self.batch_size = kwargs.get("batch_size", 16)
            self.preprocessor = Standardizer(
                torch.Tensor([[-0.5]]), torch.Tensor([[0.5]])
            )

        def forward(self, x):
            return self.linear(x)

        @staticmethod
        def get_tune_config():
            return {
                "epochs": [100, 200, 300],
                "batch_size": [16],
            }

    def setup_method(self):
        """
        Define the PyTorchBackend instance.
        """
        self.model = self.DummyModel()

    def test_fit(self):
        """
        Test the fit method of PyTorchBackend.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[2.0], [4.0], [6.0]])
        self.model.fit(x, y)

        assert isinstance(self.model.loss_history, list)
        assert len(self.model.loss_history) == 10
        assert all(isinstance(loss, float) for loss in self.model.loss_history)

    def test_predict(self):
        """
        Test the predict method of PyTorchBackend.
        """
        x_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([[2.0], [4.0], [6.0]])
        self.model.fit(x_train, y_train)

        X_test = torch.tensor([[4.0]])
        y_pred = self.model.predict(X_test)

        assert isinstance(y_pred, torch.Tensor)
        assert y_pred.shape == (1, 1)
        assert y_pred.shape == (1, 1)

    def test_tune_xy(self):
        """
        Test that Tuner accepts X,Y inputs.
        """
        x_train = torch.Tensor(np.arange(16).reshape(-1, 1))
        y_train = 2 * x_train
        tuner = Tuner(x_train, y_train, n_iter=10)
        tuner.run(self.DummyModel)

    def test_standardizer(self):
        x_train = torch.Tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        )
        x_train_preprocessed = self.model.preprocess(x_train)
        assert isinstance(x_train_preprocessed, torch.Tensor)
        assert torch.allclose(
            x_train_preprocessed,
            torch.Tensor([[3.0], [5.0], [7.0], [9.0], [11.0]]),
        )

    def test_standardizer_fail(self):
        x_train = torch.Tensor([0.1, 2.0, 6.0, 0.2])
        with pytest.raises(
            ValueError, match="Expected 2D torch.Tensor, actual shape dim 1"
        ):
            self.model.preprocess(x_train)

    def test_tune_dataset(self):
        """
        Test that Tuner accepts a single Dataset input.
        """
        x_train = torch.Tensor(np.arange(16).reshape(-1, 1))
        y_train = 2 * x_train
        dataset = self.model._convert_to_dataset(x_train, y_train)
        tuner = Tuner(x=dataset, y=None, n_iter=10)
        tuner.run(self.DummyModel)
