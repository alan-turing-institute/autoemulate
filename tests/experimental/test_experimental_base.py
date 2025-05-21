import numpy as np
import pytest
import torch
from autoemulate.experimental.data.utils import Standardizer
from autoemulate.experimental.emulators.base import PyTorchBackend
from autoemulate.experimental.tuner import Tuner
from torch import nn, optim


class TestPyTorchBackend:
    """
    Class to test the PyTorchBackend class.
    """

    class DummyModel(PyTorchBackend):
        """
        A dummy implementation of PyTorchBackend for testing purposes.
        """

        def __init__(self, x=None, y=None, random_state=None, **kwargs):
            super().__init__(random_state=random_state)
            _, _ = x, y  # unused variables
            self.linear = nn.Linear(1, 1)
            self.loss_func = nn.MSELoss()
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

        @staticmethod
        def is_multioutput():
            return False

    def setup_method(self):
        """
        Define the PyTorchBackend instance.
        """
        self.model = self.DummyModel()

    def test_fit(self):
        """
        Test the fit method of PyTorchBackend.
        """
        x = torch.Tensor(np.array([[1.0], [2.0], [3.0]]))
        y = torch.Tensor(np.array([[2.0], [4.0], [6.0]]))
        self.model.fit(x, y)

        assert isinstance(self.model.loss_history, list)
        assert len(self.model.loss_history) == 10
        assert all(isinstance(loss, float) for loss in self.model.loss_history)

    def test_predict(self):
        """
        Test the predict method of PyTorchBackend.
        """
        x_train = torch.Tensor(np.array([[1.0], [2.0], [3.0]]))
        y_train = torch.Tensor(np.array([[2.0], [4.0], [6.0]]))
        self.model.fit(x_train, y_train)

        X_test = torch.tensor([[4.0]])
        y_pred = self.model.predict(X_test)

        assert isinstance(y_pred, torch.Tensor)
        assert y_pred.shape == (1, 1)
        assert y_pred.shape == (1, 1)

    def test_tune_xy(self):
        """
        Test that Tuner accepts x,y inputs.
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

    def test_fit_predict_deterministic_with_seed(self):
        """
        Test that fitting two models with the same seed and data
        produces identical predictions.
        """
        x_train = torch.Tensor(np.array([[1.0], [2.0], [3.0]]))
        y_train = torch.Tensor(np.array([[2.0], [4.0], [6.0]]))
        x_test = torch.tensor([[4.0]])

        model1 = self.DummyModel(random_state=123)
        model1.fit(x_train, y_train)
        pred1 = model1.predict(x_test)

        model2 = self.DummyModel(random_state=123)
        model2.fit(x_train, y_train)
        pred2 = model2.predict(x_test)

        assert isinstance(pred1, torch.Tensor)
        assert isinstance(pred2, torch.Tensor)
        assert torch.allclose(pred1, pred2)
