import numpy as np
import pytest
import torch
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.tuner import Tuner
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.base import PyTorchBackend
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR


class TestPyTorchBackend:
    """
    Class to test the PyTorchBackend class.
    """

    class DummyModel(PyTorchBackend):
        """
        A dummy implementation of PyTorchBackend for testing purposes.
        """

        def __init__(self, x=None, y=None, random_seed=None, device=None, **kwargs):
            super().__init__()
            TorchDeviceMixin.__init__(self, device)
            if random_seed is not None:
                set_random_seed(random_seed)
            _, _ = x, y  # unused variables
            self.linear = nn.Linear(1, 1)
            self.loss_func = nn.MSELoss()
            self.optimizer = self.optimizer_cls(self.parameters(), lr=self.lr)  # type: ignore[call-arg]
            self.scheduler_setup(kwargs)
            self.epochs = kwargs.get("epochs", 10)
            self.batch_size = kwargs.get("batch_size", 16)

        def forward(self, x):
            return self.linear(x)

        @staticmethod
        def get_tune_params():
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
        self.model = self.DummyModel(
            scheduler_cls=ExponentialLR, scheduler_kwargs={"gamma": 0.9}
        )

    def test_model_name(self):
        """
        Test the model_name class method of Emulator.
        """
        assert self.model.model_name() == "DummyModel"

    def test_short_name(self):
        """
        Test the short_name class method of Emulator.
        """
        assert self.model.short_name() == "dm"

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
        assert not y_pred.requires_grad

        y_pred_grad = self.model.predict(X_test, with_grad=True)
        assert y_pred_grad

    def test_tune_xy(self):
        """
        Test that Tuner accepts x,y inputs.
        """
        x_train = torch.Tensor(np.arange(16).reshape(-1, 1))
        y_train = 2 * x_train
        tuner = Tuner(x_train, y_train, n_iter=10)
        tuner.run(self.DummyModel)

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

        # Set a random seed for reproducibility
        seed = 42
        model1 = self.DummyModel(random_seed=seed)
        model1.fit(x_train, y_train)
        pred1 = model1.predict(x_test)

        # Use the same seed to ensure deterministic behavior
        model2 = self.DummyModel(random_seed=seed)
        model2.fit(x_train, y_train)
        pred2 = model2.predict(x_test)

        # Use a different seed to ensure deterministic behavior
        new_seed = 43
        model3 = self.DummyModel(random_seed=new_seed)
        model3.fit(x_train, y_train)
        pred3 = model3.predict(x_test)

        assert isinstance(pred1, torch.Tensor)
        assert isinstance(pred2, torch.Tensor)
        assert isinstance(pred3, torch.Tensor)
        assert torch.allclose(pred1, pred2)
        msg = "Predictions should differ with different seeds."
        assert not torch.allclose(pred1, pred3), msg

    def test_scheduler_setup(self):
        # Should raise ValueError if kwargs is None
        with pytest.raises(ValueError, match="Provide a kwargs dictionary including"):
            self.model.scheduler_setup(None)

        # Should raise RuntimeError if optimizer is missing
        model_no_opt = self.DummyModel()
        delattr(model_no_opt, "optimizer")
        with pytest.raises(RuntimeError, match="Optimizer must be set before"):
            model_no_opt.scheduler_setup(
                {"scheduler_cls": ExponentialLR, "scheduler_kwargs": {"gamma": 0.9}}
            )

        # Should set scheduler to None if scheduler_cls is None
        model_none_sched = self.DummyModel()
        model_none_sched.scheduler_cls = None
        model_none_sched.optimizer = model_none_sched.optimizer_cls(
            model_none_sched.parameters(),
            lr=model_none_sched.lr,  # type: ignore[call-arg]
        )
        model_none_sched.scheduler_setup({"scheduler_kwargs": {}})
        assert model_none_sched.scheduler is None

        # Should set scheduler if scheduler_cls is valid
        model_valid_sched = self.DummyModel()
        model_valid_sched.scheduler_cls = ExponentialLR
        model_valid_sched.optimizer = model_valid_sched.optimizer_cls(
            model_valid_sched.parameters(),
            lr=model_valid_sched.lr,  # type: ignore[call-arg]
        )
        model_valid_sched.scheduler_setup({"scheduler_kwargs": {"gamma": 0.9}})
        assert isinstance(model_valid_sched.scheduler, ExponentialLR)
