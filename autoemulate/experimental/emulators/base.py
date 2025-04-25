from abc import ABC, abstractmethod
from typing import ClassVar

from torch import nn, optim

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import InputTypeMixin
from autoemulate.experimental.types import InputLike, OutputLike, TuneConfig


class Emulator(ABC):
    """
    The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    """

    @abstractmethod
    def __init__(
        self, x: InputLike | None = None, y: InputLike | None = None, **kwargs
    ): ...

    @classmethod
    def model_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def fit(self, x: InputLike, y: InputLike | None): ...

    @abstractmethod
    def predict(self, x: InputLike) -> OutputLike:
        pass

    def is_multioutput(self) -> bool:
        """Flag to indicate if the model is multioutput or not."""
        return False

    @staticmethod
    @abstractmethod
    def get_tune_config() -> TuneConfig:
        """
        The keys in the TuneConfig must be implemented as keyword arguments in the
        __init__ method of any subclasses.

        e.g.

        tune_config: TuneConfig = {
            "lr": list[0.01, 0.1],
            "batch_size": [16, 32],
            "mean"
        }

        model_config: ModelConfig = {
            "lr": 0.01,
            "batch_size": 16
        }

        class MySubClass(Emulator):
            def __init__(lr, batch_size):
                self.lr = lr
                self.batch_size = batch_size
        """

        ...


class PyTorchBackend(nn.Module, Emulator, InputTypeMixin, Preprocessor):
    """
    PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`
    """

    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 10
    loss_history: ClassVar[list[float]] = []
    verbose: bool = False
    preprocessor: Preprocessor | None = None
    loss_fn: nn.Module = nn.MSELoss()
    optimizer: optim.Optimizer

    def preprocess(self, x):
        if self.preprocessor is None:
            return x
        return self.preprocessor.preprocess(x)

    def fit(
        self,
        x: InputLike,
        y: InputLike | None,
    ):
        """
        Train a PyTorchBackend model.

        Parameters
        ----------
            X: InputLike
                Input features as numpy array, PyTorch tensor, or DataLoader.
            y: OutputLine or None
                Target values (not needed if x is a DataLoader).
            batch_size: int
                Batch size (used only when xis not a DataLoader).
            shuffle: bool
                Whether to shuffle the data.
            epochs: int
                Number of training epochs.
            verbose: bool
                Whether to print progress.

        Returns:
        -------
            List of loss values per epoch.
        """

        self.train()  # Set model to training mode

        # Convert input to DataLoader if not already
        dataloader = self._convert_to_dataloader(
            x, y, batch_size=self.batch_size, shuffle=self.shuffle
        )

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0

            for X_batch, y_batch in dataloader:
                # Preprocess x_batch
                # TODO: consider if this should be moved outside of dataloader iteration
                # e.g. as part of the InputTypeMixin
                x = self.preprocess(X_batch)

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batches += 1

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        x = self.preprocess(x)
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        msg = "This function is not yet implemented."
        raise NotImplementedError(msg)

    @staticmethod
    def get_tune_config():
        msg = (
            "Subclasses should implement for generating tuning config specific to "
            "each subclass."
        )
        raise NotImplementedError(msg)
