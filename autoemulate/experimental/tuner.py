import numpy as np
from torchmetrics import R2Score
from tqdm import tqdm

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import ConversionMixin, Emulator
from autoemulate.experimental.model_selection import evaluate
from autoemulate.experimental.types import (
    DeviceLike,
    InputLike,
    ModelConfig,
    TensorLike,
    TuneConfig,
)


class Tuner(ConversionMixin, TorchDeviceMixin):
    """
    Run randomised hyperparameter search for a given model.

    Parameters
    ----------
    x: InputLike
        Input features.
    y: OutputLike or None
        Target values (not needed if x is a Dataset).
    n_iter: int
        Number of parameter settings to randomly sample and test.
    """

    def __init__(
        self,
        x: InputLike,
        y: InputLike | None,
        n_iter: int = 10,
        device: DeviceLike | None = None,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        self.n_iter = n_iter

        # Convert input types, convert to tensors to ensure correct shapes, move to
        # device and convert back to dataset. TODO: consider if this is the best way to
        # do this.
        dataset = self._convert_to_dataset(x, y)
        x_tensor, y_tensor = self._convert_to_tensors(dataset)
        x_tensor, y_tensor = self._move_tensors_to_device(x_tensor, y_tensor)
        self.dataset = self._convert_to_dataset(x_tensor, y_tensor)

        # Q: should users be able to choose a different validation metric?
        self.metric = R2Score

    def run(self, model_class: type[Emulator]) -> tuple[list[float], list[ModelConfig]]:
        """
        Parameters
        ----------
        model_class: type[Emulator]
            A concrete Emulator subclass.

        Returns
        -------
        Tuple[list[float], list[ModelConfig]]
            The validation scores and parameter values used in each search iteration.
        """
        # split data into train/validation sets
        # batch size defaults to size of train data if not otherwise specified
        train_loader, val_loader = self._random_split(self.dataset)

        train_x: TensorLike
        train_y: TensorLike
        train_x, train_y = next(iter(train_loader))

        val_x: TensorLike
        val_y: TensorLike
        val_x, val_y = next(iter(val_loader))

        # get all the available hyperparameter options
        tune_config: TuneConfig = model_class.get_tune_config()

        # keep track of what parameter values tested and how they performed
        model_config_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        for _ in tqdm(range(self.n_iter)):
            # randomly sample hyperparameters and instantiate model
            model_config: ModelConfig = {
                k: v[np.random.randint(len(v))] for k, v in tune_config.items()
            }
            m = model_class(train_x, train_y, device=self.device, **model_config)
            m.fit(train_x, train_y)

            # evaluate
            y_pred = m.predict(val_x)
            score = evaluate(val_y, y_pred, self.metric, self.device)

            # record score and config
            model_config_tested.append(model_config)
            val_scores.append(score)

        return val_scores, model_config_tested
