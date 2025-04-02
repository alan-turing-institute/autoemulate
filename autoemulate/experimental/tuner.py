import gpytorch
import numpy as np
from sklearn.metrics import r2_score

from autoemulate.experimental.emulators.base import Emulator, InputTypeMixin
from autoemulate.experimental.types import (
    InputLike,
    ModelConfig,
    TuneConfig,
    ParamLike,
)


class Tuner(InputTypeMixin):
    """
    Run randomised hyperparameter search for a given model.

    Parameters
    ----------
    X: InputLike
        Input features as numpy array, PyTorch tensor, or Dataset.
    y: OutputLine or None
        Target values (not needed if x is a Dataset).
    n_iter: int
        Number of parameter settings to randomly sample and test.

    Returns
    -------
    Tuple[list[float], list[ModelConfig]]
        The validation scores and parameter values used in each search iteration.
    """

    def __init__(self, x: InputLike, y: InputLike | None, n_iter: int):
        self.n_iter = n_iter
        self.dataset = self._convert_to_dataset(x, y)
        # Q: should users be able to choose a different validation metric?
        self.score_f = r2_score

    def run(self, model_class: type[Emulator]) -> tuple[list[float], list[ModelConfig]]:
        # split data into train/validation sets
        # batch size defaults to size of train data if not otherwise specified
        train_loader, val_loader = self._random_split(self.dataset)
        train_x, train_y = next(iter(train_loader))
        val_x, val_y = next(iter(val_loader))

        # get all the available hyperparameter options
        params_all: TuneConfig = model_class.get_tune_config()

        # keep track of what parameter values tested and how they performed
        params_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        for _ in range(self.n_iter):
            # randomly sample hyperparameters and instantiate model
            params_sample: dict[str, ParamLike] = {
                k: np.random.choice(v) for k, v in params_all.items()
            }
            if isinstance(model_class, gpytorch.models.ExactGP):
                m = model_class(train_x, train_y, **params_sample)
                m.fit(train_x, train_y)
            else:
                m = model_class(**params_sample)
                # TODO: check if pass as dataloader
                m.fit(train_x, train_y)

            # evaluate
            y_pred = m.predict(val_x)
            score = self.score_f(val_y, y_pred.detach().numpy())
            params_tested.append(params_sample)
            val_scores.append(score)

        return val_scores, params_tested
