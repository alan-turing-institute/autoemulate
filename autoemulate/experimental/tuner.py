import gpytorch
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, random_split

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import (
    DistributionLike,
    ModelConfig,
    TensorLike,
    TuneConfig,
    ValueLike,
)


class Tuner:
    """
    Run randomised hyperparameter search for a given model.

    Parameters
    ----------
    dataset: Dataset
        The training data.
    n_iter: int
        Number of parameter settings to randomly sample and test.

    Returns
    -------
    Tuple[list[float], list[ModelConfig]]
        The validation scores and parameter values used in each search iteration.
    """

    def __init__(self, dataset: Dataset, n_iter: int):
        self.n_iter = n_iter
        self.dataset = dataset
        # Q: should users be able to choose a different validation metric?
        self.score_f = r2_score

    def run(self, model_class: type[Emulator]) -> tuple[list[float], list[ModelConfig]]:
        # split data into train/validation sets
        train, val = tuple(random_split(self.dataset, [0.8, 0.2]))
        train, val = (
            DataLoader(train, batch_size=len(train)),
            DataLoader(val, batch_size=len(val)),
        )
        ((train_x, train_y), (val_x, val_y)) = next(iter(train)), next(iter(val))

        # get all the available hyperparameter options
        params_all: TuneConfig = model_class.get_tune_config()

        # keep track of what parameter values tested and how they performed
        params_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        for _ in range(self.n_iter):
            # randomly sample hyperparameters and instantiate model
            params_sample: dict[str, ValueLike] = {
                k: np.random.choice(v) for k, v in params_all.items()
            }
            if isinstance(model_class, gpytorch.models.ExactGP):
                m = model_class(train_x, train_y, **params_sample)
                assert isinstance(m, Emulator)
                m.fit(train_x, train_y)
            else:
                m = model_class(**params_sample)
                assert isinstance(m, Emulator)
                # TODO: check if pass as dataloader
                m.fit(train_x, train_y)

            # evaluate
            y_pred = m.predict(val_x)
            # TODO: handle case when y_pred is not Tensor
            if isinstance(y_pred, TensorLike):
                score = self.score_f(val_y, y_pred.detach().numpy())
            elif isinstance(y_pred, DistributionLike):
                score = self.score_f(val_y, y_pred.mean.detach().numpy())
            elif (
                isinstance(y_pred, tuple)
                and len(y_pred) == 2
                and all(isinstance(item, TensorLike) for item in y_pred)
            ):
                score = self.score_f(val_y, y_pred[0].detach().numpy())
            else:
                raise ValueError(f"Score not implmented for {type(y_pred)}")
            assert isinstance(score, float)
            params_tested.append(params_sample)
            val_scores.append(score)

        return val_scores, params_tested
