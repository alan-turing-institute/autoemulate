import gpytorch
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, random_split

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import ParamsLike, ValueLike


class Tuner:
    """
    Run randomised hyperparameter search for a given model.

    Parameters
    ----------
    dataset: Dataset
        The training data.
    n_iter: int
        Number of parameter settings to randomly sample and test.
    fit_config: FitConfig
        Specifies how to fit each model.

    Returns
    -------
    Tuple[list[float], list[dict[str, ValueLike]]]
        The validation score and dictionary of parameter values used in each iteration.
    """

    def __init__(self, dataset: Dataset, n_iter: int, fit_config: FitConfig):
        self.n_iter = n_iter
        self.dataset = dataset
        self.fit_config = FitConfig()
        # Q: should users be able to choose a different validation metric?
        self.score_f = r2_score

    def run(self, model_class: type[Emulator]) -> dict[tuple[ValueLike, ...], float]:
        # split data into train/validation sets
        train, val = tuple(random_split(self.dataset, [0.8, 0.2]))
        train, val = (
            DataLoader(train, batch_size=self.fit_config.batch_size),
            DataLoader(val, batch_size=self.fit_config.batch_size),
        )

        # get all the available hyperparameter options
        params_all: ParamsLike = model_class.get_tunables()

        # keep track of what parameter values tested and how they performed
        params_tested: list[dict[str, ValueLike]] = []
        val_scores: list[float] = []

        for _ in range(self.n_iter):
            # randomly sample hyperparameters
            params_sample: dict[str, ValueLike] = {
                k: np.random.choice(v) for k, v in params_all
            }

            # fit model
            if isinstance(model_class, gpytorch.models.ExactGP):
                train_x, train_y = next(train)
                m = model_class(train_x, train_y, **s)
                assert isinstance(m, Emulator)
                m.fit(train_x, train_y, FitConfig())
            else:
                m = model_class(**params_sample)
                m.fit(train, y=None, config=self.fit_config)

            # make predictions
            y_pred = m.predict(val)

            # evaluate
            score = self.score_f(next(val)[1], y_pred)
            params_tested.append(params_sample)
            val_scores.append(score)

        return val_scores, params_tested
