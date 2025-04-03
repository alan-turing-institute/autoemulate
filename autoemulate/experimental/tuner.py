import gpytorch
import numpy as np
from sklearn.metrics import r2_score

from autoemulate.experimental.cross_validation import evaluate
from autoemulate.experimental.emulators.base import Emulator, InputTypeMixin
from autoemulate.experimental.types import InputLike, ModelConfig, TuneConfig


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
    """

    def __init__(self, x: InputLike, y: InputLike | None, n_iter: int):
        self.n_iter = n_iter
        self.dataset = self._convert_to_dataset(x, y)
        # Q: should users be able to choose a different validation metric?
        self.score_f = r2_score

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
        train_x, train_y = next(iter(train_loader))
        val_x, val_y = next(iter(val_loader))

        # get all the available hyperparameter options
        tune_config: TuneConfig = model_class.get_tune_config()

        # keep track of what parameter values tested and how they performed
        model_config_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        for _ in range(self.n_iter):
            # randomly sample hyperparameters and instantiate model
            model_config: ModelConfig = {
                k: np.random.choice(v) for k, v in tune_config.items()
            }
            if isinstance(model_class, gpytorch.models.ExactGP):
                m = model_class(train_x, train_y, **model_config)
                assert isinstance(m, Emulator)
                m.fit(train_x, train_y)
            else:
                m = model_class(**model_config)
                assert isinstance(m, Emulator)
                # TODO: check if pass as dataloader
                m.fit(train_x, train_y)

            # evaluate
            y_pred = m.predict(val_x)
            score = evaluate(val_y, y_pred, self.score_f)
            assert isinstance(score, float)
            model_config_tested.append(model_config)
            val_scores.append(score)

        return val_scores, model_config_tested
