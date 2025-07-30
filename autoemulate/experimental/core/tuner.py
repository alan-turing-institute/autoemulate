import json
import logging

import numpy as np
from sklearn.model_selection import KFold
from torchmetrics import R2Score

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.model_selection import cross_validate
from autoemulate.experimental.core.types import DeviceLike, InputLike, ModelParams
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.emulators.base import ConversionMixin, Emulator
from autoemulate.experimental.transforms.base import AutoEmulateTransform

logger = logging.getLogger("autoemulate")


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
    device: DeviceLike | None
        The device to use for model training and evaluation. If None, uses the default
        device (usually CPU or GPU).
    random_seed: int | None
        Random seed for reproducibility. If None, no seed is set.
    """

    def __init__(
        self,
        x: InputLike,
        y: InputLike | None,
        n_iter: int = 10,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
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

        if random_seed is not None:
            set_random_seed(seed=random_seed)

    def run(  # noqa: PLR0913
        self,
        model_class: type[Emulator],
        x_transforms: list[AutoEmulateTransform] | None = None,
        y_transforms: list[AutoEmulateTransform] | None = None,
        n_splits: int = 5,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> tuple[list[list[float]], list[ModelParams]]:
        """
        Run randomised hyperparameter search for a given model.

        Parameters
        ----------
        model_class: type[Emulator]
            A concrete Emulator subclass.
        x_transforms_list: list[list[AutoEmulateTransform]] | None
            An optional list of sequences of transforms to apply to the input data.
            Defaults to None, in which case the data is standardized.
        y_transforms_list: list[list[AutoEmulateTransform]] | None
            An optional list of sequences of transforms to apply to the output data.
            Defaults to None, in which case the data is standardized.
        n_splits: int
            Number of cross validation folds to split data into. Defaults to 5.
        seed: int | None
            Random seed to use in cross validation. If None, no seed is set.
        shuffle: bool
            Whether to shuffle data before splitting into cross validation folds.
            Defaults to True.

        Returns
        -------
        Tuple[list[float], list[ModelParams]]
            The validation scores and parameter values used in each search iteration.
        """
        # keep track of what parameter values tested and how they performed
        model_params_tested: list[ModelParams] = []
        val_scores: list[list[float]] = []

        # Initialize retries and maximum number of retries for consecutive failed tuning
        # iterations that would indicate that no model params is working for the
        # given dataset/model and that the tuning process should be stopped
        retries, max_retries = 0, 100
        while len(model_params_tested) < self.n_iter:
            # randomly sample hyperparameters and instantiate model
            model_params = model_class.get_random_params()
            try:
                # Perform cross-validation on randomly sampled model params
                scores = cross_validate(
                    cv=KFold(n_splits=n_splits, random_state=seed, shuffle=shuffle),
                    dataset=self.dataset,
                    x_transforms=x_transforms,
                    y_transforms=y_transforms,
                    model=model_class,
                    model_params=model_params,
                    device=self.device,
                    random_seed=None,
                )

                # Reset retries following a successful cross_validation call
                retries = 0

                # Store the model params and validation scores
                model_params_tested.append(model_params)
                val_scores.append(scores["r2"])  # type: ignore  # noqa: PGH003

                # Log the tuning iteration results
                logger.debug(
                    "tuning model: %s; iteration: %d/%d; mean (std) r2=%.3f (%.3f); "
                    "model_params: %s",
                    model_class.model_name(),
                    len(model_params_tested),
                    self.n_iter,
                    np.mean(scores["r2"]),
                    np.std(scores["r2"]),
                    str(model_params),
                )

            except Exception as e:
                # Increment retries following a failed tuning iteration and try again
                retries += 1
                logger.warning(
                    "Failed tuning iteration %d with model params: %s: %s",
                    len(model_params_tested),
                    json.dumps(model_params, default=str, separators=(",", ":")),
                    str(e),
                )
                # If many consecutive retries, log error and raise exception
                if retries > max_retries:
                    logger.error(
                        "Failed after %s with exception %s", max_retries, str(e)
                    )
                    msg = (
                        f"Failed to tune model {model_class.model_name()} after "
                        f"{max_retries} retries."
                    )
                    raise RuntimeError(msg) from e

        return val_scores, model_params_tested
