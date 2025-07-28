import json
import logging

import numpy as np
from sklearn.model_selection import KFold
from torchmetrics import R2Score

from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import ConversionMixin, Emulator
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import DeviceLike, InputLike, ModelConfig

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
        max_retries: int = 100,
    ) -> tuple[list[float], list[ModelConfig]]:
        """
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
        Tuple[list[float], list[ModelConfig]]
            The validation scores and parameter values used in each search iteration.
        """
        # keep track of what parameter values tested and how they performed
        model_config_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        i, retries = 0, 0
        while i < self.n_iter:
            # randomly sample hyperparameters and instantiate model
            model_config = model_class.get_random_config()
            try:
                scores = cross_validate(
                    cv=KFold(n_splits=n_splits, random_state=seed, shuffle=shuffle),
                    dataset=self.dataset,
                    x_transforms=x_transforms,
                    y_transforms=y_transforms,
                    model=model_class,
                    model_config=model_config,
                    device=self.device,
                    random_seed=None,
                )
                retries = 0
                model_config_tested.append(model_config)
                val_scores.append(scores["r2"])  # type: ignore  # noqa: PGH003
                i += 1
                logger.debug(
                    "tuning model: %s; iteration: %d/%d; mean (std) r2=%.3f (%.3f); "
                    "model_config: %s",
                    model_class.model_name(),
                    i + 1,
                    self.n_iter,
                    np.mean(scores["r2"]),
                    np.std(scores["r2"]),
                    str(model_config),
                )

            except Exception as e:
                # Retry with new random config parameters
                retries += 1
                logger.warning(
                    "Failed tuning iteration %d with model config: %s: %s",
                    i + 1,
                    json.dumps(model_config, default=str, separators=(",", ":")),
                    str(e),
                )
                # If many retries, log error and break
                if retries > max_retries:
                    logger.error(
                        "Failed after %s with exception %s", max_retries, str(e)
                    )
                    break

        return val_scores, model_config_tested
