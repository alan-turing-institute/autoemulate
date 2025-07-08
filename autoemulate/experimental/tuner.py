import torch
from sklearn.model_selection import KFold
from torchmetrics import R2Score
from tqdm import tqdm

from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import ConversionMixin, Emulator
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import (
    DeviceLike,
    InputLike,
    ModelConfig,
    TensorLike,
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
    ) -> tuple[list[float], list[ModelConfig]]:
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

        # keep track of what parameter values tested and how they performed
        model_config_tested: list[ModelConfig] = []
        val_scores: list[float] = []

        for _ in tqdm(range(self.n_iter)):
            # randomly sample hyperparameters and instantiate model
            model_config = model_class.get_random_config()

            scores = cross_validate(
                cv=KFold(n_splits=n_splits, random_state=seed, shuffle=shuffle),
                dataset=ConversionMixin._convert_to_dataset(
                    torch.cat([train_x, val_x], dim=0),
                    torch.cat([train_y, val_y], dim=0),
                ),
                x_transforms=x_transforms,
                y_transforms=y_transforms,
                model=model_class,
                device=self.device,
                random_seed=None,
            )
            model_config_tested.append(model_config)
            val_scores.append(scores["r2"])  # type: ignore  # noqa: PGH003

        return val_scores, model_config_tested
