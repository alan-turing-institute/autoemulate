import numpy as np
import torch
from autoemulate.experimental.types import InputLike
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


class InputTypeMixin:
    """
    Mixin class to convert input data to pytorch Datasets and DataLoaders.
    """

    def _convert_to_dataset(
        self,
        x: InputLike,
        y: InputLike | None = None,
    ) -> Dataset:
        """
        Convert input data to pytorch Dataset.
        """
        # Convert input to Dataset if not already
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(x, (torch.Tensor, np.ndarray)) and isinstance(
            y, (torch.Tensor, np.ndarray)
        ):
            dataset = TensorDataset(x, y)
        elif isinstance(x, (torch.Tensor, np.ndarray)) and y is None:
            dataset = TensorDataset(x)
        elif isinstance(x, Dataset) and y is None:
            dataset = x
        else:
            raise ValueError(
                f"Unsupported type for x ({type(x)}). Must be numpy array or PyTorch "
                "tensor."
            )

        return dataset

    def _convert_to_dataloader(
        self,
        x: InputLike,
        y: InputLike | None = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Convert input data to pytorch DataLoaders.
        """
        if isinstance(x, DataLoader) and y is None:
            dataloader = x
        elif isinstance(x, DataLoader) and y is not None:
            raise ValueError(
                f"Since x is already a DataLoader, expect y to be None, not {type(y)}."
            )
        else:
            dataset = self._convert_to_dataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def _convert_to_tensors(
        self,
        x: InputLike,
        y: InputLike | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Convert InputLike x, y to Tensor or tuple of Tensors.
        """
        dataset = self._convert_to_dataset(x, y)
        if isinstance(dataset, TensorDataset):
            if len(dataset.tensors) > 2:
                raise ValueError(
                    f"Dataset must have 2 or fewer tensors. Found "
                    f"{len(dataset.tensors)}."
                )
            if len(dataset.tensors) == 2:
                x, y = dataset.tensors
                assert x.ndim == 2
                assert y.ndim in (1, 2)
                # Ensure always 2D tensors
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                return x, y
            if len(dataset.tensors) == 1:
                (x,) = dataset.tensors
                assert x.ndim == 2
                return x
            msg = "Number of tensors returned must be greater than zero."
            raise ValueError(msg)
        raise ValueError(
            f"Unsupported type for dataset ({type(dataset)}). Must be TensorDataset."
        )

    def _convert_to_numpy(
        self,
        x: InputLike,
        y: InputLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert InputLike x, y to tuple of numpy arrays.
        """
        if isinstance(x, np.ndarray) and (y is None or isinstance(y, np.ndarray)):
            return x, y

        result = self._convert_to_tensors(x, y)
        if isinstance(result, tuple):
            x, y = result
            return x.numpy(), y.numpy()
        x = result
        return x.numpy(), None

    def _random_split(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        train_size: float = 0.8,
        test_size: float = 0.2,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Split Dataset into train/test DataLoaders.

        Parameters
        ----------
        dataset: Dataset
            The data to split.
        batch_size: int | None
            The DataLoader batch_size. If None, sets batch_size to lenth of training
            data. Defaults to None.
        """
        if train_size < 0.0 or train_size > 1.0 or test_size < 0.0 or test_size > 1.0:
            raise ValueError(
                f"Train size ({train_size}) and test size ({test_size}) must be "
                "specified as a proportion between 0 and 1"
            )
        if test_size + train_size != 1.0:
            raise ValueError(
                f"Train size ({train_size}) and test size ({test_size}) must sum to 1"
            )
        train, test = tuple(random_split(dataset, [train_size, test_size]))
        if batch_size is None:
            batch_size = len(train)
        train_loader = self._convert_to_dataloader(train, batch_size=batch_size)
        test_loader = self._convert_to_dataloader(test, batch_size=batch_size)
        return train_loader, test_loader

    # TODO: consider possible method for predict
    # def convert_x(self, y: np.ndarray | torch.Tensor | Data) -> torch.Tensor:
    #     if isinstance(y, np.ndarray):
    #         y = torch.tensor(y, dtype=torch.float32)
    #     else:
    #         raise ValueError(
    #             "Unsupported type for X. Must be numpy array, PyTorch tensor"
    #         )
    #     return y
