import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from autoemulate.experimental.types import InputLike, OutputLike


class InputTypeMixin:
    """
    Mixin class to convert input data to pytorch Datasets and DataLoaders.
    """

    def _convert_to_dataset(
        self,
        x: InputLike,
        y: OutputLike | None = None,
    ) -> Dataset:
        """
        Convert input data to pytorch Dataset.
        """
        # Convert input to DataLoader if not already
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(x, (torch.Tensor, np.ndarray)) and y is not None:
            dataset = TensorDataset(x, y)
        elif isinstance(x, Dataset) and y is None:
            dataset = x
        else:
            raise ValueError(
                f"Unsupported type for x ({type(x)}). Must be numpy array or PyTorch tensor."
            )

        return dataset

    def _convert_to_dataloader(
        self,
        x: InputLike,
        y: OutputLike | None = None,
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

    def _random_split(
        self, dataset: Dataset, batch_size: int | None = None
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
        train, test = tuple(random_split(dataset, [0.8, 0.2]))
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
    #         raise ValueError("Unsupported type for X. Must be numpy array, PyTorch tensor")
    #     return y
