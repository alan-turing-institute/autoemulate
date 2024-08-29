import numpy as np
import pytest
import torch

from autoemulate.emulators.neural_networks.datasets import cnp_collate_fn
from autoemulate.emulators.neural_networks.datasets import CNPDataset


@pytest.fixture
def sample_data():
    X = np.random.rand(30, 5)
    y = np.random.rand(30, 2)
    return X, y


def test_cnp_dataset_init(sample_data):
    X, y = sample_data
    dataset = CNPDataset(
        X, y, max_context_points=10, min_context_points=3, n_episode=16
    )

    assert isinstance(dataset.X, torch.Tensor)
    assert isinstance(dataset.y, torch.Tensor)
    assert dataset.max_context_points == 10
    assert dataset.min_context_points == 3
    assert dataset.n_episode == 16


def test_cnp_dataset_getitem(sample_data):
    X, y = sample_data
    dataset = CNPDataset(
        X, y, max_context_points=10, min_context_points=3, n_episode=16
    )

    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2  # ((context_X, context_y, target_X), y)

    X_context = item[0]["X_context"]
    y_context = item[0]["y_context"]
    X_target = item[0]["X_target"]
    y_target = item[1]

    # check that contexts are between min and max context points
    assert X_context.shape[0] <= dataset.max_context_points
    assert X_context.shape[0] >= dataset.min_context_points
    assert X_context.shape[1] == X.shape[1]
    assert y_context.shape[0] == X_context.shape[0]

    # check that targets are n_episode datapoints
    assert X_target.shape[0] == dataset.n_episode
    assert y_target.shape[0] == dataset.n_episode

    # check that context and targets have same number of features
    assert X_context.shape[1] == X.shape[1]
    assert y_context.shape[1] == y.shape[1]
    assert X_context.shape[1] == X_target.shape[1]
    assert y_context.shape[1] == y_target.shape[1]


def test_cnp_collate_fn_structure(sample_data):
    X, y = sample_data
    dataset = CNPDataset(
        X, y, max_context_points=10, min_context_points=3, n_episode=16
    )
    batch_size = 4

    batch = [dataset[i] for i in range(batch_size)]
    X_batched, y_batched = cnp_collate_fn(batch)

    X_context = X_batched["X_context"]
    y_context = X_batched["y_context"]
    X_target = X_batched["X_target"]
    y_target = y_batched

    # check batch dimension was added
    assert len(X_context.shape) == 3
    assert len(y_context.shape) == 3
    assert len(X_target.shape) == 3
    assert len(y_target.shape) == 3

    assert X_context.shape[0] == batch_size
    assert y_context.shape[0] == batch_size
    assert X_target.shape[0] == batch_size
    assert y_target.shape[0] == batch_size

    # check that X_target is fully populated (we always take all points)
    assert torch.all(X_target != 0)


def test_cnp_collate_fn_context_mask(sample_data):
    X, y = sample_data
    dataset = CNPDataset(
        X, y, max_context_points=10, min_context_points=3, n_episode=16
    )
    batch_size = 4

    batch = [dataset[i] for i in range(batch_size)]
    X_batched, y_batched = cnp_collate_fn(batch)

    X_context = X_batched["X_context"]
    y_context = X_batched["y_context"]
    context_mask = X_batched["context_mask"]

    # check mask
    assert context_mask.shape == (batch_size, X_context.shape[1])
    assert context_mask.dtype == torch.bool

    # check that number of True values in each row matches number of non-zero context points
    for i in range(batch_size):
        assert context_mask[i].sum() == (X_context[i] != 0).any(dim=1).sum()
        assert context_mask[i].sum() == (y_context[i] != 0).any(dim=1).sum()

    # check that all masked context elements are zero
    assert torch.all((X_context * (~context_mask.unsqueeze(-1))) == 0)
    assert torch.all((y_context * (~context_mask.unsqueeze(-1))) == 0)

    # check that number True values in each masked row is between min and max context points
    assert all(
        dataset.min_context_points
        <= context_mask[i].sum()
        <= dataset.max_context_points
        for i in range(batch_size)
    )
