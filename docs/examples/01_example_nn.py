# %%
from torch.utils.data import DataLoader, TensorDataset

from autoemulate.refactor.nn import PyTorchEmulator
from autoemulate.refactor.utils import sample_data_y1d


# %%
train_x, train_y = sample_data_y1d()
batch_size = 16  # Define batch size
dataset = TensorDataset(train_x, train_y)  # Wrap data in a dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = PyTorchEmulator(train_x.shape[1], 1)
model.fit(dataloader, model.criterion, model.optimizer, epochs=100)

# %%
model.cross_validate(dataset, model.criterion, model.optimizer)

# %%
model.cross_validate(dataset, model.criterion, model.optimizer)
