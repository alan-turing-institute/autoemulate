import torch
import torch.nn as nn

from autoemulate.refactor.base import BaseEmulator, InputTypeMixin, PyTorchBackend


class PyTorchEmulator(PyTorchBackend, BaseEmulator, InputTypeMixin):
    def __init__(self, input_size, output_size):
        super(PyTorchBackend, self).__init__()
        super(nn.Module, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, output_size)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.model(x)
