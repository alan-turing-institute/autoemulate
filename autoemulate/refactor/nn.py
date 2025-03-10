import torch
import torch.nn as nn

from autoemulate.refactor.base import BaseModel, InputTypeMixin


class PyTorchModel(BaseModel, InputTypeMixin, nn.Module):
    def __init__(self, input_size, output_size):
        super(PyTorchModel, self).__init__()
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, output_size)
        )
        self.criterion = nn.CrossEntropyLoss()  # or nn.MSELoss() for regression
        self.optimizer = torch.optim.Adam(self.parameters())

    def fit(self, X, y, epochs=100):
        X = self.convert_to_tensor(X)
        y = self.convert_to_tensor(y)

        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = self.convert_to_tensor(X)
        self.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.argmax(outputs, dim=1).numpy()
