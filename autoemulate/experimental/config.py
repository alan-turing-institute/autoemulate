from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass(kw_only=True)
class DeviceConfig:
    device: str = "cpu"


@dataclass(kw_only=True)
class LoggingConfig:
    verbose: bool = False


@dataclass(kw_only=True)
class FitConfig(DeviceConfig, LoggingConfig):
    epochs: int = 10
    batch_size: int = 16
    shuffle: bool = True
    criterion: type[nn.Module] = torch.nn.MSELoss
    optimizer: type[Optimizer] = torch.optim.Adam
    device: str = "cpu"


@dataclass(kw_only=True)
class PredictConfig(DeviceConfig):
    pass


@dataclass(kw_only=True)
class TuneConfig(FitConfig):
    params: dict[str, list[float]]


@dataclass(kw_only=True)
class CVConfig(FitConfig):
    pass
