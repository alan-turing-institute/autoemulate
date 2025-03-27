from dataclasses import dataclass

import numpy as np
import torch.nn as nn
from torch.optim import Optimizer


@dataclass(kw_only=True)
class DeviceConfig:
    device: str = "cpu"


@dataclass(kw_only=True)
class LoggingConfig:
    verbose: bool


@dataclass(kw_only=True)
class FitConfig(DeviceConfig, LoggingConfig):
    epochs: int
    shuffle: bool
    criterion: nn.Module
    optimizer: Optimizer
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
