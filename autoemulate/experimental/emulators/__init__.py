from .gaussian_process.exact import GaussianProcessExact
from .lightgbm import LightGBM
from .neural_processes.conditional_neural_process import CNPModule

ALL_EMULATORS = [GaussianProcessExact, LightGBM, CNPModule]
