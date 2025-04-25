from .gaussian_process.exact import GaussianProcessExact
from .lightgbm import LightGBM

ALL_EMULATORS = [GaussianProcessExact, LightGBM]
