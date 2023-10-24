from .version import version as __version__

from .GaussianProcess import GaussianProcess
try:
    from .GaussianProcessGPU import GaussianProcessGPU
    from .GaussianProcessGPU import GPUUnavailableError
except:
    pass
from .MultiOutputGP import MultiOutputGP
from .MultiOutputGP_GPU import MultiOutputGP_GPU
from .fitting import fit_GP_MAP
from .MeanFunction import MeanFunction
from .ExperimentalDesign import MonteCarloDesign, LatinHypercubeDesign, MaxiMinLHC
from .SequentialDesign import MICEDesign
from .HistoryMatching import HistoryMatching
from .DimensionReduction import gKDR
