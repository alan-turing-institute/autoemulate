from .discrete_fourier import DiscreteFourierTransform
from .pca import PCATransform
from .standardize import StandardizeTransform
from .vae import VAETransform

__all__ = [
    "DiscreteFourierTransform",
    "PCATransform",
    "StandardizeTransform",
    "VAETransform",
]

ALL_TRANSFORMS = [
    DiscreteFourierTransform,
    PCATransform,
    StandardizeTransform,
    VAETransform,
]

# Registry mapping transform names to their classes
# Add new transforms here when creating them
TRANSFORM_REGISTRY = {
    "pca": PCATransform,
    "standardize": StandardizeTransform,
    "vae": VAETransform,
    "discrete_fourier": DiscreteFourierTransform,
}
