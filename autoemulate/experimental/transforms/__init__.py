from .pca import PCATransform
from .standardize import StandardizeTransform
from .vae import VAETransform

__all__ = ["PCATransform", "StandardizeTransform", "VAETransform"]

ALL_TRANSFORMS = [PCATransform, StandardizeTransform, VAETransform]

# Registry mapping transform names to their classes
# Add new transforms here when creating them
TRANSFORM_REGISTRY = {
    "pca": PCATransform,
    "standardize": StandardizeTransform,
    "vae": VAETransform,
}
