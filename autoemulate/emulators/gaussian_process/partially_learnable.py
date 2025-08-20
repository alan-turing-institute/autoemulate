import torch
import gpytorch
from gpytorch.means import LinearMean
from autoemulate.core.types import TensorLike


class PartiallyLearnableMean(gpytorch.means.Mean):
    """
    A mixed mean module that combines a known function for one dimension
    with a learnable linear function for the remaining dimensions.
    
    This implements Universal Kriging where part of the mean function is known
    and part is learned from data.
    
    Parameters
    ----------
    mean_func : callable
        A function that takes a tensor and returns a tensor of the same shape.
        This function will be applied to the specified dimension.
    known_dim : int
        The dimension index to which the custom mean function will be applied.
        Default is 0 (first dimension).
    input_size : int
        The total number of input features.
    batch_shape : torch.Size | None
        Optional batch dimension for multi-task GPs.
    """
    
    def __init__(
        self,
        mean_func: callable,
        known_dim: int = 0,
        input_size: int = 1,
        batch_shape: torch.Size | None = None,
    ):
        super().__init__()
        
        # Store the custom function and dimension
        self.mean_func = mean_func
        self.known_dim = known_dim
        self.input_size = input_size
        
        if batch_shape is None:
            batch_shape = torch.Size()
        
        # Create indices for learnable dimensions (all except known_dim)
        self.learnable_dims = [i for i in range(input_size) if i != known_dim]
        
        # Only create linear mean if there are learnable dimensions
        self.linear_mean = LinearMean(
            input_size=len(self.learnable_dims), 
            batch_shape=batch_shape
        )

    
    def forward(self, x: TensorLike) -> TensorLike:
        """
        Forward pass through the partially learnable mean module.
        """
        # Apply custom mean function to the known dimension
        known_part = self.mean_func(x[..., self.known_dim])
        

        learnable_data = x[..., self.learnable_dims]
        learnable_part = self.linear_mean(learnable_data)
        return known_part + learnable_part
    
    def __repr__(self) -> str:
        """Return string representation of the PartiallyLearnableMean module."""
        func_name = getattr(self.mean_func, '__name__', 'mean_func')
        return (
            f"PartiallyLearnableMean("
            f"mean_func={func_name}, "
            f"known_dim={self.known_dim}, "
            f"input_size={self.input_size})"
        )