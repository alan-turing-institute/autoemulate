"""Pydantic models for configuring spatiotemporal emulation experiments."""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SimulatorType(str, Enum):
    """Available simulator types."""

    ADVECTION_DIFFUSION = "advection_diffusion"
    REACTION_DIFFUSION = "reaction_diffusion"


class DataSourceType(str, Enum):
    """Data source type - determines how data is loaded."""

    GENERATED = "generated"  # Generate data from simulator
    WELL_NATIVE = "well_native"  # Use The Well's native datasets
    FILE = "file"  # Load from existing files (HDF5/PyTorch)


class DatasetType(str, Enum):
    """Available dataset types."""

    ADVECTION_DIFFUSION = "advection_diffusion"
    REACTION_DIFFUSION = "reaction_diffusion"
    MHD = "mhd"
    BOUT = "bout"
    GENERIC = "generic"


class EmulatorType(str, Enum):
    """Available emulator types."""

    THE_WELL_FNO = "the_well_fno"
    THE_WELL_FNO_WITH_TIME = "the_well_fno_with_time"
    THE_WELL_FNO_WITH_LEARNABLE_WEIGHTS = "the_well_fno_with_learnable_weights"
    THE_WELL_AFNO = "the_well_afno"
    THE_WELL_UNET_CLASSIC = "the_well_unet_classic"
    THE_WELL_UNET_CONVNEXT = "the_well_unet_convnext"


class FormatterType(str, Enum):
    """Available data formatter types."""

    DEFAULT_CHANNELS_FIRST = "default_channels_first"
    DEFAULT_CHANNELS_FIRST_WITH_TIME = "default_channels_first_with_time"


class OptimizerType(str, Enum):
    """Available optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class LRSchedulerType(str, Enum):
    """Available learning rate scheduler types."""

    STEP_LR = "step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


class LossFunctionType(str, Enum):
    """Available loss function types."""

    VRMSE = "vrmse"
    MSE = "mse"
    MAE = "mae"


class SimulatorConfig(BaseModel):
    """Configuration for simulator."""

    type: SimulatorType = Field(
        default=SimulatorType.ADVECTION_DIFFUSION,
        description="Type of simulator to use",
    )
    parameters_range: dict[str, tuple[float, float]] = Field(
        default={"nu": (0.0001, 0.01), "mu": (0.5, 2.0)},
        description="Parameter ranges for the simulator",
    )
    output_names: list[str] = Field(
        default=["solution"], description="Names of output variables"
    )
    return_timeseries: bool = Field(
        default=True, description="Whether to return full timeseries"
    )
    n: int = Field(default=64, description="Number of spatial points per direction")
    L: float = Field(default=10.0, description="Domain size in X and Y directions")
    T: float = Field(default=10.0, description="Total simulation time")
    dt: float = Field(default=0.1, description="Time step size")


class DataConfig(BaseModel):
    """Configuration for data generation and loading."""

    # Data source - automatically determined from config fields
    source_type: DataSourceType | None = Field(
        default=None,
        description="Data source type (auto-detected if None)",
    )

    # Data paths
    data_path: str | None = Field(
        default=None,
        description="Path to data directory (for file loading) or Well datasets base path (for well_native)",
    )

    # The Well native dataset configuration
    well_dataset_name: str | None = Field(
        default=None,
        description="Name of The Well dataset (e.g., 'turbulent_radiative_layer_2D'). Set this to use Well native datasets.",
    )

    # Data generation (for generated source type)
    n_train_samples: int = Field(
        default=200, description="Number of training samples to generate"
    )
    n_valid_samples: int = Field(
        default=4, description="Number of validation samples to generate"
    )
    n_test_samples: int = Field(
        default=4, description="Number of test samples to generate"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for data generation"
    )

    # Dataset configuration
    dataset_type: DatasetType = Field(
        default=DatasetType.ADVECTION_DIFFUSION, description="Type of dataset"
    )
    n_steps_input: int = Field(default=4, description="Number of input time steps")
    n_steps_output: int = Field(default=10, description="Number of output time steps")
    stride: int = Field(default=1, description="Stride for sampling the data")
    input_channel_idxs: tuple[int, ...] | None = Field(
        default=None, description="Indices of input channels to use"
    )
    output_channel_idxs: tuple[int, ...] | None = Field(
        default=None, description="Indices of output channels to use"
    )

    # DataLoader configuration
    batch_size: int = Field(default=4, description="Batch size for DataLoader")
    dtype: str = Field(default="float32", description="Data type (float32 or float64)")

    def get_source_type(self) -> DataSourceType:
        """Automatically determine data source type from configuration."""
        if self.source_type is not None:
            return self.source_type

        # Auto-detect based on fields
        if self.well_dataset_name is not None:
            return DataSourceType.WELL_NATIVE
        elif self.data_path is not None:
            return DataSourceType.FILE
        else:
            return DataSourceType.GENERATED


class TeacherForcingConfig(BaseModel):
    """Configuration for teacher forcing schedule."""

    start: float = Field(
        default=1.0, description="Starting teacher forcing ratio", ge=0.0, le=1.0
    )
    end: float = Field(
        default=0.0, description="Ending teacher forcing ratio", ge=0.0, le=1.0
    )
    schedule_epochs: int | None = Field(
        default=None,
        description="Number of epochs for schedule (None = use total epochs)",
    )
    schedule_type: str = Field(
        default="linear",
        description="Schedule type: linear, exponential, or step",
    )
    mode: str = Field(
        default="mix",
        description="Mode: mix (tempering) or prob (stochastic)",
    )
    min_prob: float = Field(
        default=1e-6, description="Minimum probability for numerical stability"
    )


class ModelParamsConfig(BaseModel):
    """Configuration for model-specific parameters."""

    # FNO parameters
    modes1: int = Field(default=16, description="FNO modes in first spatial dimension")
    modes2: int = Field(default=16, description="FNO modes in second spatial dimension")
    modes3: int = Field(
        default=16, description="FNO modes in third spatial dimension (3D only)"
    )
    modes_time: int = Field(
        default=3, description="FNO modes in time dimension (for FNOWithTime)"
    )
    hidden_channels: int = Field(default=64, description="Number of hidden channels")
    gradient_checkpointing: bool = Field(
        default=False, description="Enable gradient checkpointing"
    )

    # AFNO parameters
    hidden_dim: int = Field(default=64, description="AFNO hidden dimension")
    n_blocks: int = Field(default=14, description="AFNO number of blocks")

    # UNet parameters
    init_features: int = Field(default=48, description="UNet initial features")
    blocks_per_stage: int = Field(
        default=2, description="UNet blocks per stage (ConvNext variant)"
    )

    # Allow extra fields for custom model parameters
    model_config = {"extra": "allow"}


class TrainerConfig(BaseModel):
    """Configuration for training."""

    # Optimizer configuration
    optimizer_type: OptimizerType = Field(
        default=OptimizerType.ADAM, description="Optimizer type"
    )
    optimizer_params: dict[str, Any] = Field(
        default_factory=lambda: {"lr": 1e-3},
        description="Optimizer parameters (lr, weight_decay, etc.)",
    )

    # LR Scheduler configuration
    lr_scheduler_type: LRSchedulerType | None = Field(
        default=None, description="Learning rate scheduler type"
    )
    lr_scheduler_params: dict[str, Any] = Field(
        default_factory=dict,
        description="LR scheduler parameters",
    )

    # Training parameters
    epochs: int = Field(default=10, description="Number of training epochs")
    checkpoint_frequency: int = Field(
        default=5, description="Save checkpoint every N epochs"
    )
    val_frequency: int = Field(default=1, description="Validate every N epochs")
    rollout_val_frequency: int = Field(
        default=1, description="Rollout validation every N epochs"
    )

    # Rollout parameters
    max_rollout_steps: int = Field(
        default=100, description="Maximum rollout steps for validation"
    )
    short_validation_length: int = Field(
        default=20, description="Short validation length"
    )
    make_rollout_videos: bool = Field(
        default=True, description="Generate rollout videos"
    )
    num_time_intervals: int = Field(
        default=5, description="Number of time intervals for metrics"
    )

    # Mixed precision
    enable_amp: bool = Field(
        default=False, description="Enable automatic mixed precision"
    )
    amp_type: str | None = Field(
        default=None, description="AMP data type (float16 or bfloat16)"
    )

    # Distributed training
    is_distributed: bool = Field(
        default=False, description="Enable distributed training"
    )

    # Device
    device: str = Field(
        default="cpu", description="Device to use (cpu, cuda, mps, cuda:0, etc.)"
    )

    # Teacher forcing
    enable_tf_schedule: bool = Field(
        default=False, description="Enable scheduled teacher forcing in training"
    )
    tf_params: TeacherForcingConfig = Field(
        default_factory=TeacherForcingConfig,
        description="Teacher forcing schedule parameters",
    )

    # Loss function
    loss_fn: LossFunctionType = Field(
        default=LossFunctionType.VRMSE, description="Loss function to use"
    )

    # Checkpoint path for resuming
    checkpoint_path: str | None = Field(
        default=None, description="Path to checkpoint to resume from"
    )


class PathsConfig(BaseModel):
    """Configuration for input/output paths."""

    output_dir: Path = Field(
        default=Path("./outputs"), description="Base output directory"
    )
    data_save_path: Path | None = Field(
        default=None, description="Path to save generated data (HDF5 or PT format)"
    )
    model_save_path: Path | None = Field(
        default=None, description="Path to save trained model state dict"
    )
    save_format: str = Field(
        default="h5",
        description="Data save format: h5 (HDF5) or pt (PyTorch)",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v


class ExperimentConfig(BaseModel):
    """Top-level configuration for a spatiotemporal emulation experiment."""

    # Experiment metadata
    experiment_name: str = Field(
        default="spatiotemporal_experiment", description="Name of the experiment"
    )
    description: str = Field(default="", description="Description of the experiment")

    # Core components
    emulator_type: EmulatorType = Field(
        default=EmulatorType.THE_WELL_FNO, description="Type of emulator to use"
    )
    formatter_type: FormatterType = Field(
        default=FormatterType.DEFAULT_CHANNELS_FIRST,
        description="Type of data formatter to use",
    )

    # Configuration sections
    simulator: SimulatorConfig | None = Field(
        default=None,
        description="Simulator configuration (required if generating data)",
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data configuration"
    )
    model_params: ModelParamsConfig = Field(
        default_factory=ModelParamsConfig,
        description="Model-specific parameters",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig, description="Training configuration"
    )
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Path configuration"
    )

    # Logging and monitoring
    verbose: bool = Field(default=False, description="Enable verbose logging")
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    model_config = {"arbitrary_types_allowed": True}

    def save_to_yaml(self, path: str | Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        config_dict = self.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)
