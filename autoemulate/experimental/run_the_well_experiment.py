"""Script to run spatiotemporal emulation experiments from YAML config.

This script provides a command-line interface for running The Well-based
spatiotemporal emulation experiments using YAML configuration files.

Example usage:
    # Create an example config
    python run_the_well_experiment.py --create-example

    # Run an experiment from config
    python run_the_well_experiment.py --config config.yaml

    # Override output directory
    python run_the_well_experiment.py --config config.yaml --output-dir ./results
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import torch
from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataModule, WellDataset
from the_well.data.data_formatter import DefaultChannelsFirstFormatter

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoemulate.experimental.config_models import (
    DatasetType,
    DataSourceType,
    EmulatorType,
    ExperimentConfig,
    FormatterType,
    LossFunctionType,
    LRSchedulerType,
    OptimizerType,
)
from autoemulate.experimental.data.spatiotemporal_dataset import (
    AdvectionDiffusionDataset,
    AutoEmulateDataModule,
    BOUTDataset,
    ReactionDiffusionDataset,
)
from autoemulate.experimental.emulators.the_well import (
    DefaultChannelsFirstFormatterWithTime,
    TheWellAFNO,
    TheWellFNO,
    TheWellFNOWithLearnableWeights,
    TheWellFNOWithTime,
    TheWellUNetClassic,
    TheWellUNetConvNext,
    TrainerParams,
)
from autoemulate.simulations.advection_diffusion import AdvectionDiffusion
from autoemulate.simulations.reaction_diffusion import ReactionDiffusion

# Set up logger
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, log_level: str = "INFO", verbose: bool = False):
    """Set up logging configuration.

    Parameters
    ----------
    output_dir : Path
        Directory to save log file
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    verbose : bool
        If True, also log to console with detailed format
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove any existing handlers
    root_logger.handlers = []

    # File handler - always detailed
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if verbose:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        console_formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return log_file


def create_simulator(config: ExperimentConfig):
    """Create simulator from configuration."""
    if config.simulator is None:
        msg = "Simulator config required when generating data"
        raise ValueError(msg)

    sim_cfg = config.simulator

    if sim_cfg.type.value == "advection_diffusion":
        return AdvectionDiffusion(
            parameters_range=sim_cfg.parameters_range,
            output_names=sim_cfg.output_names,
            return_timeseries=sim_cfg.return_timeseries,
            n=sim_cfg.n,
            L=sim_cfg.L,
            T=sim_cfg.T,
            dt=sim_cfg.dt,
        )
    elif sim_cfg.type.value == "reaction_diffusion":
        return ReactionDiffusion(
            parameters_range=sim_cfg.parameters_range,
            output_names=sim_cfg.output_names,
            return_timeseries=sim_cfg.return_timeseries,
            n=sim_cfg.n,
            L=sim_cfg.L,
            T=sim_cfg.T,
            dt=sim_cfg.dt,
        )

    msg = f"Unknown simulator type: {sim_cfg.type}"
    raise ValueError(msg)


def get_dataset_class(dataset_type: DatasetType):
    """Get dataset class from type."""
    dataset_classes = {
        DatasetType.ADVECTION_DIFFUSION: AdvectionDiffusionDataset,
        DatasetType.REACTION_DIFFUSION: ReactionDiffusionDataset,
        DatasetType.BOUT: BOUTDataset,
        # Add more dataset types as needed
    }

    if dataset_type not in dataset_classes:
        msg = f"Dataset type {dataset_type} not supported"
        raise ValueError(msg)

    return dataset_classes[dataset_type]


def generate_or_load_data(config: ExperimentConfig, simulator=None):
    """Generate or load data and create data module.

    Supports three data source types:
    1. GENERATED: Generate data from simulator
    2. FILE: Load from existing files (HDF5/PyTorch)
    3. WELL_NATIVE: Use The Well's native datasets
    """
    data_cfg = config.data
    source_type = data_cfg.get_source_type()

    logger.info("Data source type: %s", source_type.value)

    # Get dtype
    dtype = torch.float32 if data_cfg.dtype == "float32" else torch.float64

    # Handle The Well native datasets
    if source_type == DataSourceType.WELL_NATIVE:
        if data_cfg.well_dataset_name is None:
            msg = "well_dataset_name required for WELL_NATIVE source type"
            raise ValueError(msg)

        logger.info(
            "Creating WellDataModule for dataset: %s", data_cfg.well_dataset_name
        )
        logger.info("Base path: %s", data_cfg.data_path or "../data/the_well/datasets")
        logger.info(
            "n_steps_input: %d, n_steps_output: %d",
            data_cfg.n_steps_input,
            data_cfg.n_steps_output,
        )

        from the_well.data import WellDataModule, WellDataset

        datamodule = WellDataModule(
            well_base_path=data_cfg.data_path or "../data/the_well/datasets",
            well_dataset_name=data_cfg.well_dataset_name,
            n_steps_input=data_cfg.n_steps_input,
            n_steps_output=data_cfg.n_steps_output,
            batch_size=data_cfg.batch_size,
            train_dataset=WellDataset,
        )

        logger.info("Training dataset size: %d samples", len(datamodule.train_dataset))
        logger.info("Validation dataset size: %d samples", len(datamodule.val_dataset))
        logger.info("Test dataset size: %d samples", len(datamodule.test_dataset))

        return datamodule

    # Handle generated and file-based data with AutoEmulateDataModule
    dataset_cls = get_dataset_class(data_cfg.dataset_type)

    if source_type == DataSourceType.FILE:
        # Load from existing data directory
        logger.info("Loading data from %s", data_cfg.data_path)
        datamodule = AutoEmulateDataModule(
            data_path=data_cfg.data_path,
            dataset_cls=dataset_cls,
            n_steps_input=data_cfg.n_steps_input,
            n_steps_output=data_cfg.n_steps_output,
            stride=data_cfg.stride,
            input_channel_idxs=data_cfg.input_channel_idxs,
            output_channel_idxs=data_cfg.output_channel_idxs,
            batch_size=data_cfg.batch_size,
            dtype=dtype,
            verbose=config.verbose,
        )
    else:  # GENERATED
        # Generate data from simulator
        if simulator is None:
            msg = "Simulator required for GENERATED source type"
            raise ValueError(msg)

        logger.info(
            "Generating %d training, %d validation, and %d test samples",
            data_cfg.n_train_samples,
            data_cfg.n_valid_samples,
            data_cfg.n_test_samples,
        )

        # Generate splits
        logger.debug("Generating training data...")
        data_train = simulator.forward_samples_spatiotemporal(
            data_cfg.n_train_samples, data_cfg.random_seed
        )
        data_valid = simulator.forward_samples_spatiotemporal(
            data_cfg.n_valid_samples,
            data_cfg.random_seed + 1 if data_cfg.random_seed else None,
        )
        data_test = simulator.forward_samples_spatiotemporal(
            data_cfg.n_test_samples,
            data_cfg.random_seed + 2 if data_cfg.random_seed else None,
        )

        data = {"train": data_train, "valid": data_valid, "test": data_test}

        # Optionally save generated data
        if config.paths.data_save_path:
            logger.info("Saving generated data to %s", config.paths.data_save_path)
            save_data_splits(
                data, config.paths.data_save_path, config.paths.save_format
            )

        # Create data module
        datamodule = AutoEmulateDataModule(
            data_path=None,
            data=data,
            dataset_cls=dataset_cls,
            n_steps_input=data_cfg.n_steps_input,
            n_steps_output=data_cfg.n_steps_output,
            stride=data_cfg.stride,
            input_channel_idxs=data_cfg.input_channel_idxs,
            output_channel_idxs=data_cfg.output_channel_idxs,
            batch_size=data_cfg.batch_size,
            dtype=dtype,
            verbose=config.verbose,
        )

    return datamodule


def save_data_splits(data, base_path, save_format="h5"):
    """Save data splits to disk."""
    base_path = Path(base_path)
    logger.debug("Saving data in %s format", save_format)

    for split_name, split_data in data.items():
        split_dir = base_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if save_format == "h5":
            file_path = split_dir / "data.h5"
            logger.debug("Saving %s data to %s", split_name, file_path)
            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=split_data["data"].numpy())
                if split_data["constant_scalars"] is not None:
                    f.create_dataset(
                        "constant_scalars",
                        data=split_data["constant_scalars"].numpy(),
                    )
                if split_data["constant_fields"] is not None:
                    f.create_dataset(
                        "constant_fields", data=split_data["constant_fields"].numpy()
                    )
        elif save_format == "pt":
            file_path = split_dir / "data.pt"
            logger.debug("Saving %s data to %s", split_name, file_path)
            torch.save(split_data, file_path)
        else:
            msg = f"Unknown save format: {save_format}"
            raise ValueError(msg)


def get_formatter_class(formatter_type: FormatterType):
    """Get formatter class from type."""
    formatter_classes = {
        FormatterType.DEFAULT_CHANNELS_FIRST: DefaultChannelsFirstFormatter,
        FormatterType.DEFAULT_CHANNELS_FIRST_WITH_TIME: (
            DefaultChannelsFirstFormatterWithTime
        ),
    }

    return formatter_classes[formatter_type]


def get_loss_function(loss_type: LossFunctionType):
    """Get loss function from type."""
    if loss_type == LossFunctionType.VRMSE:
        return VRMSE()
    if loss_type == LossFunctionType.MSE:
        return torch.nn.MSELoss()
    if loss_type == LossFunctionType.MAE:
        return torch.nn.L1Loss()

    msg = f"Unknown loss function: {loss_type}"
    raise ValueError(msg)


def create_lr_scheduler(config: ExperimentConfig):
    """Create learning rate scheduler factory."""
    if config.trainer.lr_scheduler_type is None:
        return None

    sched_type = config.trainer.lr_scheduler_type
    params = config.trainer.lr_scheduler_params

    if sched_type == LRSchedulerType.STEP_LR:
        return lambda opt: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=params.get("step_size", 10),
            gamma=params.get("gamma", 0.1),
        )

    if sched_type == LRSchedulerType.EXPONENTIAL_LR:
        return lambda opt: torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=params.get("gamma", 0.95)
        )

    if sched_type == LRSchedulerType.COSINE_ANNEALING_LR:
        return lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=params.get("T_max", 100)
        )

    if sched_type == LRSchedulerType.REDUCE_LR_ON_PLATEAU:
        return lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.1),
            patience=params.get("patience", 10),
        )

    return None


def get_optimizer_class(optimizer_type: OptimizerType):
    """Get optimizer class from type."""
    optimizer_classes = {
        OptimizerType.ADAM: torch.optim.Adam,
        OptimizerType.ADAMW: torch.optim.AdamW,
        OptimizerType.SGD: torch.optim.SGD,
        OptimizerType.RMSPROP: torch.optim.RMSprop,
    }

    return optimizer_classes[optimizer_type]


def create_trainer_params(config: ExperimentConfig) -> TrainerParams:
    """Create TrainerParams from configuration."""
    trainer_cfg = config.trainer

    return TrainerParams(
        optimizer_cls=get_optimizer_class(trainer_cfg.optimizer_type),
        optimizer_params=trainer_cfg.optimizer_params,
        epochs=trainer_cfg.epochs,
        checkpoint_frequency=trainer_cfg.checkpoint_frequency,
        val_frequency=trainer_cfg.val_frequency,
        rollout_val_frequency=trainer_cfg.rollout_val_frequency,
        max_rollout_steps=trainer_cfg.max_rollout_steps,
        short_validation_length=trainer_cfg.short_validation_length,
        make_rollout_videos=trainer_cfg.make_rollout_videos,
        lr_scheduler=create_lr_scheduler(config),
        amp_type=trainer_cfg.amp_type,
        num_time_intervals=trainer_cfg.num_time_intervals,
        enable_amp=trainer_cfg.enable_amp,
        is_distributed=trainer_cfg.is_distributed,
        checkpoint_path=trainer_cfg.checkpoint_path,
        device=trainer_cfg.device,
        output_path=str(config.paths.output_dir),
        enable_tf_schedule=trainer_cfg.enable_tf_schedule,
        tf_params={
            "start": trainer_cfg.tf_params.start,
            "end": trainer_cfg.tf_params.end,
            "schedule_epochs": trainer_cfg.tf_params.schedule_epochs,
            "schedule_type": trainer_cfg.tf_params.schedule_type,
            "mode": trainer_cfg.tf_params.mode,
            "min_prob": trainer_cfg.tf_params.min_prob,
        },
    )


def create_emulator(config: ExperimentConfig, datamodule):
    """Create emulator from configuration."""
    emulator_type = config.emulator_type
    formatter_cls = get_formatter_class(config.formatter_type)
    loss_fn = get_loss_function(config.trainer.loss_fn)
    trainer_params = create_trainer_params(config)

    # Get model parameters as kwargs
    model_params = config.model_params.model_dump(exclude_unset=True)

    emulator_classes = {
        EmulatorType.THE_WELL_FNO: TheWellFNO,
        EmulatorType.THE_WELL_FNO_WITH_TIME: TheWellFNOWithTime,
        EmulatorType.THE_WELL_FNO_WITH_LEARNABLE_WEIGHTS: (
            TheWellFNOWithLearnableWeights
        ),
        EmulatorType.THE_WELL_AFNO: TheWellAFNO,
        EmulatorType.THE_WELL_UNET_CLASSIC: TheWellUNetClassic,
        EmulatorType.THE_WELL_UNET_CONVNEXT: TheWellUNetConvNext,
    }

    if emulator_type not in emulator_classes:
        msg = f"Unknown emulator type: {emulator_type}"
        raise ValueError(msg)

    emulator_cls = emulator_classes[emulator_type]

    logger.info("Emulator type: %s", emulator_type.value)
    logger.info("Formatter: %s", config.formatter_type.value)
    logger.info("Loss function: %s", config.trainer.loss_fn.value)
    logger.debug("Model parameters: %s", model_params)

    return emulator_cls(
        datamodule=datamodule,
        formatter_cls=formatter_cls,
        loss_fn=loss_fn,
        trainer_params=trainer_params,
        **model_params,
    )


def run_experiment(config_path: str, output_dir: str | None = None):
    """Run a complete experiment from config file."""
    # Load configuration
    logger.info("Loading configuration from %s", config_path)
    config = ExperimentConfig.load_from_yaml(config_path)

    # Override output directory if provided
    if output_dir:
        config.paths.output_dir = Path(output_dir)

    # Create output directory
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(
        config.paths.output_dir,
        log_level=config.log_level,
        verbose=config.verbose,
    )

    # Save config to output directory for reproducibility
    config.save_to_yaml(config.paths.output_dir / "config.yaml")
    logger.info("Configuration saved to %s", config.paths.output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info("Experiment: %s", config.experiment_name)
    if config.description:
        logger.info("Description: %s", config.description)
    logger.info("Emulator type: %s", config.emulator_type.value)
    logger.info("Output directory: %s", config.paths.output_dir)
    logger.info("=" * 60)

    # Create simulator if needed
    simulator = None
    if config.data.data_path is None:
        logger.info("Creating simulator...")
        simulator = create_simulator(config)
        if config.simulator:
            logger.info("Simulator type: %s", config.simulator.type.value)
            logger.info(
                "Simulator params: n=%d, T=%.1f, dt=%.2f",
                config.simulator.n,
                config.simulator.T,
                config.simulator.dt,
            )

    # Generate or load data
    logger.info("Preparing data...")
    datamodule = generate_or_load_data(config, simulator)

    # Log dataset sizes (handle both AutoEmulate and Well data modules)
    logger.info("Training dataset size: %d samples", len(datamodule.train_dataset))

    # WellDataModule uses 'val_dataset', AutoEmulateDataModule uses 'valid_dataset'
    if hasattr(datamodule, "valid_dataset"):
        logger.info(
            "Validation dataset size: %d samples", len(datamodule.valid_dataset)
        )
    elif hasattr(datamodule, "val_dataset"):
        logger.info("Validation dataset size: %d samples", len(datamodule.val_dataset))

    logger.info("Test dataset size: %d samples", len(datamodule.test_dataset))

    # Create emulator
    logger.info("Creating emulator...")
    emulator = create_emulator(config, datamodule)

    # Train emulator
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    logger.info("Device: %s", config.trainer.device)
    logger.info("Number of epochs: %d", config.trainer.epochs)
    logger.info("Batch size: %d", config.data.batch_size)
    logger.info(
        "Optimizer: %s (lr=%.2e)",
        config.trainer.optimizer_type.value,
        config.trainer.optimizer_params.get("lr", "N/A"),
    )
    logger.info("Teacher forcing enabled: %s", config.trainer.enable_tf_schedule)
    if config.trainer.enable_tf_schedule:
        logger.info(
            "  TF schedule: %s (%.2f -> %.2f)",
            config.trainer.tf_params.schedule_type,
            config.trainer.tf_params.start,
            config.trainer.tf_params.end,
        )
    logger.info("Starting training...")
    logger.info("")

    emulator.fit()

    # Evaluate emulator
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    logger.info("Evaluating on validation set...")
    valid_results = emulator.trainer.validation_loop(
        datamodule.rollout_val_dataloader(), valid_or_test="rollout_valid", full=True
    )

    logger.info("Evaluating on test set...")
    test_results = emulator.trainer.validation_loop(
        datamodule.rollout_test_dataloader(), valid_or_test="rollout_test", full=True
    )

    # Save model if path is provided
    if config.paths.model_save_path:
        save_path = Path(config.paths.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model to %s", save_path)
        torch.save(emulator.state_dict(), save_path)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Validation metrics:")
    for key, value in valid_results.items():
        logger.info("  %s: %s", key, value)
    logger.info("")
    logger.info("Test metrics:")
    for key, value in test_results.items():
        logger.info("  %s: %s", key, value)
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    logger.info("Outputs saved to: %s", config.paths.output_dir)
    logger.info("Log file saved to: %s/logs/", config.paths.output_dir)
    logger.info("")
    logger.info("Experiment complete!")

    return emulator, valid_results, test_results


def create_example_config(output_path: str = "example_config.yaml"):
    """Create an example configuration file."""
    from autoemulate.experimental.config_models import (
        DataConfig,
        EmulatorType,
        ExperimentConfig,
        FormatterType,
        ModelParamsConfig,
        PathsConfig,
        SimulatorConfig,
        TrainerConfig,
    )

    config = ExperimentConfig(
        experiment_name="advection_diffusion_the_well_example",
        description="Example configuration for advection-diffusion with The Well FNO",
        emulator_type=EmulatorType.THE_WELL_FNO,
        formatter_type=FormatterType.DEFAULT_CHANNELS_FIRST,
        simulator=SimulatorConfig(
            n=64,
            T=10.0,
            dt=0.1,
            return_timeseries=True,
        ),
        data=DataConfig(
            n_train_samples=200,
            n_valid_samples=4,
            n_test_samples=4,
            n_steps_input=4,
            n_steps_output=10,
            batch_size=4,
        ),
        model_params=ModelParamsConfig(
            modes1=16,
            modes2=16,
        ),
        trainer=TrainerConfig(
            epochs=10,
            max_rollout_steps=100,
            optimizer_params={"lr": 1e-3},
            device="cpu",  # Change to "cuda" or "mps" as needed
        ),
        paths=PathsConfig(
            output_dir=Path("./outputs/the_well_example"),
        ),
    )

    config.save_to_yaml(output_path)
    print(f"Example configuration saved to {output_path}")


def main():
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description="Run spatiotemporal emulation experiments using The Well"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create an example configuration file",
    )
    parser.add_argument(
        "--example-output",
        type=str,
        default="example_config.yaml",
        help="Output path for example config",
    )

    args = parser.parse_args()

    if args.create_example:
        create_example_config(args.example_output)
        return

    if not args.config:
        parser.error("--config is required (or use --create-example)")

    run_experiment(args.config, args.output_dir)


if __name__ == "__main__":
    main()
