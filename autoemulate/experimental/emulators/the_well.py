import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torchinfo
import wandb
from autoemulate.core.device import TorchDeviceMixin, get_torch_device
from autoemulate.core.types import DeviceLike, ModelParams, TensorLike
from autoemulate.experimental.emulators.spatiotemporal import SpatioTemporalEmulator
from the_well.benchmark import models
from the_well.benchmark.metrics import validation_metric_suite
from the_well.benchmark.trainer import Trainer
from the_well.data import DeltaWellDataset, WellDataModule
from the_well.data.data_formatter import AbstractDataFormatter
from the_well.data.datamodule import AbstractDataModule
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainerParams:
    """Parameters for the Trainer."""

    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_params: dict = field(default_factory=lambda: {"lr": 1e-3})
    epochs: int = 10
    checkpoint_frequency: int = 5
    val_frequency: int = 1
    rollout_val_frequency: int = 1
    max_rollout_steps: int = 10
    short_validation_length: int = 20
    make_rollout_videos: bool = True
    lr_scheduler: type[torch.optim.lr_scheduler._LRScheduler] | None = None
    amp_type: str = "float16"  # bfloat not supported in FFT
    num_time_intervals: int = 5
    enable_amp: bool = False
    is_distributed: bool = False
    checkpoint_path: str = ""  # Path to a checkpoint to resume from, if any
    device: DeviceLike = "cpu"
    output_path: str = "./"


class AutoEmulateTrainer(Trainer):
    """AutoEmulate trainer."""

    def __init__(
        self,
        output_path: Path | str,
        formatter_cls: type[AbstractDataFormatter],
        model: nn.Module,
        loss_fn: Callable,
        datamodule: WellDataModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: _LRScheduler | None,
        trainer_params: TrainerParams,
    ):
        """Subclass to integrate with AutoEmulate framework and extend functionality.

        Parameters
        ----------
        output_path: Path | str
            Base path to be used for outputs.
        formatter: AbstractDataFormatter
            A data formatter that handles the formatting of the data for the model.
        model: nn.Module
            A PyTorch model to train
        datamodule:
            A datamodule that provides dataloaders for each split (train, valid, and
            test).
        loss_fn: Callable
            A loss function to use for training and validation. This can also be a
            trainable nn.Module providing that it is included in the model parameters.
        trainer_params: TrainerParams
            Parameters for the trainer.
        """
        self.starting_epoch = 1  # Gets overridden on resume

        # Paths
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        self.checkpoint_folder = str(output_path / "checkpoints")
        artifact_path = output_path / "artifacts"
        self.artifact_folder = str(artifact_path)
        self.viz_folder = str(artifact_path / "viz")
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.artifact_folder, exist_ok=True)
        os.makedirs(self.viz_folder, exist_ok=True)

        # Device setup
        self.device = get_torch_device(trainer_params.device)
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.is_delta = isinstance(datamodule.train_dataset, DeltaWellDataset)
        self.validation_suite = [*validation_metric_suite, self.loss_fn]
        self.max_epoch = trainer_params.epochs
        self.checkpoint_frequency = trainer_params.checkpoint_frequency
        self.val_frequency = trainer_params.val_frequency
        self.rollout_val_frequency = trainer_params.rollout_val_frequency
        self.max_rollout_steps = trainer_params.max_rollout_steps
        self.short_validation_length = trainer_params.short_validation_length
        self.make_rollout_videos = trainer_params.make_rollout_videos
        self.num_time_intervals = trainer_params.num_time_intervals
        self.enable_amp = trainer_params.enable_amp
        self.amp_type = (
            torch.bfloat16 if trainer_params.amp_type == "bfloat16" else torch.float16
        )
        self.grad_scaler = torch.GradScaler(
            self.device.type, enabled=self.enable_amp and self.amp_type != "bfloat16"
        )
        self.is_distributed = trainer_params.is_distributed
        self.best_val_loss = None
        self.starting_val_loss = float("inf")
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if self.datamodule.train_dataset.use_normalization:
            self.dset_norm = self.datamodule.train_dataset.norm
        self.formatter = formatter_cls(self.dset_metadata)
        if len(trainer_params.checkpoint_path) > 0:
            self.load_checkpoint(trainer_params.checkpoint_path)


class TheWellEmulator(SpatioTemporalEmulator):
    """Base class for The Well emulators."""

    model: torch.nn.Module
    model_cls: type[torch.nn.Module]
    model_parameters: ClassVar[ModelParams]

    def __init__(
        self,
        datamodule: AbstractDataModule | WellDataModule,
        formatter_cls: type[AbstractDataFormatter],
        loss_fn: Callable,
        trainer_params: TrainerParams | None = None,
        **kwargs,
    ):
        # Parameters for the Trainer
        self.trainer_params = trainer_params or TrainerParams()

        # Device setup and backend init
        TorchDeviceMixin.__init__(
            self, device=get_torch_device(self.trainer_params.device)
        )
        super().__init__(**kwargs)

        # Set output path
        output_path = Path(self.trainer_params.output_path)

        # Set datamodule
        self.datamodule = datamodule

        if isinstance(datamodule, WellDataModule):
            # Load metadata from train_dataset
            metadata = datamodule.train_dataset.metadata
            self.n_steps_input = datamodule.train_dataset.n_steps_input
            self.n_steps_output = datamodule.train_dataset.n_steps_output
            # TODO: aim to be more flexible (such as time as an input channel)
            self.n_input_fields = (
                self.n_steps_input * metadata.n_fields + metadata.n_constant_fields
            )
            self.n_output_fields = metadata.n_fields
            self.model = self.model_cls(
                **self.model_parameters,
                # TODO: check if general beyond FNO
                dim_in=self.n_input_fields,
                dim_out=self.n_output_fields,
                n_spatial_dims=metadata.n_spatial_dims,
                spatial_resolution=metadata.spatial_resolution,
            )
            # TODO: update with logging
            print(torchinfo.summary(self.model, depth=5))
        else:
            msg = "Alternative datamodules not yet supported"
            raise NotImplementedError(msg)

        # Init optimizer
        optimizer = self.trainer_params.optimizer_cls(
            self.model.parameters(), **self.trainer_params.optimizer_params
        )

        # Init scheduler
        lr_scheduler = (
            self.trainer_params.lr_scheduler(optimizer)
            if self.trainer_params.lr_scheduler is not None
            else None
        )

        # Init trainer
        self.trainer = AutoEmulateTrainer(
            loss_fn=loss_fn,
            output_path=output_path,
            formatter_cls=formatter_cls,
            model=self.model,
            datamodule=datamodule,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            trainer_params=self.trainer_params,
        )

        # Move to device
        # TODO: check if this needs updating for distributed handling
        self.to(self.device)

    def _fit(
        self,
        x: TensorLike | DataLoader | None = None,  # noqa: ARG002
        y: TensorLike | None = None,  # noqa: ARG002
        **kwargs,
    ):
        """Train a spatio-temporal emulator.

        Parameters
        ----------
        x: TensorLike | DataLoader | None
            Input features as `TensorLike` or `DataLoader`.
        y: OutputLike | None
            Target values (not needed if x is a DataLoader).
        """
        # TODO: placeholder to disable wandb for now, make configurable later
        wandb.init(mode="disabled")
        self.trainer.train()

    def _predict(self, x: TensorLike | DataLoader | None, with_grad=False):
        """Predict using the spatio-temporal emulator."""
        self.eval()
        with torch.set_grad_enabled(with_grad):
            if isinstance(x, DataLoader):
                dataloader = x
            else:
                msg = "x must be a DataLoader"
                raise ValueError(msg)
            preds, refs = [], []
            for batch in dataloader:
                pred = self.trainer.rollout_model(
                    self.model, batch, self.trainer.formatter, train=False
                )
                preds.append(pred[0])
                refs.append(pred[1])
            # Just return preds for now but could also retiurn refs if needed
            return torch.cat(preds)
            # return torch.cat(preds), torch.cat(refs)

    def predict_autoregressive(  # noqa: D102 # type: ignore [override] # (n_steps not used) and initial state derived from dataloader
        self, x: TensorLike | DataLoader | None, with_grad=False
    ):
        return self._predict(x, with_grad)

    @staticmethod
    def is_multioutput() -> bool:
        """Check if the model is multi-output."""
        return True


class TheWellFNO(TheWellEmulator):
    """The Well FNO emulator."""

    model_cls: type[torch.nn.Module] = models.FNO
    model_parameters: ClassVar[ModelParams] = {
        "modes1": 16,
        "modes2": 16,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# TODO: fix this as not initializing correctly at the moment
class TheWellAFNO(TheWellEmulator):
    """The Well AFNO emulator."""

    model_cls: type[torch.nn.Module] = models.AFNO
    model_parameters: ClassVar[ModelParams] = {
        "hidden_dim": 64,
        "n_blocks": 14,
    }

    def __init__(
        self,
        datamodule: AbstractDataModule | WellDataModule,
        formatter_cls: type[AbstractDataFormatter],
        loss_fn: Callable,
        trainer_params: TrainerParams | None = None,
        **kwargs,
    ):
        super().__init__(
            loss_fn=loss_fn,
            datamodule=datamodule,
            formatter_cls=formatter_cls,
            trainer_params=trainer_params,
            **kwargs,
        )


class TheWellUNetClassic(TheWellEmulator):
    """The Well UNet Classic emulator."""

    model_cls: type[torch.nn.Module] = models.UNetClassic
    model_parameters: ClassVar[ModelParams] = {"init_features": 48}

    def __init__(
        self,
        datamodule: AbstractDataModule | WellDataModule,
        formatter_cls: type[AbstractDataFormatter],
        loss_fn: Callable,
        trainer_params: TrainerParams | None = None,
        **kwargs,
    ):
        super().__init__(
            loss_fn=loss_fn,
            datamodule=datamodule,
            formatter_cls=formatter_cls,
            trainer_params=trainer_params,
            **kwargs,
        )


class TheWellUNetConvNext(TheWellEmulator):
    """The Well UNet ConvNext emulator."""

    model_cls: type[torch.nn.Module] = models.UNetConvNext
    model_parameters: ClassVar[ModelParams] = {
        "init_features": 48,
        "blocks_per_stage": 2,
    }

    def __init__(
        self,
        datamodule: AbstractDataModule | WellDataModule,
        formatter_cls: type[AbstractDataFormatter],
        loss_fn: Callable,
        trainer_params: TrainerParams | None = None,
        **kwargs,
    ):
        super().__init__(
            loss_fn=loss_fn,
            datamodule=datamodule,
            formatter_cls=formatter_cls,
            trainer_params=trainer_params,
            **kwargs,
        )
