import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
import torchinfo
import wandb
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, ModelParams, TensorLike
from autoemulate.experimental.emulators.spatiotemporal import SpatioTemporalEmulator
from the_well.benchmark import models
from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.trainer import Trainer
from the_well.data import WellDataModule
from the_well.data.datamodule import AbstractDataModule
from torch.utils.data import DataLoader


@dataclass
class TrainerParams:
    """Parameters for the Trainer."""

    # TODO: make the below configurable
    loss_fn_cls: type[torch.nn.Module] = VRMSE
    epochs: int = 10
    checkpoint_frequency: int = 5
    val_frequency: int = 1
    rollout_val_frequency: int = 1
    max_rollout_steps: int = 10
    short_validation_length: int = 20
    make_rollout_videos: bool = True
    lr_scheduler: type[torch.optim.lr_scheduler._LRScheduler] | None = None
    amp_type: str = "float16"  # bfloat not supported in FFT
    enable_amp: bool = False
    is_distributed: bool = False
    checkpoint_path: str = ""  # Path to a checkpoint to resume from, if any


class TheWellEmulator(SpatioTemporalEmulator):
    """Base class for The Well emulators."""

    model: torch.nn.Module
    model_cls: type[torch.nn.Module]
    model_parameters: ClassVar[ModelParams]

    def __init__(
        self,
        datamodule: AbstractDataModule | WellDataModule,
        output_path: str | Path | None,
        device: DeviceLike = "cpu",
        trainer_params: TrainerParams | None = None,
        *args,
        **kwargs,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        super().__init__(*args, **kwargs)

        # TODO: update path handling
        output_path = Path(output_path) if output_path else Path("./")
        checkpoint_path = output_path / "checkpoints"
        artifact_path = output_path / "artifacts"
        viz_path = artifact_path / "viz"

        # Parameters for the Trainer
        self.trainer_params = trainer_params or TrainerParams()

        # Make paths
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(artifact_path, exist_ok=True)
        os.makedirs(viz_path, exist_ok=True)
        self.datamodule = datamodule

        if isinstance(datamodule, WellDataModule):
            # Load metadata from train_dataset
            metadata = datamodule.train_dataset.metadata
            n_steps_input = datamodule.train_dataset.n_steps_input
            n_steps_output = datamodule.train_dataset.n_steps_output
            # TODO: aim to be more flexible (such as time as an input channel)
            self.n_input_fields = (
                n_steps_input * metadata.n_fields + metadata.n_constant_fields
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
            metadata = datamodule.train_dataset.metadata
            n_steps_input = datamodule.train_dataset.n_steps_input
            n_steps_output = datamodule.train_dataset.n_steps_output

            self.model = self.model_cls(
                **self.model_parameters,
                # TODO: check if general beyond FNO
                dim_in=self.n_input_fields,
                dim_out=self.n_output_fields,
                n_spatial_dims=metadata.n_spatial_dims,
                spatial_resolution=metadata.spatial_resolution,
            )
            print(torchinfo.summary(self.model, depth=5))
        else:
            msg = "Alternative datamodules not yet supported"
            raise NotImplementedError(msg)

        # init optimizer
        optimizer: torch.optim.Optimizer = kwargs.get(
            "optimizer_cls", torch.optim.Adam
        )(self.model.parameters(), lr=kwargs.get("lr", 0.001))

        # init scheduler
        lr_scheduler = (
            self.trainer_params.lr_scheduler(optimizer)
            if self.trainer_params.lr_scheduler is not None
            else None
        )
        self.trainer = Trainer(
            checkpoint_folder=str(checkpoint_path),
            artifact_folder=str(artifact_path),
            viz_folder=str(viz_path),
            formatter="channels_first_default",  # "channels_last_default" for others
            model=self.model,
            datamodule=datamodule,
            optimizer=optimizer,
            # TODO: make the below configurable
            loss_fn=self.trainer_params.loss_fn_cls(),
            epochs=self.trainer_params.epochs,
            checkpoint_frequency=self.trainer_params.checkpoint_frequency,
            val_frequency=self.trainer_params.val_frequency,
            rollout_val_frequency=self.trainer_params.rollout_val_frequency,
            max_rollout_steps=self.trainer_params.max_rollout_steps,
            short_validation_length=self.trainer_params.short_validation_length,
            make_rollout_videos=self.trainer_params.make_rollout_videos,
            num_time_intervals=n_steps_output,
            lr_scheduler=lr_scheduler,
            device=self.device,
            is_distributed=self.trainer_params.is_distributed,
            enable_amp=self.trainer_params.enable_amp,
            amp_type=self.trainer_params.amp_type,  # bfloat not supported in FFT
            checkpoint_path=self.trainer_params.checkpoint_path,
        )
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TheWellUNetClassic(TheWellEmulator):
    """The Well UNet Classic emulator."""

    model_cls: type[torch.nn.Module] = models.UNetClassic
    model_parameters: ClassVar[ModelParams] = {"init_features": 48}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TheWellUNetConvNext(TheWellEmulator):
    """The Well UNet ConvNext emulator."""

    model_cls: type[torch.nn.Module] = models.UNetConvNext
    model_parameters: ClassVar[ModelParams] = {
        "init_features": 48,
        "blocks_per_stage": 2,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
