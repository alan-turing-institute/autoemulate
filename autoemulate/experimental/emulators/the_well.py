import os
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

        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3
        )
        lr_scheduler = None

        self.trainer = Trainer(
            checkpoint_folder=str(checkpoint_path),
            artifact_folder=str(artifact_path),
            viz_folder=str(viz_path),
            # TODO: make the below configurable
            formatter="channels_first_default",  # "channels_last_default" for others
            model=self.model,
            datamodule=datamodule,
            optimizer=optimizer,
            loss_fn=VRMSE(),
            epochs=10,
            checkpoint_frequency=5,
            val_frequency=1,
            rollout_val_frequency=5,
            max_rollout_steps=10,
            short_validation_length=20,
            make_rollout_videos=False,
            num_time_intervals=n_steps_output,
            lr_scheduler=lr_scheduler,
            device=self.device,
            is_distributed=False,
            enable_amp=False,
            amp_type="float16",  # bfloat not supported in FFT
            checkpoint_path="",  # Path to a checkpoint to resume from, if any
        )

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
            preds = []
            for batch in dataloader:
                preds.append(
                    self.trainer.rollout_model(
                        self.model, batch, self.trainer.formatter, train=False
                    )
                )
            return torch.cat(preds, 0)

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
