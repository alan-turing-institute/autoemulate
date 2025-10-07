import inspect
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
from einops import rearrange
from the_well.benchmark import models
from the_well.benchmark.metrics import validation_metric_suite
from the_well.benchmark.models.common import BaseModel
from the_well.benchmark.trainer import Trainer
from the_well.data import DeltaWellDataset, WellDataModule
from the_well.data.data_formatter import AbstractDataFormatter
from the_well.data.datamodule import AbstractDataModule
from the_well.data.datasets import WellMetadata
from torch import nn
from torch.nn import functional as F
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
    # Enable scheduled teacher forcing in training
    enable_tf_schedule: bool = False
    #  start, end, schedule_epochs, schedule_type, mode, min_prob
    tf_params: dict = field(
        default_factory=lambda: {
            "start": 1.0,
            "end": 0.0,
            "schedule_epochs": None,  # fallback to total epochs
            "schedule_type": "linear",  # linear | exponential | step
            "mode": "mix",  # mix (tempering) | prob (stochastic)
            "min_prob": 1e-6,
        }
    )


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

        # Assign model, datamodule, optimizer
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

        # Flag controlling whether TF schedule is active
        self.enable_tf_schedule = trainer_params.enable_tf_schedule

        # Remaining trainer params
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
            self.device.type,
            enabled=self.enable_amp and trainer_params.amp_type != "bfloat16",
        )
        self.is_distributed = trainer_params.is_distributed
        self.best_val_loss = None
        self.starting_val_loss = float("inf")
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if self.datamodule.train_dataset.use_normalization:
            self.dset_norm = self.datamodule.train_dataset.norm
        self.formatter = formatter_cls(self.dset_metadata)
        if (
            trainer_params.checkpoint_path is not None
            and len(trainer_params.checkpoint_path) > 0
        ):
            self.load_checkpoint(trainer_params.checkpoint_path)

        # Teacher Forcing scheduler setup
        tf_params = trainer_params.tf_params
        self.tf_start = float(tf_params.get("start", 1.0))
        self.tf_end = float(tf_params.get("end", 0.0))
        self.tf_mode = tf_params.get("mode", "mix")
        self.tf_type = tf_params.get("schedule_type", "linear")
        self.tf_epochs = tf_params.get("schedule_epochs") or self.max_epoch
        self.tf_min_prob = float(tf_params.get("min_prob", 1e-6))

        # Initialize current_epoch so scheduling logic has a defined value pre-training
        self.current_epoch = self.starting_epoch

    def train_one_epoch(self, epoch: int, dataloader) -> float:
        """Override to expose the current epoch to scheduling utilities.

        Sets `self.current_epoch` before delegating to the base implementation so
        `_teacher_forcing_ratio` can derive a reliable epoch index without
        re-implementing the full training loop from the upstream Trainer.
        """
        self.current_epoch = epoch

        # Defer to base trainer to perform training and collect logs
        result = super().train_one_epoch(epoch, dataloader)
        if isinstance(result, tuple) and len(result) == 2:
            epoch_loss, train_logs = result
        else:  # Upstream type hint mismatch safeguard
            epoch_loss, train_logs = result, {}

        # Augment logs with scheduled TF ratio (0 if disabled)
        try:
            train_logs["tf/ratio_scheduled"] = float(self._teacher_forcing_ratio())
        # pragma: no cover - defensive
        except Exception:
            train_logs["tf/ratio_scheduled"] = float("nan")

        return epoch_loss, train_logs  # type: ignore as this is the upstream signature

    def _teacher_forcing_ratio(self) -> float:
        """Compute the scheduled teacher forcing ratio for the current epoch.

        Returns 0.0 immediately if scheduling is disabled.
        """
        if not self.enable_tf_schedule:
            return 0.0
        e = max(int(self.current_epoch) - 1, 0)
        total = max(self.tf_epochs - 1, 1)
        progress = min(e / total, 1.0)
        # Base linear interpolation used as default
        linear_val = self.tf_start + (self.tf_end - self.tf_start) * progress

        if self.tf_type == "exponential":
            # Solve start * gamma^e = end at e=total => gamma = (end/start)^(1/total)
            if self.tf_start > 0 and self.tf_end > 0:
                gamma = (self.tf_end / self.tf_start) ** (1 / total)
                r = self.tf_start * (gamma**e)
            else:  # fall back to linear if invalid bounds
                r = linear_val
        elif self.tf_type == "step":
            r = self.tf_start if e < total else self.tf_end
        else:  # linear or unknown -> use linear interpolation
            r = linear_val
        # Clamp to [0.0, 1.0] numerical floor
        return float(max(min(r, 1.0), 0.0))

    def rollout_model(
        self,
        model,
        batch,
        formatter,
        train: bool = True,
        teacher_forcing: bool = False,
        tf_ratio: float | None = None,
    ):
        """Roll out the model.

        Parameters
        ----------
        model: nn.Module
            The model to evaluate.
        batch: dict
            Batch produced by the dataloader (contains `input_fields` and optional
            `constant_fields` plus targets).
        formatter: AbstractDataFormatter
            Formatter that converts batch tensors to model inputs / outputs.
        train: bool
            If True, uses the scheduled teacher forcing ratio (unless `tf_ratio` is
            explicitly provided). If False, no schedule is applied. Defaults to True.
        teacher_forcing: bool
            Backwards-compatible flag. When evaluating (`train=False`) and `tf_ratio` is
            not provided, `teacher_forcing=True` implies full teacher forcing
            (ratio = 1.0). During training this flag is ignored because scheduled
            teacher forcing is always active. Defaults to False.
        tf_ratio: float | None
            Explicit teacher forcing ratio override in `[0, 1]`. Highest precedence: if
            provided it is used directly (clamped) regardless of mode (train/eval) or
            schedule. This allows ad-hoc evaluation at a fixed ratio or reproducing the
            original full teacher forcing rollout with `tf_ratio=1.0`. Defaults to None.

        Notes
        -----
        Precedence:
        1. `tf_ratio` argument (if not None)
        2. Training schedule (`train=True`)
        3. Full TF on eval if `teacher_forcing=True`
        4. No teacher forcing
        """
        inputs, y_ref = formatter.process_input(batch)
        rollout_steps = min(
            y_ref.shape[1], self.max_rollout_steps
        )  # Number of timesteps in target
        y_ref = y_ref[:, :rollout_steps].to(self.device)
        # Create a moving batch of one step at a time
        moving_batch = batch
        moving_batch["input_fields"] = moving_batch["input_fields"].to(self.device)
        if "constant_fields" in moving_batch:
            moving_batch["constant_fields"] = moving_batch["constant_fields"].to(
                self.device
            )
        y_preds = []

        # Calculate the tf_ratio using precedence rules
        def _resolve_tf_ratio():
            if tf_ratio is not None:
                return float(max(min(tf_ratio, 1.0), 0.0))
            if train and self.enable_tf_schedule:
                return self._teacher_forcing_ratio()
            if teacher_forcing:
                return 1.0
            return 0.0

        effective_tf_ratio = _resolve_tf_ratio()

        use_tf = effective_tf_ratio > 0.0

        for i in range(rollout_steps):
            if not train:
                moving_batch = self.normalize(moving_batch)

            inputs, _ = formatter.process_input(moving_batch)
            inputs = [x.to(self.device) for x in inputs]
            y_pred = model(*inputs)

            y_pred = formatter.process_output_channel_last(y_pred)

            if not train:
                moving_batch, y_pred = self.denormalize(moving_batch, y_pred)

            if (not train) and self.is_delta:
                # TODO: update to handle case when more than single time step
                assert {
                    moving_batch["input_fields"][:, -1, ...].shape == y_pred.shape
                }, (
                    f"Mismatching shapes between last input timestep "
                    f"{moving_batch[:, -1, ...].shape} and prediction {y_pred.shape}"
                )
                y_pred = moving_batch["input_fields"][:, -1, ...] + y_pred
            y_pred = formatter.process_output_expand_time(y_pred)
            # If not last step, update moving batch
            if i != rollout_steps - 1:
                next_moving_batch_tail = moving_batch["input_fields"][:, 1:]
                if use_tf:
                    if self.tf_mode == "prob":
                        # Sample Bernoulli per batch element deciding to use GT vs pred
                        mask = (
                            torch.rand(
                                y_pred.shape[0],  # generate a mask per batch
                                1,
                                *([1] * (y_pred.dim() - 2)),
                                device=y_pred.device,
                            )
                            < effective_tf_ratio
                        ).to(y_pred.dtype)
                        mixed = mask * y_ref[:, i : i + 1] + (1 - mask) * y_pred
                        next_moving_batch = torch.cat(
                            [next_moving_batch_tail, mixed], dim=1
                        )
                    else:  # mix/tempering
                        mixed = (
                            effective_tf_ratio * y_ref[:, i : i + 1]
                            + (1 - effective_tf_ratio) * y_pred
                        )
                        next_moving_batch = torch.cat(
                            [next_moving_batch_tail, mixed], dim=1
                        )
                else:
                    # Fully free running
                    next_moving_batch = torch.cat(
                        [next_moving_batch_tail, y_pred], dim=1
                    )
                moving_batch["input_fields"] = next_moving_batch
            y_preds.append(y_pred)
        y_pred_out = torch.cat(y_preds, dim=1)
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref


class TheWellEmulator(SpatioTemporalEmulator):
    """Base class for The Well emulators."""

    model: torch.nn.Module
    model_cls: type[torch.nn.Module]
    model_parameters: ClassVar[ModelParams]
    with_time: bool = False

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
        # Init base without nn.Module kwargs
        super().__init__()

        # Split incoming kwargs into those intended for the model class vs others.
        # Anything matching the model's __init__ signature (excluding self) is
        # treated as a model override and merged with class-level model_parameters.
        model_sig = inspect.signature(self.model_cls.__init__).parameters
        allowed_model_keys = {
            name
            for name, p in model_sig.items()
            if name != "self"
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        provided_model_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in allowed_model_keys
        }
        # Combine defaults with provided overrides
        self._model_kwargs = {**self.model_parameters, **provided_model_kwargs}

        # Set output path
        output_path = Path(self.trainer_params.output_path)

        # Set datamodule
        self.datamodule = datamodule

        if isinstance(datamodule, WellDataModule):
            # Load metadata from train_dataset
            metadata = datamodule.train_dataset.metadata
            self.n_steps_input = datamodule.train_dataset.n_steps_input
            self.n_steps_output = datamodule.train_dataset.n_steps_output

            # Determine whether the model expects time as an explicit dimension
            # For such models, channels should repr fields(+constants), not time*fields
            if self.with_time:
                self.n_input_fields = metadata.n_fields + metadata.n_constant_fields
            else:
                self.n_input_fields = (
                    self.n_steps_input * metadata.n_fields + metadata.n_constant_fields
                )
            self.n_output_fields = metadata.n_fields
            self.model = self.model_cls(
                **self._model_kwargs,
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


class FNOWithTime(BaseModel):
    """FNO with time."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        modes_time: int,
        modes1: int,
        modes2: int,
        modes3: int = 16,
        hidden_channels: int = 64,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.modes_time = modes_time
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False
        self.gradient_checkpointing = gradient_checkpointing

        if self.n_spatial_dims == 2:
            self.n_modes = (self.modes_time, self.modes1, self.modes2)
        elif self.n_spatial_dims == 3:
            self.n_modes = (self.modes_time, self.modes1, self.modes2, self.modes3)

        self.model = models.fno.NeuralOpsCheckpointWrapper(
            n_modes=self.n_modes,
            in_channels=self.dim_in,
            out_channels=self.dim_out,
            hidden_channels=self.hidden_channels,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, input) -> torch.Tensor:  # noqa: D102
        return self.model(input)  # type: ignore  # noqa: PGH003


class TheWellFNO(TheWellEmulator):
    """The Well FNO emulator."""

    model_cls: type[torch.nn.Module] = models.FNO
    model_parameters: ClassVar[ModelParams] = {
        "modes1": 16,
        "modes2": 16,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TheWellFNOWithTime(TheWellEmulator):
    """The Well FNO emulator."""

    with_time: bool = True
    model_cls: type[torch.nn.Module] = FNOWithTime
    model_parameters: ClassVar[ModelParams] = {
        "modes_time": 3,
        "modes1": 16,
        "modes2": 16,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DefaultChannelsFirstFormatterWithTime(AbstractDataFormatter):
    """
    Default preprocessor for data in channels first format.

    Stacks time as individual channel.
    """

    def process_input(self, data: dict) -> tuple:  # noqa: D102
        x = data["input_fields"]
        x = rearrange(x, "b ... c -> b c ...")  # Move channels to before batch
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], "b ... c -> b c 1 ...")
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return rearrange(output, "b c ... -> b ... c")

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # Time does not need to be expanded as it is already included
        output = rearrange(output, "b ... c -> b ... c")
        # Only take the first temporal slice at the moment since predictions are only
        # for one step ahead
        return output[:, :1, ...]


class TheWellFNOWithLearnableWeights(TheWellEmulator):
    """The Well FNO emulator with learnable weights."""

    model_cls: type[torch.nn.Module] = models.FNO
    model_parameters: ClassVar[ModelParams] = {
        "modes1": 16,
        "modes2": 16,
    }

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
        # Skip TheWellEmulator init as overriden here
        SpatioTemporalEmulator.__init__(self, **kwargs)

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

        # Learnable weights for loss function
        self.weights = nn.Parameter(
            torch.ones(self.n_steps_output, device=self.device), requires_grad=True
        )

        # Assign given metric as base loss function
        self.base_loss_func = loss_fn

        # Init trainer
        self.trainer = AutoEmulateTrainer(
            loss_fn=self.custom_loss_fn,  # use custom loss fn as callable in trainer
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

    def custom_loss_fn(self, y_pred, y_target, meta: WellMetadata):
        """Loss function that uses parameters constructed at init."""
        # Make positive
        w = F.softplus(self.weights)

        # Normalize so mean(w) == 1
        w = w / (w.mean() + 1e-12)

        # Reshape to (1, n_steps, spatial_dims, channels) for broadcasting
        w = w.view(1, -1, *([1] * meta.n_spatial_dims), 1)

        # Pad with 1s along the time dimension if needed to match y_pred shape
        if w.shape[1] != y_pred.shape[1]:
            extra = y_pred.shape[1] - w.shape[1]
            pad_shape = (1, extra, *([1] * meta.n_spatial_dims), 1)
            ones = torch.ones(pad_shape, device=w.device, dtype=w.dtype)
            w = torch.cat([w, ones], dim=1)

        # Return a weighted loss
        return self.base_loss_func(w * y_pred, w * y_target, meta)


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
