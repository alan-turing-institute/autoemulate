from pathlib import Path

import h5py
import torch
from autoemulate.core.types import TensorLike
from the_well.data.datamodule import AbstractDataModule, WellDataModule  # noqa: F401
from the_well.data.datasets import WellMetadata
from torch.utils.data import DataLoader, Dataset


class SimpleZScoreNormalization:
    """Simple Z-score normalization for AutoEmulate datasets.

    Computes mean and std from the data and applies (x - mean) / std normalization.
    Compatible with The Well's Trainer interface.
    """

    def __init__(self, data: torch.Tensor, min_std: float = 1e-6):
        """Initialize normalization with data statistics.

        Parameters
        ----------
        data : torch.Tensor
            Training data tensor of shape [N, T, W, H, C]
        min_std : float
            Minimum standard deviation to avoid division by zero
        """
        # Compute statistics over all dimensions except channels
        # Shape: [C] - one mean/std per channel
        self.mean = data.mean(dim=(0, 1, 2, 3))
        self.std = data.std(dim=(0, 1, 2, 3)).clamp(min=min_std)
        self.min_std = min_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., C]

        Returns
        -------
        torch.Tensor
            Normalized tensor
        """
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor of shape [..., C]

        Returns
        -------
        torch.Tensor
            Denormalized tensor
        """
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def normalize_flattened(
        self,
        x: torch.Tensor,
        mode: str = "variable",  # noqa: ARG002
    ) -> torch.Tensor:
        """Normalize tensor where fields are flattened as channels.

        Compatible with The Well's Trainer interface.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., C]
        mode : str
            Mode for normalization ("variable" or "constant").
            Currently both use same stats.

        Returns
        -------
        torch.Tensor
            Normalized tensor
        """
        return self.normalize(x)

    def denormalize_flattened(
        self,
        x: torch.Tensor,
        mode: str = "variable",  # noqa: ARG002
    ) -> torch.Tensor:
        """Denormalize tensor where fields are flattened as channels.

        Compatible with The Well's Trainer interface.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor of shape [..., C]
        mode : str
            Mode for denormalization ("variable" or "constant").
            Currently both use same stats.

        Returns
        -------
        torch.Tensor
            Denormalized tensor
        """
        return self.denormalize(x)

    def delta_denormalize_flattened(
        self,
        x: torch.Tensor,
        mode: str = "variable",  # noqa: ARG002
    ) -> torch.Tensor:
        """Delta denormalize tensor (for delta models).

        For simple Z-score normalization, delta denormalization only
        requires std scaling (no mean shift needed).

        Parameters
        ----------
        x : torch.Tensor
            Normalized delta tensor of shape [..., C]
        mode : str
            Mode for denormalization ("variable" or "constant").
            Currently both use same stats.

        Returns
        -------
        torch.Tensor
            Denormalized delta tensor
        """
        return x * self.std.to(x.device)


class AutoEmulateDataset(Dataset):
    """A class for spatio-temporal datasets."""

    def __init__(
        self,
        data_path: str | None,
        data: dict | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        # TODO: support for passing data from dict
        input_channel_idxs: tuple[int, ...] | None = None,
        output_channel_idxs: tuple[int, ...] | None = None,
        full_trajectory_mode: bool = False,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        use_normalization: bool = False,
        norm: SimpleZScoreNormalization | None = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path: str
            Path to the HDF5 file containing the dataset.
        n_steps_input: int
            Number of input time steps.
        n_steps_output: int
            Number of output time steps.
        stride: int
            Stride for sampling the data.
        data: dict | None
            Preloaded data. Defaults to None.
        input_channel_idxs: tuple[int, ...] | None
            Indices of input channels to use. Defaults to None.
        output_channel_idxs: tuple[int, ...] | None
            Indices of output channels to use. Defaults to None.
        full_trajectory_mode: bool
            If True, use full trajectories without creating subtrajectories.
        dtype: torch.dtype
            Data type for tensors. Defaults to torch.float32.
        verbose: bool
            If True, print dataset information.
        use_normalization: bool
            Whether to apply Z-score normalization. Defaults to False.
        norm: SimpleZScoreNormalization | None
            Normalization object (computed from training data). Defaults to None.
        """
        self.dtype = dtype
        self.verbose = verbose
        self.use_normalization = use_normalization
        self.norm = norm

        # Read or parse data
        self.read_data(data_path) if data_path is not None else self.parse_data(data)

        self.full_trajectory_mode = full_trajectory_mode
        self.n_steps_input = n_steps_input
        self.n_steps_output = (
            n_steps_output
            if not self.full_trajectory_mode
            # TODO: make more robust and flexible for different trajectory lengths
            else self.data.shape[1] - self.n_steps_input
        )
        self.stride = stride
        self.input_channel_idxs = input_channel_idxs
        self.output_channel_idxs = output_channel_idxs

        # Destructured here
        (
            self.n_trajectories,
            self.n_timesteps,
            self.width,
            self.height,
            self.n_channels,
        ) = self.data.shape

        # Pre-compute all subtrajectories for efficient indexing
        self.all_input_fields = []
        self.all_output_fields = []
        self.all_constant_scalars = []
        self.all_constant_fields = []

        for traj_idx in range(self.n_trajectories):
            # Create subtrajectories for this trajectory
            fields = (
                self.data[traj_idx]
                .unfold(0, self.n_steps_input + self.n_steps_output, self.stride)
                .permute(0, -1, 1, 2, 3)  # [num_subtrajectories, T_in + T_out, W, H, C]
            )

            # Split into input and output
            input_fields = fields[
                :, : self.n_steps_input, ...
            ]  # [num_subtrajectories, T_in, W, H, C]
            output_fields = fields[
                :, self.n_steps_input :, ...
            ]  # [num_subtrajectories, T_out, W, H, C]

            # Store each subtrajectory separately
            for sub_idx in range(input_fields.shape[0]):
                self.all_input_fields.append(
                    input_fields[sub_idx].to(self.dtype)
                )  # [T_in, W, H, C]
                self.all_output_fields.append(
                    output_fields[sub_idx].to(self.dtype)
                )  # [T_out, W, H, C]

                # Handle constant scalars
                if self.constant_scalars is not None:
                    self.all_constant_scalars.append(
                        self.constant_scalars[traj_idx].to(self.dtype)
                    )

                # Handle constant fields
                if self.constant_fields is not None:
                    self.all_constant_fields.append(
                        self.constant_fields[traj_idx].to(self.dtype)
                    )

        if self.verbose:
            print(f"Created {len(self.all_input_fields)} subtrajectory samples")
            print(f"Each input sample shape: {self.all_input_fields[0].shape}")
            print(f"Each output sample shape: {self.all_output_fields[0].shape}")
            print(f"Data type: {self.all_input_fields[0].dtype}")

    def _from_f(self, f):
        assert "data" in f, "HDF5 file must contain 'data' dataset"
        self.data: TensorLike = torch.Tensor(f["data"][:]).to(self.dtype)  # type: ignore # [N, T, W, H, C]  # noqa: PGH003
        if self.verbose:
            print(f"Loaded data shape: {self.data.shape}")
        # TODO: add the constant scalars
        self.constant_scalars = (
            torch.Tensor(f["constant_scalars"][:]).to(self.dtype)  # type: ignore  # noqa: PGH003
            if "constant_scalars" in f
            else None
        )  # [N, C]

        # Constant fields
        self.constant_fields = (
            torch.Tensor(f["constant_fields"][:]).to(  # type: ignore # noqa: PGH003
                self.dtype
            )  # [N, W, H, C]
            if "constant_fields" in f and f["constant_fields"] != {}
            else None
        )

    def read_data(self, data_path: str):
        """Read data.

        By default assumes HDF5 format in `data_path` with correct shape and fields.
        """
        self.data_path = data_path
        if self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
            with h5py.File(self.data_path, "r") as f:
                self._from_f(f)
        if self.data_path.endswith(".pt"):
            self._from_f(torch.load(self.data_path))

    def parse_data(self, data: dict | None):
        """Parse data from a dictionary."""
        if data is not None:
            self.data = (
                data["data"].to(self.dtype)
                if torch.is_tensor(data["data"])
                else torch.tensor(data["data"], dtype=self.dtype)
            )
            self.constant_scalars = data.get("constant_scalars", None)
            self.constant_fields = data.get("constant_fields", None)
            return
        msg = "No data provided to parse."
        raise ValueError(msg)

    def __len__(self):  # noqa: D105
        return len(self.all_input_fields)

    def __getitem__(self, idx):  # noqa: D105
        input_fields = self.all_input_fields[idx]
        output_fields = self.all_output_fields[idx]

        item = {
            "input_fields": input_fields,
            "output_fields": output_fields,
        }
        if len(self.all_constant_scalars) > 0:
            item["constant_scalars"] = self.all_constant_scalars[idx]
        if len(self.all_constant_fields) > 0:
            item["constant_fields"] = self.all_constant_fields[idx]

        return item


class MHDDataset(AutoEmulateDataset):
    """PyTorch Dataset for MHD data."""

    def __init__(self, data_path: str, t_in: int = 5, t_out: int = 10, stride: int = 1):
        super().__init__(
            data_path, n_steps_input=t_in, n_steps_output=t_out, stride=stride
        )


class ReactionDiffusionDataset(AutoEmulateDataset):
    """Reaction-Diffusion dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = WellMetadata(
            dataset_name="ReactionDiffusion",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=["beta", "d"],
            field_names={0: ["U", "V"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


class AdvectionDiffusionDataset(AutoEmulateDataset):
    """Advection-Diffusion dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = WellMetadata(
            dataset_name="AdvectionDiffusion",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=["nu", "mu"],
            field_names={0: ["vorticity"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


class BOUTDataset(AutoEmulateDataset):
    """BOUT++ dataset."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = WellMetadata(
            dataset_name="BOUT++",
            n_spatial_dims=2,
            spatial_resolution=self.data.shape[-3:-1],
            scalar_names=[],
            constant_scalar_names=[
                f"const{i}"
                for i in range(self.constant_scalars.shape[-1])  # type: ignore  # noqa: PGH003
            ],
            field_names={0: ["vorticity"]},
            constant_field_names={},
            boundary_condition_types=["periodic", "periodic"],
            n_files=0,
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[self.data.shape[1]] * self.data.shape[0],
            grid_type="cartesian",
        )


# class AutoEmulateDataModule(AbstractDataModule):
class AutoEmulateDataModule(WellDataModule):
    """A class for spatio-temporal data modules."""

    def __init__(
        self,
        data_path: str | None,
        data: dict[str, dict] | None = None,
        dataset_cls: type[AutoEmulateDataset] = AutoEmulateDataset,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        stride: int = 1,
        # TODO: support for passing data from dict
        input_channel_idxs: tuple[int, ...] | None = None,
        output_channel_idxs: tuple[int, ...] | None = None,
        batch_size: int = 4,
        dtype: torch.dtype = torch.float32,
        ftype: str = "torch",
        verbose: bool = False,
        use_normalization: bool = False,
    ):
        self.verbose = verbose
        self.use_normalization = use_normalization

        base_path = Path(data_path) if data_path is not None else None
        suffix = ".pt" if ftype == "torch" else ".h5"
        fname = f"data{suffix}"
        train_path = base_path / "train" / fname if base_path is not None else None
        valid_path = base_path / "valid" / fname if base_path is not None else None
        test_path = base_path / "test" / fname if base_path is not None else None

        # Create training dataset first (without normalization)
        self.train_dataset = dataset_cls(
            data_path=str(train_path) if train_path is not None else None,
            data=data["train"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=False,  # Temporarily disable to compute stats
            norm=None,
        )

        # Compute normalization from training data if requested
        norm = None
        if self.use_normalization:
            if self.verbose:
                print("Computing normalization statistics from training data...")
            norm = SimpleZScoreNormalization(self.train_dataset.data)
            if self.verbose:
                print(f"  Mean (per channel): {norm.mean}")
                print(f"  Std (per channel): {norm.std}")

            # Now enable normalization for training dataset
            self.train_dataset.use_normalization = True
            self.train_dataset.norm = norm

        self.val_dataset = dataset_cls(
            data_path=str(valid_path) if valid_path is not None else None,
            data=data["valid"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.test_dataset = dataset_cls(
            data_path=str(test_path) if test_path is not None else None,
            data=data["test"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.rollout_val_dataset = dataset_cls(
            data_path=str(train_path) if train_path is not None else None,
            data=data["train"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            full_trajectory_mode=True,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.rollout_test_dataset = dataset_cls(
            data_path=str(test_path) if test_path is not None else None,
            data=data["test"] if data is not None else None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
            full_trajectory_mode=True,
            dtype=dtype,
            verbose=self.verbose,
            use_normalization=self.use_normalization,
            norm=norm,
        )
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """DataLoader for training."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader for standard validation (not full trajectory rollouts)."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def rollout_val_dataloader(self) -> DataLoader:
        """DataLoader for full trajectory rollouts on validation data."""
        return DataLoader(
            self.rollout_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader for testing."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def rollout_test_dataloader(self) -> DataLoader:
        """DataLoader for full trajectory rollouts on test data."""
        return DataLoader(
            self.rollout_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
