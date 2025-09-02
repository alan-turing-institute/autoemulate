import h5py
import torch
from autoemulate.core.types import TensorLike
from torch.utils.data import Dataset


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

        """
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.stride = stride
        self.input_channel_idxs = input_channel_idxs
        self.output_channel_idxs = output_channel_idxs

        # Read or parse data
        self.read_data(data_path) if data_path is not None else self.parse_data(data)

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
                self.all_input_fields.append(input_fields[sub_idx])  # [T_in, W, H, C]
                self.all_output_fields.append(
                    output_fields[sub_idx]
                )  # [T_out, W, H, C]

                # Handle constant scalars
                if self.constant_scalars is not None:
                    self.all_constant_scalars.append(self.constant_scalars[traj_idx])
                else:
                    self.all_constant_scalars.append(torch.tensor([]))

        print(f"Created {len(self.all_input_fields)} subtrajectory samples")
        print(f"Each input sample shape: {self.all_input_fields[0].shape}")
        print(f"Each output sample shape: {self.all_output_fields[0].shape}")

    def read_data(self, data_path: str):
        """Read data.

        By default assumes HDF5 format in `data_path` with correct shape and fields.
        """
        # TODO: support passing as dict
        # Load data
        self.data_path = data_path
        with h5py.File(self.data_path, "r") as f:
            assert "data" in f, "HDF5 file must contain 'data' dataset"
            self.data: TensorLike = torch.Tensor(f["data"][:])  # type: ignore # [N, T, W, H, C]  # noqa: PGH003
        print(f"Loaded data shape: {self.data.shape}")
        # TODO: add the constant scalars
        self.constant_scalars = (
            torch.Tensor(f["constant_scalars"][:])  # type: ignore  # noqa: PGH003
            if "constant_scalars" in f
            else None
        )  # [N, C]
        # TODO: add the constant fields
        # self.constant_fields = torch.Tensor(f['data'][:])  # [N, W, H, C]

    def parse_data(self, data: dict | None):
        """Parse data from a dictionary."""
        if data is not None:
            self.data = data["data"]
            self.constant_scalars = data.get("constant_scalars", None)
            self.constant_fields = data.get("constant_fields", None)
            return
        msg = "No data provided to parse."
        raise ValueError(msg)

    def __len__(self):  # noqa: D105
        return len(self.all_input_fields)

    def __getitem__(self, idx):  # noqa: D105
        return {
            "input_fields": self.all_input_fields[idx],
            "output_fields": self.all_output_fields[idx],
            "constant_scalars": self.all_constant_scalars[idx],
            # TODO: add this
            # "constant_fields": self.all_constant_fields[idx],
        }


class ReactionDiffusionDataset(AutoEmulateDataset):
    """PyTorch Dataset for Reaction-Diffusion data."""

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
    ):
        super().__init__(
            data_path,
            data=data,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            stride=stride,
            input_channel_idxs=input_channel_idxs,
            output_channel_idxs=output_channel_idxs,
        )

class MHDDataset(AutoEmulateDataset):
    """PyTorch Dataset for MHD data."""

    def __init__(self, data_path: str, t_in: int = 5, t_out: int = 10, stride: int = 1):
        super().__init__(
            data_path, n_steps_input=t_in, n_steps_output=t_out, stride=stride
        )
