import torch
from autoemulate.experimental.types import TensorLike
from the_well.data import WellDataset


class AutoEmulateWellDatasetTabular(WellDataset):
    """
    Well data by default has dimensions:
        (n_traj, n_steps, coord1, coord2, (coord3), n_fields)
        -> (n_simulations, n_timesteps, n)
        where n_traj = n_simulations, n_steps = n_timesteps
    """

    def __init__(self, field_indices: list[int], **kwargs):
        # TODO: add a way to ensure
        super().__init__(
            # TODO: add handling depending on whether specified n_step_inputs
            n_steps_input=81,
            n_steps_output=0,
            **kwargs,
        )
        self.field_indices = field_indices

    def __getitem__(self, index):
        x = super().__getitem__(index)
        x, y = self._convert_to_autoemulate(x)
        return {"x": x, "y": y}

    def _convert_to_autoemulate(self, x):
        constant_scalars = x["constant_scalars"]
        input_fields = x["input_fields"]
        constant_scalars_interleave = torch.repeat_interleave(
            constant_scalars.unsqueeze(0), self.n_steps_input, 0
        )
        ts = torch.arange(self.n_steps_input)
        x = torch.concat([constant_scalars_interleave, ts.reshape(-1, 1)], 1)
        chosen_field_ints = [0]
        input_fields_reshaped = input_fields[..., chosen_field_ints].view(
            -1, input_fields.shape[-3], input_fields.shape[-2], len(chosen_field_ints)
        )
        y = input_fields_reshaped.view(input_fields_reshaped.shape[0], -1)
        return x, y

    def get_data(self) -> tuple[TensorLike, TensorLike]:
        xs, ys = [], []
        for item in self:
            x = item["x"]
            y = item["y"]
            xs.append(x)
            ys.append(y)
        # TODO: consider using concat instead of stack
        return torch.stack(xs, 0), torch.stack(ys, 0)
