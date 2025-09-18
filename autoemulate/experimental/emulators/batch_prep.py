import torch

def prepare_batch_fno(sample, channels=(0,), with_constants=True, with_time=False):
    """Prepare a batch of input and output data."""
    # Get input fields, constant scalars and output fields
    x = sample["input_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]
    constant_scalars = sample["constant_scalars"]  # [batch, n_constants]
    y = sample["output_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]

    # Permute both x and y
    x = x.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]
    y = y.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]

    # Only add constants to input, not output
    if with_constants:
        # Assign spatio-temporal dims to constants
        time_window, height, width = x.shape[2], x.shape[3], x.shape[4]
        n_constants = constant_scalars.shape[-1]

        # Add spatio-temporal dims to constants
        c_broadcast = constant_scalars.reshape(1, n_constants, 1, 1, 1).expand(
            1, n_constants, time_window, height, width
        )

        # Concatenate along channel dimension
        x = torch.cat([x, c_broadcast], dim=1)

    if not with_time:
        # Take last time step for both input and output
        return x[:, :, -1, :, :], y[:, :, -1, :, :]
    # Otherwise include time
    return x, y
