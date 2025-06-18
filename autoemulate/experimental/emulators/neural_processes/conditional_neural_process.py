import functools

import numpy as np
import torch
import torch.utils
import torch.utils.data
from autoemulate.experimental.device import TorchDeviceMixin, get_torch_device
from autoemulate.experimental.emulators.base import PyTorchBackend
from autoemulate.experimental.types import DeviceLike, DistributionLike, TensorLike
from torch import nn
from torch.utils.data import Dataset


class CNPDataset(Dataset, TorchDeviceMixin):
    """
    Dataset for Conditional Neural Process (CNP).
    Samples context points and target points for
    each episode.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        min_context_points: int,
        max_context_points: int,
        n_episode: int,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        x: (n_samples, x_dim)
            Input data
        y: (n_samples, y_dim)
            Output data
        min_context_points: int
            Minimum number of context points to sample
        max_context_points: int
            Maximum number of context points to sample
        n_episode: int
            Number of episodes to sample. Must be greater than max_context_points.
        """
        TorchDeviceMixin.__init__(self, device=device)
        self.x, self.y = self._move_tensors_to_device(x, y)
        if max_context_points >= n_episode:
            msg = "max_context_points must be less than n_episode"
            raise ValueError(msg)
        self.max_context_points = max_context_points
        self.min_context_points = min_context_points
        self.n_samples = len(self.x)
        # self.x_dim = self.x[0][0].shape[-1]
        self.n_episode = n_episode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.sample()

    def sample(self):
        """
        From the full dataset, sample context and target points.
        """

        x = self.x
        y = self.y

        # sample context points
        num_context = np.random.randint(
            self.min_context_points,
            self.max_context_points + 1,
        )

        # randomly select context
        perm = torch.randperm(self.n_samples)
        context_idxs = perm[:num_context]
        target_idxs = perm[
            : self.n_episode
        ]  # including context points in targets helps with training

        x_context = x[context_idxs]
        y_context = y[context_idxs]
        x_target = x[target_idxs]
        y_target = y[target_idxs]

        x = {
            "x_context": x_context,
            "y_context": y_context,
            "x_target": x_target,
        }

        return (x, y_target)


def cnp_collate_fn(batch, device: DeviceLike | None = None):
    """
    Collate function for CNP.

    Handles different dimensions in each sample due to different numbers of context
    points.

    TODO:
    this currently just adds zeros. These shouldn't have an impact on the output as long
    as we don't do layernorm or attention, but we should modify cnp etc. to handle this
    better.
    """
    X, y = zip(*batch, strict=False)
    device = get_torch_device(device)

    # Get the maximum number of context and target points in the batch
    max_context = max(x["x_context"].shape[0] for x in X)
    max_target = max(x["x_target"].shape[0] for x in X)

    # Initialize tensors to hold the batched data
    x_context_batched = torch.zeros(len(batch), max_context, X[0]["x_context"].shape[1])
    y_context_batched = torch.zeros(len(batch), max_context, X[0]["y_context"].shape[1])
    x_target_batched = torch.zeros(len(batch), max_target, X[0]["x_target"].shape[1])
    y_batched = torch.zeros(len(batch), max_target, y[0].shape[1])

    # Create masks for context and target
    context_mask = torch.zeros(len(batch), max_context, dtype=torch.bool)
    target_mask = torch.zeros(len(batch), max_target, dtype=torch.bool)

    # Fill in the batched tensors
    for i, (x, yi) in enumerate(zip(X, y, strict=False)):
        n_context = x["x_context"].shape[0]
        n_target = x["x_target"].shape[0]

        x_context_batched[i, :n_context] = x["x_context"]
        y_context_batched[i, :n_context] = x["y_context"]
        x_target_batched[i, :n_target] = x["x_target"]
        y_batched[i, :n_target] = yi

        context_mask[i, :n_context] = True
        target_mask[i, :n_target] = True

    # Create a dictionary for X
    x_batched = {
        "x_context": x_context_batched.to(device),
        "y_context": y_context_batched.to(device),
        "x_target": x_target_batched.to(device),
        "context_mask": context_mask.to(device),
    }

    return x_batched, y_batched.to(device)


class Encoder(nn.Module):
    """
    Deterministic encoder for conditional neural process model.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers_enc: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        layers = [nn.Linear(input_dim + output_dim, hidden_dim), activation()]
        for _ in range(hidden_layers_enc):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_context: TensorLike,
        y_context: TensorLike,
        context_mask: TensorLike | None = None,
    ):
        """
        Encode context

        Parameters
        ----------
        x_context: (batch_size, n_context_points, input_dim)
        y_context: (batch_size, n_context_points, output_dim)
        context_mask: (batch_size, n_context_points)

        Returns
        -------
        r: (batch_size, 1, latent_dim)
        """
        x = torch.cat([x_context, y_context], dim=-1)
        x = self.net(x)

        if context_mask is not None:
            masked_x = x * context_mask.unsqueeze(-1)
            r = masked_x.sum(dim=1, keepdim=True) / context_mask.sum(
                dim=1, keepdim=True
            ).unsqueeze(-1)  # mean over valid context points
        else:
            r = x.mean(dim=1, keepdim=True)  # mean over context points
        return r


class Decoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers_dec: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        layers = [nn.Linear(latent_dim + input_dim, hidden_dim), activation()]
        for _ in range(hidden_layers_dec):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, r: TensorLike, x_target: TensorLike):
        """
        Decode using representation r and target points x_target

        Parameters
        ----------
        r: (batch_size, 1, latent_dim)
        x_target: (batch_size, n_points, input_dim)

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """
        _, n, _ = x_target.shape  # batch_size, n_points, input_dim
        r_expanded = r.expand(-1, n, -1)
        x = torch.cat([r_expanded, x_target], dim=-1)
        hidden = self.net(x)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


class CNPModule(PyTorchBackend):
    """ "
    Implemntation of Conditional Neural Process (CNP) model.
    The model is a deterministic encoder-decoder architecture.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        hidden_layers_enc: int = 2,
        hidden_layers_dec: int = 2,
        activation: type[nn.Module] = nn.ReLU,
        min_context_points: int = 2,
        offset_context_points: int = 2,
        n_episodes: int = 12,
        batch_size: int = 4,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        x: (n_points, input_dim)
            Input data of shape (n_points, input_dim).
        y: (n_points, output_dim)
            Output data of shape (n_points, output_dim).
        hidden_dim: int
            Hidden dimension of the encoder and decoder.
        latent_dim: int
            Latent dimension of the encoder and decoder.
        hidden_layers_enc: int
            Number of hidden layers in the encoder.
        hidden_layers_dec: int
            Number of hidden layers in the decoder.
        activation: type[nn.Module]
            Activation function to use in the encoder and decoder.
        min_context_points: int
            Minimum number of context points to sample.
        offset_context_points: int
            Offset for the maximum number of context points to sample.
            max_context_points = min_context_points + offset_context_points
        n_episodes: int
            This is the length of the target sequence.
            It must be greater than max_context_points.
        batch_size: int
            Batch size for training.
        device: DeviceLike | None
            Device to use for training. If None, use the default device.
        """
        if random_seed is not None:
            self.set_random_seed(random_seed, deterministic=True)
        super().__init__()
        TorchDeviceMixin.__init__(self, device=device)
        x, y = self._move_tensors_to_device(x, y)

        x, y = self._convert_to_tensors(x, y)
        self.input_dim = x.shape[1]
        self.output_dim = y.shape[1]

        self.encoder = Encoder(
            self.input_dim,
            self.output_dim,
            hidden_dim,
            latent_dim,
            hidden_layers_enc,
            activation,
        )
        self.decoder = Decoder(
            self.input_dim,
            latent_dim,
            hidden_dim,
            self.output_dim,
            hidden_layers_dec,
            activation,
        )

        # Move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.to(self.device)

        self.min_context_points = min_context_points
        self.max_context_points = self.min_context_points + offset_context_points
        self.n_episode = n_episodes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.x_train = None
        self.y_train = None
        self.batch_size = batch_size

    def forward(
        self,
        context_x: TensorLike,
        context_y: TensorLike,
        target_x: TensorLike,
        context_mask: TensorLike | None = None,
    ) -> torch.distributions.Independent:
        """

        Parameters
        ----------
        context_x: (batch_size, n_context_points, input_dim)
        context_y: (batch_size, n_context_points, output_dim)
        target_x: (batch_size, n_points, input_dim)
        context_mask: (batch_size, n_context_points), currently unused,
        as we pad with 0's and don't have attention, layernorm yet.

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """

        r = self.encoder(context_x, context_y, context_mask)
        mean, logvar = self.decoder(r, target_x)

        return torch.distributions.Independent(
            torch.distributions.Normal(mean, torch.exp(0.5 * logvar)),
            reinterpreted_batch_ndims=1,
        )

    def _fit(self, x: TensorLike, y: TensorLike):
        """
        Fit the model to the data.
        Note the batching of data is done internally in the method.
        Parameters
        ----------
        x: (n_points, input_dim)
            Input data of shape (n_points, input_dim).
        y: (n_points, output_dim)
            Output data of shape (n_points, output_dim).
        """
        self.train()
        x, y = self._move_tensors_to_device(x, y)

        # Save off all X_train and y_train
        self.x_train, self.y_train = self._convert_to_tensors(x, y)
        self.x_train, self.y_train = self._move_tensors_to_device(
            self.x_train, self.y_train
        )

        # Convert dataset to CNP Dataset
        dataset = CNPDataset(
            self.x_train,
            self.y_train,
            min_context_points=self.min_context_points,
            max_context_points=self.max_context_points,
            n_episode=self.n_episode,
        )

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0

            for x_batch, y_target in torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=functools.partial(cnp_collate_fn, device=self.device),
            ):
                x_context = x_batch["x_context"]
                y_context = x_batch["y_context"]
                x_target = x_batch["x_target"]

                # Preprocess x_batch
                x_context = self.preprocess(x_context)
                x_target = self.preprocess(x_target)
                assert isinstance(x_context, TensorLike)
                assert isinstance(x_target, TensorLike)

                # Forward pass
                y_pred = self.forward(x_context, y_context, x_target)
                loss = -y_pred.log_prob(y_target).sum(1).mean()

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batches += 1

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

    def _predict(self, x: TensorLike) -> DistributionLike:
        """
        Predict uses the training data as the context data and the input x as the target
        data. The data is preprocessed within the method.

        Parameters
        ----------
        x: (n_points, input_dim)
            Input data of shape (n_points, input_dim).

        Returns
        -------
        distribution: torch.distributions.Independent
            Distribution of the output data.
            Note the distribution is a single tensor of shape (n_points, output_dim).

        """

        self.eval()
        x = x.to(self.device)
        x = self.preprocess(x)

        x_target = self._convert_to_tensors(x)

        # Sort splitting into context and target
        # TODO: Do we need a conversion for single tensors?
        if not isinstance(x_target, TensorLike):
            msg = "x_target must be a single tensor"
            raise ValueError(msg)

        # Unsqueeze the batch dimension for x_train, y_train and x_target
        assert isinstance(self.x_train, TensorLike)
        assert isinstance(self.y_train, TensorLike)
        x_train = self.x_train.unsqueeze(0)
        y_train = self.y_train.unsqueeze(0)
        x_target = x_target.unsqueeze(0)

        # Forward pass and drop the batch dimension
        distribution = self.forward(x_train, y_train, x_target)
        mean = distribution.mean.squeeze(0)  # Remove batch dimension
        variance = distribution.variance.squeeze(0)  # Remove batch dimension
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, variance.sqrt()),
            reinterpreted_batch_ndims=1,
        )

    @staticmethod
    def is_multioutput() -> bool:
        """
        Check if the model is a multi-output model.
        """
        return True

    @staticmethod
    def get_tune_config():
        return {
            "hidden_dim": [16, 32, 64],
            "latent_dim": np.arange(8, 64, 8),
            "hidden_layers_enc": [1, 2, 4],
            "hidden_layers_dec": [1, 2, 4],
            "activation": [nn.ReLU],
            "min_context_points": [4, 5, 6],
            "offset_context_points": [4, 5],
            # max_context_points must be less than n_episodes
            "n_episodes": [12, 13, 14],
        }
