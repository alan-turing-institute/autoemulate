import functools

import neuralprocesses as nps
import torch
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike
from autoemulate.emulators.base import Emulator, PyTorchBackend
from autoemulate.emulators.neural_process.dataset import CNPDataset, cnp_collate_fn

# from torch import nn


class NeuralProceess(Emulator, TorchDeviceMixin):
    """Neural Process base class."""

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        # hidden_dim: int = 32,
        # latent_dim: int = 16,
        # hidden_layers_enc: int = 2,
        # hidden_layers_dec: int = 2,
        # activation: type[nn.Module] = nn.ReLU,
        min_context_points: int = 2,
        offset_context_points: int = 2,
        n_episodes: int = 12,
        batch_size: int = 4,
        device: DeviceLike | None = None,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        # nn.Module.__init__(self)

        self.min_context_points = min_context_points
        self.max_context_points = self.min_context_points + offset_context_points
        self.n_episode = n_episodes

        self.x_train = None
        self.y_train = None
        self.batch_size = batch_size

        # Construct a ConvCNP.
        self.convcnp = nps.construct_convgnp(
            dim_x=x.shape[1],
            dim_y=y.shape[1],
            likelihood="het",  # TODO: make configurable
            # likelihood="lowrank",  # TODO: what is this?
        )
        self.optimizer = torch.optim.Adam(self.convcnp.parameters(), lr=0.001)
        self.epochs = 100
        # self.to(self.device)

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        # self.train()
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

                # # Forward pass
                # y_pred = self.forward(x_context, y_context, x_target)
                # loss = -y_pred.log_prob(y_target).sum(1).mean()

                # # Backward pass and optimize
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                # Compute the loss and update the model parameters.
                loss = -torch.mean(
                    nps.loglik(
                        self.convcnp,
                        x_context,
                        y_context,
                        x_target,
                        y_target,
                        normalise=True,
                    )
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batches += 1

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            # self.loss_history.append(avg_epoch_loss)

            # if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
            #     print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")  # noqa: E501

    def forward(
        self, x_context: TensorLike, y_context: TensorLike, x_target: TensorLike
    ) -> TensorLike:
        """...."""
        return self.convcnp(
            x_context,
            y_context,
            x_target,
        )

    def _predict(self, x: TensorLike, with_grad: bool = False) -> DistributionLike:
        """...."""
        # Testing: make some predictions.
        # From: https://github.com/wesselb/neuralprocesses?tab=readme-ov-file#tldr-just-get-me-started
        mean, var, noiseless_samples, noisy_samples = nps.predict(
            self.convcnp,
            self.x_train,
            self.y_train,
            x,
        )
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, var.sqrt()), reinterpreted_batch_ndims=1
        )

    @staticmethod
    def is_multioutput() -> bool:
        """Check if the backend supports multi-output emulation."""
        return True
