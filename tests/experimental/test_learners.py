from collections.abc import Iterable

import gpytorch
import numpy as np
import pytest
import torch
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
    constant_mean,
    rbf,
)
from autoemulate.experimental.learners import (
    Simulator,
    stream,
)
from autoemulate.experimental_design import LatinHypercube
from autoemulate.simulations.projectile import simulate_projectile_multioutput
from tqdm import tqdm


@pytest.fixture
def emulator_config():
    return {
        "likelihood_cls": gpytorch.likelihoods.MultitaskGaussianLikelihood,
        "mean_module_fn": constant_mean,
        "covar_module_fn": rbf,
    }


def get_emulator(x, y, config):
    return GaussianProcessExact(x=x, y=y, **config)


def test_learners(emulator_config):
    # Define a simple sine simulator.
    class Sin(Simulator):
        def sample_forward(self, X: torch.Tensor) -> torch.Tensor:
            return torch.sin(X)

        def sample_inputs(self, n: int) -> torch.Tensor:
            return torch.Tensor(LatinHypercube([(0.0, 50.0)]).sample(n))

    class Projectile(Simulator):
        def sample_forward(self, X: torch.Tensor) -> torch.Tensor:
            return torch.tensor([simulate_projectile_multioutput(x) for x in X])

        def sample_inputs(self, n: int) -> torch.Tensor:
            return torch.Tensor(LatinHypercube([(-5.0, 1.0), (0.0, 1000.0)]).sample(n))

    # Define an emulator using a dummy Gaussian Process.
    class GP(Emulator):
        def __init__(self):
            self.model_cls = GaussianProcessExact
            self.model_config = emulator_config

        def fit_forward(self, X: torch.Tensor, Y: torch.Tensor):
            self.model = self.model_cls(X, Y, **self.model_config)
            self.model.fit(X, Y)

        def sample_forward(self, X: torch.Tensor):
            return torch.from_numpy(np.array(self.model.predict(X).mean))

    def learners(
        *, simulator: Simulator, n_initial_samples: int, adaptive_only: bool
    ) -> Iterable:
        X_train = simulator.sample_inputs(n_initial_samples)
        Y_train = simulator.sample(X_train)
        yield stream.Random(
            simulator=simulator,
            emulator=get_emulator(X_train, Y_train, emulator_config),
            X_train=X_train,
            Y_train=Y_train,
            p_query=0.25,
        )
        if not adaptive_only:
            yield stream.Distance(
                simulator=simulator,
                emulator=get_emulator(X_train, Y_train, emulator_config),
                X_train=X_train,
                Y_train=Y_train,
                threshold=0.5,
            )
            yield stream.A_Optimal(
                simulator=simulator,
                emulator=get_emulator(X_train, Y_train, emulator_config),
                X_train=X_train,
                Y_train=Y_train,
                threshold=1.0,
            )
            yield stream.D_Optimal(
                simulator=simulator,
                emulator=get_emulator(X_train, Y_train, emulator_config),
                X_train=X_train,
                Y_train=Y_train,
                threshold=-4.2,
            )
            yield stream.E_Optimal(
                simulator=simulator,
                emulator=get_emulator(X_train, Y_train, emulator_config),
                X_train=X_train,
                Y_train=Y_train,
                threshold=1.0,
            )
        yield stream.Adaptive_Distance(
            simulator=simulator,
            emulator=get_emulator(X_train, Y_train, emulator_config),
            X_train=X_train,
            Y_train=Y_train,
            threshold=0.5,
            Kp=1.0,
            Ki=1.0,
            Kd=1.0,
            key="rate",
            target=0.25,
            min_threshold=0.0,  # if isinstance(simulator, Sin) else None,
            max_threshold=2.0 if isinstance(simulator, Sin) else None,
            window_size=10,
        )
        yield stream.Adaptive_A_Optimal(
            simulator=simulator,
            emulator=get_emulator(X_train, Y_train, emulator_config),
            X_train=X_train,
            Y_train=Y_train,
            threshold=1e-1,
            Kp=2.0,
            Ki=1.0,
            Kd=2.0,
            key="rate",
            target=0.25,
            min_threshold=0.0,  # if isinstance(simulator, Sin) else None,
            max_threshold=1.0 if isinstance(simulator, Sin) else None,
            window_size=10,
        )
        yield stream.Adaptive_D_Optimal(
            simulator=simulator,
            emulator=get_emulator(X_train, Y_train, emulator_config),
            X_train=X_train,
            Y_train=Y_train,
            threshold=-4.0,
            Kp=2.0,
            Ki=1.0,
            Kd=2.0,
            key="rate",
            target=0.25,
            min_threshold=-5 if isinstance(simulator, Sin) else None,
            max_threshold=0 if isinstance(simulator, Sin) else None,
            window_size=10,
        )
        yield stream.Adaptive_E_Optimal(
            simulator=simulator,
            emulator=get_emulator(X_train, Y_train, emulator_config),
            X_train=X_train,
            Y_train=Y_train,
            threshold=0.75 if isinstance(simulator, Sin) else 1000,
            Kp=2.0,
            Ki=1.0,
            Kd=2.0,
            key="rate",
            target=0.25,
            min_threshold=0.0,  # if isinstance(simulator, Sin) else None,
            max_threshold=1.0 if isinstance(simulator, Sin) else None,
            window_size=10,
        )

    def run_experiment(
        *,
        simulator: Simulator,
        seeds: list[int],
        n_initial_samples: int,
        n_stream_samples: int,
        adaptive_only: bool,
    ) -> tuple[list[dict], list[dict]]:
        metrics, summary = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            X_stream = simulator.sample_inputs(n_stream_samples)
            tqdm.write(f"Trial with seed {seed}")
            for learner in learners(
                simulator=simulator,
                n_initial_samples=n_initial_samples,
                adaptive_only=adaptive_only,
            ):
                learner.fit_samples(X_stream)
                metrics.append(dict(name=learner.__class__.__name__, **learner.metrics))
                summary.append(dict(name=learner.__class__.__name__, **learner.summary))
        return metrics, summary

    metrics, summary = run_experiment(
        simulator=Sin(),
        seeds=[0],
        n_initial_samples=5,
        n_stream_samples=100,
        adaptive_only=True,
    )
