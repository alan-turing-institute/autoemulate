from collections.abc import Iterable

import numpy as np
import pytest
import torch
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcess,
)
from autoemulate.experimental.learners import stream
from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.simulations.projectile import ProjectileMultioutput
from autoemulate.experimental.types import DeviceLike, TensorLike
from tqdm import tqdm


# Define a simple sine simulator.
class Sin(Simulator):
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def learners(
    *,
    simulator: Simulator,
    n_initial_samples: int,
    adaptive_only: bool,
    device: DeviceLike,
) -> Iterable:
    x_train = simulator.sample_inputs(n_initial_samples).to(device)
    y_train = simulator.forward_batch(x_train).to(device)
    yield stream.Random(
        simulator=simulator,
        emulator=GaussianProcess(x_train, y_train, device=device),
        x_train=x_train,
        y_train=y_train,
        p_query=0.25,
    )
    if not adaptive_only:
        yield stream.Distance(
            simulator=simulator,
            emulator=GaussianProcess(x_train, y_train, device=device),
            x_train=x_train,
            y_train=y_train,
            threshold=0.5,
        )
        yield stream.A_Optimal(
            simulator=simulator,
            emulator=GaussianProcess(x_train, y_train, device=device),
            x_train=x_train,
            y_train=y_train,
            threshold=1.0,
        )
        yield stream.D_Optimal(
            simulator=simulator,
            emulator=GaussianProcess(x_train, y_train, device=device),
            x_train=x_train,
            y_train=y_train,
            threshold=-4.2,
        )
        yield stream.E_Optimal(
            simulator=simulator,
            emulator=GaussianProcess(x_train, y_train, device=device),
            x_train=x_train,
            y_train=y_train,
            threshold=1.0,
        )
    yield stream.Adaptive_Distance(
        simulator=simulator,
        emulator=GaussianProcess(x_train, y_train, device=device),
        x_train=x_train,
        y_train=y_train,
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
        emulator=GaussianProcess(x_train, y_train, device=device),
        x_train=x_train,
        y_train=y_train,
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
        emulator=GaussianProcess(x_train, y_train, device=device),
        x_train=x_train,
        y_train=y_train,
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
        emulator=GaussianProcess(x_train, y_train, device=device),
        x_train=x_train,
        y_train=y_train,
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


def run_experiment(  # noqa: PLR0913
    *,
    simulator: Simulator,
    seeds: list[int],
    n_initial_samples: int,
    n_stream_samples: int,
    adaptive_only: bool,
    device: DeviceLike,
) -> tuple[list[dict], list[dict]]:
    metrics, summary = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_stream = simulator.sample_inputs(n_stream_samples)
        tqdm.write(f"Trial with seed {seed}")
        for learner in learners(
            simulator=simulator,
            n_initial_samples=n_initial_samples,
            adaptive_only=adaptive_only,
            device=device,
        ):
            print("input: ", x_stream.type())
            learner.fit_samples(x_stream)
            metrics.append(dict(name=learner.__class__.__name__, **learner.metrics))
            summary.append(dict(name=learner.__class__.__name__, **learner.summary))
    return metrics, summary


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_learners_sin(device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    metrics, summary = run_experiment(
        simulator=Sin(parameters_range={"x": (0, 50.0)}, output_names=["y"]),
        seeds=[0],
        n_initial_samples=5,
        n_stream_samples=100,
        adaptive_only=True,
        device=device,
    )


def test_learners_projectile():
    metrics, summary = run_experiment(
        simulator=ProjectileMultioutput(),
        seeds=[0],
        n_initial_samples=5,
        n_stream_samples=100,
        adaptive_only=True,
        device="cpu",
    )
