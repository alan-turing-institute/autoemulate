from dataclasses import asdict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autoemulate.emulators import GaussianProcess
from autoemulate.experimental.learners import base
from autoemulate.experimental.learners import Emulator
from autoemulate.experimental.learners import membership
from autoemulate.experimental.learners import pool
from autoemulate.experimental.learners import Simulator
from autoemulate.experimental.learners import stream
from autoemulate.experimental_design import LatinHypercube
from autoemulate.simulations.projectile import simulate_projectile_multioutput

# Import core classes from the source code.


def test_learners():
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
            self.model = GaussianProcess()

        def fit_forward(self, X: torch.Tensor, Y: torch.Tensor):
            self.model.fit(X, Y)

        def sample_forward(self, X: torch.Tensor):
            return torch.from_numpy(np.array(self.model.predict(X, return_std=True)))

    def learners(
        *, simulator: Simulator, n_initial_samples: int, adaptive_only: bool
    ) -> Iterable:
        X_train = simulator.sample_inputs(n_initial_samples)
        Y_train = simulator.sample(X_train)
        yield stream.Random(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            p_query=0.25,
        )
        if not adaptive_only:
            yield stream.Distance(
                simulator=simulator,
                emulator=GP(),
                X_train=X_train,
                Y_train=Y_train,
                threshold=0.5,
            )
            yield stream.A_Optimal(
                simulator=simulator,
                emulator=GP(),
                X_train=X_train,
                Y_train=Y_train,
                threshold=1.0,
            )
            yield stream.D_Optimal(
                simulator=simulator,
                emulator=GP(),
                X_train=X_train,
                Y_train=Y_train,
                threshold=-4.2,
            )
            yield stream.E_Optimal(
                simulator=simulator,
                emulator=GP(),
                X_train=X_train,
                Y_train=Y_train,
                threshold=1.0,
            )
        yield stream.Adaptive_Distance(
            simulator=simulator,
            emulator=GP(),
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
            emulator=GP(),
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
            emulator=GP(),
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
            emulator=GP(),
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
        seeds: List[int],
        n_initial_samples: int,
        n_stream_samples: int,
        adaptive_only: bool,
    ) -> Tuple[List[Dict], List[Dict]]:
        metrics, summary = list(), list()
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
