from autoemulate.emulators import GaussianProcess
from autoemulate.learners.base import Simulator, Emulator, Random, A_Optimal, D_Optimal, E_Optimal
from autoemulate.learners.base import PID_A_Optimal, PID_D_Optimal, PID_E_Optimal
from autoemulate.experimental_design import LatinHypercube
import matplotlib.pyplot as plt
import torch

def test_learners():

    class Sin(Simulator):
        def sample_forward(self, X: torch.Tensor) -> torch.Tensor:
            return torch.sin(X)

    class GP(Emulator):
        def __init__(self):
            self.model = GaussianProcess()
        def fit_forward(self, X: torch.Tensor, Y: torch.Tensor):
            self.model.fit(X, Y)
        def sample_forward(self, X: torch.Tensor):
            return torch.tensor(self.model.predict(X, return_std=True))
        
    # Initialize simulator and training data
    simulator = Sin()
    X_train = torch.linspace(0, 50, 5).reshape(-1, 1)
    Y_train = simulator.sample(X_train)

    learners = [
        Random(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            p_query=0.25
        ),
        A_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=1e-6
        ),
        D_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=-4.1
        ),
        E_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=1e-6
        ),
        PID_A_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=1e-1,
            Kp=1.0,
            Ki=1.0,
            Kd=1.0,
            key="rate",
            target=0.25,
            min_threshold=0.0,
            max_threshold=1.0,
            window_size=10
        ),
        PID_D_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=-4.0,
            Kp=1.0,
            Ki=1.0,
            Kd=1.0,
            key="rate",
            target=0.25,
            min_threshold=None,
            max_threshold=None,
            window_size=10
        ),
        PID_E_Optimal(
            simulator=simulator,
            emulator=GP(),
            X_train=X_train,
            Y_train=Y_train,
            threshold=0.75,
            Kp=1.0,
            Ki=1.0,
            Kd=1.0,
            key="rate",
            target=0.25,
            min_threshold=0.0,
            max_threshold=1.0,
            window_size=10
        ),
    ]

    # Generate samples for streaming using Latin Hypercube sampling
    X_stream = torch.tensor(LatinHypercube([(0, 50)]).sample(250))
    for learner in learners:
        learner.fit_samples(X_stream)

    # Report metrics
    for learner in learners:
        print(learner.summary)