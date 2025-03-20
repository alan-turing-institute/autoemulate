# %%

from dataclasses import dataclass, field, asdict, InitVar
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, List, Any
import numpy as np, torch
from torcheval.metrics.functional import r2_score, mean_squared_error

@dataclass(kw_only=True)
class Base(ABC):

    @staticmethod
    def check_vector(X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(X)}")
        elif X.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got {X.ndim}D")
        else:
            return X

    @staticmethod
    def check_matrix(X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(X)}")
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {X.ndim}D")
        else:
            return X

    @staticmethod
    def check_pair(X: torch.Tensor, Y: torch.Tensor):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        else:
            return X, Y
        
    @staticmethod
    def check_covariance(Y: torch.Tensor, Sigma: torch.Tensor):
        # Full covariance matrix
        if Sigma.shape == (Y.shape[0], Y.shape[1], Y.shape[1]):
            return Sigma
        # Diagonal covariance matrix
        elif Sigma.shape == (Y.shape[0], Y.shape[1]):
            return Sigma
        # Scalar covariance matrix
        elif Sigma.shape == (Y.shape[0],):
            return Sigma
        else:
            raise ValueError("Invalid covariance matrix shape")
        
    @staticmethod
    def trace(Sigma: torch.Tensor, d: int) -> torch.Tensor:
        # A-optimal design criterion
        if Sigma.dim() == 3 and Sigma.shape[1:] == (d, d):
            return torch.diagonal(Sigma, dim1=1, dim2=2).sum(dim=1).mean()
        elif Sigma.dim() == 2 and Sigma.shape[1] == d:
            return Sigma.sum(dim=1).mean()
        elif Sigma.dim() == 1:
            return d * Sigma.mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")
        
    @staticmethod
    def logdet(Sigma: torch.Tensor, dim: int):
        # D-optimal design criterion
        if len(Sigma.shape) == 3 and Sigma.shape[1:] == (dim, dim):
            return torch.logdet(Sigma).mean()
        elif len(Sigma.shape) == 2 and Sigma.shape[1] == dim:
            return torch.sum(torch.log(Sigma), dim=1).mean()
        elif len(Sigma.shape) == 1:
            return dim * torch.log(Sigma).mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")
        
    @staticmethod
    def max_eigval(Sigma: torch.Tensor) -> torch.Tensor:
        # D-optimal design criterion
        if Sigma.dim() == 3 and Sigma.shape[1:] == (Sigma.shape[1], Sigma.shape[1]):
            eigvals = torch.linalg.eigvalsh(Sigma)
            return eigvals[:, -1].mean() # Eigenvalues are sorted
        elif Sigma.dim() == 2:
            return Sigma.max(dim=1).values.mean()
        elif Sigma.dim() == 1:
            return Sigma.mean()
        else:
            raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

@dataclass(kw_only=True)
class Simulator(Base): 

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.check_matrix(self.sample_forward(self.check_matrix(X)))
        X, Y = self.check_pair(X, Y)
        return Y
    
    @abstractmethod
    def sample_forward(self, X: torch.Tensor):
        pass

@dataclass(kw_only=True)
class Emulator(Base): 

    def sample(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.check_matrix(X)
        Y, Sigma = self.sample_forward(X)
        Y = self.check_matrix(Y)
        X, Y = self.check_pair(X, Y)
        Sigma = self.check_covariance(Y, Sigma)
        return Y, Sigma
    
    @abstractmethod
    def sample_forward(self, X: torch.Tensor):
        pass

    def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        self.check_matrix(X_train)
        self.check_matrix(Y_train)
        self.check_pair(X_train, Y_train)
        self.fit_forward(X_train, Y_train)

    @abstractmethod
    def fit_forward(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        pass

@dataclass(kw_only=True)
class Learner(Base):
    simulator: Simulator
    emulator: Emulator
    in_dim: int = field(init=False)
    out_dim: int = field(init=False)
    # To not get X_train or Y_train in asdict(obj)
    X_train: InitVar[torch.Tensor]
    Y_train: InitVar[torch.Tensor]

    def __post_init__(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        # Store the initialization variables in private attributes
        self.X_train = X_train
        self.Y_train = Y_train
        # Now call emulator.fit on the private attributes
        self.emulator.fit(X_train, Y_train)
        self.in_dim = self.X_train.shape[1]
        self.out_dim = self.Y_train.shape[1]


@dataclass(kw_only=True)
class Active(Learner):
    metrics: Dict[str, List[Any]] = field(init=False)

    def __post_init__(self, X_train, Y_train):
        super().__post_init__(X_train, Y_train)
        self.metrics = dict(mse=list(), r2=list(), rate=list())
        # Private attributes
        self.Y_true = list()
        self.Y_pred = list()
        # NOTE: only relevant for Stream learners
        self.queries = list()

    def fit(self, *args: torch.Tensor | None):
        
        # Update emulator
        X, Y_pred, Sigma, metrics = self.query(*args)
        if X is not None:
            Y = self.simulator.sample(X)
            self.X_train = torch.cat([self.X_train, X])
            self.Y_train = torch.cat([self.Y_train, Y])
            self.emulator.fit(self.X_train, self.Y_train)

        # Record private items
        self.Y_true.append(Y)
        self.Y_pred.append(Y_pred)
        self.queries.append(False if X is None else True)

        # Compute general metrics
        Y_pred = torch.cat(self.Y_pred)
        Y_true = torch.cat(self.Y_true)
        mse = mean_squared_error(Y_true, Y_pred)
        r2 = r2_score(Y_true, Y_pred)
        rate = torch.sum(self.queries) / len(self.queries) if len(self.queries) > 0 else 0

        # Update general metrics
        self.metrics['mse'].append(mse.item())
        self.metrics['r2'].append(r2.item())
        self.metrics['rate'].append(rate.item())

        # Update specific metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = [value]
            else:
                self.metrics[key].append(value)
    
    @abstractmethod
    def query(self, *arg: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor,  Dict[str, List[Any]]]:
        pass

@dataclass(kw_only=True)
class Pool(Active):

    @abstractmethod
    def query(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,  Dict[str, List[Any]]]:
        pass

@dataclass(kw_only=True)
class Membership(Active):

    @abstractmethod
    def query(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,  Dict[str, List[Any]]]:
        pass

@dataclass(kw_only=True)
class Stream(Active):
    
    @abstractmethod
    def query(self, X: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor,  Dict[str, List[Any]]]:
        pass

@dataclass(kw_only=True)
class Random(Stream):
    p_query: float

    def query(self, X):
        Y, Sigma = self.emulator.sample(X)
        logdet = self.logdet(Sigma, self.out_dim)
        X = X if np.random.rand() < self.p_query else None
        return X, Y, Sigma, dict(logdet=logdet.item())
    
@dataclass(kw_only=True)
class Threshold(Stream):
    threshold: float

    @abstractmethod
    def score(self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        pass

    def query(self, X):
        Y, Sigma = self.emulator.sample(X)
        score = self.score(X, Y, Sigma)
        X = X if score > self.threshold else None
        logdet = self.logdet(Sigma, self.out_dim)
        return X, Y, Sigma, dict(logdet=logdet.item(), score=score.item())
    
@dataclass(kw_only=True)
class Input(Threshold):
    pass

@dataclass(kw_only=True)
class Distance(Input):

    def score(self, X: torch.Tensor) -> float:
        distances = torch.cdist(X, self.X_train)
        min_dists, _ = distances.min(dim=1)
        return min_dists.mean()

@dataclass(kw_only=True)
class Output(Threshold):
    pass
    
@dataclass(kw_only=True)
class A_Optimal(Output):

    def score(self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        return self.trace(Sigma, self.out_dim)
    
@dataclass(kw_only=True)
class D_Optimal(Output):

    def score(self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        return self.logdet(Sigma, self.out_dim)
    
@dataclass(kw_only=True)
class E_Optimal(Output):

    def score(self, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        return self.max_eigval(Sigma)
    
@dataclass(kw_only=True)
class Adaptive:
    Kp: float
    Ki: float
    Kd: float
    key: str
    target: float
    max_threshold: float

    def query(self, X: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor,  Dict[str, List[Any]]]:

        # PID terms
        X, Y, Sigma, metrics = super().query(X)
        e = self.metrics[self.key] - self.target
        ep, ei, ed = e[-1], e.sum(), e[-1] - e[-2]

        # Control policy
        self.threshold += self.kp * ep + self.ki * ei + self.kd * ed
        self.threshold = torch.clip(self.threshold, 0.0, self.max_threshold)
        metrics.update(dict(threshold=self.threshold, ep=ep, ei=ei, ed=ed))

for cls in [A_Optimal, D_Optimal, E_Optimal]:
    globals()[f"PID_{cls.__name__}"] = dataclass(type(f"PID_{cls.__name__}", (Adaptive, cls), {}), kw_only=True)

if __name__ == "__main__":

    from autoemulate.emulators import GaussianProcess
    from autoemulate.experimental_design import LatinHypercube
    import tqdm
    import matplotlib.pyplot as plt

    class Sin(Simulator):
        def sample_forward(self, X):
            return np.sin(X)
    
    class GP(Emulator):
        def __init__(self):
            self.model = GaussianProcess()
        def fit_forward(self, X, Y):
            self.model.fit(X, Y)
        def sample_forward(self, X):
            return self.model.predict(X, return_std=True)
        
    simulator = Sin()
    emulator = GP()
    X_train = torch.linspace(0, 50, 5).reshape(-1, 1)
    Y_train = simulator.sample(X_train)

    learners = [
        Random(
            simulator=simulator,
            emulator=emulator,
            X_train=X_train,
            Y_train=Y_train,
            p_query=0.2
        )
    ]
    inputs = torch.tensor(LatinHypercube([(0, 50)]).sample(1))