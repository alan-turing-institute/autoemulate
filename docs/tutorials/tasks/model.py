import torch

# probabilistic programming
import pyro 

# autoemulate imports
from autoemulate.simulations.epidemic import Epidemic

# suppress warnings in notebook for readability
import os
import warnings

# ignore warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# random seed for reproducibility
random_seed = 42

from autoemulate.data.utils import set_random_seed
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

simulator = Epidemic(log_level="error")
x = simulator.sample_inputs(1000)
y, _ = simulator.forward_batch(x)

true_beta = 0.3
true_gamma = 0.15 

# simulator expects inputs of shape [1, number of inputs]
params = torch.tensor([true_beta, true_gamma]).view(1, -1)
true_infection_rate = simulator.forward(params)

n_obs = 100
stdev = 0.05
noise = torch.normal(mean=0, std=stdev, size=(n_obs,))
observed_infection_rates = true_infection_rate[0] + noise

observations = {"infection_rate": observed_infection_rates}

import pyro.distributions as dist

# define the probabilistic model
def model():
    # uniform priors on parameters range
    beta = pyro.sample("beta", dist.Uniform(0.1, 0.5))
    gamma = pyro.sample("gamma", dist.Uniform(0.01, 0.2))
    
    mean = simulator.forward(torch.tensor([[beta, gamma]]))

    with pyro.plate(f"data", n_obs):
        pyro.sample(
            "infection_rate",
            dist.Normal(mean, stdev),
            obs=observations["infection_rate"],
        )