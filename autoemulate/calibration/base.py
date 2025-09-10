import logging
from collections.abc import Callable

import arviz as az
import numpy as np
from getdist import MCSamples
from pyro.infer import HMC, MCMC, NUTS, Predictive
from pyro.infer.mcmc import RandomWalkKernel

from autoemulate.core.types import TensorLike


class BayesianMixin:
    """Mixin class for Bayesian calibration methods."""

    logger: logging.Logger
    model: Callable
    observations: dict[str, TensorLike] | None

    def _get_kernel(
        self,
        sampler: str,
        model_kwargs: dict[str, TensorLike] | None = None,
        **sampler_kwargs,
    ):
        """Get the appropriate MCMC kernel based on sampler choice."""
        # TODO: consider how to pass model args, functools.partial?
        model_kwargs = model_kwargs or {}
        sampler = sampler.lower()
        if sampler == "nuts":
            self.logger.debug("Using NUTS kernel.")
            return NUTS(self.model, **sampler_kwargs)
        if sampler == "hmc":
            step_size = sampler_kwargs.pop("step_size", 0.01)
            trajectory_length = sampler_kwargs.pop("trajectory_length", 1.0)
            self.logger.debug(
                "Using HMC kernel with step_size=%s, trajectory_length=%s",
                step_size,
                trajectory_length,
            )
            return HMC(
                self.model,
                step_size=step_size,
                trajectory_length=trajectory_length,
                **sampler_kwargs,
            )
        if sampler == "metropolis":
            self.logger.debug("Using Metropolis (RandomWalkKernel).")
            return RandomWalkKernel(self.model, **sampler_kwargs)
        self.logger.error("Unknown sampler: %s", sampler)
        raise ValueError(f"Unknown sampler: {sampler}")

    def run_mcmc(
        self,
        warmup_steps: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        initial_params: dict[str, TensorLike] | None = None,
        model_kwargs: dict | None = None,
        sampler: str = "nuts",
        **sampler_kwargs,
    ) -> MCMC:
        """
        Run Markov Chain Monte Carlo (MCMC). Defaults to using the NUTS sampler.

        Parameters
        ----------
        warmup_steps: int
            Number of warm up steps to run per chain (i.e., burn-in). These samples are
            discarded. Defaults to 500.
        num_samples: int
            Number of samples to draw after warm up. Defaults to 1000.
        num_chains: int
            Number of parallel chains to run. Defaults to 1.
        initial_params: dict[str, TensorLike] | None
            Optional dictionary specifiying initial values for each calibration
            parameter per chain. The tensors must be of length `num_chains`.
        model_kwargs: dict | None
            Optional dictionary of keyword arguments to pass to the model.
        sampler: str
            The MCMC kernel to use, one of "hmc", "nuts" or "metropolis".
        **sampler_kwargs
            Additional keyword arguments to pass to the MCMC kernel.

        Returns
        -------
        MCMC
            The Pyro MCMC object. Methods include `summary()` and `get_samples()`.
        """
        # Check initial param values match number of chains

        if initial_params is not None:
            for param, init_vals in initial_params.items():
                if init_vals.shape[0] != num_chains:
                    msg = (
                        "An initial value must be provided for each chain, parameter "
                        f"{param} tensor only has {init_vals.shape[0]} values."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
            self.logger.debug(
                "Initial parameters provided for MCMC: %s", initial_params
            )

        # Run NUTS
        kernel = self._get_kernel(sampler, model_kwargs=model_kwargs, **sampler_kwargs)
        mcmc = MCMC(
            kernel,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            num_chains=num_chains,
            # If None, init values are sampled from the prior.
            initial_params=initial_params,
            # Multiprocessing
            mp_context="spawn" if num_chains > 1 else None,
        )
        self.logger.info("Starting MCMC run.")
        mcmc.run()
        self.logger.info("MCMC run completed.")
        return mcmc

    def posterior_predictive(self, mcmc: MCMC) -> TensorLike:
        """
        Return posterior predictive samples.

        Parameters
        ----------
        mcmc: MCMC
            The MCMC object.

        Returns
        -------
        TensorLike
            Tensor of posterior predictive samples [n_mcmc_samples, n_obs, n_outputs].
        """
        posterior_samples = mcmc.get_samples()
        posterior_predictive = Predictive(self.model, posterior_samples)
        samples = posterior_predictive(predict=True)
        self.logger.debug("Posterior predictive samples generated.")
        return samples

    def to_arviz(
        self, mcmc: MCMC, posterior_predictive: bool = False
    ) -> az.InferenceData:
        """
        Convert MCMC object to Arviz InferenceData object for plotting.

        Parameters
        ----------
        mcmc: MCMC
            The MCMC object.
        posterior_predictive: bool
            Whether to include posterior predictive samples. Defaults to False.

        Returns
        -------
        az.InferenceData
        """
        pp_samples = None
        if posterior_predictive:
            self.logger.info("Including posterior predictive samples in Arviz output.")
            pp_samples = self.posterior_predictive(mcmc)

        # Need to create dataset manually for Metropolis Hastings
        # This is because az.from_pyro expects kernel with `divergences`
        if isinstance(mcmc.kernel, RandomWalkKernel):
            self.logger.debug(
                "Using manual conversion for Metropolis (RandomWalkKernel) kernel."
            )
            if posterior_predictive:
                if self.observations is None:
                    msg = (
                        "Observations must be provided to include observed_data in "
                        "Arviz InferenceData."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                az_data = az.InferenceData(
                    posterior=az.convert_to_dataset(
                        mcmc.get_samples(group_by_chain=True)
                    ),
                    posterior_predictive=az.convert_to_dataset(pp_samples),
                    observed_data=az.convert_to_dataset(self.observations),
                )
            else:
                az_data = az.InferenceData(
                    posterior=az.convert_to_dataset(
                        mcmc.get_samples(group_by_chain=True)
                    ),
                )
        else:
            self.logger.debug("Using az.from_pyro for conversion.")
            az_data = az.from_pyro(mcmc, posterior_predictive=pp_samples)

        self.logger.info("Arviz InferenceData conversion complete.")
        return az_data

    @staticmethod
    def to_getdist(
        data: MCMC | az.InferenceData,
        label: str,
        use_weights: bool = True,
        weight_name: str = "weight",
    ) -> MCSamples:
        """Convert Pyro MCMC or ArviZ InferenceData to GetDist MCSamples.

        This lightweight helper extends the original implementation to also accept
        SMC / other results already converted to ArviZ InferenceData. If a weight
        variable (default: smc_weight) is present in sample_stats it will be
        used as importance weights.

        Parameters
        ----------
        data: MCMC | az.InferenceData
            The Pyro MCMC object or an ArviZ InferenceData object containing posterior
            samples.
        label: str
            Label for the MCSamples object.
        use_weights: bool
            If True and `data` is an `InferenceData` with `weight_name` in
            `sample_stats` then those weights are applied. Defaults to True.
        weight_name: str
            Name of the weight variable inside `sample_stats` to look up.

        Returns
        -------
        MCSamples
            The GetDist MCSamples object.
        """
        if isinstance(data, MCMC):
            samples_dict = data.get_samples()
            arr = np.array(list(samples_dict.values())).T
            names = list(samples_dict.keys())
            weights = None
        else:
            posterior = data.posterior  # type: ignore[attr-defined]
            names = list(posterior.data_vars)
            cols = []
            for name in names:
                vals = np.asarray(posterior[name].values)
                # Expect shape (chain, draw) for scalar parameters
                if vals.ndim != 2:
                    msg = (
                        f"Posterior variable '{name}' has shape {vals.shape}; "
                        "only scalar parameter sites (chain, draw) supported here."
                    )
                    raise ValueError(msg)
                cols.append(vals.reshape(-1))
            arr = np.vstack(cols).T  # (n_total_draws, n_params)
            weights = None
            sample_stats = getattr(data, "sample_stats", None)  # type: ignore[attr-defined]
            if use_weights and sample_stats is not None and weight_name in sample_stats:
                w = np.asarray(sample_stats[weight_name].values)
                if w.ndim == 2:  # (chain, draw)
                    weights = w.reshape(-1)
        return MCSamples(samples=arr, names=names, label=label, weights=weights)
