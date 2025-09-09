import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from getdist import MCSamples
from pyro.distributions import (
    Normal,
    TransformedDistribution,  # type: ignore since this is a valid import
    constraints,
)
from pyro.distributions.transforms import (
    AffineTransform,
    SigmoidTransform,
    Transform,
)
from pyro.infer import HMC, MCMC, NUTS, Predictive
from pyro.infer.mcmc import RandomWalkKernel
from torch.special import ndtr

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.emulators.base import Emulator


class BoundedDomainTransform(Transform):
    """Transform to map from unbounded domain to bounded domain."""

    domain = constraints.real
    codomain = constraints.interval(0.0, 1.0)
    bijective = True

    def __init__(self, domain_min, domain_max):
        super().__init__()
        self.domain_min = torch.as_tensor(domain_min)
        self.domain_max = torch.as_tensor(domain_max)
        self._sigmoid = SigmoidTransform()
        self._affine = AffineTransform(
            self.domain_min, (self.domain_max - self.domain_min)
        )

    def _call(self, x):
        u = self._sigmoid(x)
        return self._affine(u)

    def _inverse(self, y):
        u = (y - self.domain_min) / (self.domain_max - self.domain_min)
        return self._sigmoid.inv(u)

    def log_abs_det_jacobian(self, x, y):
        """Compute the log absolute determinant of the Jacobian."""
        u = self._sigmoid(x)
        return self._sigmoid.log_abs_det_jacobian(
            x, u
        ) + self._affine.log_abs_det_jacobian(u, y)


class IntervalExcursionSetCalibration(TorchDeviceMixin):
    """
    Interval excursion set calibration using MC methods.

    Interval excursion set calibration identifies the set of input parameters that lead
    to model outputs that are within specified bands of observed data.
    """

    MIN_VAR = 1e-12

    def __init__(
        self,
        emulator: Emulator,
        parameter_range: dict[str, tuple[float, float]],
        y_lower: TensorLike,
        y_upper: TensorLike,
        y_labels: list[str] | None = None,
        device: DeviceLike | None = None,
        log_level: str = "progress_bar",
    ):
        """
        Initialize the calibration object.

        Parameters
        ----------
        emulator: Emulator
            Fitted Emulator object.
        parameters_range : dict[str, tuple[float, float]]
            A dictionary mapping input parameter names to their (min, max) ranges.
        y_lower: TensorLike
            A tensor of lower bounds for each output.
        y_upper: TensorLike
            A tensor of upper bounds for each output.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        log_level: str
            Logging level for the calibration. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
        """
        TorchDeviceMixin.__init__(self, device=device)
        self.parameter_range = parameter_range

        # Set domain lower and upper bounds as tensors
        self.domain_min = torch.tensor([b[0] for b in self.parameter_range])
        self.domain_max = torch.tensor([b[1] for b in self.parameter_range])

        self._y_lower = y_lower
        self._y_upper = y_upper
        self.calibration_params = list(parameter_range.keys())
        self.d = len(self.parameter_range)
        self.emulator = emulator
        self.emulator.device = self.device
        self.output_names = y_labels or [f"y{i}" for i in range(len(y_lower))]
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.logger.info(
            "Initializing BayesianCalibration with parameters: %s",
            self.calibration_params,
        )

        # TODO: add input handling for y_lower and y_upper as floats or lists
        self.logger.info("Processed observations for outputs: %s", self.output_names)

    @property
    def y_band_low(self) -> TensorLike:
        """Return lower bounds as a tensor."""
        return self._y_lower

    @property
    def y_band_high(self) -> TensorLike:
        """Return upper bounds as a tensor."""
        return self._y_upper

    @torch.no_grad()
    def band_prob_from_mu_sigma(
        self,
        mu: torch.Tensor,
        var_or_cov: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        aggregate: str = "geomean",
    ):
        """
        Per-sample band probability across tasks given GP mean and variance/covariance.

        Parameters
        ----------
        mu : (N, T) or (T,)
        var_or_cov : (N, T) variance per task OR (N, T, T) covariance across tasks
        y1, y2 : (T,) lower/upper bounds per task
        aggregate : 'geomean' | 'sumlog' | 'none'
            - 'geomean': returns geometric mean across tasks (shape N,)
            - 'sumlog': returns sum of log-probs across tasks (shape N,)
            - 'none': returns per-task probabilities (shape N, T)

        Notes
        -----
        - This helper derives per-task standard deviations as sqrt(diag(cov)) when a
        full covariance is provided, or sqrt(variance) when per-task variances are
        provided.
        """
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)

        # Derive per-task std devs from variance / covariance
        if var_or_cov.dim() == 3:
            var_diag = torch.diagonal(var_or_cov, dim1=-2, dim2=-1).clamp_min(
                self.MIN_VAR
            )
            sigma = var_diag.sqrt()
        else:
            # (N, T) variance or (T,) variance for single sample
            if var_or_cov.dim() == 1:
                var_or_cov = var_or_cov.unsqueeze(0)
            sigma = var_or_cov.clamp_min(self.MIN_VAR).sqrt()

        # Broadcast bounds to (N, T)
        y1v = y1.view(1, -1)
        y2v = y2.view(1, -1)

        # Stable CDF differences
        a = ((y1v - mu) / sigma).clamp(-30.0, 30.0)
        b = ((y2v - mu) / sigma).clamp(-30.0, 30.0)
        p_task = (ndtr(b.double()) - ndtr(a.double())).clamp_min(1e-12).to(mu.dtype)

        if aggregate == "none":
            return p_task
        log_p = p_task.log()
        if aggregate == "geomean":
            return log_p.mean(dim=-1).exp()
        if aggregate == "sumlog":
            return log_p.sum(dim=-1)
        msg = "aggregate must be one of {'geomean', 'sumlog', 'none'}"
        raise ValueError(msg)

    def band_logprob(
        self,
        mu: TensorLike,
        var: TensorLike,
        y1: TensorLike,
        y2: TensorLike,
        temp=1.0,
        softness: float | None = None,
        mix=1.0,
    ):
        """
        Multi-task band log-probability with optional soft surrogate and mixing.

        Parameters
        ----------
        mu: TensorLike
            Predicted mean. Shape (N, T) or (T,)
        var: TensorLike
            Predicted variance per task (N, T) or covariance (N, T, T)
        y1: TensorLike
            lower bounds (T,)
        y2: TensorLike
            upper bounds (T,)

        Returns
        -------
        TensorLike
            Scalar per sample (sums over tasks) and then summed over batch here only
            when inputs are 1-sample; callers can keep per-sample if needed by removing
            .sum().
        """
        # Ensure batch: (N, T)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)

        # Broadcast bounds to (N, T) for the soft surrogate below
        y1v = y1.view(1, -1)
        y2v = y2.view(1, -1)

        # Exact likelihood across tasks via shared helper (sum of log-probs), with
        # temperature
        log_p_exact = temp * self.band_prob_from_mu_sigma(
            mu, var, y1, y2, aggregate="sumlog"
        )

        if softness is None:
            # Return per-sample log-prob; callers can sum if needed
            return log_p_exact.sum()

        # For the soft surrogate we need per-task std; reuse helper path to std
        if var.dim() == 3:
            var_diag = torch.diagonal(var, dim1=-2, dim2=-1).clamp_min(self.MIN_VAR)
            sigma = var_diag.sqrt()
        else:
            if var.dim() == 1:
                var = var.unsqueeze(0)
            sigma = var.clamp_min(self.MIN_VAR).sqrt()

        # Soft surrogate per task
        lo = torch.sigmoid((mu - y1v) / (softness * sigma))
        hi = torch.sigmoid((y2v - mu) / (softness * sigma))
        p_soft = (lo * hi).clamp_min(1e-12)
        log_p_soft = p_soft.log().sum(dim=-1)

        # Mixture in log-space per sample
        mix_t = torch.tensor(mix, dtype=mu.dtype, device=mu.device)
        log_p_mix = torch.logaddexp(
            torch.log1p(-mix_t) + log_p_soft,
            torch.log(mix_t) + log_p_exact,
        )
        return log_p_mix.sum()

    def make_interval_band_model(
        self,
        y_band_low: TensorLike,
        y_band_high: TensorLike,
        temp=1.0,
        softness=None,
        mix=1.0,
    ):
        """Make a Pyro model for interval excursion set calibration."""

        def model():
            base = Normal(0.0, 1.0).expand([1, self.d]).to_event(2)
            transform = BoundedDomainTransform(self.domain_min, self.domain_max)
            x_star = pyro.sample("x_star", TransformedDistribution(base, [transform]))
            mu, var = self.emulator.predict_mean_and_variance(x_star)
            assert isinstance(var, TensorLike)
            pyro.factor(
                "band_logp",
                self.band_logprob(
                    mu,
                    var,
                    y_band_low,
                    y_band_high,
                    temp=temp,
                    softness=softness,
                    mix=mix,
                ),
            )

        return model

    def plot_samples(self, samples, num_samples):
        """Plot samples with band probabilities and GP mean."""
        # Ensure correct shape
        samples = samples.reshape(num_samples, -1)
        with torch.no_grad():
            mu_s, var_s = self.emulator.predict_mean_and_variance(samples)
            assert isinstance(var_s, TensorLike)
            # Use shared helper for band probability aggregation, passing var directly
            p_band = self.band_prob_from_mu_sigma(
                mu_s, var_s, self.y_band_low, self.y_band_high, aggregate="geomean"
            )

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Probability in band
        sc1 = ax1.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            c=p_band.cpu(),
            cmap="viridis",
            s=6,
            alpha=0.7,
        )
        ax1.set_title("Band Probability (agg across tasks)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        plt.colorbar(sc1, ax=ax1, label="P[y in band]")

        # Predicted mean
        mu_color = mu_s.mean(dim=-1) if mu_s.dim() > 1 else mu_s
        sc2 = ax2.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            c=mu_color.cpu(),
            cmap="viridis",
            s=6,
            alpha=0.7,
        )
        ax2.set_title("GP Mean (avg across tasks)")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        plt.colorbar(sc2, ax=ax2, label="GP Mean")

        plt.tight_layout()
        plt.show()

    # TODO: refactor to base class since duplicated in BayesianCalibration
    def _get_kernel(
        self, sampler: str, model_kwargs: dict | None = None, **sampler_kwargs
    ):
        """Get the appropriate MCMC kernel based on sampler choice."""
        sampler = sampler.lower()

        if sampler == "nuts":
            self.logger.debug("Using NUTS kernel.")
            return NUTS(
                self.make_interval_band_model(
                    self.y_band_low, self.y_band_high, **model_kwargs or {}
                ),
                **sampler_kwargs,
            )
        if sampler == "hmc":
            step_size = sampler_kwargs.pop("step_size", 0.01)
            trajectory_length = sampler_kwargs.pop("trajectory_length", 1.0)
            self.logger.debug(
                "Using HMC kernel with step_size=%s, trajectory_length=%s",
                step_size,
                trajectory_length,
            )
            return HMC(
                self.make_interval_band_model(
                    self.y_band_low, self.y_band_high, **model_kwargs or {}
                ),
                step_size=step_size,
                trajectory_length=trajectory_length,
                **sampler_kwargs,
            )
        if sampler == "metropolis":
            self.logger.debug("Using Metropolis (RandomWalkKernel).")
            return RandomWalkKernel(
                self.make_interval_band_model(
                    self.y_band_low, self.y_band_high, **model_kwargs or {}
                ),
                **sampler_kwargs,
            )
        self.logger.error("Unknown sampler: %s", sampler)
        raise ValueError(f"Unknown sampler: {sampler}")

    # TODO: refactor to base class since duplicated in BayesianCalibration
    def run_mcmc(
        self,
        warmup_steps: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        initial_params: dict[str, TensorLike] | None = None,
        sampler: str = "metropolis",
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
        sampler: str
            The MCMC kernel to use, one of "hmc", "nuts" or "metropolis".

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
        kernel = self._get_kernel(sampler, **sampler_kwargs)
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

    # TODO: refactor to base class since duplicated in BayesianCalibration
    def posterior_predictive(
        self, mcmc: MCMC, model_kwargs: dict | None = None
    ) -> TensorLike:
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
        posterior_predictive = Predictive(
            self.make_interval_band_model(
                self.y_band_low, self.y_band_high, **model_kwargs or {}
            ),
            posterior_samples,
        )
        samples = posterior_predictive(predict=True)
        self.logger.debug("Posterior predictive samples generated.")
        return samples

    @torch.no_grad()
    def run_smc(  # noqa: PLR0915
        self,
        n_particles=4000,
        ess_target_frac=0.7,
        max_steps=60,
        move_steps=2,
        rw_step=0.3,
        seed=0,
    ):
        """SMC with adaptive tempering for band posterior.

        Includes vectorized random-walk Metropolis rejuvenation in the whitened space.

        Parameters
        ----------
        n_particles: int
            Number of particles to use. Defaults to 4000.
        ess_target_frac: float
            Target effective sample size (ESS) as a fraction of n_particles.
            Defaults to 0.7.
        max_steps: int
            Maximum number of tempering steps. Defaults to 60.
        move_steps: int
            Number of random-walk Metropolis steps per rejuvenation. Defaults to 2.
        rw_step: float
            Standard deviation of the random-walk proposal in whitened space.
            Defaults to 0.
        seed: int
            Random seed for reproducibility. Defaults to 0.

        Returns
        -------
        Tuple[TensorLike, TensorLike, TensorLike, TensorLike, int]
            x_final, weights, beta_schedule, ess_history, unique_count
        """
        torch.manual_seed(seed)
        transform = BoundedDomainTransform(self.domain_min, self.domain_max)
        device = self.domain_min.device
        dtype = self.domain_min.dtype

        # Work in whitened space z ~ N(0, I_d)
        z = torch.randn(n_particles, self.d, device=device, dtype=dtype)

        def compute_ll(z_batch: torch.Tensor) -> torch.Tensor:
            x = transform(z_batch)  # (N, d)
            assert isinstance(x, TensorLike)
            mu, var = self.emulator.predict_mean_and_variance(x)
            assert isinstance(var, TensorLike)
            return self.band_prob_from_mu_sigma(
                mu, var, self.y_band_low, self.y_band_high, aggregate="sumlog"
            )  # (N,)

        # Initial log-likelihoods
        ll = compute_ll(z)

        # Initialize weights at beta=0
        logw = torch.zeros(n_particles, device=device, dtype=dtype)
        beta = 0.0
        betas: list[float] = [beta]
        ess_hist: list[float] = []

        def ess_from_logw(lw: torch.Tensor) -> float:
            w = (lw - lw.logsumexp(0)).exp()
            return float(1.0 / (w.pow(2).sum() + 1e-12))

        def systematic_resample(w: torch.Tensor) -> torch.Tensor:
            N = w.numel()
            u0 = torch.rand((), device=w.device, dtype=w.dtype) / N
            cdf = torch.cumsum(w, dim=0)
            thresholds = u0 + torch.arange(N, device=w.device, dtype=w.dtype) / N
            idx = torch.searchsorted(cdf, thresholds)
            return idx.clamp_max(N - 1).long()

        def rejuvenate_rw(
            z: torch.Tensor, ll: torch.Tensor, beta: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Vectorized RW-MH in z-space with tempered target: prior + beta * ll
            for _ in range(move_steps):
                z_prop = z + rw_step * torch.randn_like(z)
                ll_prop = compute_ll(z_prop)
                dprior = -0.5 * (z_prop.pow(2).sum(dim=1) - z.pow(2).sum(dim=1))
                dlike = beta * (ll_prop - ll)
                log_alpha = dprior + dlike
                accept = torch.log(
                    torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
                ).le(log_alpha)
                if accept.any():
                    z = torch.where(accept.unsqueeze(1), z_prop, z)
                    ll = torch.where(accept, ll_prop, ll)
            return z, ll

        # Adaptive tempering loop
        for _ in range(max_steps):
            target_ess = ess_target_frac * n_particles
            # Find delta via binary search
            low, high = 0.0, float(1.0 - beta)
            for _ in range(25):
                if high <= 1e-6:
                    break
                delta = 0.5 * (low + high)
                lw_try = logw + delta * ll
                ess_try = ess_from_logw(lw_try)
                if ess_try < target_ess:
                    high = delta
                else:
                    low = delta
            delta = low
            if delta <= 1e-8 and beta < 1.0:
                delta = min(1e-3, 1.0 - beta)
            beta = float(min(1.0, beta + delta))
            logw = logw + delta * ll

            # Normalize and compute ESS
            lw_norm = logw - logw.logsumexp(0)
            w = lw_norm.exp()
            ess = float(1.0 / (w.pow(2).sum() + 1e-12))
            betas.append(beta)
            ess_hist.append(ess)

            need_resample = (ess < target_ess) or (beta >= 1.0 - 1e-6)
            if need_resample:
                idx = systematic_resample(w)
                z = z[idx]
                ll = ll[idx]
                logw.zero_()
                z, ll = rejuvenate_rw(z, ll, beta)

            if beta >= 1.0 - 1e-6:
                break

        x_final = transform(z)
        lw_norm = logw - logw.logsumexp(0)
        w_final = lw_norm.exp()
        unique_count = torch.unique(x_final, dim=0).shape[0]

        return (
            x_final,
            w_final,
            torch.tensor(betas),
            torch.tensor(ess_hist),
            int(unique_count),
        )

    # TODO: refactor to base class since duplicated in BayesianCalibration
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
            az_data = az.InferenceData(
                posterior=az.convert_to_dataset(mcmc.get_samples(group_by_chain=True)),
            )
        else:
            self.logger.debug("Using az.from_pyro for conversion.")
            az_data = az.from_pyro(mcmc, posterior_predictive=pp_samples)

        self.logger.info("Arviz InferenceData conversion complete.")
        return az_data

    # TODO: refactor to base class since duplicated in BayesianCalibration
    @staticmethod
    def to_getdist(mcmc: MCMC, label: str):
        """
        Convert Pyro MCMC object to GetDist MCSamples object for plotting.

        Parameters
        ----------
        mcmc: MCMC
            The Pyro MCMC object.
        label: str
            Label for the MCSamples object.

        Returns
        -------
        MCSamples
            The GetDist MCSamples object.
        """
        samples = mcmc.get_samples()
        return MCSamples(
            samples=np.array(list(samples.values())).T,
            names=list(samples.keys()),
            label=label,
        )
