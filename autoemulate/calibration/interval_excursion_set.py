import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from pyro.distributions import (
    MultivariateNormal,
    Normal,
    TransformedDistribution,  # type: ignore since this is a valid import
    Uniform,
    constraints,
)
from pyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    SigmoidTransform,
    Transform,
)
from pyro.infer import MCMC
from torch.special import ndtr

from autoemulate.calibration.base import BayesianMixin
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.base import ProbabilisticEmulator


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


class IntervalExcursionSetCalibration(TorchDeviceMixin, BayesianMixin):
    """Interval excursion set calibration using MC methods.

    Interval excursion set calibration identifies the set of input parameters that lead
    to model outputs that are within specified bands of observed data.

    Attributes
    ----------
    MIN_VAR : float
        Minimum variance to avoid numerical issues.
    """

    MIN_VAR = 1e-12

    def __init__(
        self,
        emulator: ProbabilisticEmulator,
        parameters_range: dict[str, tuple[float, float]],
        output_bounds: dict[str, tuple[float, float]],
        output_names: list[str],
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
        output_bounds: dict[str, tuple[float, float]],
            A dictionary of lower and upper bounds for each output.
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
        self.parameters_range = parameters_range

        # Set domain lower and upper bounds as tensors
        self.domain_min = torch.tensor([b[0] for b in self.parameters_range.values()])
        self.domain_max = torch.tensor([b[1] for b in self.parameters_range.values()])

        # Process output bounds into tensors, ensuring order matches output_names so
        # that we can subset emulator outputs with idxs in the same order
        y_lower = torch.tensor(
            [output_bounds[name][0] for name in output_names if name in output_bounds]
        ).to(self.device)
        y_upper = torch.tensor(
            [output_bounds[name][1] for name in output_names if name in output_bounds]
        ).to(self.device)
        self.y_idxs = torch.tensor(
            [idx for idx, name in enumerate(output_names) if name in output_bounds]
        ).to(torch.int32)
        self._y_lower = y_lower
        self._y_upper = y_upper
        self.calibration_params = list(parameters_range.keys())
        self.d = len(self.parameters_range)
        self.emulator = emulator
        self.emulator.device = self.device
        # TODO: we might want to check that the len equals the number of tasks returned
        self.output_names = output_names
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.logger.info(
            "Initializing IntervalExcursionSetCalibration with parameters: %s",
            self.calibration_params,
        )

        # TODO: add input handling for y_lower and y_upper as floats or lists
        self.logger.info("Processed observations for outputs: %s", self.output_names)

    @property
    def y_lower(self) -> TensorLike:
        """Return lower bounds as a tensor."""
        return self._y_lower

    @property
    def y_upper(self) -> TensorLike:
        """Return upper bounds as a tensor."""
        return self._y_upper

    def interval_prob_from_mu_sigma(
        self,
        mu: TensorLike,
        var_or_cov: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        aggregate: str = "geomean",
    ):
        """
        Interval probability across tasks given GP mean and variance/covariance.

        Parameters
        ----------
        mu: TensorLike
            Mean per task (N, T) or (T,)
        var_or_cov: TensorLike
            (N, T) variance per task OR (N, T, T) covariance across tasks
        y1, y2 : TensorLike
            (T,) lower/upper bounds per task
        aggregate : str
            'geomean' | 'sumlog' | 'none'
                - 'geomean': returns geometric mean across tasks (shape N,)
                - 'sumlog': returns sum of log-probs across tasks (shape N,)
                - 'none': returns per-task probabilities (shape N, T)

        """
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)

        # Derive per-task std devs from variance / covariance
        if var_or_cov.dim() == 3:
            var_diag = torch.diagonal(var_or_cov, dim1=-2, dim2=-1).clamp_min(
                IntervalExcursionSetCalibration.MIN_VAR
            )
            sigma = var_diag.sqrt()
        else:
            # (N, T) variance or (T,) variance for single sample
            if var_or_cov.dim() == 1:
                var_or_cov = var_or_cov.unsqueeze(0)
            sigma = var_or_cov.clamp_min(IntervalExcursionSetCalibration.MIN_VAR).sqrt()

        # Broadcast bounds to (N, T)
        y1v = y1.view(1, -1)
        y2v = y2.view(1, -1)

        # Subset mu and sigma to only the outputs being calibrated
        mu = mu[:, self.y_idxs]
        sigma = sigma[:, self.y_idxs]

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

    def interval_logprob(
        self,
        mu: TensorLike,
        var: TensorLike,
        y1: TensorLike,
        y2: TensorLike,
        temperature: float = 1.0,
    ):
        """
        Multi-task interval log-probability (sum of log-probs across tasks).

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
        temperature: float
            Optional temperature scaling. Defaults to 1.0.

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

        # Exact likelihood across tasks is sum log-probs with temperature
        log_p_exact = temperature * self.interval_prob_from_mu_sigma(
            mu, var, y1, y2, aggregate="sumlog"
        )

        # Return total log-prob summed over the batch (sum over samples).
        return log_p_exact.sum()

    def model(self, temperature=1.0, uniform_prior: bool = True, predict: bool = False):
        """Pyro model for interval excursion set calibration."""
        if not uniform_prior:
            # Sample each parameter as its own transformed site so names are preserved
            xs = []
            for i, name in enumerate(self.calibration_params):
                lo = self.domain_min[i]
                hi = self.domain_max[i]
                scale = hi - lo
                td = TransformedDistribution(
                    Normal(0.0, 1.0),
                    [SigmoidTransform(), AffineTransform(lo, scale)],
                )
                xi = pyro.sample(name, td)
                xs.append(xi)
            x = torch.stack(xs, dim=0).unsqueeze(0).to(self.device).to(torch.float32)
        else:
            # Sample each parameter uniformly within its bounds
            param_list = []
            for param in self.parameters_range:
                if param in self.calibration_params:
                    # Sample from uniform prior
                    min_val, max_val = self.parameters_range[param]
                    sampled_val = pyro.sample(param, Uniform(min_val, max_val))
                    param_list.append(sampled_val.to(self.device))
            x = torch.stack(param_list, dim=0).unsqueeze(0).float()

        # Calculate mean and variance at x with grad for HMC/NUTS
        mu, var = self.emulator.predict_mean_and_variance(x, with_grad=True)
        assert isinstance(var, TensorLike)

        if not predict:
            # Add log-prob factor for posterior calculation
            log_prob = self.interval_logprob(
                mu, var, self.y_lower, self.y_upper, temperature=temperature
            )
            pyro.factor("band_logp", log_prob)
        else:
            # Predictive sampling: draws from the multivariate normal predictive
            # distribution, y_pred ~ p(y | x_post)
            y_pred = pyro.sample(
                "y_pred",
                MultivariateNormal(
                    # mu and var are both (1, T)
                    mu,
                    covariance_matrix=torch.diag_embed(var),
                ).to_event(1),
            )
            for i, name in enumerate(self.output_names):
                pyro.deterministic(name, y_pred[..., i])

    def plot_samples(
        self, data: MCMC | az.InferenceData | TensorLike, x_idx=0, y_idx=1
    ):
        """Plot samples with band probabilities and GP mean."""
        if isinstance(data, az.InferenceData):
            samples = data.posterior.to_dataframe()[  # type: ignore  # noqa: PGH003
                self.parameters_range.keys()
            ].to_numpy()
        elif isinstance(data, MCMC):
            try:
                samples = torch.stack(
                    [data.get_samples()[key] for key in self.calibration_params], dim=-1
                )
            except Exception:
                msg = (
                    "Parameter keys do not match calibration_params, stacking all "
                    "samples"
                )
                self.logger.warning(msg)
                samples = torch.stack(list(data.get_samples().values()), dim=-1)
            samples = samples.reshape(data.num_chains * data.num_samples, -1)
        elif isinstance(data, TensorLike):
            samples = data.reshape(data.shape[0], -1)
        else:
            msg = "data must be MCMC, InferenceData, or TensorLike"
            raise ValueError(msg)

        samples = torch.tensor(samples)
        with torch.no_grad():
            mu_s, var_s = self.emulator.predict_mean_and_variance(samples)
            assert isinstance(var_s, TensorLike)
            p_band = self.interval_prob_from_mu_sigma(
                mu_s, var_s, self.y_lower, self.y_upper, aggregate="geomean"
            )

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Probability in band
        sc1 = ax1.scatter(
            samples[:, x_idx].cpu(),
            samples[:, y_idx].cpu(),
            c=p_band.cpu(),
            cmap="viridis",
            s=6,
            alpha=0.7,
            lw=0,
        )
        ax1.set_title("Band Probability (agg across tasks)")
        ax1.set_xlabel(self.calibration_params[x_idx])
        ax1.set_ylabel(self.calibration_params[y_idx])
        ax1.set_xlim(self.domain_min[x_idx].item(), self.domain_max[x_idx].item())
        ax1.set_ylim(self.domain_min[y_idx].item(), self.domain_max[y_idx].item())
        plt.colorbar(sc1, ax=ax1, label="P[y in band]")

        # Predicted mean
        mu_color = mu_s.mean(dim=-1) if mu_s.dim() > 1 else mu_s
        sc2 = ax2.scatter(
            samples[:, x_idx].cpu(),
            samples[:, y_idx].cpu(),
            c=mu_color.cpu(),
            cmap="viridis",
            s=6,
            alpha=0.7,
            lw=0,
        )
        ax2.set_title("GP Mean (avg across tasks)")
        ax2.set_xlabel(self.calibration_params[x_idx])
        ax2.set_ylabel(self.calibration_params[y_idx])
        ax2.set_xlim(self.domain_min[x_idx].item(), self.domain_max[x_idx].item())
        ax2.set_ylim(self.domain_min[y_idx].item(), self.domain_max[y_idx].item())
        plt.colorbar(sc2, ax=ax2, label="GP Mean")

        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def run_smc(  # noqa: PLR0915
        self,
        n_particles: int = 4000,
        ess_target_frac: float = 0.7,
        max_steps: int = 60,
        move_steps: int = 2,
        rw_step: float = 0.3,
        seed: int | None = None,
        uniform_prior: bool = True,
        plot_diagnostics: bool = True,
        return_az_data: bool = True,
    ) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike, int] | az.InferenceData:
        """SMC with adaptive tempering for band posterior.

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
        seed: int | None
            Random seed for reproducibility. Defaults to None.
        uniform_prior: bool
            If True, use uniform prior over the bounded domain. If False, use standard
            normal prior in the whitened space (z ~ N(0, I_d)). Defaults to True.
        plot_diagnostics: bool
            Whether to plot diagnostic plots of the final particles, temperature
            schedule, and ESS history. Defaults to True.
        return_az_data: bool
            Whether to return ArviZ InferenceData object. Defaults to True.

        Returns
        -------
        Tuple[TensorLike, TensorLike, TensorLike, TensorLike, int] | az.InferenceData
            A tuple of x_final, weights, beta_schedule, ess_history, unique_count or
            an ArviZ InferenceData object if return_az_data is True.

        References
        ----------
        - See Del Moral et al. (2006) <https://doi.org/10.1111/j.1467-9868.2006.00553.x>
        "Sequential Monte Carlo samplers" for details on the SMC algorithm.

        """
        if seed is not None:
            set_random_seed(seed)
        transform = (
            BoundedDomainTransform(self.domain_min, self.domain_max)
            if not uniform_prior
            else ComposeTransform([])
        )
        device = self.domain_min.device
        dtype = self.domain_min.dtype

        # Sample from whitened space z ~ N(0, I_d) or original space if uniform prior
        z = (
            torch.randn(n_particles, self.d, device=device, dtype=dtype)
            if not uniform_prior
            else Uniform(self.domain_min, self.domain_max).sample((n_particles,))
        )

        def compute_ll(z_batch: torch.Tensor) -> torch.Tensor:
            x = transform(z_batch)  # (N, d)
            assert isinstance(x, TensorLike)
            mu, var = self.emulator.predict_mean_and_variance(x)
            assert isinstance(var, TensorLike)
            # Sum of log-probs across tasks
            return self.interval_prob_from_mu_sigma(
                mu, var, self.y_lower, self.y_upper, aggregate="sumlog"
            )  # (N,)

        # Initial log-likelihoods
        ll = compute_ll(z)

        # Initialize weights at beta=0 (prior)
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
            """Vectorized RW-MH with tempered target.

            - If uniform_prior: proposals outside [domain_min, domain_max] are rejected
              and prior term cancels (constant inside domain).
            - Else (whitened Gaussian): include standard normal prior term in MH ratio.
            """
            for _ in range(move_steps):
                # Create Gaussian random-walk proposals
                z_prop = z + rw_step * torch.randn_like(z)

                if uniform_prior:
                    # Reject out-of-bounds proposals under uniform prior
                    inside = (
                        (z_prop >= self.domain_min) & (z_prop <= self.domain_max)
                    ).all(dim=1)

                    # Evaluate ll_prop for all (keeps vectorization); masked later
                    ll_prop = compute_ll(z_prop)
                    dprior = torch.zeros_like(ll)
                    dlike = beta * (ll_prop - ll)
                    log_alpha = dprior + dlike
                    ulog = torch.log(
                        torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
                    )
                    accept = (ulog <= log_alpha) & inside
                else:
                    ll_prop = compute_ll(z_prop)
                    dprior = -0.5 * (z_prop.pow(2).sum(dim=1) - z.pow(2).sum(dim=1))
                    dlike = beta * (ll_prop - ll)
                    log_alpha = dprior + dlike
                    ulog = torch.log(
                        torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
                    )
                    accept = ulog <= log_alpha

                if accept.any():
                    z = torch.where(accept.unsqueeze(1), z_prop, z)
                    ll = torch.where(accept, ll_prop, ll)
            return z, ll

        # Adaptive tempering loop from prior (beta=0) to posterior (beta=1)
        for _ in range(max_steps):
            # Target ESS as number of particles
            target_ess = ess_target_frac * n_particles

            # Find largest delta via binary search that gives ess >= target_ess
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

            # Resample and rejuvenate if ESS below target or at final beta=1 so that
            # the final output is unweighted and represents the posterior
            need_resample = (ess < target_ess) or (beta >= 1.0 - 1e-6)
            if need_resample:
                idx = systematic_resample(w)
                z = z[idx]
                ll = ll[idx]
                logw.zero_()
                z, ll = rejuvenate_rw(z, ll, beta)

            # Break if reached beta=1, i.e. full posterior
            if beta >= 1.0 - 1e-6:
                break

        x_final = transform(z)
        assert isinstance(x_final, TensorLike)
        lw_norm = logw - logw.logsumexp(0)
        w_final = lw_norm.exp()
        unique_count = torch.unique(x_final, dim=0).shape[0]

        if plot_diagnostics:
            # Diagnostic plots
            plt.figure(figsize=(5, 4))
            plt.scatter(
                x_final[:, 0].cpu(), x_final[:, 1].cpu(), s=4, alpha=0.4, c="tab:orange"
            )
            plt.title(
                f"SMC particles (final), unique={unique_count}/{x_final.shape[0]}"
            )
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.tight_layout()

            plt.figure(figsize=(6, 3))
            plt.plot(betas, "-o", ms=3)
            plt.ylabel("beta")
            plt.xlabel("step")
            plt.title("Temperatures")
            plt.tight_layout()

            plt.figure(figsize=(6, 3))
            plt.plot(ess_hist, "-o", ms=3)
            plt.ylabel("ESS")
            plt.xlabel("step")
            plt.title("ESS over steps")
            plt.tight_layout()

        if return_az_data:
            # Convert to SMC-specific ArviZ InferenceData
            x_np = x_final.detach().cpu().numpy()
            w_np = w_final.detach().cpu().numpy()
            lw_np = lw_norm.detach().cpu().numpy()

            # Posterior: store each calibrated parameter as its own variable
            posterior = {
                name: x_np[None, :, i]  # shape (chain=1, draw=N)
                for i, name in enumerate(self.calibration_params)
            }

            # Sample stats: normalized weights and log-weights per draw
            sample_stats = {
                "weight": w_np[None, :],  # (chain=1, draw=N)
                "log_weight": lw_np[None, :],
            }

            # Constant data: beta schedule, ESS history, and unique_count
            constant_data = {
                "beta_schedule": np.asarray(betas, dtype=float),
                "ess_history": np.asarray(ess_hist, dtype=float),
                "unique_count": np.array(int(unique_count), dtype=int),
            }

            return az.from_dict(
                posterior=posterior,
                sample_stats=sample_stats,
                constant_data=constant_data,
                coords={
                    "tempering_step": np.arange(len(betas)),
                    "tempering_step_ess": np.arange(len(ess_hist)),
                },
                dims={
                    # beta schedule has one entry per tempering step
                    "beta_schedule": ["tempering_step"],
                    # ess history typically logged after each step
                    "ess_history": ["tempering_step_ess"],
                    # per-parameter posterior vars are scalars per draw (no extra dims)
                },
            )
        return (
            x_final,
            w_final,
            torch.tensor(betas),
            torch.tensor(ess_hist),
            int(unique_count),
        )
