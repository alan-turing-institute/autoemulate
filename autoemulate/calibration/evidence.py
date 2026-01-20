"""Bayesian evidence computation using the Harmonic method."""

from collections.abc import Callable
from typing import Any

import harmonic as hm
import numpy as np
import torch
from pyro.infer import MCMC

from autoemulate.calibration.bayes import extract_log_probabilities
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.types import DeviceLike


class EvidenceComputation(TorchDeviceMixin):
    """
    Bayesian evidence computation using the Harmonic method.

    This class estimates the Bayesian evidence (marginal likelihood) from MCMC
    posterior samples using learnable harmonic mean estimators with normalizing
    flows, as implemented in the Harmonic package.

    The evidence is a key quantity in Bayesian model comparison, representing
    the probability of the data given the model, integrated over all parameter
    values weighted by the prior. It naturally penalizes model complexity through
    the integral over the prior.

    Parameters
    ----------
    mcmc : MCMC
        Fitted Pyro MCMC object with posterior samples. Should have been run with
        multiple chains for reliable evidence estimates.
    model : Callable
        The Pyro probabilistic model used in MCMC sampling. This should be the
        same model function passed to the MCMC kernel during sampling.
    training_proportion : float, optional
        Proportion of samples used for training the normalizing flow model
        (remaining samples used for evidence estimation). Default is 0.5.
    temperature : float, optional
        Temperature parameter for flow training, controlling the smoothness of
        the learned distribution. Lower values make training more stable but may
        underfit. Default is 0.8.
    flow_model : str, optional
        Type of normalizing flow model to use. Currently supports "RQSpline"
        (Rational Quadratic Spline flows). Default is "RQSpline".
    device : DeviceLike | None, optional
        Device for computations (e.g., 'cpu', 'cuda'). If None, uses the default
        device. Default is None.
    log_level : str, optional
        Logging verbosity level. Options: "debug", "info", "warning", "error",
        "critical", "progress_bar". Default is "info".

    Attributes
    ----------
    samples : torch.Tensor
        Posterior samples of shape (num_chains, num_samples_per_chain, ndim).
    log_probs : torch.Tensor
        Log probabilities of shape (num_chains, num_samples_per_chain).
    chains : harmonic.Chains
        Harmonic Chains object containing samples and log probabilities.
    chains_train : harmonic.Chains
        Training subset of chains.
    chains_infer : harmonic.Chains
        Inference subset of chains (used for evidence computation).
    flow : harmonic.model
        Trained normalizing flow model.
    evidence : harmonic.Evidence
        Evidence estimator object.

    Examples
    --------
    >>> from autoemulate.calibration import BayesianCalibration, EvidenceComputation
    >>> # Run MCMC calibration
    >>> bc = BayesianCalibration(emulator, param_range, observations)
    >>> mcmc = bc.run_mcmc(num_samples=1000, num_chains=4)
    >>>
    >>> # Compute evidence
    >>> ec = EvidenceComputation(mcmc, bc.model)
    >>> results = ec.compute_evidence(epochs=30)
    >>>
    >>> print(f"Log Evidence: {results['ln_evidence']:.2f}")
    >>> print(f"Error: [{results['error_lower']:.3f}, {results['error_upper']:.3f}]")

    Notes
    -----
    The Harmonic method learns a normalizing flow to approximate the posterior
    distribution, then uses this learned distribution to compute importance
    sampling weights for evidence estimation. This approach is more robust than
    traditional harmonic mean estimators and provides stable error estimates.

    The evidence computation requires well-converged MCMC samples. It is
    recommended to:
    - Use multiple chains (at least 4)
    - Check convergence diagnostics (R-hat ≈ 1, sufficient ESS)
    - Use sufficient samples (at least 1000 per chain after warmup)

    References
    ----------
    .. [1] McEwen, J. D., et al. (2021). "Robust Bayesian Evidence Calculation via
           the Learned Harmonic Mean Estimator." arXiv:2111.12720.
    .. [2] Harmonic documentation: https://github.com/astro-informatics/harmonic

    See Also
    --------
    BayesianCalibration : MCMC-based Bayesian calibration
    extract_log_probabilities : Extract log probabilities from MCMC samples
    """

    def __init__(
        self,
        mcmc: MCMC,
        model: Callable,
        training_proportion: float = 0.5,
        temperature: float = 0.8,
        flow_model: str = "RQSpline",
        device: DeviceLike | None = None,
        log_level: str = "info",
    ):
        """Initialize evidence computation."""
        TorchDeviceMixin.__init__(self, device=device)
        self.logger, self.progress_bar = get_configured_logger(log_level)

        # Validate and store parameters
        self._validate_parameters(training_proportion, temperature, flow_model)
        self.mcmc = mcmc
        self.model = model
        self.training_proportion = training_proportion
        self.temperature = temperature
        self.flow_model_type = flow_model

        # Extract and validate samples
        self._extract_and_validate_samples()

        # Initialize Harmonic components (will be populated in compute_evidence)
        self._initialize_harmonic_components()

        self.logger.info("EvidenceComputation initialized successfully")

    def _validate_parameters(
        self, training_proportion: float, temperature: float, flow_model: str
    ):
        """Validate initialization parameters."""
        if not 0 < training_proportion < 1:
            msg = "training_proportion must be between 0 and 1"
            self.logger.error(msg)
            raise ValueError(msg)

        if temperature <= 0:
            msg = "temperature must be positive"
            self.logger.error(msg)
            raise ValueError(msg)

        if flow_model not in ["RQSpline"]:
            msg = (
                f"Unsupported flow_model: {flow_model}. "
                "Currently only 'RQSpline' is supported."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Initializing EvidenceComputation")
        self.logger.info("Training proportion: %s", training_proportion)
        self.logger.info("Temperature: %s", temperature)
        self.logger.info("Flow model: %s", flow_model)

    def _extract_and_validate_samples(self):
        """Extract samples from MCMC and validate them."""
        self.logger.info("Extracting log probabilities from MCMC samples...")
        try:
            self.samples, self.log_probs = extract_log_probabilities(
                self.mcmc, self.model, device=self.device
            )
        except Exception as e:
            msg = f"Failed to extract log probabilities: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        self.logger.info("Samples shape: %s", self.samples.shape)
        self.logger.info("Log probabilities shape: %s", self.log_probs.shape)

        # Validate training_proportion for number of chains
        num_chains = self.samples.shape[0]
        min_train_chains = int(num_chains * self.training_proportion)
        if min_train_chains < 1:
            msg = (
                f"training_proportion ({self.training_proportion:.2f}) is too small "
                f"for {num_chains} chains. This would result in "
                f"{min_train_chains} training chains. Increase num_chains or "
                f"training_proportion to ensure at least 1 training chain."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Check for numerical issues
        if torch.isnan(self.log_probs).any():
            msg = "Log probabilities contain NaN values"
            self.logger.error(msg)
            raise ValueError(msg)

        if torch.isinf(self.log_probs).any():
            msg = "Log probabilities contain Inf values"
            self.logger.error(msg)
            raise ValueError(msg)

    def _initialize_harmonic_components(self):
        """Initialize Harmonic library components."""
        self.chains: hm.Chains | None = None  # type: ignore[name-defined]
        self.chains_train: hm.Chains | None = None  # type: ignore[name-defined]
        self.chains_infer: hm.Chains | None = None  # type: ignore[name-defined]
        self.flow: hm.model.RQSplineModel | None = None  # type: ignore[name-defined]
        self.evidence: hm.Evidence | None = None  # type: ignore[name-defined]

    def compute_evidence(
        self,
        epochs: int = 30,
        verbose: bool = False,
    ) -> dict:
        """
        Compute the Bayesian evidence.

        This method trains a normalizing flow on a subset of MCMC samples, then
        uses the trained flow to compute the evidence on the remaining samples.

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs for the normalizing flow model. More epochs
            may improve accuracy but increase computation time. Default is 30.
        verbose : bool, optional
            Whether to print training progress and details during flow training.
            Default is False.

        Returns
        -------
        dict
            Dictionary containing evidence estimation results:
            - "ln_evidence" : float
                Natural logarithm of the evidence (log marginal likelihood).
            - "ln_inv_evidence" : float
                Natural logarithm of the inverse evidence (as computed by Harmonic).
            - "error_lower" : float
                Lower bound of the error estimate (asymmetric).
            - "error_upper" : float
                Upper bound of the error estimate (asymmetric).
            - "samples_shape" : tuple
                Shape of the samples tensor used.
            - "num_chains" : int
                Number of MCMC chains.
            - "num_samples_per_chain" : int
                Number of samples per chain.
            - "num_parameters" : int
                Number of parameters in the model.

        Raises
        ------
        RuntimeError
            If flow training or evidence computation fails.

        Notes
        -----
        The returned "ln_evidence" is the most commonly used quantity for model
        comparison via Bayes factors:
            BF = exp(ln_evidence_model1 - ln_evidence_model2)

        The error bounds are asymmetric and represent the uncertainty in the
        log inverse evidence. To interpret these errors:
        - Tight errors (< 0.1) indicate reliable estimation
        - Large errors (> 1.0) suggest more samples or longer training needed

        Examples
        --------
        >>> ec = EvidenceComputation(mcmc, model)
        >>> results = ec.compute_evidence(epochs=50, verbose=True)
        >>> ln_ev = results["ln_evidence"]
        >>> err_lower = results["error_lower"]
        >>> err_upper = results["error_upper"]
        >>> print(f"ln(Evidence) = {ln_ev:.2f} (+{err_upper:.3f}, {err_lower:.3f})")
        """
        self.logger.info("Starting evidence computation")

        # Create Harmonic Chains object
        self.logger.info("Creating Harmonic Chains...")
        ndim = self.samples.shape[2]
        self.chains = hm.Chains(ndim)
        assert self.chains is not None  # for type checker
        self.chains.add_chains_3d(
            self.samples.cpu().numpy(), self.log_probs.cpu().numpy()
        )

        # Split into training and inference sets
        self.logger.info(
            "Splitting chains (training proportion: %s)...",
            self.training_proportion,
        )
        self.chains_train, self.chains_infer = hm.utils.split_data(
            self.chains, training_proportion=self.training_proportion
        )
        assert self.chains_train is not None  # for type checker
        assert self.chains_infer is not None  # for type checker

        self.logger.info(
            "Training chains: %s chains, %s total samples",
            self.chains_train.nchains,
            self.chains_train.samples.shape[0],
        )
        self.logger.info(
            "Inference chains: %s chains, %s total samples",
            self.chains_infer.nchains,
            self.chains_infer.samples.shape[0],
        )

        # Train normalizing flow
        self.logger.info(
            "Training %s flow model (epochs=%s)...", self.flow_model_type, epochs
        )
        try:
            if self.flow_model_type == "RQSpline":
                self.flow = hm.model.RQSplineModel(
                    ndim, standardize=True, temperature=self.temperature
                )
            assert self.flow is not None  # for type checker
            self.flow.fit(
                np.asarray(self.chains_train.samples),  # pyright: ignore[reportArgumentType]
                epochs=epochs,
                verbose=verbose,
            )
        except Exception as e:
            msg = f"Flow training failed: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        self.logger.info("Flow training completed")

        # Compute evidence
        self.logger.info("Computing evidence...")
        try:
            self.evidence = hm.Evidence(self.chains_infer.nchains, self.flow)
            assert self.evidence is not None  # for type checker
            self.evidence.add_chains(self.chains_infer)

            ln_inv_evidence = self.evidence.ln_evidence_inv
            error_bounds = self.evidence.compute_ln_inv_evidence_errors()

        except Exception as e:
            msg = f"Evidence computation failed: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        # Prepare results
        ln_evidence = -ln_inv_evidence
        error_lower = float(error_bounds[0])  # Negative error
        error_upper = float(error_bounds[1])  # Positive error

        results = {
            "ln_evidence": float(ln_evidence),
            "ln_inv_evidence": float(ln_inv_evidence),
            "error_lower": error_lower,
            "error_upper": error_upper,
            "samples_shape": tuple(self.samples.shape),
            "num_chains": self.samples.shape[0],
            "num_samples_per_chain": self.samples.shape[1],
            "num_parameters": self.samples.shape[2],
        }

        self.logger.info("Evidence computation completed")
        self.logger.info("ln(Evidence) = %.4f", ln_evidence)
        self.logger.info("ln(Inverse Evidence) = %.4f", ln_inv_evidence)
        self.logger.info("Error bounds: [%.4f, %.4f]", error_lower, error_upper)

        return results

    def get_chains(self) -> Any:
        """
        Return the Harmonic Chains object for advanced usage.

        Returns
        -------
        harmonic.Chains
            The Chains object containing all samples and log probabilities.

        Raises
        ------
        RuntimeError
            If compute_evidence has not been called yet.
        """
        if self.chains is None:
            msg = "Chains not initialized. Call compute_evidence() first."
            self.logger.error(msg)
            raise RuntimeError(msg)
        return self.chains

    def get_flow_model(self) -> Any:
        """
        Return the trained normalizing flow model for inspection.

        Returns
        -------
        harmonic.model
            The trained flow model object.

        Raises
        ------
        RuntimeError
            If compute_evidence has not been called yet or flow training failed.
        """
        if self.flow is None:
            msg = "Flow model not trained. Call compute_evidence() first."
            self.logger.error(msg)
            raise RuntimeError(msg)
        return self.flow

    def get_evidence_object(self) -> Any:
        """
        Return the Harmonic Evidence object for advanced diagnostics.

        Returns
        -------
        harmonic.Evidence
            The Evidence estimator object.

        Raises
        ------
        RuntimeError
            If compute_evidence has not been called yet.
        """
        if self.evidence is None:
            msg = "Evidence not computed. Call compute_evidence() first."
            self.logger.error(msg)
            raise RuntimeError(msg)
        return self.evidence
