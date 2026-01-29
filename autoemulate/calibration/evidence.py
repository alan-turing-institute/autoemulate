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
        (Rational Quadratic Spline flows) and "RealNVP" (Real-valued Non-Volume
        Preserving transformations). Default is "RQSpline".
    flow_kwargs : dict | None, optional
        Additional keyword arguments to pass to the flow model constructor.
        These override the default values. Common parameters include:

        For RQSplineModel:
            - standardize : bool - Whether to standardize inputs (default: True)
            - n_layers : int - Number of coupling layers (default: 8)
            - n_bins : int - Number of spline bins (default: 8)
            - hidden_size : list[int] - Hidden layer sizes (default: [64, 64])
            - spline_range : tuple[float, float] - Spline range (default: (-10, 10))
            - learning_rate : float - Optimizer learning rate (default: 0.001)
            - momentum : float - Optimizer momentum (default: 0.9)

        For RealNVPModel:
            - standardize : bool - Whether to standardize inputs (default: True)
            - n_scaled_layers : int - Number of scaled layers (default: 2)
            - n_unscaled_layers : int - Number of unscaled layers (default: 4)
            - learning_rate : float - Optimizer learning rate (default: 0.001)
            - momentum : float - Optimizer momentum (default: 0.9)

        Note: temperature is set separately via the temperature parameter.
        Default is None (uses Harmonic's defaults with standardize=True).
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
    ndim : int
        Number of parameters (dimensions) in the model.
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
    Basic usage:

    >>> from autoemulate.calibration import BayesianCalibration, EvidenceComputation
    >>> # Run MCMC calibration
    >>> bc = BayesianCalibration(emulator, param_range, observations)
    >>> mcmc = bc.run_mcmc(num_samples=1000, num_chains=4)
    >>>
    >>> # Compute evidence with defaults
    >>> ec = EvidenceComputation(mcmc, bc.model)
    >>> results = ec.run(epochs=30)
    >>>
    >>> print(f"Log Evidence: {results['ln_evidence']:.2f}")
    >>> print(f"Error: [{results['error_lower']:.3f}, {results['error_upper']:.3f}]")

    Using custom flow model parameters:

    >>> # Customize RQSpline flow model
    >>> flow_kwargs = {
    ...     "n_layers": 12,
    ...     "n_bins": 16,
    ...     "hidden_size": [128, 128],
    ...     "learning_rate": 0.0005,
    ... }
    >>> ec = EvidenceComputation(mcmc, bc.model, flow_kwargs=flow_kwargs)
    >>> results = ec.run(epochs=50)
    >>>
    >>> # Use RealNVP model instead
    >>> ec = EvidenceComputation(
    ...     mcmc,
    ...     bc.model,
    ...     flow_model="RealNVP",
    ...     flow_kwargs={"n_scaled_layers": 4, "n_unscaled_layers": 6},
    ... )
    >>> results = ec.run(epochs=30)

    Notes
    -----
    The Harmonic method learns a normalizing flow to approximate the normalizedposterior
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
    .. [1] McEwen, J. D., et al. (2021). "Machine learning assisted Bayesian model
           comparison: learnt harmonic mean estimator." arXiv:2111.12720.
    .. [2] Harmonic documentation: https://github.com/astro-informatics/harmonic

    See Also
    --------
    BayesianCalibration : MCMC-based Bayesian calibration
    extract_log_probabilities : Extract log probabilities from MCMC samples
    """

    # Supported normalizing flow models
    SUPPORTED_MODELS = ("rqspline", "realnvp")

    def __init__(
        self,
        mcmc: MCMC,
        model: Callable,
        training_proportion: float = 0.5,
        temperature: float = 0.8,
        flow_model: str = "RQSpline",
        flow_kwargs: dict | None = None,
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
        self.flow_model_type = flow_model.lower()
        self.flow_kwargs = flow_kwargs if flow_kwargs is not None else {}

        # Extract and validate samples
        self._extract_and_validate_samples()

        # Initialize Harmonic components (will be populated in compute_evidence)
        self._initialize_harmonic_components()

        self.logger.debug("EvidenceComputation initialized successfully")

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

        if flow_model.lower() not in self.SUPPORTED_MODELS:
            msg = (
                f"Unsupported flow_model: {flow_model}. "
                f"Currently supports: {', '.join(self.SUPPORTED_MODELS)}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug("Initializing EvidenceComputation")
        self.logger.debug("Training proportion: %s", training_proportion)
        self.logger.debug("Temperature: %s", temperature)
        self.logger.debug("Flow model: %s", flow_model)

    def _extract_and_validate_samples(self):
        """Extract samples from MCMC and validate them."""
        self.logger.debug("Extracting log probabilities from MCMC samples...")
        try:
            self.samples, self.log_probs = extract_log_probabilities(
                self.mcmc, self.model, device=self.device
            )
        except Exception as e:
            msg = f"Failed to extract log probabilities: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        self.logger.debug("Samples shape: %s", self.samples.shape)
        self.logger.debug("Log probabilities shape: %s", self.log_probs.shape)

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

        # Save number of dimensions for later use
        self.ndim = self.samples.shape[2]
        self.logger.debug("Number of parameters: %d", self.ndim)

    def _initialize_harmonic_components(self):
        """Initialize Harmonic library components."""
        self.chains: hm.Chains | None = None  # type: ignore[name-defined]
        self.chains_train: hm.Chains | None = None  # type: ignore[name-defined]
        self.chains_infer: hm.Chains | None = None  # type: ignore[name-defined]
        self.flow: Any = None  # Can be RQSplineModel or RealNVPModel
        self.evidence: hm.Evidence | None = None  # type: ignore[name-defined]

    def _create_flow_model(self, ndim: int) -> Any:
        """
        Create a normalizing flow model based on flow_model_type.

        Parameters
        ----------
        ndim : int
            Number of dimensions (parameters) in the model.

        Returns
        -------
        Any
            A Harmonic flow model instance (RQSplineModel or RealNVPModel).

        Raises
        ------
        ValueError
            If flow_model_type is not supported (should not occur if
            _validate_parameters was called).
        """
        # Set default kwargs with standardize and temperature
        default_kwargs = {"standardize": True, "temperature": self.temperature}

        # Merge with user-provided flow_kwargs (user values override defaults)
        model_kwargs = {**default_kwargs, **self.flow_kwargs}

        if self.flow_model_type == "rqspline":
            return hm.model.RQSplineModel(ndim, **model_kwargs)
        if self.flow_model_type == "realnvp":
            return hm.model.RealNVPModel(ndim, **model_kwargs)
        msg = f"Unsupported flow_model: {self.flow_model_type}"
        self.logger.error(msg)
        raise ValueError(msg)

    def split_data(self) -> tuple:
        """
        Split MCMC samples into training and inference sets.

        Returns
        -------
        tuple
            A tuple containing (chains_train, chains_infer) where each is a
            harmonic.Chains object.

        Notes
        -----
        This creates a Harmonic Chains object from the MCMC samples and splits
        it according to the training_proportion specified during initialization.
        The training set is used to fit the normalizing flow, while the inference
        set is used for evidence computation.
        """
        self.chains = hm.Chains(self.ndim)
        assert self.chains is not None  # for type checker
        self.chains.add_chains_3d(
            self.samples.cpu().numpy(), self.log_probs.cpu().numpy()
        )

        self.chains_train, self.chains_infer = hm.utils.split_data(
            self.chains, training_proportion=self.training_proportion
        )

        return self.chains_train, self.chains_infer

    def fit_flow(self, epochs: int = 30, verbose: bool = False) -> None:
        """
        Train the normalizing flow model on training data.

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 30.
        verbose : bool, optional
            Whether to print training progress. Default is False.

        Raises
        ------
        RuntimeError
            If flow training fails.

        Notes
        -----
        This method trains a normalizing flow on the training chains to learn
        the normalized posterior distribution. The trained flow is then used for
        evidence computation. The flow model type is determined by the flow_model
        parameter passed during initialization.
        """
        if self.chains_train is None:
            msg = "Must call split_data() before fit_flow()"
            raise RuntimeError(msg)

        self.flow = self._create_flow_model(self.ndim)
        assert self.flow is not None  # for type checker
        self.flow.fit(
            np.asarray(self.chains_train.samples),  # pyright: ignore[reportArgumentType]
            epochs=epochs,
            verbose=verbose,
        )

    def compute_ln_evidence(self) -> dict:
        """
        Compute the Bayesian evidence using the trained flow.

        Returns
        -------
        dict
            Dictionary containing:
            - "ln_evidence" : float - Natural log of the evidence
            - "ln_inv_evidence" : float - Natural log of inverse evidence
            - "error_lower" : float - Lower error bound
            - "error_upper" : float - Upper error bound
            - "samples_shape" : tuple - Shape of samples tensor
            - "num_chains" : int - Number of MCMC chains
            - "num_samples_per_chain" : int - Samples per chain
            - "num_parameters" : int - Number of parameters

        Raises
        ------
        RuntimeError
            If evidence computation fails or required methods haven't been called.

        Notes
        -----
        The evidence is computed using the learned harmonic mean estimator.
        The flow must be trained (fit_flow) and data must be split (split_data)
        before calling this method.
        """
        if self.chains_infer is None or self.flow is None:
            msg = "Must call split_data() and fit_flow() before compute_ln_evidence()"
            raise RuntimeError(msg)

        self.evidence = hm.Evidence(self.chains_infer.nchains, self.flow)
        assert self.evidence is not None  # for type checker
        self.evidence.add_chains(self.chains_infer)

        ln_inv_evidence = self.evidence.ln_evidence_inv
        error_bounds = self.evidence.compute_ln_inv_evidence_errors()

        ln_evidence = -ln_inv_evidence

        return {
            "ln_evidence": float(ln_evidence),
            "ln_inv_evidence": float(ln_inv_evidence),
            "error_lower": float(error_bounds[0]),
            "error_upper": float(error_bounds[1]),
            "samples_shape": tuple(self.samples.shape),
            "num_chains": self.samples.shape[0],
            "num_samples_per_chain": self.samples.shape[1],
            "num_parameters": self.samples.shape[2],
        }

    def run(
        self,
        epochs: int = 30,
        verbose: bool = False,
    ) -> dict:
        """
        Run the complete Bayesian evidence computation pipeline.

        Compute the Bayesian evidence.

        This method runs the complete evidence computation pipeline:
        1. Split data into training/inference sets
        2. Train normalizing flow on training set
        3. Compute evidence on inference set

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs for the normalizing flow. Default is 30.
        verbose : bool, optional
            Whether to print training progress. Default is False.

        Returns
        -------
        dict
            Dictionary containing evidence estimation results. See compute_ln_evidence()
            for details.

        Examples
        --------
        >>> ec = EvidenceComputation(mcmc, model)
        >>> results = ec.run(epochs=50, verbose=True)
        >>> print(f"ln(Evidence) = {results['ln_evidence']:.2f}")

        See Also
        --------
        compute_ln_evidence : Compute evidence from already-trained flow
        split_data : Split data into training/inference sets
        fit_flow : Train the normalizing flow
        """
        self.logger.debug("Starting evidence computation pipeline")
        self.split_data()
        self.fit_flow(epochs=epochs, verbose=verbose)
        results = self.compute_ln_evidence()
        self.logger.info("Evidence computed: ln(Z) = %.4f", results["ln_evidence"])
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
            If run has not been called yet.
        """
        if self.chains is None:
            msg = "Chains not initialized. Call run() first."
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
            If run has not been called yet or flow training failed.
        """
        if self.flow is None:
            msg = "Flow model not trained. Call run() first."
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
            If run has not been called yet.
        """
        if self.evidence is None:
            msg = "Evidence not computed. Call run() first."
            self.logger.error(msg)
            raise RuntimeError(msg)
        return self.evidence
