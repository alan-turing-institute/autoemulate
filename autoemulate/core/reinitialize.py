import inspect

from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators import Emulator, TransformedEmulator, get_emulator_class


def fit_from_reinitialized(
    x: TensorLike,
    y: TensorLike,
    emulator: Emulator,
    transformed_emulator_params: dict | None = None,
    device: DeviceLike | None = None,
    random_seed: int | None = None,
):
    """
    Fit a fresh model with reinitialized parameters using the best configuration.

    This method creates a new model instance with the same configuration as the
    best (or specified) model from the comparison, but with freshly initialized
    parameters fitted on the provided data.

    Parameters
    ----------
    x: TensorLike
        Input features for training the fresh model.
    y: TensorLike
        Target values for training the fresh model.
    emulator: Emulator
        An Emulator object containing the pre-trained emulator.
    transformed_emulator_params: None | TransformedEmulatorParams
        Parameters for the transformed emulator. When None, the same parameters as
        used when identifying the best model are used. Defaults to None.
    device: str | None
        Device to use for model fitting (e.g., 'cpu' or 'cuda'). If None, the default
        device is used. Defaults to None.
    random_seed: int | None
        Random seed for parameter initialization. Defaults to None.

    Returns
    -------
    TransformedEmulator
        A new model instance with the same configuration but fresh parameters
        fitted on the provided data.

    Notes
    -----
    Unlike TransformedEmulator.refit() which retrains an existing model,
    this method creates a completely new model instance with reinitialized
    parameters. This ensures that when fitting on new data that the same
    initialization conditions are applied. This can have an affect for example
    given kernel initialization in Gaussian Processes or weight initialization in
    neural networks.
    """
    if random_seed is not None:
        set_random_seed(seed=random_seed)

    # Extract emulator and its parameters from Emulator instance
    if isinstance(emulator, TransformedEmulator):
        model = emulator.model
        emulator_name = emulator.untransformed_model_name
        x_transforms = emulator.x_transforms
        y_transforms = emulator.y_transforms
    else:
        model = emulator
        emulator_name = emulator.model_name()
        x_transforms = None
        y_transforms = None

    # Extract parameters from the provided emulator instance
    model_cls = get_emulator_class(emulator_name)
    init_sig = inspect.signature(model_cls.__init__)
    emulator_params = {}
    for param_name in init_sig.parameters:
        if param_name in ["self", "x", "y", "device"]:
            continue
        # NOTE: some emulators have standardize_x/y params option
        # this is different to TransformedEmulator x/y transforms
        if param_name == "standardize_x":
            emulator_params["standardize_x"] = bool(model.x_transform)
        if param_name == "standardize_y":
            emulator_params["standardize_y"] = bool(model.y_transform)
        if hasattr(model, param_name):
            emulator_params[param_name] = getattr(model, param_name)

    transformed_emulator_params = transformed_emulator_params or {}

    new_emulator = TransformedEmulator(
        x.float(),
        y.float(),
        model=get_emulator_class(emulator_name),
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        device=device,
        **emulator_params,
    )

    new_emulator.fit(x, y)
    return new_emulator
