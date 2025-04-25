from autoemulate.experimental.emulators.neural_processes.conditional_neural_process import (  # noqa: E501
    CNPModule,
)
from autoemulate.experimental.types import DistributionLike


def test_predict_with_cnp_module(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d

    cnp = CNPModule(x, y)  # Assuming x is of shape (n_samples, x_dim)

    cnp.fit(x, y)

    x, y = new_data_y1d
    distribution = cnp.predict(x)
    assert isinstance(distribution, DistributionLike)
