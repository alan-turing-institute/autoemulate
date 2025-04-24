from autoemulate.experimental.emulators.neural_processes.conditional_neural_process import (
    CNPModule,
)


def test_predict_with_cnp_module(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d

    cnp = CNPModule(input_dim=x.shape[1])  # Assuming x is of shape (n_samples, x_dim)

    cnp.fit(x, y)

    x, y = new_data_y1d
    predictions = cnp.predict(x)
    print(predictions.shape)
