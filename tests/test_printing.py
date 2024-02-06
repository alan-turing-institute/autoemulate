import numpy as np
import pandas as pd
import pytest

from autoemulate.emulators import GaussianProcessSk, RandomForest
from autoemulate.printing import print_cv_results
from autoemulate.utils import get_model_name

# prep inputs
MODEL_REGISTRY = {"GaussianProcessSk": GaussianProcessSk, "RandomForest": RandomForest}
models = [MODEL_REGISTRY[model]() for model in MODEL_REGISTRY.keys()]

# make scores_df
metrics = ["rmse", "r2"]
model_names = [get_model_name(model) for model in models]
data = []
for model in model_names:
    for metric in metrics:
        for fold in range(5):
            score = (
                np.random.uniform(-5000, 5000)
                if metric == "rmse"
                else np.random.uniform(-1, 1)
            )
            data.append(
                {"model": model, "metric": metric, "fold": fold, "score": score}
            )
scores_df = pd.DataFrame(data)


def test_print_results_all_models(capsys):
    print_cv_results(models, scores_df)
    captured = capsys.readouterr()
    assert "Average scores across all models:" in captured.out
    assert "model" in captured.out


def test_print_results_single_model(capsys):
    print_cv_results(models, scores_df, model="GaussianProcessSk")
    captured = capsys.readouterr()
    assert "Scores for GaussianProcessSk across all folds:" in captured.out
    assert "fold" in captured.out
    assert "metric" in captured.out


def test_print_results_invalid_model():
    with pytest.raises(ValueError):
        print_cv_results(models, scores_df, model="InvalidModel")
