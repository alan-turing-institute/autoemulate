import re

import numpy as np
import pandas as pd
import pytest

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import RandomForest
from autoemulate.utils import get_short_model_name

models = [GaussianProcessSklearn(), RandomForest()]

# make scores_df
metrics = ["rmse", "r2"]
model_names = [model.model_name for model in models]
data = []
for model in model_names:
    for metric in metrics:
        for fold in range(5):
            score = (
                np.random.uniform(-5000, 5000)
                if metric == "rmse"
                else np.random.uniform(-1, 1)
            )
            short = "".join(re.findall(r"[A-Z]", model)).lower()
            data.append(
                {
                    "model": model,
                    "short": short,
                    "metric": metric,
                    "fold": fold,
                    "score": score,
                }
            )
scores_df = pd.DataFrame(data)
