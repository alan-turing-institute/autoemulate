import pandas as pd
import pytest
from autoemulate.core.metrics import R2, RMSE
from autoemulate.core.results import Result, Results
from autoemulate.emulators.transformed.base import TransformedEmulator


class DummyEmulator(TransformedEmulator):
    def __init__(self, x_transforms, y_transforms):
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms


def make_result(
    idx, r2, r2_std, r2_train, r2_train_std, rmse, rmse_std, rmse_train, rmse_train_std
):
    return Result(
        id=idx,
        model_name=f"model{idx}",
        model=DummyEmulator(x_transforms=[f"x{idx}"], y_transforms=[f"y{idx}"]),
        params={"param": idx},
        test_metrics={
            R2: (r2, r2_std),
            RMSE: (rmse, rmse_std),
        },
        train_metrics={
            R2: (r2_train, r2_train_std),
            RMSE: (rmse_train, rmse_train_std),
        },
    )


def test_result_attributes():
    emu = DummyEmulator(["x"], ["y"])
    res = Result(
        id=0,
        model_name="mymodel",
        model=emu,
        params={"foo": 1},
        test_metrics={R2: (0.9, 0.01), RMSE: (0.1, 0.02)},
        train_metrics={R2: (0.95, 0.015), RMSE: (0.05, 0.025)},
    )
    assert res.id == 0
    assert res.model_name == "mymodel"
    assert res.model is emu
    assert res.x_transforms == ["x"]
    assert res.y_transforms == ["y"]
    assert res.params == {"foo": 1}
    assert res.test_metrics[R2] == (0.9, 0.01)
    assert res.test_metrics[RMSE] == (0.1, 0.02)


def test_result_metadata_df():
    res = make_result(42, 0.8, 0.02, 0.85, 0.025, 0.12, 0.03, 0.07, 0.035)
    df = res.metadata_df()
    assert isinstance(df, pd.DataFrame)
    # Verify expected columns are present (order may vary due to dict iteration)
    expected_base_columns = [
        "id",
        "model_name",
        "x_transforms",
        "y_transforms",
        "params",
    ]
    expected_metric_columns = [
        "r2_test",
        "r2_test_std",
        "rmse_test",
        "rmse_test_std",
        "r2_train",
        "r2_train_std",
        "rmse_train",
        "rmse_train_std",
    ]
    for col in expected_base_columns + expected_metric_columns:
        assert col in df.columns
    assert df.loc[0, "id"] == 42
    assert df.loc[0, "model_name"] == "model42"
    assert df.loc[0, "x_transforms"] == ["x42"]
    assert df.loc[0, "y_transforms"] == ["y42"]
    assert df.loc[0, "params"] == "{'param': 42}"
    assert df.loc[0, "r2_test"] == 0.8
    assert df.loc[0, "rmse_test"] == 0.12
    assert df.loc[0, "r2_test_std"] == 0.02
    assert df.loc[0, "rmse_test_std"] == 0.03
    assert df.loc[0, "r2_train"] == 0.85
    assert df.loc[0, "rmse_train"] == 0.07
    assert df.loc[0, "r2_train_std"] == 0.025
    assert df.loc[0, "rmse_train_std"] == 0.035


def test_results_add_and_get():
    r1 = make_result(1, 0.5, 0.01, 0.55, 0.02, 1.0, 0.1, 0.9, 0.05)
    r2 = make_result(2, 0.7, 0.015, 0.75, 0.025, 0.8, 0.08, 0.7, 0.04)
    res = Results()
    res.add_result(r1)
    res.add_result(r2)
    assert res.get_result(1) is r1
    assert res.get_result(2) is r2


def test_results_best_result():
    r1 = make_result(1, 0.5, 0.01, 0.55, 0.02, 1.0, 0.1, 0.9, 0.05)
    r2 = make_result(2, 0.7, 0.015, 0.75, 0.025, 0.8, 0.08, 0.7, 0.04)
    r3 = make_result(3, 0.6, 0.012, 0.65, 0.018, 0.9, 0.09, 0.8, 0.045)
    res = Results([r1, r2, r3])
    best = res.best_result()
    assert best is r2


def test_results_best_result_empty():
    res = Results()
    with pytest.raises(ValueError, match="No results available"):
        res.best_result()


def test_results_get_result_not_found():
    r1 = make_result(1, 0.5, 0.01, 0.55, 0.02, 1.0, 0.1, 0.9, 0.05)
    res = Results([r1])
    with pytest.raises(ValueError, match="No result found with ID: 3"):
        res.get_result(3)


def test_results_summarize():
    r1 = make_result(1, 0.5, 0.01, 0.55, 0.02, 1.0, 0.1, 0.9, 0.05)
    r2 = make_result(
        2, 0.7, 0.015, 0.75, 0.025, 0.8, 0.08, 0.7, 0.04
    )  # r2_score higher
    res = Results([r1, r2])

    summary = res.summarize()
    assert isinstance(summary, pd.DataFrame)

    # Verify DataFrame is sorted by r2_test descending (r2 first, then r1)
    assert summary.iloc[0]["model_name"] == "model2"
    assert summary.iloc[1]["model_name"] == "model1"

    # Verify all expected columns are present
    assert "model_name" in summary.columns
    assert "x_transforms" in summary.columns
    assert "y_transforms" in summary.columns
    assert "params" in summary.columns
    assert "r2_test" in summary.columns
    assert "r2_test_std" in summary.columns
    assert "rmse_test" in summary.columns
    assert "rmse_test_std" in summary.columns
    assert "r2_train" in summary.columns
    assert "r2_train_std" in summary.columns
    assert "rmse_train" in summary.columns
    assert "rmse_train_std" in summary.columns

    # Verify values for r2
    assert summary.iloc[0]["r2_test"] == 0.7
    assert summary.iloc[1]["r2_test"] == 0.5
    assert summary.iloc[0]["rmse_test"] == 0.8
    assert summary.iloc[1]["rmse_test"] == 1.0
