import pandas as pd
import pytest
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.results import Result, Results


class DummyEmulator(TransformedEmulator):
    def __init__(self, x_transforms, y_transforms):
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms


def make_result(idx, r2, rmse):
    return Result(
        id=f"id{idx}",
        model_name=f"model{idx}",
        model=DummyEmulator(x_transforms=[f"x{idx}"], y_transforms=[f"y{idx}"]),
        config={"param": idx},
        r2_score=r2,
        rmse_score=rmse,
    )


def test_result_attributes():
    emu = DummyEmulator(["x"], ["y"])
    res = Result(
        id="abc",
        model_name="mymodel",
        model=emu,
        config={"foo": 1},
        r2_score=0.9,
        rmse_score=0.1,
    )
    assert res.id == "abc"
    assert res.model_name == "mymodel"
    assert res.model is emu
    assert res.x_transforms == ["x"]
    assert res.y_transforms == ["y"]
    assert res.config == {"foo": 1}
    assert res.r2_score == 0.9
    assert res.rmse_score == 0.1


def test_results_add_and_get():
    r1 = make_result(1, 0.5, 1.0)
    r2 = make_result(2, 0.7, 0.8)
    res = Results()
    res.add_result(r1)
    res.add_result(r2)
    assert res.get_result("id1") is r1
    assert res.get_result("id2") is r2


def test_results_best_result():
    r1 = make_result(1, 0.5, 1.0)
    r2 = make_result(2, 0.7, 0.8)
    r3 = make_result(3, 0.6, 0.9)
    res = Results([r1, r2, r3])
    best = res.best_result()
    assert best is r2


def test_results_best_result_empty():
    res = Results()
    with pytest.raises(ValueError, match="No results available"):
        res.best_result()


def test_results_get_result_not_found():
    r1 = make_result(1, 0.5, 1.0)
    res = Results([r1])
    with pytest.raises(ValueError, match="No result found with ID: fakeID"):
        res.get_result("fakeID")


def test_results_summarize():
    r1 = make_result(1, 0.5, 1.0)
    r2 = make_result(2, 0.7, 0.8)  # r2_score higher
    res = Results([r1, r2])
    df = res.summarize()
    assert isinstance(df, pd.DataFrame)
    assert set(df["id"]) == {"id1", "id2"}
    # Should be sorted by r2_score descending
    assert list(df["id"]) == ["id2", "id1"]
