"""Check optional evidence imports in fresh Python processes."""

import subprocess
import sys
import textwrap

import pytest


def run_with_harmonic_import_error(error, code):
    """Block harmonic without changing imports in the pytest process."""
    setup = f"""
import importlib.abc
import sys

class BlockHarmonic(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "harmonic":
            raise {error}

sys.meta_path.insert(0, BlockHarmonic())
"""
    result = subprocess.run(
        [sys.executable, "-c", setup + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_and_calibration_import_without_harmonic():
    run_with_harmonic_import_error(
        'AssertionError("harmonic must not be imported")',
        """
        import autoemulate
        from autoemulate.calibration.bayes import (
            BayesianCalibration, extract_log_probabilities,
        )
        from autoemulate.calibration.history_matching import HistoryMatching
        from autoemulate.calibration.evidence import EvidenceComputation
        import pyro
        import pyro.distributions as dist
        import torch

        class Samples:
            def get_samples(self, group_by_chain=False):
                return {"x": torch.zeros(2, 3)}

        def model():
            pyro.sample("x", dist.Normal(0.0, 1.0))

        samples, log_probs = extract_log_probabilities(Samples(), model, device="cpu")
        assert samples.shape == (2, 3, 1)
        assert torch.isfinite(log_probs).all()
        assert "harmonic" not in sys.modules
        """,
    )


def test_missing_harmonic_has_installation_hint():
    run_with_harmonic_import_error(
        'ModuleNotFoundError("No module named harmonic", name="harmonic")',
        """
        import pytest
        from autoemulate.calibration.evidence import EvidenceComputation

        with pytest.raises(ImportError, match="pip install") as exc:
            EvidenceComputation(None, lambda: None, device="cpu")
        assert "autoemulate[evidence]" in str(exc.value)
        assert isinstance(exc.value.__cause__, ModuleNotFoundError)
        assert exc.value.__cause__.name == "harmonic"
        """,
    )


@pytest.mark.parametrize(
    "error",
    [
        'ModuleNotFoundError("broken subdependency", name="jax")',
        'ModuleNotFoundError("broken submodule", name="harmonic.model")',
        'ImportError("broken binary extension")',
    ],
)
def test_broken_harmonic_import_is_not_relabelled(error):
    run_with_harmonic_import_error(
        error,
        f"""
        import pytest
        from autoemulate.calibration.evidence import EvidenceComputation

        expected = {error}
        with pytest.raises(type(expected)) as exc:
            EvidenceComputation(None, lambda: None, device="cpu")
        assert str(exc.value) == str(expected)
        assert exc.value.name == expected.name
        assert "autoemulate[evidence]" not in str(exc.value)
        """,
    )
