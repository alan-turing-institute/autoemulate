import pytest
import torch
from autoemulate.experimental.data.preprocessors import Standardizer
from autoemulate.experimental.types import TorchDefaultDType


class TestStandardizer:
    def test_init_valid(self):
        """
        Test Standardizer initialization with valid mean and std.
        """
        mean = torch.zeros((2, 2))
        std = torch.ones((2, 2))
        s = Standardizer(mean, std)
        assert torch.equal(s.mean, mean)
        assert torch.equal(s.std, std)

    def test_init_invalid_mean_shape(self):
        """
        Test Standardizer raises ValueError if mean is not 2D.
        """
        mean = torch.zeros((2,))
        std = torch.ones((2, 2))
        with pytest.raises(ValueError, match="mean is expected to be 2D"):
            Standardizer(mean, std)

    def test_init_invalid_std_shape(self):
        """
        Test Standardizer raises ValueError if std is not 2D.
        """
        mean = torch.zeros((2, 2))
        std = torch.ones((2,))
        with pytest.raises(ValueError, match="std is expected to be 2D"):
            Standardizer(mean, std)

    def test_small_std_replaced(self):
        """
        Test that small std values are replaced with 1.0.
        """
        mean = torch.zeros((1, 2))
        std = torch.tensor([[1e-40, 2.0]], dtype=TorchDefaultDType)
        s = Standardizer(mean, std)
        assert s.std[0, 0] == 1.0
        assert s.std[0, 1] == 2.0

    def test_preprocess_valid(self):
        """
        Test Standardizer preprocess with valid input.
        """
        mean = torch.tensor([[1.0, 2.0]])
        std = torch.tensor([[2.0, 4.0]])
        s = Standardizer(mean, std)
        x = torch.tensor([[3.0, 6.0]])
        out = s.preprocess(x)
        expected = (x - mean) / std
        assert torch.allclose(out, expected)

    def test_preprocess_invalid_type(self):
        """
        Test Standardizer preprocess raises ValueError if input is not a TensorLike.
        """
        mean = torch.zeros((1, 2))
        std = torch.ones((1, 2))
        s = Standardizer(mean, std)
        msg = "Expected 2D TensorLike, actual type <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            s.preprocess([[1.0, 2.0]])  # type: ignore PGH003

    def test_preprocess_invalid_dim(self):
        """
        Test Standardizer preprocess raises ValueError if input is not 2D.
        """
        mean = torch.zeros((1, 2))
        std = torch.ones((1, 2))
        s = Standardizer(mean, std)
        x = torch.tensor([1.0, 2.0])
        msg = "Expected 2D TensorLike, actual shape dim 1"
        with pytest.raises(ValueError, match=msg):
            s.preprocess(x)
