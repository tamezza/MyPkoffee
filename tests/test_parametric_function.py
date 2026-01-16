"""Tests for parametric functions."""

import numpy as np
import pytest

from pkoffee.data import data_dtype
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    Peak2Model,
    PeakModel,
    Quadratic,
)


class TestQuadratic:
    """Tests for Quadratic function."""

    def test_call(self):
        """Test quadratic function evaluation."""
        quad = Quadratic()
        x = np.array([0, 1, 2], dtype=data_dtype)
        result = quad(x, a0=data_dtype(1.0), a1=data_dtype(2.0), a2=data_dtype(3.0))
        expected = np.array([1.0, 6.0, 17.0], dtype=data_dtype)  # 1 + 2x + 3x²
        np.testing.assert_array_almost_equal(result, expected)

    def test_param_guess(self):
        """Test parameter guess generation."""
        params = Quadratic.param_guess(y_min=data_dtype(0.5))
        assert "a0" in params
        assert "a1" in params
        assert "a2" in params
        assert params["a0"] == data_dtype(0.5)
        assert params["a1"] == data_dtype(0.0)
        assert params["a2"] == data_dtype(0.01)

    def test_param_bounds(self):
        """Test parameter bounds."""
        bounds = Quadratic.param_bounds()
        assert "a0" in bounds.min and "a0" in bounds.max
        assert "a1" in bounds.min and "a1" in bounds.max
        assert "a2" in bounds.min and "a2" in bounds.max


class TestMichaelisMentenSaturation:
    """Tests for Michaelis-Menten function."""

    def test_call(self):
        """Test Michaelis-Menten function evaluation."""
        mm = MichaelisMentenSaturation()
        x = np.array([0, 1, 10], dtype=data_dtype)
        result = mm(x, v_max=data_dtype(10.0), k=data_dtype(1.0), y0=data_dtype(0.0))
        # At x=0: y0 + 10*0/(1+0) = 0
        # At x=1: 0 + 10*1/(1+1) = 5
        # At x=10: 0 + 10*10/(1+10) ≈ 9.09
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(5.0, abs=1e-5)
        assert result[2] == pytest.approx(9.09, abs=1e-2)

    def test_param_guess(self):
        """Test parameter guess generation."""
        params = MichaelisMentenSaturation.param_guess(
            x_min=data_dtype(0.0),
            x_max=data_dtype(10.0),
            y_min=data_dtype(0.5),
            y_max=data_dtype(5.0),
        )
        assert "v_max" in params
        assert "k" in params
        assert "y0" in params
        assert params["v_max"] > 0
        assert params["k"] > 0
        assert params["y0"] == data_dtype(0.5)

    def test_param_bounds(self):
        """Test parameter bounds."""
        bounds = MichaelisMentenSaturation.param_bounds()
        assert bounds.min["k"] == data_dtype(0.0)  # k must be non-negative


class TestLogistic:
    """Tests for Logistic function."""

    def test_call(self):
        """Test logistic function evaluation."""
        logistic = Logistic()
        x = np.array([0, 5, 10], dtype=data_dtype)
        result = logistic(
            x, L=data_dtype(10.0), k=data_dtype(1.0), x0=data_dtype(5.0), y0=data_dtype(0.0)
        )
        # At x=x0=5: y0 + L/(1+e^0) = 0 + 10/2 = 5
        assert result[1] == pytest.approx(5.0, abs=1e-5)
        # Function should be symmetric around x0
        assert result[0] < result[1] < result[2]

    def test_param_guess(self):
        """Test parameter guess generation."""
        params = Logistic.param_guess(
            x_min=data_dtype(0.0),
            x_max=data_dtype(10.0),
            y_min=data_dtype(0.5),
            y_max=data_dtype(5.0),
        )
        assert "L" in params
        assert "k" in params
        assert "x0" in params
        assert "y0" in params
        assert params["x0"] == pytest.approx(5.0, abs=1e-5)  # Midpoint

    def test_param_bounds(self):
        """Test parameter bounds."""
        bounds = Logistic.param_bounds()
        assert bounds.min["k"] == data_dtype(0.0)  # k must be non-negative


class TestPeakModel:
    """Tests for Peak model."""

    def test_call(self):
        """Test peak function evaluation."""
        peak = PeakModel()
        x = np.array([0, 1, 2], dtype=data_dtype)
        result = peak(x, a=data_dtype(1.0), b=data_dtype(1.0))
        # f(x) = a*x*exp(-x/b)
        # At x=0: 0
        # At x=1: 1*1*exp(-1) ≈ 0.368
        # At x=2: 1*2*exp(-2) ≈ 0.271
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(np.exp(-1), abs=1e-3)
        assert result[2] == pytest.approx(2 * np.exp(-2), abs=1e-3)

    def test_peak_location(self):
        """Test that function has a peak at x=b."""
        peak = PeakModel()
        b_val = data_dtype(2.0)
        x = np.linspace(0, 10, 100, dtype=data_dtype)
        result = peak(x, a=data_dtype(1.0), b=b_val)
        max_idx = np.argmax(result)
        # Peak should be near x=b
        assert x[max_idx] == pytest.approx(b_val, abs=0.2)

    def test_param_guess(self):
        """Test parameter guess generation."""
        params = PeakModel.param_guess(
            x_min=data_dtype(0.0), x_max=data_dtype(10.0), y_max=data_dtype(5.0)
        )
        assert "a" in params
        assert "b" in params

    def test_param_bounds(self):
        """Test parameter bounds."""
        bounds = PeakModel.param_bounds()
        assert bounds.min["b"] == data_dtype(0.0)  # b must be non-negative


class TestPeak2Model:
    """Tests for Peak² model."""

    def test_call(self):
        """Test peak² function evaluation."""
        peak2 = Peak2Model()
        x = np.array([0, 1, 2], dtype=data_dtype)
        result = peak2(x, a=data_dtype(1.0), b=data_dtype(1.0))
        # f(x) = a*x²*exp(-x/b)
        # At x=0: 0
        # At x=1: 1*1*exp(-1) ≈ 0.368
        # At x=2: 1*4*exp(-2) ≈ 0.541
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(np.exp(-1), abs=1e-3)
        assert result[2] == pytest.approx(4 * np.exp(-2), abs=1e-3)

    def test_peak_location(self):
        """Test that function has a peak at x=2b."""
        peak2 = Peak2Model()
        b_val = data_dtype(2.0)
        x = np.linspace(0, 10, 100, dtype=data_dtype)
        result = peak2(x, a=data_dtype(1.0), b=b_val)
        max_idx = np.argmax(result)
        # Peak should be near x=2b
        assert x[max_idx] == pytest.approx(2 * b_val, abs=0.3)

    def test_param_guess(self):
        """Test parameter guess generation."""
        params = Peak2Model.param_guess(
            x_min=data_dtype(0.0), x_max=data_dtype(10.0), y_max=data_dtype(5.0)
        )
        assert "a" in params
        assert "b" in params

    def test_param_bounds(self):
        """Test parameter bounds."""
        bounds = Peak2Model.param_bounds()
        assert bounds.min["b"] == data_dtype(0.0)  # b must be non-negative
