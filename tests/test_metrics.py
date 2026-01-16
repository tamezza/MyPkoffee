"""Test the model evaluation metrics for assessing fit quality."""

import numpy as np
import pytest

from pkoffee.data import data_dtype
from pkoffee.metrics import (
    SizeMismatchError,
    check_size_match,
    compute_mae,
    compute_r2,
    compute_rmse,
)


class TestCheckSizeMatch:
    """Tests for check_size_match function."""

    def test_same_size_arrays(self):
        """Test that same-sized arrays pass validation."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        check_size_match(a, b)  # Should not raise

    def test_different_size_arrays(self):
        """Test that different-sized arrays raise SizeMismatchError."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        with pytest.raises(SizeMismatchError):
            check_size_match(a, b)

    def test_empty_arrays(self):
        """Test that empty arrays of same size pass."""
        a = np.array([])
        b = np.array([])
        check_size_match(a, b)  # Should not raise


class TestComputeR2:
    """Tests for compute_r2 function."""

    def test_perfect_fit(self):
        """Test R² = 1.0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        r2 = compute_r2(y_true, y_pred)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"

    def test_good_fit(self):
        """Test R² for a good but not perfect fit."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        r2 = compute_r2(y_true, y_pred)
        assert 0.9 < r2 < 1.0, f"Expected R² between 0.9 and 1.0, got {r2}"

    def test_poor_fit(self):
        """Test R² for a poor fit (can be negative)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 10.0, 10.0, 10.0])
        r2 = compute_r2(y_true, y_pred)
        assert r2 < 0, f"Expected negative R², got {r2}"

    def test_constant_true_values(self):
        """Test R² returns NaN when all true values are the same."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0, 5.0])
        r2 = compute_r2(y_true, y_pred)
        assert np.isnan(r2), f"Expected NaN for constant y_true, got {r2}"

    def test_size_mismatch(self):
        """Test that size mismatch raises error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(SizeMismatchError):
            compute_r2(y_true, y_pred)


class TestComputeRMSE:
    """Tests for compute_rmse function."""

    def test_perfect_fit(self):
        """Test RMSE = 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        rmse = compute_rmse(y_true, y_pred)
        assert np.isclose(rmse, 0.0), f"Expected RMSE=0.0, got {rmse}"

    def test_known_rmse(self):
        """Test RMSE calculation with known result."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        rmse = compute_rmse(y_true, y_pred)
        expected = 0.1
        assert np.isclose(rmse, expected, atol=1e-5), f"Expected RMSE≈{expected}, got {rmse}"

    def test_size_mismatch(self):
        """Test that size mismatch raises error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(SizeMismatchError):
            compute_rmse(y_true, y_pred)


class TestComputeMAE:
    """Tests for compute_mae function."""

    def test_perfect_fit(self):
        """Test MAE = 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        mae = compute_mae(y_true, y_pred)
        assert np.isclose(mae, 0.0), f"Expected MAE=0.0, got {mae}"

    def test_known_mae(self):
        """Test MAE calculation with known result."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        mae = compute_mae(y_true, y_pred)
        expected = 0.1
        assert np.isclose(mae, expected, atol=1e-5), f"Expected MAE≈{expected}, got {mae}"

    def test_asymmetric_errors(self):
        """Test MAE handles both positive and negative errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.5, 2.5, 2.5, 4.5])
        mae = compute_mae(y_true, y_pred)
        expected = 0.5
        assert np.isclose(mae, expected), f"Expected MAE={expected}, got {mae}"

    def test_size_mismatch(self):
        """Test that size mismatch raises error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(SizeMismatchError):
            compute_mae(y_true, y_pred)
