"""Tests for data loading and preprocessing utilities."""

import errno
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkoffee.data import (
    ColumnTypeError,
    CSVReadError,
    MissingColumnsError,
    RequiredColumn,
    curate,
    data_dtype,
    extract_arrays,
    load_csv,
    validate,
)


class TestValidate:
    """Tests for data validation."""

    def test_valid_data(self):
        """Test validation passes for valid data."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        validate(data)  # Should not raise

    def test_missing_cups_column(self):
        """Test error when cups column is missing."""
        data = pd.DataFrame({
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        with pytest.raises(MissingColumnsError):
            validate(data)

    def test_missing_productivity_column(self):
        """Test error when productivity column is missing."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
        })
        with pytest.raises(MissingColumnsError):
            validate(data)

    def test_missing_both_columns(self):
        """Test error when both required columns are missing."""
        data = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(MissingColumnsError):
            validate(data)

    def test_invalid_cups_type(self):
        """Test error when cups column has invalid type."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: ["one", "two", "three"],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        with pytest.raises(ColumnTypeError):
            validate(data)

    def test_invalid_productivity_type(self):
        """Test error when productivity column has invalid type."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
            RequiredColumn.PRODUCTIVITY: ["low", "medium", "high"],
        })
        with pytest.raises(ColumnTypeError):
            validate(data)


class TestCurate:
    """Tests for data curation."""

    def test_no_nan_values(self):
        """Test that data without NaN is unchanged."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        curated = curate(data)
        assert len(curated) == len(data)
        pd.testing.assert_frame_equal(curated, data)

    def test_removes_nan_in_cups(self):
        """Test that rows with NaN in cups are removed."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, np.nan, 3],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        curated = curate(data)
        assert len(curated) == 2
        assert not curated[RequiredColumn.CUPS].isna().any()

    def test_removes_nan_in_productivity(self):
        """Test that rows with NaN in productivity are removed."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
            RequiredColumn.PRODUCTIVITY: [0.5, np.nan, 1.5],
        })
        curated = curate(data)
        assert len(curated) == 2
        assert not curated[RequiredColumn.PRODUCTIVITY].isna().any()

    def test_removes_multiple_nan_rows(self):
        """Test that multiple rows with NaN are removed."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, np.nan, 3, 4, np.nan],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, np.nan, 1.8, 2.0],
        })
        curated = curate(data)
        assert len(curated) == 2  # Only rows 0 and 4 are valid


class TestLoadCSV:
    """Tests for CSV loading."""

    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("cups,productivity\n")
            f.write("1,0.5\n")
            f.write("2,1.0\n")
            f.write("3,1.5\n")
            temp_path = Path(f.name)

        try:
            data = load_csv(temp_path)
            assert len(data) == 3
            assert RequiredColumn.CUPS in data.columns
            assert RequiredColumn.PRODUCTIVITY in data.columns
        finally:
            temp_path.unlink()

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_csv(Path("nonexistent_file.csv"))

    def test_invalid_csv_format(self):
        """Test error when CSV is malformed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid csv content without proper structure\n")
            f.write("more invalid content\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(MissingColumnsError):
                load_csv(temp_path)
        finally:
            temp_path.unlink()

    def test_csv_with_nan_values(self):
        """Test that CSV with NaN values is automatically curated."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("cups,productivity\n")
            f.write("1,0.5\n")
            f.write("2,\n")  # Missing productivity
            f.write("3,1.5\n")
            temp_path = Path(f.name)

        try:
            data = load_csv(temp_path)
            assert len(data) == 2  # Row with missing value should be removed
        finally:
            temp_path.unlink()


class TestExtractArrays:
    """Tests for array extraction."""

    def test_extract_arrays(self):
        """Test extraction of numpy arrays from DataFrame."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [1, 2, 3],
            RequiredColumn.PRODUCTIVITY: [0.5, 1.0, 1.5],
        })
        cups, productivity = extract_arrays(data)
        
        assert isinstance(cups, np.ndarray)
        assert isinstance(productivity, np.ndarray)
        assert cups.dtype == data_dtype
        assert productivity.dtype == data_dtype
        assert len(cups) == 3
        assert len(productivity) == 3
        np.testing.assert_array_equal(cups, np.array([1, 2, 3], dtype=data_dtype))
        np.testing.assert_array_equal(productivity, np.array([0.5, 1.0, 1.5], dtype=data_dtype))

    def test_extract_preserves_order(self):
        """Test that extraction preserves row order."""
        data = pd.DataFrame({
            RequiredColumn.CUPS: [5, 2, 8, 1],
            RequiredColumn.PRODUCTIVITY: [2.5, 1.0, 4.0, 0.5],
        })
        cups, productivity = extract_arrays(data)
        
        np.testing.assert_array_equal(cups, np.array([5, 2, 8, 1], dtype=data_dtype))
        np.testing.assert_array_equal(productivity, np.array([2.5, 1.0, 4.0, 0.5], dtype=data_dtype))
