"""Tests for model fitting functionality."""

import numpy as np
import pytest

from pkoffee.data import data_dtype
from pkoffee.fit_model import Model, fit_model
from pkoffee.parametric_function import Quadratic


class TestModel:
    """Tests for Model class."""

    def test_model_creation(self):
        """Test creating a model instance."""
        quad = Quadratic()
        model = Model(
            name="TestQuad",
            function=quad,
            params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.01)},
            bounds=Quadratic.param_bounds(),
        )
        assert model.name == "TestQuad"
        assert model.r_squared == -data_dtype(np.inf)

    def test_predict(self):
        """Test model prediction."""
        quad = Quadratic()
        model = Model(
            name="TestQuad",
            function=quad,
            params={"a0": data_dtype(1.0), "a1": data_dtype(2.0), "a2": data_dtype(3.0)},
            bounds=Quadratic.param_bounds(),
        )
        x = np.array([0, 1, 2], dtype=data_dtype)
        predictions = model.predict(x)
        expected = np.array([1.0, 6.0, 17.0], dtype=data_dtype)
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_model_sort(self):
        """Test sorting models by R²."""
        quad = Quadratic()
        models = [
            Model(
                name="Low",
                function=quad,
                params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
                bounds=Quadratic.param_bounds(),
                r_squared=data_dtype(0.3),
            ),
            Model(
                name="High",
                function=quad,
                params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
                bounds=Quadratic.param_bounds(),
                r_squared=data_dtype(0.9),
            ),
            Model(
                name="Mid",
                function=quad,
                params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
                bounds=Quadratic.param_bounds(),
                r_squared=data_dtype(0.6),
            ),
        ]
        
        Model.sort(models)
        assert models[0].name == "High"
        assert models[1].name == "Mid"
        assert models[2].name == "Low"

    def test_model_sort_with_infinity(self):
        """Test sorting handles -inf R² values."""
        quad = Quadratic()
        models = [
            Model(
                name="Failed",
                function=quad,
                params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
                bounds=Quadratic.param_bounds(),
                r_squared=-data_dtype(np.inf),
            ),
            Model(
                name="Good",
                function=quad,
                params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
                bounds=Quadratic.param_bounds(),
                r_squared=data_dtype(0.8),
            ),
        ]
        
        Model.sort(models)
        assert models[0].name == "Good"
        assert models[1].name == "Failed"


class TestFitModel:
    """Tests for fit_model function."""

    def test_fit_quadratic_perfect_data(self):
        """Test fitting quadratic to perfect quadratic data."""
        # Generate perfect quadratic data: y = 1 + 2x + 0.5x²
        x = np.linspace(0, 10, 50, dtype=data_dtype)
        y = 1 + 2 * x + 0.5 * x**2
        
        quad = Quadratic()
        initial_model = Model(
            name="Quadratic",
            function=quad,
            params={"a0": data_dtype(0.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
            bounds=Quadratic.param_bounds(),
        )
        
        fitted_model, predictions = fit_model(x, y, initial_model)
        
        # Check R² is very close to 1
        assert fitted_model.r_squared > 0.999
        
        # Check fitted parameters are close to true values
        assert fitted_model.params["a0"] == pytest.approx(1.0, abs=1e-3)
        assert fitted_model.params["a1"] == pytest.approx(2.0, abs=1e-3)
        assert fitted_model.params["a2"] == pytest.approx(0.5, abs=1e-3)

    def test_fit_with_noise(self):
        """Test fitting to noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50, dtype=data_dtype)
        y = 1 + 2 * x + 0.5 * x**2 + np.random.normal(0, 0.5, 50).astype(data_dtype)
        
        quad = Quadratic()
        initial_model = Model(
            name="Quadratic",
            function=quad,
            params={"a0": data_dtype(1.0), "a1": data_dtype(2.0), "a2": data_dtype(0.5)},
            bounds=Quadratic.param_bounds(),
        )
        
        fitted_model, predictions = fit_model(x, y, initial_model)
        
        # R² should still be quite high for low noise
        assert fitted_model.r_squared > 0.9
        assert len(predictions) == len(x)

    def test_fit_with_nan_raises_error(self):
        """Test that NaN in data raises error."""
        x = np.array([1, 2, np.nan], dtype=data_dtype)
        y = np.array([1, 2, 3], dtype=data_dtype)
        
        quad = Quadratic()
        initial_model = Model(
            name="Quadratic",
            function=quad,
            params={"a0": data_dtype(1.0), "a1": data_dtype(0.0), "a2": data_dtype(0.0)},
            bounds=Quadratic.param_bounds(),
        )
        
        with pytest.raises(ValueError):
            fit_model(x, y, initial_model)

    def test_predictions_match_input_length(self):
        """Test that predictions have same length as input."""
        x = np.linspace(0, 10, 30, dtype=data_dtype)
        y = 1 + 2 * x + 0.5 * x**2
        
        quad = Quadratic()
        initial_model = Model(
            name="Quadratic",
            function=quad,
            params={"a0": data_dtype(1.0), "a1": data_dtype(2.0), "a2": data_dtype(0.5)},
            bounds=Quadratic.param_bounds(),
        )
        
        fitted_model, predictions = fit_model(x, y, initial_model)
        assert len(predictions) == len(x)
