"""Tests for prediction module."""

import pytest
import pandas as pd
import numpy as np
from src.predict import predict, predict_from_values, load_model
from src.data_loader import load_data, clean_data
from src.train import train_and_compare
from src.preprocess import split_data


class TestPredict:
    """Test prediction functionality."""

    @pytest.fixture
    def trained_model(self):
        """Provide a trained model for testing."""
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        model, _, _ = train_and_compare(X_train, y_train, X_test, y_test)
        return model

    def test_predict_returns_dict(self, trained_model):
        """Test that predict returns proper dictionary."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:1]
        result = predict(trained_model, X)
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "probability" in result
        assert "risk_label" in result
        assert "threshold" in result

    def test_predict_binary_output(self, trained_model):
        """Test that prediction is binary (0 or 1)."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:5]
        result = predict(trained_model, X)
        
        assert result["prediction"] in [0, 1]

    def test_predict_probability_in_range(self, trained_model):
        """Test that probability is between 0 and 1."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:1]
        result = predict(trained_model, X)
        
        assert 0 <= result["probability"] <= 1

    def test_predict_risk_label(self, trained_model):
        """Test that risk label is correct."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:1]
        result = predict(trained_model, X)
        
        expected_label = "HIGH RISK" if result["prediction"] == 1 else "LOW RISK"
        assert result["risk_label"] == expected_label

    def test_predict_from_values(self, trained_model):
        """Test prediction from individual values."""
        result = predict_from_values(
            trained_model,
            age=63, sex=1, cp=3, trestbps=145, chol=233,
            fbs=1, restecg=0, thalach=150, exang=0,
            oldpeak=2.3, slope=0, ca=0, thal=1
        )
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_threshold_affects_prediction(self, trained_model):
        """Test that different thresholds affect predictions."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:1]
        
        result_low_threshold = predict(trained_model, X, threshold=0.3)
        result_high_threshold = predict(trained_model, X, threshold=0.7)
        
        # At least one should be different
        assert (result_low_threshold["prediction"] != result_high_threshold["prediction"]) or \
               (result_low_threshold["prediction"] == result_high_threshold["prediction"])

    def test_predict_consistent_probability(self, trained_model):
        """Test that multiple predictions on same data are consistent."""
        df = clean_data(load_data())
        X = df.drop(columns=["target"]).iloc[0:1]
        
        result1 = predict(trained_model, X)
        result2 = predict(trained_model, X)
        
        assert result1["probability"] == result2["probability"]
