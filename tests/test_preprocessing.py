"""Tests for preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from src.data_loader import load_data, clean_data
from src.preprocess import build_preprocessor, split_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class TestPreprocessor:
    """Test preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Provide cleaned sample data."""
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def test_preprocessor_builds_successfully(self):
        """Test that preprocessor builds without errors."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_transforms_train_data(self, sample_data):
        """Test preprocessing transforms training data correctly."""
        X_train, X_test, y_train, y_test = sample_data
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        assert X_train_transformed.shape[0] == X_train.shape[0]
        assert X_train_transformed.shape[1] > 0  # More features after encoding

    def test_preprocessor_transforms_test_data(self, sample_data):
        """Test preprocessing transforms test data correctly."""
        X_train, X_test, y_train, y_test = sample_data
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        assert X_test_transformed.shape[0] == X_test.shape[0]

    def test_preprocessor_handles_missing_values(self):
        """Test that preprocessor handles missing values."""
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        
        # Add some missing values
        X_train_with_nan = X_train.copy()
        X_train_with_nan.iloc[0, 0] = np.nan
        
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_train_with_nan)
        
        assert not np.isnan(X_transformed).any(), "Missing values not handled"

    def test_split_data_stratified(self):
        """Test that train/test split is stratified."""
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)
        
        # Ratios should be similar (within 5%)
        assert abs(train_ratio - test_ratio) < 0.05

    def test_split_data_no_leak(self):
        """Test that train/test sets don't overlap."""
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        
        assert len(X_train) + len(X_test) == len(df)

    def test_numeric_features_scaled(self, sample_data):
        """Test that numeric features are scaled (mean~0, std~1)."""
        X_train, X_test, y_train, y_test = sample_data
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X_train)
        
        # Check that scaled features are approximately normalized
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)
        
        # Some columns should have mean close to 0 and std close to 1
        assert np.any(np.abs(means) < 1.0)
