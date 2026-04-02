"""Tests for data_loader.py"""

import io
import sys

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_data, clean_data


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_data(self):
        """Test that data loads successfully."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[1] == 14  # 13 features + 1 target

    def test_data_columns(self):
        """Test that required columns exist."""
        df = load_data()
        expected_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_clean_data_removes_duplicates(self):
        """Test that duplicate rows are removed."""
        df = load_data()
        original_len = len(df)
        df_clean = clean_data(df)
        assert len(df_clean) <= original_len
        assert df_clean.duplicated().sum() == 0

    def test_clean_data_preserves_columns(self):
        """Test that columns are preserved after cleaning."""
        df = load_data()
        df_clean = clean_data(df)
        assert df.columns.equals(df_clean.columns)

    def test_clean_data_prints_with_cp1252_stdout(self, monkeypatch):
        """Test that duplicate summary prints safely on Windows cp1252 consoles."""
        df = load_data()
        fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", fake_stdout)

        df_clean = clean_data(df)

        assert len(df_clean) > 0

    def test_target_values_are_binary(self):
        """Test that target is binary (0 or 1)."""
        df = load_data()
        df_clean = clean_data(df)
        assert set(df_clean["target"].unique()).issubset({0, 1})

    def test_no_null_values_after_clean(self):
        """Test that there are no null values in cleaned data."""
        df = load_data()
        df_clean = clean_data(df)
        assert df_clean.isnull().sum().sum() == 0

    def test_numeric_features_are_numeric(self):
        """Test that numeric features have correct dtype."""
        df = load_data()
        numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        for feat in numeric_features:
            assert pd.api.types.is_numeric_dtype(df[feat]), f"{feat} is not numeric"

    def test_categorical_features_in_valid_range(self):
        """Test that categorical features are in expected ranges."""
        df = load_data()
        assert df["sex"].isin([0, 1]).all(), "sex should be 0 or 1"
        assert df["cp"].isin([0, 1, 2, 3]).all(), "cp should be 0-3"
        assert df["fbs"].isin([0, 1]).all(), "fbs should be 0 or 1"
        assert df["restecg"].isin([0, 1, 2]).all(), "restecg should be 0-2"
        assert df["exang"].isin([0, 1]).all(), "exang should be 0 or 1"
        assert df["slope"].isin([0, 1, 2]).all(), "slope should be 0-2"
        assert df["ca"].isin([0, 1, 2, 3, 4]).all(), "ca should be 0-4"
        assert df["thal"].isin([0, 1, 2, 3]).all(), "thal should be 0-3"
