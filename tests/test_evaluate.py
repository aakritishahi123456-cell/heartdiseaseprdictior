"""Tests for evaluation utilities."""

import numpy as np

from src.data_loader import clean_data, load_data
from src.evaluate import load_model, plot_feature_importance
from src.preprocess import split_data


class DummyPermutationResult:
    """Lightweight stand-in for sklearn permutation importance results."""

    def __init__(self, size):
        self.importances_mean = np.linspace(0.1, 0.01, size)
        self.importances_std = np.linspace(0.01, 0.001, size)


def test_plot_feature_importance_uses_original_feature_names(tmp_path, monkeypatch):
    """Test that permutation importance aligns with the original training columns."""
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    model = load_model()

    monkeypatch.setattr("src.evaluate.FIGURES_DIR", tmp_path)
    monkeypatch.setattr(
        "src.evaluate.permutation_importance",
        lambda *args, **kwargs: DummyPermutationResult(len(X_train.columns)),
    )

    importance_df = plot_feature_importance(model, X_train, y_train)

    assert importance_df["Feature"].tolist() == list(X_train.columns)
    assert (tmp_path / "feature_importance.png").exists()
