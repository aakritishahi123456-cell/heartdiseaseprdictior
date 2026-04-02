"""
evaluate.py
Evaluates the trained model with classification metrics,
confusion matrix, and threshold optimization analysis.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from data_loader import load_data, clean_data
from preprocess import split_data


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def load_model():
    """Load saved model pipeline."""
    return joblib.load(MODEL_PATH)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model with standard classification metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }

    print(f"\n--- Evaluation (threshold = {threshold}) ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, save=True):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_yticklabels(["Low Risk", "High Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'confusion_matrix.png'}")

    plt.close(fig)


def plot_roc_curve(y_test, y_prob, save=True):
    """Plot and optionally save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'roc_curve.png'}")

    plt.close(fig)


def threshold_analysis(y_test, y_prob):
    """Analyze different thresholds and their impact on recall/precision."""
    print("\n--- Threshold Analysis ---")
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Accuracy':<10}")
    print("-" * 52)

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        r = recall_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        a = accuracy_score(y_test, y_pred)
        print(f"  {t:<10.2f} {r:<10.3f} {p:<10.3f} {f:<10.3f} {a:<10.3f}")


if __name__ == "__main__":
    # Load data and model
    df = clean_data(load_data())
    _, X_test, _, y_test = split_data(df)
    model = load_model()

    # Evaluate at default and optimized thresholds
    metrics_default, y_pred_default, y_prob = evaluate_model(model, X_test, y_test, threshold=0.5)
    metrics_optimized, y_pred_optimized, _ = evaluate_model(model, X_test, y_test, threshold=0.4)

    # Plots
    plot_confusion_matrix(y_test, y_pred_optimized)
    plot_roc_curve(y_test, y_prob)

    # Threshold analysis
    threshold_analysis(y_test, y_prob)
