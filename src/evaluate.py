"""
evaluate.py
Evaluates the trained model with classification metrics,
confusion matrix, threshold optimization, cross-validation, and feature importance.
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
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data, clean_data
from src.preprocess import split_data


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


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform k-fold cross-validation on the model.

    Parameters:
        model: sklearn Pipeline
        X_train: Training features
        y_train: Training target
        cv: Number of folds (default 5)

    Returns:
        Dictionary with cross-validation scores
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

    print(f"\n--- Cross-Validation Results ({cv}-fold) ---")
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        print(f"  {metric.capitalize():12}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"    Fold scores: {[f'{s:.4f}' for s in scores]}")

    return cv_results


def plot_feature_importance(model, X_train, y_train):
    """
    Calculate and plot feature importance using permutation importance.

    Parameters:
        model: trained sklearn Pipeline
        X_train: Training features
        y_train: Training target
    """
    # Permutation importance on a fitted pipeline scores the original input columns.
    feature_names = X_train.columns.tolist()

    # Calculate permutation importance
    result = permutation_importance(
        model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
    )

    # Sort by importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    }).sort_values('Importance', ascending=False)

    print("\n--- Feature Importance (Top 10) ---")
    print(importance_df.head(10).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(10)
    ax.barh(range(len(top_features)), top_features['Importance'], xerr=top_features['Std'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Most Important Features')
    ax.invert_yaxis()
    plt.tight_layout()

    if FIGURES_DIR.exists() or FIGURES_DIR.mkdir(parents=True, exist_ok=True) or True:
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        print(f"\nSaved: {FIGURES_DIR / 'feature_importance.png'}")

    plt.close(fig)
    return importance_df


def plot_metrics_comparison(thresholds, metrics_dict):
    """Plot how metrics change across different thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Metrics vs Classification Threshold', fontsize=14, fontweight='bold')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[t][metric] for t in thresholds]
        axes_flat[idx].plot(thresholds, values, marker='o', linewidth=2)
        axes_flat[idx].set_xlabel('Threshold')
        axes_flat[idx].set_ylabel(metric)
        axes_flat[idx].set_title(metric)
        axes_flat[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "metrics_vs_threshold.png", dpi=150)
    print(f"Saved: {FIGURES_DIR / 'metrics_vs_threshold.png'}")

    plt.close(fig)


def comprehensive_evaluation(model, X_train, y_train, X_test, y_test):
    """
    Run comprehensive evaluation including CV, feature importance, and threshold analysis.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    # Cross-validation
    cross_validate_model(model, X_train, y_train, cv=5)

    # Standard evaluation at 0.5 threshold
    metrics_default, y_pred_default, y_prob = evaluate_model(model, X_test, y_test, threshold=0.5)

    # Optimized threshold evaluation (0.4 for recall priority in screening)
    metrics_optimized, y_pred_optimized, _ = evaluate_model(model, X_test, y_test, threshold=0.4)

    # Feature importance
    importance_df = plot_feature_importance(model, X_train, y_train)

    # Confusion matrices
    plot_confusion_matrix(y_test, y_pred_default, save=True)

    # ROC curve
    plot_roc_curve(y_test, y_prob, save=True)

    # Threshold analysis
    threshold_analysis(y_test, y_prob)

    # Metrics comparison across thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    metrics_across_thresholds = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics_across_thresholds[t] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
        }

    plot_metrics_comparison(thresholds, metrics_across_thresholds)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Load data and model
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    model = load_model()

    # Run comprehensive evaluation
    comprehensive_evaluation(model, X_train, y_train, X_test, y_test)
