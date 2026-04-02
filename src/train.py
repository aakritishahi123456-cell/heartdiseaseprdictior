"""
train.py
Trains Logistic Regression, KNN, and Random Forest models inside
a leakage-free Pipeline. Saves the best model to models/heart_model.pkl.
Uses GridSearchCV with cross-validation for comprehensive hyperparameter tuning.
"""

import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report

from src.data_loader import load_data, clean_data
from src.preprocess import build_preprocessor, split_data


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def get_pipelines(preprocessor):
    """Return a dict of model name → Pipeline."""
    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "KNN": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier())
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ]),
    }


def train_and_compare(X_train, y_train, X_test, y_test):
    """Train all models, print scores, and return the best pipeline."""
    preprocessor = build_preprocessor()
    pipelines = get_pipelines(preprocessor)

    print("\n--- Initial Model Comparison (Accuracy) ---")
    best_name = None
    best_score = 0
    best_pipeline = None
    results = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        results[name] = score
        print(f"  {name:25}: {score:.4f}")

        if score >= best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe

    print(f"\nBest initial model: {best_name} (accuracy = {best_score:.4f})")
    return best_pipeline, best_name, results


def tune_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Comprehensive hyperparameter tuning for Logistic Regression using GridSearchCV.
    Optimizes for RECALL (screening priority) with GridSearchCV on recall metric.
    """
    print("\n--- Tuning: Logistic Regression (with Recall Priority) ---")

    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "liblinear", "saga"],
        "model__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="recall",  # Optimize for recall (screening priority)
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")

    # Evaluate on test set
    test_recall = grid.score(X_test, y_test)
    print(f"  Test recall: {test_recall:.4f}")

    return grid.best_estimator_


def tune_random_forest(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Random Forest."""
    print("\n--- Tuning: Random Forest (with Recall Priority) ---")

    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")

    test_recall = grid.score(X_test, y_test)
    print(f"  Test recall: {test_recall:.4f}")

    return grid.best_estimator_


def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Gradient Boosting."""
    print("\n--- Tuning: Gradient Boosting (with Recall Priority) ---")

    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")

    test_recall = grid.score(X_test, y_test)
    print(f"  Test recall: {test_recall:.4f}")

    return grid.best_estimator_


def evaluate_tuned_models(models_dict, X_test, y_test):
    """Evaluate all tuned models and compare performance."""
    print("\n" + "="*60)
    print("TUNED MODELS COMPARISON")
    print("="*60)

    best_name = None
    best_score = 0
    best_model = None

    for name, model in models_dict.items():
        from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.4).astype(int)  # Use optimized threshold

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

        # Use recall as the comparison metric (screening priority)
        if recall >= best_score:
            best_score = recall
            best_name = name
            best_model = model

    print("\n" + "="*60)
    print(f"Best model for screening: {best_name} (Recall = {best_score:.4f})")
    print("="*60)

    return best_model, best_name


def save_model(pipeline, filename="heart_model.pkl"):
    """Save trained pipeline to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(pipeline, path)
    print(f"\n✅ Model saved to {path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HEART DISEASE RISK MODEL TRAINING PIPELINE")
    print("="*60)

    # 1. Load and clean
    print("\n[1/4] Loading and cleaning data...")
    df = clean_data(load_data())

    # 2. Split
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Compare initial models
    print("\n[3/4] Training initial models...")
    best_pipe, best_name, results = train_and_compare(X_train, y_train, X_test, y_test)

    # 4. Hyperparameter tuning on all models
    print("\n[4/4] Hyperparameter tuning...")
    lr_tuned = tune_logistic_regression(X_train, y_train, X_test, y_test)
    rf_tuned = tune_random_forest(X_train, y_train, X_test, y_test)
    gb_tuned = tune_gradient_boosting(X_train, y_train, X_test, y_test)

    # 5. Compare tuned models
    tuned_models = {
        "Logistic Regression (Tuned)": lr_tuned,
        "Random Forest (Tuned)": rf_tuned,
        "Gradient Boosting (Tuned)": gb_tuned,
    }

    best_model, best_model_name = evaluate_tuned_models(tuned_models, X_test, y_test)

    # 6. Save the best model
    save_model(best_model)

    print("\n✅ Training pipeline complete!")
