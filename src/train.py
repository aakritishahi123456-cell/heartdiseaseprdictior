"""
train.py
Trains Logistic Regression, KNN, and Random Forest models inside
a leakage-free Pipeline. Saves the best model to models/heart_model.pkl.
"""

import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data_loader import load_data, clean_data
from preprocess import build_preprocessor, split_data


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
            ("model", RandomForestClassifier(random_state=42))
        ]),
    }


def train_and_compare(X_train, y_train, X_test, y_test):
    """Train all models, print scores, and return the best pipeline."""
    preprocessor = build_preprocessor()
    pipelines = get_pipelines(preprocessor)

    best_name = None
    best_score = 0
    best_pipeline = None
    results = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        results[name] = score
        print(f"  {name}: accuracy = {score:.4f}")

        if score >= best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe

    print(f"\nBest model: {best_name} (accuracy = {best_score:.4f})")
    return best_pipeline, best_name, results


def tune_logistic_regression(X_train, y_train):
    """Hyperparameter tuning for Logistic Regression via GridSearchCV."""
    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_


def save_model(pipeline, filename="heart_model.pkl"):
    """Save trained pipeline to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # 1. Load and clean
    df = clean_data(load_data())

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Compare models
    print("\n--- Model Comparison ---")
    best_pipe, best_name, results = train_and_compare(X_train, y_train, X_test, y_test)

    # 4. Hyperparameter tuning on Logistic Regression
    print("\n--- Hyperparameter Tuning (Logistic Regression) ---")
    tuned_pipe = tune_logistic_regression(X_train, y_train)

    # 5. Save tuned model
    save_model(tuned_pipe)
