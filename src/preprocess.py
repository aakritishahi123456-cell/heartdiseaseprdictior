"""
preprocess.py
Defines the preprocessing pipeline using ColumnTransformer.
Ensures leakage-free transformation by fitting only on training data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ---------- Feature definitions ----------

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET = "target"


# ---------- Pipeline builders ----------

def build_preprocessor():
    """
    Build a ColumnTransformer with:
      - Numeric: SimpleImputer(median) → StandardScaler
      - Categorical: SimpleImputer(most_frequent) → OneHotEncoder
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def split_data(df, test_size=0.2, random_state=42):
    """Split dataframe into train/test sets."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_loader import load_data, clean_data

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Transformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test shape:  {X_test_transformed.shape}")
