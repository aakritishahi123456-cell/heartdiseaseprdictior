"""Input validation module for patient data."""

import pandas as pd
from typing import Dict, List, Tuple


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PatientDataValidator:
    """Validates patient clinical data before model prediction."""

    # Valid ranges for clinical features
    VALID_RANGES = {
        "age": (20, 100),
        "trestbps": (60, 250),  # Resting blood pressure
        "chol": (50, 600),      # Cholesterol
        "thalach": (50, 220),   # Max heart rate achieved
        "oldpeak": (0, 10),     # ST depression
    }

    VALID_VALUES = {
        "sex": [0, 1],
        "cp": [0, 1, 2, 3],     # Chest pain type
        "fbs": [0, 1],          # Fasting blood sugar > 120
        "restecg": [0, 1, 2],   # Resting ECG result
        "exang": [0, 1],        # Exercise induced angina
        "slope": [0, 1, 2],     # ST slope
        "ca": [0, 1, 2, 3, 4],  # Major vessels colored by fluoroscopy
        "thal": [0, 1, 2, 3],   # Thalassemia type
    }

    REQUIRED_FEATURES = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    @classmethod
    def validate_single_patient(cls, patient_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single patient's data.

        Parameters:
            patient_dict: Dictionary with feature names as keys

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check for required features
        missing = set(cls.REQUIRED_FEATURES) - set(patient_dict.keys())
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check numeric ranges
        for feature, (min_val, max_val) in cls.VALID_RANGES.items():
            if feature in patient_dict:
                val = patient_dict[feature]
                try:
                    val = float(val)
                    if val < min_val or val > max_val:
                        errors.append(
                            f"{feature} = {val} out of range [{min_val}, {max_val}]"
                        )
                except (TypeError, ValueError):
                    errors.append(f"{feature} is not a valid number: {val}")

        # Check categorical values
        for feature, valid_vals in cls.VALID_VALUES.items():
            if feature in patient_dict:
                val = patient_dict[feature]
                try:
                    val = int(val)
                    if val not in valid_vals:
                        errors.append(
                            f"{feature} = {val} not in {valid_vals}"
                        )
                except (TypeError, ValueError):
                    errors.append(f"{feature} is not a valid integer: {val}")

        return len(errors) == 0, errors

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame of patient data.

        Parameters:
            df: DataFrame with patient records

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check for required columns
        missing_cols = set(cls.REQUIRED_FEATURES) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            return False, errors

        # Check each feature
        for feature in cls.REQUIRED_FEATURES:
            col = df[feature]

            # Check for nulls
            if col.isnull().any():
                errors.append(f"{feature}: {col.isnull().sum()} null values")

            # Check numeric ranges
            if feature in cls.VALID_RANGES:
                min_val, max_val = cls.VALID_RANGES[feature]
                out_of_range = (col < min_val) | (col > max_val)
                if out_of_range.any():
                    errors.append(
                        f"{feature}: {out_of_range.sum()} values out of range "
                        f"[{min_val}, {max_val}]"
                    )

            # Check categorical values
            if feature in cls.VALID_VALUES:
                valid_vals = cls.VALID_VALUES[feature]
                invalid = ~col.isin(valid_vals)
                if invalid.any():
                    errors.append(
                        f"{feature}: {invalid.sum()} invalid values "
                        f"(expected {valid_vals})"
                    )

        return len(errors) == 0, errors

    @classmethod
    def sanitize_patient_dict(cls, patient_dict: Dict) -> Dict:
        """
        Sanitize and convert patient dictionary to proper types.

        Parameters:
            patient_dict: Raw patient data

        Returns:
            Sanitized dictionary
        """
        sanitized = {}

        for feature in cls.REQUIRED_FEATURES:
            if feature not in patient_dict:
                raise ValidationError(f"Missing required feature: {feature}")

            val = patient_dict[feature]

            # Convert numeric features
            if feature in cls.VALID_RANGES:
                try:
                    sanitized[feature] = float(val)
                except (TypeError, ValueError):
                    raise ValidationError(f"Invalid value for {feature}: {val}")

            # Convert categorical features
            elif feature in cls.VALID_VALUES:
                try:
                    sanitized[feature] = int(val)
                except (TypeError, ValueError):
                    raise ValidationError(f"Invalid value for {feature}: {val}")

        return sanitized


if __name__ == "__main__":
    # Test validation
    patient = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    is_valid, errors = PatientDataValidator.validate_single_patient(patient)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
