"""Master validation script for the heart disease ML project."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data, clean_data
from src.schema_validator import DataSchemaValidator
from src.validators import PatientDataValidator


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def validate_data():
    """Validate raw data."""
    print_header("1. DATA LOADING & CLEANING")

    df = load_data()
    print(f"✅ Data loaded successfully")
    print(f"   Original size: {df.shape}")

    df_clean = clean_data(df)
    print(f"✅ Data cleaned successfully")
    print(f"   Final size: {df_clean.shape}")

    return df_clean


def validate_schema(df):
    """Validate data schema."""
    print_header("2. DATA SCHEMA VALIDATION")

    is_valid = DataSchemaValidator.print_validation_report(df)
    return is_valid


def validate_patient_input():
    """Validate patient input validation."""
    print_header("3. PATIENT INPUT VALIDATION")

    test_patient = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    is_valid, errors = PatientDataValidator.validate_single_patient(test_patient)

    if is_valid:
        print("✅ Patient input validation passed")
    else:
        print("❌ Patient input validation failed:")
        for error in errors:
            print(f"   - {error}")

    return is_valid


def run_unit_tests():
    """Run unit tests with pytest."""
    print_header("4. UNIT TESTS (pytest)")

    try:
        import pytest
        retcode = pytest.main([
            "tests/",
            "-v",
            "--tb=short",
            "--color=yes",
            "-x"  # Stop on first failure
        ])
        if retcode == 0:
            print("\n✅ All unit tests passed")
        else:
            print(f"\n❌ Some tests failed (exit code: {retcode})")
        return retcode == 0
    except ImportError:
        print("⚠️  pytest not installed. Run: pip install pytest pytest-cov")
        return False


def validate_model():
    """Validate trained model."""
    print_header("5. MODEL VALIDATION")

    try:
        import joblib
        from src.preprocess import split_data

        model_path = Path(__file__).resolve().parent / "models" / "heart_model.pkl"

        if not model_path.exists():
            print(f"⚠️  Model not found at {model_path}")
            print("   Run: python src/train.py")
            return False

        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")

        # Test prediction
        df = clean_data(load_data())
        X_train, X_test, y_train, y_test = split_data(df)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        print(f"✅ Predictions generated")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Prediction range: [{y_prob[:, 1].min():.4f}, {y_prob[:, 1].max():.4f}]")

        return True

    except FileNotFoundError:
        print(f"⚠️  Model file not found")
        return False
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "█"*70)
    print("  HEART DISEASE ML PROJECT - COMPREHENSIVE VALIDATION")
    print("█"*70)

    results = {}

    # 1. Data validation
    try:
        df = validate_data()
        results["Data Loading"] = True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        results["Data Loading"] = False
        return results

    # 2. Schema validation
    try:
        results["Schema Validation"] = validate_schema(df)
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        results["Schema Validation"] = False

    # 3. Input validation
    try:
        results["Input Validation"] = validate_patient_input()
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
        results["Input Validation"] = False

    # 4. Unit tests
    try:
        results["Unit Tests"] = run_unit_tests()
    except Exception as e:
        print(f"⚠️  Unit tests error: {e}")
        results["Unit Tests"] = False

    # 5. Model validation
    try:
        results["Model Validation"] = validate_model()
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        results["Model Validation"] = False

    # Summary
    print_header("VALIDATION SUMMARY")
    all_passed = True
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check:.<40} {status}")
        if not result:
            all_passed = False

    print("\n" + "█"*70)
    if all_passed:
        print("  ✅ ALL VALIDATIONS PASSED - PROJECT IS HEALTHY")
    else:
        print("  ❌ SOME VALIDATIONS FAILED - CHECK ISSUES ABOVE")
    print("█"*70 + "\n")

    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if all(results.values()) else 1)
