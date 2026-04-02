"""
Data schema validation module for enforcing data quality constraints.
Validates raw data against expected schema before processing.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    dtype: str  # 'int', 'float', 'object', 'bool'
    nullable: bool = False
    min_value: float = None
    max_value: float = None
    allowed_values: List[Any] = None


class DataSchemaValidator:
    """Validates dataframes against a predefined schema."""

    # Define the expected schema for heart disease dataset
    HEART_DISEASE_SCHEMA = [
        ColumnSchema("age", "int", nullable=False, min_value=20, max_value=100),
        ColumnSchema("sex", "int", nullable=False, allowed_values=[0, 1]),
        ColumnSchema("cp", "int", nullable=False, allowed_values=[0, 1, 2, 3]),
        ColumnSchema("trestbps", "int", nullable=False, min_value=60, max_value=250),
        ColumnSchema("chol", "int", nullable=False, min_value=50, max_value=600),
        ColumnSchema("fbs", "int", nullable=False, allowed_values=[0, 1]),
        ColumnSchema("restecg", "int", nullable=False, allowed_values=[0, 1, 2]),
        ColumnSchema("thalach", "int", nullable=False, min_value=50, max_value=220),
        ColumnSchema("exang", "int", nullable=False, allowed_values=[0, 1]),
        ColumnSchema("oldpeak", "float", nullable=False, min_value=0, max_value=10),
        ColumnSchema("slope", "int", nullable=False, allowed_values=[0, 1, 2]),
        ColumnSchema("ca", "int", nullable=False, allowed_values=[0, 1, 2, 3, 4]),
        ColumnSchema("thal", "int", nullable=False, allowed_values=[0, 1, 2, 3]),
        ColumnSchema("target", "int", nullable=False, allowed_values=[0, 1]),
    ]

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        schema: List[ColumnSchema] = None,
        raise_errors: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate a dataframe against a schema.

        Parameters:
            df: DataFrame to validate
            schema: List of ColumnSchema objects (uses default if None)
            raise_errors: Raise exception if validation fails

        Returns:
            (is_valid, errors_list)
        """
        if schema is None:
            schema = cls.HEART_DISEASE_SCHEMA

        errors = []

        # Check for required columns
        expected_cols = {col.name for col in schema}
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Extra columns: {extra_cols}")

        # Validate each column
        for col_schema in schema:
            if col_schema.name not in df.columns:
                continue

            col = df[col_schema.name]

            # Check nullability
            null_count = col.isnull().sum()
            if null_count > 0 and not col_schema.nullable:
                errors.append(f"'{col_schema.name}': {null_count} null values (not nullable)")

            # Check dtype
            if col_schema.dtype == "int" and not pd.api.types.is_integer_dtype(col):
                errors.append(f"'{col_schema.name}': expected int, got {col.dtype}")
            elif col_schema.dtype == "float" and not pd.api.types.is_numeric_dtype(col):
                errors.append(f"'{col_schema.name}': expected float, got {col.dtype}")

            # Check range
            if col_schema.min_value is not None and col_schema.max_value is not None:
                out_of_range = ((col < col_schema.min_value) | (col > col_schema.max_value)).sum()
                if out_of_range > 0:
                    errors.append(
                        f"'{col_schema.name}': {out_of_range} values out of range "
                        f"[{col_schema.min_value}, {col_schema.max_value}]"
                    )

            # Check allowed values
            if col_schema.allowed_values is not None:
                invalid_count = (~col.isin(col_schema.allowed_values)).sum()
                if invalid_count > 0:
                    errors.append(
                        f"'{col_schema.name}': {invalid_count} invalid values "
                        f"(expected {col_schema.allowed_values})"
                    )

        if errors and raise_errors:
            raise ValueError("\n".join(errors))

        return len(errors) == 0, errors

    @classmethod
    def print_validation_report(cls, df: pd.DataFrame, schema: List[ColumnSchema] = None) -> bool:
        """Print detailed validation report."""
        is_valid, errors = cls.validate(df, schema, raise_errors=False)

        print("\n" + "="*60)
        print("DATA SCHEMA VALIDATION REPORT")
        print("="*60)

        if is_valid:
            print("✅ PASS: Dataset matches schema")
        else:
            print("❌ FAIL: Schema validation errors detected")
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")

        print("="*60 + "\n")

        return is_valid


if __name__ == "__main__":
    from src.data_loader import load_data, clean_data

    # Test schema validation
    df = clean_data(load_data())
    is_valid = DataSchemaValidator.print_validation_report(df)

    if not is_valid:
        print("⚠️  Fix schema issues before proceeding!")
