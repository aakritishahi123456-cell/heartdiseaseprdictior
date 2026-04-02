import pandas as pd
import pytest
from unittest.mock import Mock

import src.supabase_client as supabase_client
from src.supabase_client import (
    build_prediction_record,
    save_prediction_record,
    get_supabase_settings,
    is_supabase_configured,
)


def test_get_supabase_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")

    settings = get_supabase_settings()

    assert settings == {
        "url": "https://example.supabase.co",
        "key": "test-key",
    }


def test_is_supabase_configured_false_when_key_missing(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    assert is_supabase_configured() is False


def test_build_prediction_record_includes_inputs_and_result():
    input_frame = pd.DataFrame(
        [{
            "age": 54,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }]
    )

    record = build_prediction_record(
        input_frame=input_frame,
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
    )

    assert record["age"] == 54
    assert record["chol"] == 240
    assert record["threshold"] == 0.4
    assert record["probability"] == 0.82
    assert record["prediction"] == 1
    assert record["risk_label"] == "high"


def test_build_prediction_record_preserves_integer_types():
    input_frame = pd.DataFrame(
        [{
            "age": 54,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }]
    )

    record = build_prediction_record(
        input_frame=input_frame,
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
    )

    assert isinstance(record["age"], int)
    assert isinstance(record["sex"], int)
    assert isinstance(record["chol"], int)
    assert isinstance(record["prediction"], int)


def test_build_prediction_record_preserves_low_risk_label():
    input_frame = pd.DataFrame(
        [{
            "age": 40,
            "sex": 0,
            "cp": 0,
            "trestbps": 120,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }]
    )

    record = build_prediction_record(
        input_frame=input_frame,
        threshold=0.4,
        probability=0.2,
        prediction=0,
        risk_label="low",
    )

    assert record["risk_label"] == "low"
    assert record["prediction"] == 0


def test_build_prediction_record_rejects_empty_frame():
    input_frame = pd.DataFrame(
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    )

    with pytest.raises(ValueError, match="exactly one row"):
        build_prediction_record(
            input_frame=input_frame,
            threshold=0.4,
            probability=0.82,
            prediction=1,
            risk_label="high",
        )


def test_build_prediction_record_rejects_multi_row_frame():
    input_frame = pd.DataFrame(
        [
            {
                "age": 54,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 240,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 2,
            },
            {
                "age": 60,
                "sex": 0,
                "cp": 3,
                "trestbps": 140,
                "chol": 250,
                "fbs": 1,
                "restecg": 0,
                "thalach": 140,
                "exang": 1,
                "oldpeak": 2.0,
                "slope": 2,
                "ca": 1,
                "thal": 3,
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one row"):
        build_prediction_record(
            input_frame=input_frame,
            threshold=0.4,
            probability=0.82,
            prediction=1,
            risk_label="high",
        )


def test_save_prediction_record_returns_disabled_when_not_configured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    result = save_prediction_record(
        input_frame=pd.DataFrame([{"age": 54}]),
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
    )

    assert result == {
        "saved": False,
        "status": "disabled",
        "message": "Supabase is not configured.",
    }


def test_save_prediction_record_inserts_payload(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")

    execute = Mock(return_value={"data": [{"id": "123"}]})
    insert = Mock(return_value=Mock(execute=execute))
    table = Mock(return_value=Mock(insert=insert))
    client = Mock(table=table)

    result = save_prediction_record(
        input_frame=pd.DataFrame([{"age": 54, "sex": 1}]),
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
        client=client,
    )

    table.assert_called_once_with("predictions")
    insert.assert_called_once_with(
        {
            "age": 54,
            "sex": 1,
            "threshold": 0.4,
            "probability": 0.82,
            "prediction": 1,
            "risk_label": "high",
        },
        returning="minimal",
    )
    assert result["saved"] is True
    assert result["status"] == "saved"


def test_save_prediction_record_returns_error_when_insert_fails(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")

    execute = Mock(side_effect=RuntimeError("insert failed"))
    insert = Mock(return_value=Mock(execute=execute))
    table = Mock(return_value=Mock(insert=insert))
    client = Mock(table=table)

    result = save_prediction_record(
        input_frame=pd.DataFrame([{"age": 54, "sex": 1}]),
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
        client=client,
    )

    assert result["saved"] is False
    assert result["status"] == "error"
    assert "insert failed" in result["message"]


def test_save_prediction_record_returns_error_when_supabase_dependency_is_missing(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setattr(supabase_client, "_create_supabase_client", None)

    result = save_prediction_record(
        input_frame=pd.DataFrame([{"age": 54, "sex": 1}]),
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
    )

    assert result["saved"] is False
    assert result["status"] == "error"
    assert "Supabase client is unavailable" in result["message"]


def test_save_prediction_record_returns_error_when_client_creation_fails(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")

    def create_client(*args, **kwargs):
        raise RuntimeError("client boom")

    monkeypatch.setattr(supabase_client, "_create_supabase_client", create_client)

    result = save_prediction_record(
        input_frame=pd.DataFrame([{"age": 54, "sex": 1}]),
        threshold=0.4,
        probability=0.82,
        prediction=1,
        risk_label="high",
    )

    assert result["saved"] is False
    assert result["status"] == "error"
    assert "client boom" in result["message"]
