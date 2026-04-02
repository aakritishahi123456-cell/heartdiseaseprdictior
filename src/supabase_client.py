"""Helpers for reading Supabase settings and building prediction payloads."""

from __future__ import annotations

import os

try:
    from supabase import create_client as _create_supabase_client
except ImportError:  # pragma: no cover - dependency may be absent in some environments
    _create_supabase_client = None


def get_supabase_settings():
    """Read Supabase connection settings from the environment."""
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "key": os.getenv("SUPABASE_KEY", "").strip(),
    }


def is_supabase_configured():
    """Return True when both Supabase settings are present."""
    settings = get_supabase_settings()
    return bool(settings["url"] and settings["key"])


def build_prediction_record(input_frame, threshold, probability, prediction, risk_label):
    """Build a single-row prediction payload from an input DataFrame."""
    if len(input_frame) != 1:
        raise ValueError("input_frame must contain exactly one row")

    record = input_frame.to_dict(orient="records")[0]
    record["threshold"] = float(threshold)
    record["probability"] = float(probability)
    record["prediction"] = int(prediction)
    record["risk_label"] = str(risk_label)
    return record


def create_supabase_client():
    """Create a Supabase client when configuration and dependency are available."""
    settings = get_supabase_settings()
    if not settings["url"] or not settings["key"]:
        return None

    if _create_supabase_client is None:
        return None

    return _create_supabase_client(settings["url"], settings["key"])


def save_prediction_record(
    input_frame,
    threshold,
    probability,
    prediction,
    risk_label,
    client=None,
):
    """Persist a prediction record to Supabase when storage is configured."""
    if not is_supabase_configured():
        return {
            "saved": False,
            "status": "disabled",
            "message": "Supabase is not configured.",
        }

    record = build_prediction_record(
        input_frame=input_frame,
        threshold=threshold,
        probability=probability,
        prediction=prediction,
        risk_label=risk_label,
    )

    try:
        active_client = client or create_supabase_client()
        if active_client is None:
            raise RuntimeError("Supabase client is unavailable.")

        active_client.table("predictions").insert(record, returning="minimal").execute()
        return {
            "saved": True,
            "status": "saved",
            "message": "Prediction saved to Supabase.",
        }
    except Exception as exc:
        return {
            "saved": False,
            "status": "error",
            "message": f"Supabase save failed: {exc}",
        }
