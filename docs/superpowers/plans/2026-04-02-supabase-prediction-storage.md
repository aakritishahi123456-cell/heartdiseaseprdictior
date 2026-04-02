# Supabase Prediction Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every successful Streamlit prediction to a Supabase `predictions` table without breaking local prediction flow when Supabase is missing or unavailable.

**Architecture:** Keep model inference entirely local in the Streamlit app, then hand a compact prediction payload to a new `src.supabase_client` integration module. The app will treat persistence as best-effort: successful predictions remain visible even if Supabase is not configured or insert operations fail.

**Tech Stack:** Python, Streamlit, pandas, pytest, unittest.mock, Supabase Python client

---

## File Structure

- Create: `src/supabase_client.py`
  Responsibility: Supabase environment lookup, payload construction, and prediction-row inserts.
- Create: `tests/test_supabase_client.py`
  Responsibility: unit tests for configuration checks, payload creation, and insert behavior.
- Modify: `app/streamlit_app.py`
  Responsibility: call the persistence layer after a successful prediction and surface storage success/warning messages.
- Modify: `requirements.txt`
  Responsibility: add the Supabase dependency.
- Modify: `README.md`
  Responsibility: document environment variables, SQL schema, and local run steps.

### Task 1: Build and Test the Supabase Persistence Module

**Files:**
- Create: `src/supabase_client.py`
- Test: `tests/test_supabase_client.py`

- [ ] **Step 1: Write the failing tests for configuration and payload creation**

```python
import pandas as pd

from src.supabase_client import (
    build_prediction_record,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_supabase_client.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `src.supabase_client`.

- [ ] **Step 3: Write the minimal persistence module**

```python
import os


def get_supabase_settings():
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "key": os.getenv("SUPABASE_KEY", "").strip(),
    }


def is_supabase_configured():
    settings = get_supabase_settings()
    return bool(settings["url"] and settings["key"])


def build_prediction_record(input_frame, threshold, probability, prediction, risk_label):
    patient_record = input_frame.iloc[0].to_dict()
    patient_record["threshold"] = float(threshold)
    patient_record["probability"] = float(probability)
    patient_record["prediction"] = int(prediction)
    patient_record["risk_label"] = str(risk_label)
    return patient_record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_supabase_client.py -v`
Expected: PASS for the three tests added in Step 1.

- [ ] **Step 5: Commit**

```bash
git add src/supabase_client.py tests/test_supabase_client.py
git commit -m "feat: add supabase prediction payload helpers"
```

### Task 2: Add Insert Logic With Safe Failure Behavior

**Files:**
- Modify: `src/supabase_client.py`
- Modify: `tests/test_supabase_client.py`

- [ ] **Step 1: Write the failing tests for insert behavior**

```python
from unittest.mock import Mock

import pandas as pd
import pytest

from src.supabase_client import save_prediction_record


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
    insert.assert_called_once()
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_supabase_client.py -v`
Expected: FAIL with `ImportError` or assertion failures because `save_prediction_record` does not exist yet.

- [ ] **Step 3: Extend the module with client creation and save behavior**

```python
from supabase import create_client


def create_supabase_client():
    settings = get_supabase_settings()
    if not settings["url"] or not settings["key"]:
        return None
    return create_client(settings["url"], settings["key"])


def save_prediction_record(input_frame, threshold, probability, prediction, risk_label, client=None):
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

    active_client = client or create_supabase_client()

    try:
        active_client.table("predictions").insert(record).execute()
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_supabase_client.py -v`
Expected: PASS for all six tests in `tests/test_supabase_client.py`.

- [ ] **Step 5: Commit**

```bash
git add src/supabase_client.py tests/test_supabase_client.py requirements.txt
git commit -m "feat: add safe supabase prediction persistence"
```

### Task 3: Integrate Persistence Into the Streamlit App

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_supabase_client.py`

- [ ] **Step 1: Write the failing unit test for risk interpretation used by persistence**

```python
from src.supabase_client import build_prediction_record


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
```

- [ ] **Step 2: Run the targeted test to verify the current baseline**

Run: `pytest tests/test_supabase_client.py::test_build_prediction_record_preserves_low_risk_label -v`
Expected: PASS once Task 2 is complete. If it fails, fix the helper before editing the app.

- [ ] **Step 3: Update the Streamlit app to save predictions after successful inference**

```python
from src.supabase_client import is_supabase_configured, save_prediction_record


def validate_patient_input(patient_dict):
    is_valid, errors = PatientDataValidator.validate_single_patient(patient_dict)
    return is_valid, errors


def persist_prediction(input_frame, threshold, probability, prediction, risk_level):
    return save_prediction_record(
        input_frame=input_frame,
        threshold=threshold,
        probability=probability,
        prediction=prediction,
        risk_label=risk_level,
    )


if st.button("Calculate Risk Score", use_container_width=True, type="primary"):
    is_valid, errors = validate_patient_input(input_data.iloc[0].to_dict())

    if not is_valid:
        ...
    else:
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability >= custom_threshold)
        interpretation, risk_level = get_risk_interpretation(probability, custom_threshold)

        storage_result = persist_prediction(
            input_frame=input_data,
            threshold=custom_threshold,
            probability=probability,
            prediction=prediction,
            risk_level=risk_level,
        )

        if storage_result["status"] == "saved":
            st.caption("Prediction saved to Supabase.")
        elif storage_result["status"] == "error":
            st.warning(storage_result["message"])
        elif is_supabase_configured():
            st.info(storage_result["message"])
```

Implementation notes for this step:
- Replace `input_data[0].to_dict()` with `input_data.iloc[0].to_dict()` to avoid a broken column lookup.
- Compute `interpretation, risk_level` once and reuse them for UI and persistence.
- Keep the existing prediction display flow intact.
- Do not call `st.stop()` or raise if persistence fails.

- [ ] **Step 4: Run the app-adjacent tests to verify nothing regressed**

Run: `pytest tests/test_supabase_client.py tests/test_predict.py -v`
Expected: PASS for Supabase tests and existing prediction tests.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/test_supabase_client.py src/supabase_client.py
git commit -m "feat: save streamlit predictions to supabase"
```

### Task 4: Add Dependency and Setup Documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

- [ ] **Step 1: Add the failing documentation/setup checklist to the README draft**

```markdown
## Supabase Setup

1. Create a Supabase project.
2. Create a `predictions` table with the SQL below.
3. Set `SUPABASE_URL` and `SUPABASE_KEY`.
4. Run the Streamlit app and submit a prediction.
```

This step is expected to be incomplete until the exact SQL and run instructions are added in Step 3.

- [ ] **Step 2: Add the Supabase package dependency**

```text
supabase==2.15.3
```

Append the dependency to `requirements.txt` without removing existing packages.

- [ ] **Step 3: Expand the README with exact setup instructions**

```sql
create extension if not exists pgcrypto;

create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default timezone('utc', now()),
    age integer not null,
    sex integer not null,
    cp integer not null,
    trestbps integer not null,
    chol integer not null,
    fbs integer not null,
    restecg integer not null,
    thalach integer not null,
    exang integer not null,
    oldpeak double precision not null,
    slope integer not null,
    ca integer not null,
    thal integer not null,
    threshold double precision not null,
    probability double precision not null,
    prediction integer not null,
    risk_label text not null
);
```

```powershell
$env:SUPABASE_URL="https://your-project-ref.supabase.co"
$env:SUPABASE_KEY="your-supabase-key"
streamlit run app/streamlit_app.py
```

Add one short note explaining that the app still works locally if these variables are not set, but records will not be saved.

- [ ] **Step 4: Run focused verification**

Run: `pytest tests/test_supabase_client.py -v`
Expected: PASS

Run: `python -m compileall src app`
Expected: success output listing compiled files with no syntax errors

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: add supabase setup instructions"
```

### Task 5: Final Verification

**Files:**
- Modify: none
- Verify: `app/streamlit_app.py`, `src/supabase_client.py`, `tests/test_supabase_client.py`, `README.md`, `requirements.txt`

- [ ] **Step 1: Run the full targeted test suite**

Run: `pytest tests/test_supabase_client.py tests/test_predict.py tests/test_data_loader.py tests/test_preprocessing.py -v`
Expected: PASS

- [ ] **Step 2: Run a local app smoke test**

Run: `streamlit run app/streamlit_app.py`
Expected:
- the app loads without import errors
- a successful prediction still renders the result cards
- if Supabase env vars are missing, the app shows the prediction and does not crash

- [ ] **Step 3: Verify README instructions match implementation**

Checklist:
- `SUPABASE_URL` and `SUPABASE_KEY` are the exact variable names used in code
- table name is `predictions`
- saved columns match `build_prediction_record`
- the fallback behavior described in docs matches the app

- [ ] **Step 4: Review git diff before handoff**

Run: `git diff -- app/streamlit_app.py src/supabase_client.py tests/test_supabase_client.py README.md requirements.txt`
Expected: only the planned Supabase integration changes appear

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py src/supabase_client.py tests/test_supabase_client.py README.md requirements.txt
git commit -m "feat: connect streamlit predictions to supabase"
```

## Self-Review

### Spec Coverage

- Supabase client configuration via environment variables: covered in Task 1 and Task 4.
- `predictions` table design: covered in Task 4.
- Insert logic after successful predictions: covered in Task 2 and Task 3.
- Graceful failure when Supabase is unavailable: covered in Task 2, Task 3, and Task 5.
- Local setup documentation: covered in Task 4.

### Placeholder Scan

No `TODO`, `TBD`, or “implement later” placeholders remain. Each code-writing step includes explicit code and each verification step includes exact commands and expected results.

### Type Consistency

The plan consistently uses:
- `build_prediction_record(input_frame, threshold, probability, prediction, risk_label)`
- `save_prediction_record(input_frame, threshold, probability, prediction, risk_label, client=None)`
- environment variables `SUPABASE_URL` and `SUPABASE_KEY`
- table name `predictions`
