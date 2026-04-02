# Supabase Prediction Storage Design

## Goal

Connect the existing Streamlit heart disease risk app to Supabase so that every successful prediction stores the full entered patient data and the generated prediction result in a Supabase Postgres table.

## Scope

This phase adds backend persistence for prediction events only.

Included:
- Supabase client configuration via environment variables
- A `predictions` table in Supabase
- Application-side insert logic after successful predictions
- Graceful failure behavior when Supabase is unavailable
- Setup documentation for local configuration

Not included:
- User authentication
- Row-level security customization beyond basic safe defaults
- Loading training data from Supabase
- Model retraining from stored records
- Batch analytics dashboards

## Context

The current project is a local-first Streamlit application that:
- loads a serialized model from `models/heart_model.pkl`
- collects 13 clinical input fields in `app/streamlit_app.py`
- validates those fields locally
- computes a local prediction and risk interpretation

There is no backend integration layer yet, so the least disruptive design is to keep inference local and add a persistence step after prediction succeeds.

## Approaches Considered

### 1. Store prediction events in Supabase after local inference

This keeps the model workflow unchanged and adds durable storage for all entered inputs and outputs.

Pros:
- Smallest change to the existing codebase
- Low risk to current prediction behavior
- Immediate value for auditing and later analysis
- Easy to extend later with auth or dashboards

Cons:
- Training data remains separate from Supabase for now

Recommendation: use this approach.

### 2. Move the source dataset into Supabase first

Pros:
- Centralizes data management
- Could support future admin workflows

Cons:
- More migration work
- Does not improve the Streamlit user flow directly
- Adds complexity before persistence is proven useful

### 3. Move both app event storage and training data at once

Pros:
- One broader migration step

Cons:
- Higher setup friction
- More moving parts to debug
- Harder to verify incrementally

## Proposed Design

### Architecture

The Streamlit app will continue to:
1. collect patient inputs
2. validate them locally
3. run the local model

After a successful prediction, the app will attempt to insert one record into Supabase.

Flow:

`Streamlit form -> local validation -> local model inference -> Supabase insert -> user feedback`

If the insert fails, prediction still succeeds and the user sees a warning that the prediction was not stored.

### Data Model

Create a single `predictions` table with one row per successful prediction.

Suggested columns:
- `id` UUID primary key with generated default
- `created_at` timestamp with time zone default now
- `age` integer
- `sex` integer
- `cp` integer
- `trestbps` integer
- `chol` integer
- `fbs` integer
- `restecg` integer
- `thalach` integer
- `exang` integer
- `oldpeak` numeric
- `slope` integer
- `ca` integer
- `thal` integer
- `threshold` numeric
- `probability` numeric
- `prediction` integer
- `risk_label` text

This stores the full encoded model input and the downstream result. For this project, storing encoded values is acceptable because the app already converts human-friendly selections into model-ready fields before prediction.

### Application Changes

Add a small Supabase integration module under `src/` with responsibilities:
- create a client from `SUPABASE_URL` and `SUPABASE_KEY`
- report whether Supabase is configured
- insert a prediction row into `predictions`

Update the Streamlit app to:
- load Supabase configuration lazily
- save the prediction row only after validation and inference succeed
- keep the prediction UX non-blocking if Supabase insert fails
- show a small success or warning message for storage state

### Configuration

Use environment variables:
- `SUPABASE_URL`
- `SUPABASE_KEY`

For Streamlit, these can be provided either through the shell environment or a local secrets/config approach documented in the README.

The app should treat missing configuration as "storage disabled" rather than a fatal startup error.

### Error Handling

Expected cases:
- Missing environment variables
- Network/API failure
- Table missing or schema mismatch

Behavior:
- never block prediction generation because of storage issues
- catch and surface storage errors as warnings
- keep the model prediction result visible to the user

### Security and Privacy

This phase stores all entered patient data because that was the selected product direction.

Constraints:
- No authentication means anyone who can run the app can trigger inserts
- The key used in the app should be the least-privileged key that still supports inserts for the intended deployment model
- For local development, this is acceptable for a portfolio project, but production hardening would require auth, RLS review, and clearer consent handling

### Testing

Add focused tests around the new integration layer:
- payload construction from prediction results
- behavior when Supabase config is missing
- behavior when insert raises an exception

Avoid tests that depend on live Supabase connectivity. Mock the client instead.

## Implementation Notes

Likely file changes:
- `app/streamlit_app.py`
- new `src/supabase_client.py` or similar
- `requirements.txt`
- `README.md`
- optional new tests for persistence logic

Likely dependency:
- `supabase`

## Success Criteria

The feature is complete when:
- the app still performs predictions locally
- each successful prediction attempts to save one row to Supabase
- missing or broken Supabase configuration does not break predictions
- setup steps are documented clearly
- automated tests cover the new persistence behavior at a unit level

## Out of Scope Follow-Ups

Future expansions can include:
- auth and user ownership of predictions
- dashboards over stored records
- exporting records for retraining
- storing human-readable labels alongside encoded values
- moving source datasets into Supabase
