"""Threshold-aware risk interpretation helpers."""


def classify_risk(probability, threshold):
    """Return a threshold-consistent risk classification and UI headline."""
    prediction = int(probability >= threshold)

    if prediction == 0:
        return {
            "prediction": 0,
            "risk_label": "low",
            "headline": "🟢 **Low Risk** - Below risk threshold",
        }

    if probability >= 0.9:
        return {
            "prediction": 1,
            "risk_label": "critical",
            "headline": "🔴 **Very High Risk** - Strong positive signal",
        }

    if probability >= 0.75:
        return {
            "prediction": 1,
            "risk_label": "high",
            "headline": "🟠 **High Risk** - Strong positive signal",
        }

    return {
        "prediction": 1,
        "risk_label": "moderate",
        "headline": "🟡 **Moderate Risk** - Positive signal",
    }
