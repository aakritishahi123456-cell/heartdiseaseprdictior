from src.risk_interpretation import classify_risk


def test_classify_risk_returns_low_below_high_custom_threshold():
    result = classify_risk(probability=0.8, threshold=0.85)

    assert result["prediction"] == 0
    assert result["risk_label"] == "low"
    assert "Low Risk" in result["headline"]


def test_classify_risk_keeps_richer_positive_category_at_or_above_threshold():
    result = classify_risk(probability=0.92, threshold=0.85)

    assert result["prediction"] == 1
    assert result["risk_label"] == "critical"
    assert "Very High Risk" in result["headline"]
