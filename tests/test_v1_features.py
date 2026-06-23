import pytest
import pandas as pd
from openlift.core.diagnostics import DataValidator
from openlift.core.economics import calculate_economics, recommend_budget
from openlift.core.evidence import calculate_evidence_strength

def test_diagnostics():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "geo": ["A", "A", "A", "A"],
        "outcome": [10, 0, 15, 20],  # One zero -> moderate sparsity
        "spend": [100, 100, 100, None] # One missing spend
    })
    
    val = DataValidator(df, "date", "geo", "outcome", "spend")
    res = val.evaluate()
    
    assert res["score"] < 100
    assert any("sparsity" in w.lower() for w in res["warnings"])
    assert any("missing values in input col" in w.lower() for w in res["warnings"])
    
def test_diagnostics_perfect():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "geo": ["A", "A"],
        "outcome": [10, 15],
        "spend": [100, 100]
    })
    
    val = DataValidator(df, "date", "geo", "outcome", "spend")
    res = val.evaluate()
    
    assert res["score"] == 100
    assert len(res["warnings"]) == 0

def test_economics():
    # Outcome is conversions
    res = calculate_economics(incremental_outcome=100, input_change_abs=2000, outcome_is_revenue=False, margin_pct=0.8, ltv=50)
    assert res["incremental_cac"] == 2000 / 100
    assert res["incremental_roas"] == (100 * 50) / 2000
    assert res["incremental_profit"] == (100 * 50 * 0.8) - 2000
    
    # Outcome is revenue
    res2 = calculate_economics(incremental_outcome=5000, input_change_abs=2000, outcome_is_revenue=True, margin_pct=0.5)
    assert res2["incremental_cac"] is None
    assert res2["incremental_roas"] == 5000 / 2000
    assert res2["incremental_profit"] == (5000 * 0.5) - 2000

def test_recommend_budget():
    assert "Hold & Retest" in recommend_budget(0.7, 5.0, 1000)
    assert "Scale Aggressively" in recommend_budget(0.95, 3.0, 1000, target_roas=1.0)
    assert "Reduce Spend" in recommend_budget(0.9, 0.5, -500, target_roas=1.0)

def test_evidence_strength():
    assert calculate_evidence_strength(0.6) == "Weak"
    assert calculate_evidence_strength(0.96) == "Very Strong"
    
    # Downgrade due to poor match
    assert calculate_evidence_strength(0.96, match_score=2.0) == "Moderate"
    
    # Downgrade due to wide HDI
    assert calculate_evidence_strength(0.96, relative_hdi_width=2.5) == "Moderate"
