import pandas as pd

from openlift.core.creative import analyze_creative_lift


def test_analyze_creative_lift_uses_did_when_columns_exist():
    df = pd.DataFrame({
        "creative_id": ["c1", "c1", "c1", "c1"],
        "outcome": [10, 20, 8, 9],
        "spend": [100, 100, 50, 50],
        "treatment": [True, True, False, False],
        "period": ["pre", "post", "pre", "post"],
    })
    creative = pd.DataFrame({
        "creative_id": ["c1"],
        "hook_type": ["Problem"],
    })

    result = analyze_creative_lift(df, creative)

    assert result.loc[0, "creative_lift_score"] == 9
    assert result.loc[0, "hook_type"] == "Problem"


def test_analyze_creative_lift_handles_same_performance_file_without_suffixes():
    df = pd.DataFrame({
        "Campaign name": ["A", "A", "B"],
        "Day": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "Amount spent (GBP)": [10, 20, 15],
        "Results": [2, 5, 3],
    })

    result = analyze_creative_lift(
        df,
        df,
        join_col="Campaign name",
        outcome_col="Results",
        spend_col="Amount spent (GBP)",
        group_col="Campaign name",
        treatment_col="__missing_treatment__",
        period_col="__missing_period__",
    )

    assert list(result.columns)[:3] == ["Campaign name", "total_spend", "total_outcome"]
    assert result["total_outcome"].sum() == 10
