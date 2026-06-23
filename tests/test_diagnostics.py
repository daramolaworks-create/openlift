import pandas as pd
from openlift.core.diagnostics import DataValidator


def _make_df(n=50, zeros=0):
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=n)
    outcomes = [0] * zeros + list(range(1, n - zeros + 1))
    return pd.DataFrame({"date": dates, "geo": "A", "outcome": outcomes[:n]})


def test_clean_data_scores_high():
    df = _make_df()
    result = DataValidator(df, "date", "geo", "outcome").evaluate()
    assert result["score"] >= 70
    assert result["status"] in {"Excellent", "Good"}


def test_high_sparsity_lowers_score():
    df = _make_df(n=50, zeros=40)
    result = DataValidator(df, "date", "geo", "outcome").evaluate()
    assert result["score"] < DataValidator(_make_df(), "date", "geo", "outcome").evaluate()["score"]


def test_status_tiers():
    df_clean = _make_df()
    result = DataValidator(df_clean, "date", "geo", "outcome").evaluate()
    # Status should be one of the four valid tiers
    assert result["status"] in {"Excellent", "Good", "Fair", "Poor"}


def test_short_pre_period_triggers_warning():
    df = _make_df(n=100)
    result = DataValidator(
        df, "date", "geo", "outcome",
        pre_start="2024-01-01", pre_end="2024-01-10",
        post_start="2024-01-11", post_end="2024-01-20",
    ).evaluate()
    assert any("Pre-period" in w for w in result["warnings"])
