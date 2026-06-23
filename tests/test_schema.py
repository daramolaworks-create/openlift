import pandas as pd

from openlift.core.schema import validate_growth_schema


def _minimal_df():
    return pd.DataFrame({"date": ["2024-01-01"], "geo": ["A"], "outcome": [1]})


def _full_df():
    return pd.DataFrame({
        "date": ["2024-01-01"],
        "geo": ["A"],
        "outcome": [1],
        "spend": [100],
        "treatment": ["true"],
        "period": ["pre"],
    })


# ---- Non-strict (default): valid means the model can run ----

def test_non_strict_valid_when_core_columns_present():
    result = validate_growth_schema(_minimal_df())
    assert result.valid is True


def test_non_strict_reports_missing_decision_columns():
    result = validate_growth_schema(_minimal_df())
    assert "spend" in result.missing_required
    assert "treatment" in result.missing_required
    assert "period" in result.missing_required


def test_non_strict_invalid_when_core_columns_missing():
    df = pd.DataFrame({"date": ["2024-01-01"], "geo": ["A"]})  # no outcome
    result = validate_growth_schema(df)
    assert result.valid is False


# ---- Strict mode: valid means the full PRD schema is satisfied ----

def test_strict_invalid_when_decision_columns_missing():
    result = validate_growth_schema(_minimal_df(), strict=True)
    assert result.valid is False


def test_strict_valid_when_all_minimum_columns_present():
    result = validate_growth_schema(_full_df(), strict=True)
    assert result.valid is True
    assert result.missing_required == []


# ---- Column mapping ----

def test_column_map_remaps_canonical_names():
    df = pd.DataFrame({"dt": ["2024-01-01"], "market": ["A"], "sales": [1]})
    result = validate_growth_schema(df, column_map={"date": "dt", "geo": "market", "outcome": "sales"})
    assert result.valid is True


# ---- Warnings ----

def test_treatment_non_boolean_triggers_warning():
    df = _full_df().copy()
    df["treatment"] = ["maybe"]
    result = validate_growth_schema(df)
    assert any("non-boolean" in w for w in result.warnings)


def test_period_invalid_value_triggers_warning():
    df = _full_df().copy()
    df["period"] = ["during"]
    result = validate_growth_schema(df)
    assert any("Period column" in w for w in result.warnings)
