from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class SchemaCheck:
    valid: bool
    missing_required: List[str]
    missing_recommended: List[str]
    warnings: List[str]
    column_map: Dict[str, str]


# The model engine only needs these three to run.
CORE_COLUMNS = ["date", "geo", "outcome"]

# Full PRD schema — required for the decision / economics layer.
MINIMUM_COLUMNS = ["date", "geo", "outcome", "spend", "treatment", "period"]
RECOMMENDED_COLUMNS = ["channel", "campaign"]
OPTIONAL_COLUMNS = ["creative_id", "creative_name", "product", "category"]


def validate_growth_schema(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> SchemaCheck:
    """
    Validate uploaded data against the PRD growth schema.

    Non-strict (default): ``valid`` is True when the model can run, i.e. the
    three core columns (date, geo, outcome) are present.  Missing decision
    columns (spend, treatment, period) are reported in ``missing_required`` but
    do not invalidate the check.

    Strict: ``valid`` is True only when every MINIMUM_COLUMNS entry is present.
    Use this for workflows that require the full decision schema.
    """
    column_map = column_map or {}
    resolved = {canonical: column_map.get(canonical, canonical) for canonical in MINIMUM_COLUMNS}

    missing_required = [
        canonical
        for canonical, actual in resolved.items()
        if actual not in df.columns
    ]

    missing_core = [c for c in CORE_COLUMNS if column_map.get(c, c) not in df.columns]

    missing_recommended = [
        col for col in RECOMMENDED_COLUMNS if column_map.get(col, col) not in df.columns
    ]

    warnings = []
    if missing_recommended:
        warnings.append(
            "Recommended decision columns are missing: "
            + ", ".join(missing_recommended)
            + "."
        )

    if "treatment" in resolved and resolved["treatment"] in df.columns:
        unique = set(df[resolved["treatment"]].dropna().astype(str).str.lower())
        valid_values = {"true", "false", "1", "0", "yes", "no"}
        if unique and not unique <= valid_values:
            warnings.append("Treatment column contains non-boolean-looking values.")

    if "period" in resolved and resolved["period"] in df.columns:
        valid_periods = {"pre", "treatment", "post"}
        observed = set(df[resolved["period"]].dropna().astype(str).str.lower())
        if observed and not observed <= valid_periods:
            warnings.append("Period column should use pre, treatment, or post.")

    # Non-strict: can the model run? Strict: is the full schema satisfied?
    valid = (len(missing_core) == 0) if not strict else (len(missing_required) == 0)

    return SchemaCheck(
        valid=valid,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        warnings=warnings,
        column_map=resolved,
    )
