from typing import Any, Dict, Optional

import pandas as pd

from .design import GeoMatcher, PowerAnalysis


def recommend_next_experiment(
    df: pd.DataFrame,
    date_col: str,
    geo_col: str,
    outcome_col: str,
    input_col: Optional[str] = None,
    expected_lift: float = 0.10,
    target_power: float = 0.8,
    max_duration: int = 60,
    lookback_days: int = 60,
) -> Dict[str, Any]:
    """
    Recommend a concrete next geo experiment from available historical data.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    summary = (
        work.groupby(geo_col)[outcome_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_outcome", "std": "volatility", "count": "rows"})
    )
    summary["stability"] = summary["avg_outcome"] / summary["volatility"].replace(0, 1)
    summary = summary.sort_values(["stability", "avg_outcome"], ascending=False)

    if summary.empty:
        return {"error": "No geos available for experiment recommendation."}

    matcher = GeoMatcher(work, date_col, geo_col, outcome_col)
    pa = PowerAnalysis(work, date_col, geo_col, outcome_col)

    for _, row in summary.iterrows():
        test_geo = row[geo_col]
        controls = matcher.find_controls(test_geo, lookback_days=lookback_days, n_controls=5)
        control_geos = [geo for geo, _ in controls]
        if len(control_geos) < 2:
            continue

        duration = pa.recommend_duration(
            test_geo,
            control_geos,
            expected_lift,
            target_power=target_power,
            max_duration=max_duration,
        )
        if duration == -1:
            duration = max_duration

        mde = pa.find_mde(
            test_geo,
            control_geos,
            test_duration_days=duration,
            target_power=target_power,
            lookback_days=lookback_days,
        )
        required_input = {}
        if input_col and input_col != "None":
            required_input = pa.estimate_required_input(
                test_geo,
                input_col,
                mde if mde > 0 else expected_lift,
                duration,
                lookback_days=lookback_days,
            )

        return {
            "objective": "Validate incremental growth in a stable, high-signal geo.",
            "hypothesis": f"A controlled campaign in {test_geo} will create measurable incremental {outcome_col}.",
            "test_geo": test_geo,
            "control_geos": control_geos,
            "control_scores": [{"geo": geo, "distance": float(score)} for geo, score in controls],
            "duration_days": int(duration),
            "expected_lift_pct": expected_lift,
            "minimum_detectable_effect_pct": mde,
            "primary_metric": outcome_col,
            "required_input": required_input,
            "success_threshold": f"Posterior probability of positive lift >= {target_power:.0%}",
            "risks": [
                "Avoid overlapping campaigns in test and control geos.",
                "Check holdout geos for spillover before launch.",
                "Keep measurement definitions stable across the test.",
            ],
            "interpretation_plan": "Scale cautiously if lift is positive and evidence is Moderate or stronger; otherwise retest with longer duration or cleaner controls.",
        }

    return {"error": "Could not find a geo with at least two viable controls."}
