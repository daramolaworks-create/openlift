import pandas as pd


def infer_creative_columns(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    lower = {c: c.lower() for c in columns}

    def first_match(keywords, default=None):
        for col, name in lower.items():
            if any(k in name for k in keywords):
                return col
        return default

    period = None
    for col in columns:
        values = set(df[col].dropna().astype(str).str.lower().head(100).tolist())
        if values and values <= {"pre", "post", "treatment", "test", "control"}:
            period = col
            break

    treatment = None
    for col in columns:
        values = set(df[col].dropna().astype(str).str.lower().head(100).tolist())
        if values and values <= {"true", "false", "1", "0", "yes", "no", "treated", "control"}:
            treatment = col
            break

    return {
        "join": first_match(["creative_id", "creative id", "ad id", "campaign"], columns[0] if columns else None),
        "group": first_match(["hook", "angle", "format", "offer", "creative", "ad name", "campaign"], columns[0] if columns else None),
        "spend": first_match(["spend", "spent", "cost", "amount"], None),
        "outcome": first_match(["purchase", "conversion", "result", "revenue", "sales", "lead"], None),
        "period": period,
        "treatment": treatment,
    }


def analyze_creative_lift(
    df: pd.DataFrame,
    creative_df: pd.DataFrame,
    join_col: str = "creative_id",
    date_col: str = "date",
    outcome_col: str = "outcome",
    spend_col: str = "spend",
    treatment_col: str = "treatment",
    period_col: str = "period",
    group_col: str = "hook_type",
) -> pd.DataFrame:
    """
    Estimate creative-level lift.

    If treatment and period columns exist, uses a simple difference-in-differences
    style score by creative group:

        (treated post - treated pre) - (control post - control pre)

    Otherwise falls back to descriptive performance aggregation.
    """
    merged = _build_analysis_frame(df, creative_df, join_col)

    if group_col not in merged.columns:
        group_col = join_col

    if all(c in merged.columns for c in [group_col, outcome_col, treatment_col, period_col]):
        work = merged.copy()
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0)
        if spend_col in work.columns:
            work[spend_col] = pd.to_numeric(work[spend_col], errors="coerce").fillna(0)
        else:
            work[spend_col] = 0.0

        work["_is_treated"] = work[treatment_col].astype(str).str.lower().isin(["true", "1", "yes", "treated"])
        work["_period"] = work[period_col].astype(str).str.lower()
        rows = []

        for group, gdf in work.groupby(group_col, dropna=False):
            treated_pre = _mean_outcome(gdf, True, "pre", outcome_col)
            treated_post = _mean_outcome(gdf, True, "post", outcome_col)
            control_pre = _mean_outcome(gdf, False, "pre", outcome_col)
            control_post = _mean_outcome(gdf, False, "post", outcome_col)

            treated_delta = treated_post - treated_pre
            control_delta = control_post - control_pre
            lift_score = treated_delta - control_delta
            baseline = abs(treated_pre) if treated_pre else 0
            lift_pct = lift_score / baseline if baseline else None
            total_spend = float(gdf[spend_col].sum())

            rows.append({
                group_col: group,
                "treated_pre_mean": treated_pre,
                "treated_post_mean": treated_post,
                "control_pre_mean": control_pre,
                "control_post_mean": control_post,
                "creative_lift_score": lift_score,
                "creative_lift_pct": lift_pct,
                "total_spend": total_spend,
                "cost_per_incremental_outcome": total_spend / lift_score if lift_score > 0 else None,
                "rows": len(gdf),
            })

        return pd.DataFrame(rows).sort_values("creative_lift_score", ascending=False)

    if group_col in merged.columns and outcome_col in merged.columns and spend_col in merged.columns:
        agg = merged.groupby(group_col).agg(
            total_spend=(spend_col, "sum"),
            total_outcome=(outcome_col, "sum")
        ).reset_index()
        
        agg["roas"] = agg["total_outcome"] / agg["total_spend"]
        return agg.sort_values("roas", ascending=False)
        
    return pd.DataFrame()


def _mean_outcome(df: pd.DataFrame, treated: bool, period: str, outcome_col: str) -> float:
    mask = (df["_is_treated"] == treated) & (df["_period"] == period)
    if not mask.any():
        return 0.0
    return float(df.loc[mask, outcome_col].mean())


def _build_analysis_frame(
    df: pd.DataFrame,
    creative_df: pd.DataFrame,
    join_col: str,
) -> pd.DataFrame:
    """
    Prefer the uploaded creative/performance file as-is. If it is only metadata,
    add missing metric columns from the main dataset without suffixing away
    shared column names.
    """
    if join_col not in creative_df.columns:
        raise ValueError(f"Join column {join_col} missing in creative dataframe.")

    merged = creative_df.copy()
    if join_col not in df.columns:
        return merged

    missing_from_creative = [c for c in df.columns if c not in merged.columns]
    if not missing_from_creative:
        return merged

    supplemental = df[[join_col] + missing_from_creative].copy()
    return merged.merge(supplemental, on=join_col, how="left")
