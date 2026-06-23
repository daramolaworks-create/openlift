from typing import Dict, Any, Optional, List

def calculate_economics(
    incremental_outcome: float,
    input_change_abs: float,
    outcome_is_revenue: bool = False,
    margin_pct: float = 1.0,
    ltv: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates Incremental CAC, Incremental ROAS, and Incremental Profit.
    """
    res = {
        "incremental_roas": None,
        "incremental_cac": None,
        "incremental_profit": None
    }
    
    if input_change_abs <= 0 or incremental_outcome <= 0:
        return res

    if outcome_is_revenue:
        # Outcome is already revenue. No LTV multiplier applied by default, but margin matters for profit.
        res["incremental_roas"] = incremental_outcome / input_change_abs
        res["incremental_profit"] = (incremental_outcome * margin_pct) - input_change_abs
        res["payback_period_months"] = calculate_payback_period(
            incremental_revenue=incremental_outcome,
            input_change_abs=input_change_abs,
            margin_pct=margin_pct,
        )
    else:
        # Outcome is conversions (e.g. signups, purchases)
        res["incremental_cac"] = input_change_abs / incremental_outcome
        incremental_revenue = incremental_outcome * ltv
        res["incremental_roas"] = incremental_revenue / input_change_abs
        res["incremental_profit"] = (incremental_revenue * margin_pct) - input_change_abs
        res["payback_period_months"] = calculate_payback_period(
            incremental_revenue=incremental_revenue,
            input_change_abs=input_change_abs,
            margin_pct=margin_pct,
        )

    return res


def calculate_payback_period(
    incremental_revenue: float,
    input_change_abs: float,
    margin_pct: float = 1.0,
    revenue_period_days: int = 30,
) -> Optional[float]:
    gross_profit = incremental_revenue * margin_pct
    if input_change_abs <= 0 or gross_profit <= 0:
        return None
    monthly_gross_profit = gross_profit * (30 / max(revenue_period_days, 1))
    if monthly_gross_profit <= 0:
        return None
    return input_change_abs / monthly_gross_profit


def build_payback_curve(
    incremental_revenue: float,
    input_change_abs: float,
    margin_pct: float = 1.0,
    months: int = 12,
) -> List[Dict[str, float]]:
    monthly_gross_profit = incremental_revenue * margin_pct
    rows = []
    for month in range(1, months + 1):
        cumulative_profit = monthly_gross_profit * month
        rows.append({
            "month": month,
            "cumulative_gross_profit": cumulative_profit,
            "remaining_payback": input_change_abs - cumulative_profit,
            "paid_back": cumulative_profit >= input_change_abs,
        })
    return rows

def recommend_budget(
    p_positive: float,
    incremental_roas: Optional[float],
    incremental_profit: Optional[float],
    target_roas: float = 1.0
) -> str:
    """
    Provides a budget recommendation based on model confidence and economics.
    """
    if p_positive < 0.8:
        return "Hold & Retest: The evidence of lift is too weak to scale. Retest to gain confidence."
        
    if incremental_roas is not None:
        if incremental_roas > target_roas * 1.5 and p_positive >= 0.9:
            return "Scale Aggressively: Strong evidence of highly profitable lift."
        elif incremental_roas >= target_roas:
            return "Scale Cautiously: The campaign is profitable and incremental."
        else:
            return "Reduce Spend: The campaign is driving incremental volume, but it is not hitting ROAS targets."
    else:
        # No economics provided, base purely on lift confidence
        if p_positive >= 0.95:
            return "Scale: Very strong evidence of lift."
        elif p_positive >= 0.85:
            return "Scale Cautiously: Good evidence of lift."
        else:
            return "Hold & Retest: Directional evidence only."


def simulate_budget_scenarios(
    current_input: float,
    incremental_roas: Optional[float],
    incremental_cac: Optional[float] = None,
    incremental_profit: Optional[float] = None,
    p_positive: float = 0.5,
    risk_tolerance: str = "medium",
    steps: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a simple confidence-weighted budget scenario table.

    This is intentionally rule-based for V0.1: it does not assume a learned
    saturation curve, but it makes the scale/hold/cut recommendation concrete.
    """
    if steps is None:
        steps = [-0.25, -0.10, 0.0, 0.10, 0.25]

    risk_multiplier = {"low": 0.6, "medium": 0.8, "high": 1.0}.get(risk_tolerance, 0.8)
    effective_confidence = p_positive * risk_multiplier
    scenarios = []

    for change in steps:
        new_input = current_input * (1 + change)
        input_delta = new_input - current_input
        expected_incremental_revenue = None
        expected_incremental_profit = None

        if incremental_roas is not None:
            expected_incremental_revenue = input_delta * incremental_roas * effective_confidence
            if incremental_profit is not None and current_input:
                profit_rate = incremental_profit / current_input
                expected_incremental_profit = input_delta * profit_rate * effective_confidence

        scenarios.append({
            "change_pct": change,
            "new_input": new_input,
            "input_delta": input_delta,
            "confidence_weight": effective_confidence,
            "expected_incremental_revenue": expected_incremental_revenue,
            "expected_incremental_profit": expected_incremental_profit,
            "expected_incremental_cac": incremental_cac,
        })

    return scenarios
