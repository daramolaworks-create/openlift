from typing import Any, Dict, List, Optional

from .economics import recommend_budget
from .evidence import calculate_evidence_strength


def build_limitations(
    metrics: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None,
    match_score: Optional[float] = None,
    pre_period_days: Optional[int] = None,
    post_period_days: Optional[int] = None,
) -> List[str]:
    limitations: List[str] = []

    if data_quality:
        for warning in data_quality.get("warnings", [])[:4]:
            limitations.append(warning)

    if match_score is not None and match_score > 1.5:
        limitations.append("Control match quality appears weak; interpret lift conservatively.")

    if pre_period_days is not None and pre_period_days < 28:
        limitations.append("Pre-period is shorter than four weeks, so seasonality may be under-learned.")

    if post_period_days is not None and post_period_days < 14:
        limitations.append("Treatment period is short, which can widen uncertainty.")

    hdi = metrics.get("incremental_outcome_hdi_90")
    mean_lift = metrics.get("incremental_outcome_mean", 0)
    if hdi and mean_lift:
        relative_width = abs(hdi[1] - hdi[0]) / max(abs(mean_lift), 1e-9)
        if relative_width > 2:
            limitations.append("Credible interval is wide relative to estimated lift.")

    if metrics.get("p_positive", 0) < 0.8:
        limitations.append("Posterior probability of positive lift is below the usual scale threshold.")

    if not limitations:
        limitations.append("No major automated limitations detected; review experiment assumptions before scaling.")

    return limitations


def build_decision_summary(
    metrics: Dict[str, Any],
    economics: Optional[Dict[str, Any]] = None,
    data_quality: Optional[Dict[str, Any]] = None,
    match_score: Optional[float] = None,
    pre_period_days: Optional[int] = None,
    post_period_days: Optional[int] = None,
    target_roas: float = 1.0,
) -> Dict[str, Any]:
    hdi = metrics.get("incremental_outcome_hdi_90", [0, 0])
    mean_lift = metrics.get("incremental_outcome_mean", 0)
    relative_hdi_width = None
    if mean_lift:
        relative_hdi_width = abs(hdi[1] - hdi[0]) / max(abs(mean_lift), 1e-9)

    evidence = calculate_evidence_strength(
        metrics.get("p_positive", 0),
        match_score=match_score,
        relative_hdi_width=relative_hdi_width,
    )
    economics = economics or {}
    recommendation = recommend_budget(
        metrics.get("p_positive", 0),
        economics.get("incremental_roas"),
        economics.get("incremental_profit"),
        target_roas=target_roas,
    )

    p_positive = metrics.get("p_positive", 0)
    roas = economics.get("incremental_roas")
    if p_positive >= 0.9 and (roas is None or roas >= target_roas):
        decision = "Scale cautiously"
        scale_range = "15-25%"
    elif p_positive >= 0.8:
        decision = "Hold or retest before scaling"
        scale_range = "0-10%"
    elif mean_lift < 0:
        decision = "Reduce or stop"
        scale_range = "Decrease 10-25%"
    else:
        decision = "Retest"
        scale_range = "No scale recommended"

    limitations = build_limitations(
        metrics,
        data_quality=data_quality,
        match_score=match_score,
        pre_period_days=pre_period_days,
        post_period_days=post_period_days,
    )

    return {
        "decision": decision,
        "scale_range": scale_range,
        "evidence_strength": evidence,
        "recommendation": recommendation,
        "limitations": limitations,
        "next_action": _next_action(decision, evidence),
    }


def _next_action(decision: str, evidence: str) -> str:
    if decision == "Scale cautiously" and evidence in {"Strong", "Very Strong"}:
        return "Increase budget gradually while monitoring the same outcome metric."
    if decision == "Scale cautiously":
        return "Run a cautious scale-up and plan a confirmatory matched-market test."
    if decision == "Reduce or stop":
        return "Reduce spend and investigate whether platform-attributed results were non-incremental."
    return "Run a follow-up experiment with stronger controls, longer duration, or cleaner data."
