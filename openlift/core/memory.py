import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentRegistry:
    def __init__(self, path: str = ".openlift/experiments.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []

    def add_result(self, result: Dict[str, Any]) -> None:
        records = self.list()
        metrics = result.get("metrics", {})
        decision = result.get("decision", {})
        records.append({
            "experiment": result.get("experiment"),
            "test_geo": result.get("test_geo"),
            "control_geos": result.get("control_geos", []),
            "pre_period": result.get("pre_period"),
            "post_period": result.get("post_period"),
            "incremental_outcome_mean": metrics.get("incremental_outcome_mean"),
            "lift_pct_mean": metrics.get("lift_pct_mean"),
            "p_positive": metrics.get("p_positive"),
            "evidence_strength": decision.get("evidence_strength"),
            "decision": decision.get("decision"),
            "next_action": decision.get("next_action"),
        })
        self.path.write_text(json.dumps(records[-100:], indent=2))

    def scorecard(self) -> Dict[str, Any]:
        records = self.list()
        completed = len(records)
        positive = [r for r in records if (r.get("p_positive") or 0) >= 0.8]
        avg_lift = None
        lift_values = [r["lift_pct_mean"] for r in records if r.get("lift_pct_mean") is not None]
        if lift_values:
            avg_lift = sum(lift_values) / len(lift_values)
        return {
            "completed_experiments": completed,
            "positive_experiments": len(positive),
            "positive_rate": len(positive) / completed if completed else 0,
            "average_lift_pct": avg_lift,
            "records": records,
        }
