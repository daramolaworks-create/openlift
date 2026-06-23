import numpy as np
import pandas as pd

from openlift.core.experiment_designer import recommend_next_experiment


def test_recommend_next_experiment_returns_plan():
    dates = pd.date_range("2024-01-01", periods=90)
    rows = []
    for geo, offset in [("A", 0), ("B", 1), ("C", 2)]:
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "geo": geo,
                "outcome": 100 + i * 0.1 + offset + np.sin(i / 7),
                "spend": 10,
            })
    df = pd.DataFrame(rows)

    plan = recommend_next_experiment(
        df,
        "date",
        "geo",
        "outcome",
        input_col="spend",
        max_duration=21,
        lookback_days=30,
    )

    assert "test_geo" in plan
    assert len(plan["control_geos"]) >= 2
    assert plan["duration_days"] > 0
