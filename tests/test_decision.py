from openlift.core.decision import build_decision_summary, build_limitations


def test_build_decision_summary_scales_cautiously():
    metrics = {
        "incremental_outcome_mean": 100,
        "incremental_outcome_hdi_90": [40, 160],
        "p_positive": 0.95,
    }
    economics = {"incremental_roas": 2.0, "incremental_profit": 500}

    decision = build_decision_summary(metrics, economics=economics)

    assert decision["decision"] == "Scale cautiously"
    assert decision["evidence_strength"] in {"Strong", "Very Strong"}
    assert decision["limitations"]


def test_build_limitations_flags_short_periods():
    metrics = {
        "incremental_outcome_mean": 10,
        "incremental_outcome_hdi_90": [-50, 70],
        "p_positive": 0.6,
    }

    limitations = build_limitations(metrics, pre_period_days=14, post_period_days=7)

    assert any("Pre-period" in item for item in limitations)
    assert any("Treatment period" in item for item in limitations)
    assert any("Posterior probability" in item for item in limitations)
