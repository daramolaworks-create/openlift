from openlift.core.economics import calculate_payback_period, simulate_budget_scenarios


def test_simulate_budget_scenarios_returns_scale_rows():
    scenarios = simulate_budget_scenarios(
        current_input=1000,
        incremental_roas=2.0,
        incremental_profit=500,
        p_positive=0.9,
    )

    assert len(scenarios) == 5
    assert scenarios[0]["change_pct"] == -0.25
    assert scenarios[-1]["new_input"] == 1250
    assert scenarios[-1]["expected_incremental_revenue"] > 0


def test_calculate_payback_period():
    assert calculate_payback_period(1000, 500, margin_pct=0.5) == 1.0
