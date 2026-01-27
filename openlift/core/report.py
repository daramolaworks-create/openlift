from typing import Dict, Any

def format_json_output(
    experiment_config: Dict[str, Any],
    model_config: Dict[str, Any],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "experiment": experiment_config.get("name"),
        "test_geo": experiment_config.get("test_geo"),
        "control_geos": experiment_config.get("control_geos"),
        "pre_period": {
            "start_date": str(experiment_config.get("pre_period").get("start_date")),
            "end_date": str(experiment_config.get("pre_period").get("end_date"))
        },
        "post_period": {
             "start_date": str(experiment_config.get("post_period").get("start_date")),
             "end_date": str(experiment_config.get("post_period").get("end_date"))
        },
        "metrics": metrics,
        "model": model_config
    }

def print_summary(results: Dict[str, Any]):
    m = results["metrics"]
    print("\n========= OpenLift Results =========")
    print(f"Experiment: {results['experiment']}")
    print(f"Test Geo: {results['test_geo']}")
    print(f"Incremental Outcome: {m['incremental_outcome_mean']:.2f} (HDI 90%: {m['incremental_outcome_hdi_90'][0]:.2f}, {m['incremental_outcome_hdi_90'][1]:.2f})")
    print(f"Lift %: {m['lift_pct_mean']:.2%} (HDI 90%: {m['lift_pct_hdi_90'][0]:.2%}, {m['lift_pct_hdi_90'][1]:.2%})")
    print(f"Prob(Lift > 0): {m['p_positive']:.2%}")
    print("===================================\n")
