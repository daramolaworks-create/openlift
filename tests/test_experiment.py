import pytest
import yaml
from openlift.core.experiment import load_experiment_config

def test_experiment_config_valid(tmp_path):
    config = {
        "experiment": {
            "name": "test",
            "test_geo": "A",
            "control_geos": ["B", "C"],
            "pre_period": {"start_date": "2024-01-01", "end_date": "2024-02-01"}, # 32 days
            "post_period": {"start_date": "2024-02-02", "end_date": "2024-02-10"}
        },
        "data": {
            "path": "test.csv",
            "date_col": "date",
            "geo_col": "geo",
            "outcome_col": "outcome"
        }
    }
    
    p = tmp_path / "exp.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
        
    cfg = load_experiment_config(str(p))
    assert cfg.experiment.name == "test"
    assert len(cfg.experiment.control_geos) == 2

def test_experiment_short_pre(tmp_path):
    config = {
        "experiment": {
            "name": "test",
            "test_geo": "A",
            "control_geos": ["B", "C"],
            "pre_period": {"start_date": "2024-01-01", "end_date": "2024-01-05"},
            "post_period": {"start_date": "2024-01-06", "end_date": "2024-01-10"}
        },
        "data": {"path": "x", "date_col": "d", "geo_col": "g", "outcome_col": "o"}
    }
    p = tmp_path / "exp_bad.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
        
    with pytest.raises(ValueError, match="Pre-period must be at least 30 days"):
        load_experiment_config(str(p))
