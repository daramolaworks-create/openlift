from typing import Dict, Any, List
import pandas as pd
import os

from .io import load_data
from .experiment import load_experiment_config, ExperimentConfig
from .features import make_features
from .model_pymc import build_model, fit_model
from .inference import compute_lift
from .report import format_json_output

def run_geo_lift(experiment_path: str) -> Dict[str, Any]:
    """
    Run an end-to-end geo-lift experiment from a YAML config file.
    """
    full_config = load_experiment_config(experiment_path)
    exp = full_config.experiment
    data_cfg = full_config.data
    
    if not os.path.isabs(data_cfg.path):
        base_dir = os.path.dirname(experiment_path)
        data_path = os.path.join(base_dir, data_cfg.path)
    else:
        data_path = data_cfg.path
        
    df_wide = load_data(data_path, data_cfg.date_col, data_cfg.geo_col, data_cfg.outcome_col)
    
    return _run_pipeline(df_wide, exp)

def run_geo_lift_df(
    df: pd.DataFrame, 
    test_geo: str, 
    control_geos: List[str], 
    pre_start: str, 
    pre_end: str, 
    post_start: str, 
    post_end: str,
    date_col: str = 'date',
    geo_col: str = 'geo',
    outcome_col: str = 'outcome',
    name: str = "custom_experiment"
) -> Dict[str, Any]:
    df[date_col] = pd.to_datetime(df[date_col])
    df_wide = df.pivot(index=date_col, columns=geo_col, values=outcome_col).sort_index()
    
    from datetime import datetime
    def pars(d): return datetime.strptime(d, "%Y-%m-%d").date()
    
    exp_config = {
        "name": name,
        "test_geo": test_geo,
        "control_geos": control_geos,
        "pre_period": {"start_date": pars(pre_start), "end_date": pars(pre_end)},
        "post_period": {"start_date": pars(post_start), "end_date": pars(post_end)}
    }
    
    exp = ExperimentConfig(**exp_config)
    
    return _run_pipeline(df_wide, exp)

def _run_pipeline(df_wide: pd.DataFrame, exp: ExperimentConfig) -> Dict[str, Any]:
    y_pre, X_pre, dow_pre, y_post, X_post, dow_post, scaler = make_features(
        df_wide, 
        exp.test_geo, 
        exp.control_geos, 
        pd.Timestamp(exp.pre_period.start_date),
        pd.Timestamp(exp.pre_period.end_date),
        pd.Timestamp(exp.post_period.start_date),
        pd.Timestamp(exp.post_period.end_date)
    )
    
    model = build_model(y_pre, X_pre, dow_pre)
    
    draws = 1000
    tune = 1000
    chains = 2
    target_accept = 0.9
    
    idata = fit_model(
        model, 
        draws=draws, 
        tune=tune, 
        chains=chains, 
        target_accept=target_accept
    )
    
    metrics = compute_lift(idata, y_post, X_post, dow_post)
    
    model_config = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept
    }
    
    exp_dict = {
        "name": exp.name,
        "test_geo": exp.test_geo,
        "control_geos": exp.control_geos,
        "pre_period": {"start_date": exp.pre_period.start_date, "end_date": exp.pre_period.end_date},
        "post_period": {"start_date": exp.post_period.start_date, "end_date": exp.post_period.end_date},
    }
    
    return format_json_output(exp_dict, model_config, metrics)
