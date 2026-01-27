import numpy as np
import arviz as az
import xarray as xr
from openlift.core.inference import compute_lift

def test_inference_calc():
    # Mock inference data
    # 2 chains, 10 samples
    chains = 2
    draws = 10
    
    # Post period
    T_post = 5
    K = 2
    
    # X_post
    X_post = np.ones((T_post, K))
    dow_post = np.zeros(T_post, dtype=int)
    
    # True outcomes
    y_post = np.ones(T_post) * 10
    
    # Posterior samples
    # alpha = 1
    # beta = [1, 1]
    # dow = 0
    # Expected y = 1 + 1*1 + 1*1 + 0 = 3
    # Lift = 10 - 3 = 7 per day -> 35 total
    
    posterior = xr.Dataset({
        "alpha": (("chain", "draw"), np.ones((chains, draws))),
        "beta": (("chain", "draw", "k_dim"), np.ones((chains, draws, K))),
        "dow_effect": (("chain", "draw", "dow_dim"), np.zeros((chains, draws, 7)))
    })
    idata = az.InferenceData(posterior=posterior)
    
    metrics = compute_lift(idata, y_post, X_post, dow_post)
    
    # Check mean lift
    expected_daily = 7.0
    expected_total = expected_daily * T_post
    
    assert np.isclose(metrics["incremental_outcome_mean"], expected_total)
    assert metrics["p_positive"] == 1.0
