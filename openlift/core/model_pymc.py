import pymc as pm
import numpy as np
import arviz as az
from typing import Optional

def build_model(
    y_pre: np.ndarray,
    X_pre: np.ndarray,
    dow_pre: np.ndarray,
    Z_pre: Optional[np.ndarray] = None
) -> pm.Model:
    """
    Build the PyMC model for the pre-period.
    
    Parameters
    ----------
    y_pre : target geo outcome in pre-period
    X_pre : control geo features (scaled)
    dow_pre : day-of-week indices
    Z_pre : optional external covariates (holidays, weather)
    """
    T, K = X_pre.shape
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=K)
        dow_effect = pm.Normal("dow_effect", mu=0, sigma=0.5, shape=7)
        
        sigma = pm.HalfNormal("sigma", sigma=1)
        nu = pm.Exponential("nu", lam=1/30) + 1 
        
        # Expected value
        mu = alpha + pm.math.dot(X_pre, beta) + dow_effect[dow_pre]
        
        # External covariates (holidays, weather)
        if Z_pre is not None and Z_pre.shape[1] > 0:
            n_cov = Z_pre.shape[1]
            gamma = pm.Normal("gamma", mu=0, sigma=0.5, shape=n_cov)
            mu = mu + pm.math.dot(Z_pre, gamma)
        
        # Likelihood
        _y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y_pre)
        
    return model

def fit_model(
    model: pm.Model, 
    draws: int = 1000, 
    tune: int = 1000, 
    chains: int = 2, 
    target_accept: float = 0.9,
    random_seed: int = 42
) -> az.InferenceData:
    """
    Sample from the model.
    """
    with model:
        idata = pm.sample(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True
        )
    return idata
