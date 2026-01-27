import arviz as az
import numpy as np
from typing import Dict, Any

def compute_lift(
    idata: az.InferenceData,
    y_post: np.ndarray,
    X_post: np.ndarray,
    dow_post: np.ndarray
) -> Dict[str, Any]:
    """
    Compute counterfactual predictions for post-period and calculate lift metrics.
    """
    posterior = idata.posterior
    
    alpha_samples = posterior["alpha"].values  # (C, D)
    beta_samples = posterior["beta"].values    # (C, D, K)
    dow_effect_samples = posterior["dow_effect"].values # (C, D, 7)
    
    n_samples = alpha_samples.size
    alpha_flat = alpha_samples.reshape(n_samples) 
    beta_flat = beta_samples.reshape(n_samples, -1) 
    dow_effect_flat = dow_effect_samples.reshape(n_samples, -1) 
    
    term1 = alpha_flat[:, np.newaxis]
    term2 = np.dot(X_post, beta_flat.T).T
    term3 = dow_effect_flat[:, dow_post]
    
    y_hat_samples = term1 + term2 + term3
    
    y_post_total = np.sum(y_post)
    y_hat_total_samples = np.sum(y_hat_samples, axis=1)
    
    lift_samples = y_post_total - y_hat_total_samples
    lift_pct_samples = lift_samples / y_hat_total_samples
    
    metrics = {}
    metrics["observed_outcome_sum"] = float(y_post_total)
    metrics["predicted_outcome_mean"] = float(np.mean(y_hat_total_samples))
    metrics["incremental_outcome_mean"] = float(np.mean(lift_samples))
    metrics["incremental_outcome_hdi_90"] = [float(x) for x in az.hdi(lift_samples, hdi_prob=0.9)]
    metrics["lift_pct_mean"] = float(np.mean(lift_pct_samples))
    metrics["lift_pct_hdi_90"] = [float(x) for x in az.hdi(lift_pct_samples, hdi_prob=0.9)]
    metrics["p_positive"] = float(np.mean(lift_samples > 0))
    metrics["model_intercept"] = float(np.mean(alpha_flat))
    metrics["model_coefficients"] = [float(x) for x in np.mean(beta_flat, axis=0)]
    metrics["dow_effect"] = [float(x) for x in np.mean(dow_effect_flat, axis=0)]
    
    return metrics
