import arviz as az
import numpy as np
from typing import Dict, Any, List

def compute_lift(
    idata: az.InferenceData,
    y_post: np.ndarray,
    X_post: np.ndarray,
    dow_post: np.ndarray,
    Z_post: "Optional[np.ndarray]" = None,
    covariate_names: "Optional[List[str]]" = None,
) -> Dict[str, Any]:
    """
    Compute counterfactual predictions for post-period and calculate lift metrics.
    
    Parameters
    ----------
    Z_post : optional covariate matrix for the post-period  
    covariate_names : optional list of names for each covariate column
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
    
    # Add covariate term if present
    covariate_effects = {}
    if Z_post is not None and "gamma" in posterior:
        gamma_samples = posterior["gamma"].values  # (C, D, n_cov)
        gamma_flat = gamma_samples.reshape(n_samples, -1)
        term4 = np.dot(Z_post, gamma_flat.T).T
        y_hat_samples = y_hat_samples + term4
        
        # Report per-covariate effect sizes
        gamma_means = np.mean(gamma_flat, axis=0)
        names = covariate_names if covariate_names else [f"cov_{i}" for i in range(len(gamma_means))]
        for i, name in enumerate(names):
            covariate_effects[name] = float(gamma_means[i])
    
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
    
    if covariate_effects:
        metrics["covariate_effects"] = covariate_effects
    
    return metrics


def compare_cells(
    cell_a_result: Dict[str, Any],
    cell_b_result: Dict[str, Any],
    label_a: str = "Cell A",
    label_b: str = "Cell B",
) -> Dict[str, Any]:
    """
    Compare two experiment cells using their lift metrics.

    Uses the incremental lift values and confidence intervals to compute
    the probability that one cell outperforms the other.

    Parameters
    ----------
    cell_a_result, cell_b_result : dicts returned by _run_pipeline
    label_a, label_b : human-readable labels

    Returns
    -------
    Dict with comparison metrics.
    """
    try:
        a_metrics = cell_a_result["metrics"]
        b_metrics = cell_b_result["metrics"]

        a_lift = a_metrics["incremental_outcome_mean"]
        b_lift = b_metrics["incremental_outcome_mean"]

        a_lift_pct = a_metrics["lift_pct_mean"]
        b_lift_pct = b_metrics["lift_pct_mean"]

        a_confidence = a_metrics["p_positive"]
        b_confidence = b_metrics["p_positive"]

        # Use HDI to estimate uncertainty overlap
        a_hdi = a_metrics.get("incremental_outcome_hdi_90", [0, 0])
        b_hdi = b_metrics.get("incremental_outcome_hdi_90", [0, 0])

        # Approximate P(A > B) using a Normal approximation
        # from the mean and HDI width
        a_std = (a_hdi[1] - a_hdi[0]) / (2 * 1.645)  # 90% HDI ≈ ±1.645σ
        b_std = (b_hdi[1] - b_hdi[0]) / (2 * 1.645)

        # P(A > B) where A ~ N(μ_a, σ_a²), B ~ N(μ_b, σ_b²)
        # A - B ~ N(μ_a - μ_b, σ_a² + σ_b²)
        diff_mean = a_lift - b_lift
        diff_std = np.sqrt(a_std**2 + b_std**2)

        if diff_std > 0:
            from scipy.stats import norm
            p_a_greater = float(norm.cdf(diff_mean / diff_std))
        else:
            p_a_greater = 1.0 if diff_mean > 0 else 0.0

        # Determine winner
        if p_a_greater > 0.9:
            winner = label_a
            confidence_level = "high"
        elif p_a_greater < 0.1:
            winner = label_b
            confidence_level = "high"
        elif p_a_greater > 0.7:
            winner = label_a
            confidence_level = "moderate"
        elif p_a_greater < 0.3:
            winner = label_b
            confidence_level = "moderate"
        else:
            winner = "Inconclusive"
            confidence_level = "low"

        return {
            f"{label_a}_lift": a_lift,
            f"{label_b}_lift": b_lift,
            f"{label_a}_lift_pct": a_lift_pct,
            f"{label_b}_lift_pct": b_lift_pct,
            f"p_{label_a}_greater": p_a_greater,
            f"p_{label_b}_greater": 1.0 - p_a_greater,
            "absolute_delta": abs(a_lift - b_lift),
            "winner": winner,
            "confidence_level": confidence_level,
        }

    except Exception as e:
        return {"error": f"Comparison failed: {e}"}
