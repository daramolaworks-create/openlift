import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero if std is zero (constant feature)
        self.std[self.std == 0] = 1.0
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return (X - self.mean) / self.std

def make_features(
    df_wide: pd.DataFrame, 
    test_geo: str, 
    control_geos: List[str], 
    pre_start: pd.Timestamp, 
    pre_end: pd.Timestamp, 
    post_start: pd.Timestamp, 
    post_end: pd.Timestamp,
    country_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, "StandardScaler", Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Prepare features for modeling.
    
    Returns
    -------
    y_pre, X_pre, dow_pre, y_post, X_post, dow_post, scaler,
    Z_pre (optional covariates), Z_post (optional covariates),
    covariate_names (list of covariate column names)
    """
    # Filter by date range
    expected_pre_idx = pd.date_range(start=pre_start, end=pre_end, freq='D')
    expected_post_idx = pd.date_range(start=post_start, end=post_end, freq='D')
    
    try:
        # Reindex to ensure daily frequency. This introduces NaNs for missing days.
        pre_df = df_wide.reindex(expected_pre_idx)
        post_df = df_wide.reindex(expected_post_idx)
        
        # User request: Treat missing days as 0 (e.g. no conversions/spend that day)
        pre_df = pre_df.fillna(0)
        post_df = post_df.fillna(0)
        
    except Exception as e:
        raise ValueError(f"Error reindexing dates: {e}")
        
    # Validation removed as we now explicitly handle missing data as 0.

    # Extract Y (test geo)
    y_pre = pre_df[test_geo].values
    y_post = post_df[test_geo].values
    
    # Extract X (control geos)
    X_pre_raw = pre_df[control_geos].values
    X_post_raw = post_df[control_geos].values
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_pre_raw)
    X_pre = scaler.transform(X_pre_raw)
    X_post = scaler.transform(X_post_raw)
    
    # DoW features (0=Monday, 6=Sunday)
    dow_pre = pre_df.index.dayofweek.values
    dow_post = post_df.index.dayofweek.values
    
    # --- External Covariates ---
    Z_pre = None
    Z_post = None
    covariate_names = []
    
    if country_code or (latitude is not None and longitude is not None):
        from .covariates import build_covariates
        
        # Build for full range (pre + post)
        full_idx = expected_pre_idx.append(expected_post_idx)
        
        result = build_covariates(
            date_index=full_idx,
            country_code=country_code,
            latitude=latitude,
            longitude=longitude,
            geo_name=test_geo,
        )
        
        if result is not None:
            Z_full, covariate_names = result
            n_pre = len(expected_pre_idx)
            Z_pre = Z_full[:n_pre]
            Z_post = Z_full[n_pre:]
    
    return y_pre, X_pre, dow_pre, y_post, X_post, dow_post, scaler, Z_pre, Z_post, covariate_names
