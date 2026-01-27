import pandas as pd
import numpy as np
from typing import Tuple, List

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
    post_end: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare features for modeling.
    """
    # Filter by date range
    expected_pre_idx = pd.date_range(start=pre_start, end=pre_end, freq='D')
    expected_post_idx = pd.date_range(start=post_start, end=post_end, freq='D')
    
    try:
        pre_df = df_wide.reindex(expected_pre_idx)
        post_df = df_wide.reindex(expected_post_idx)
    except Exception as e:
        raise ValueError(f"Error reindexing dates: {e}")
        
    cols_to_check = [test_geo] + control_geos
    
    if pre_df[cols_to_check].isnull().any().any():
        missing_dates = pre_df[pre_df[cols_to_check].isnull().any(axis=1)].index.tolist()
        raise ValueError(f"Missing data in PRE period for required geos. Dates: {missing_dates[:5]}...")
        
    if post_df[cols_to_check].isnull().any().any():
        missing_dates = post_df[post_df[cols_to_check].isnull().any(axis=1)].index.tolist()
        raise ValueError(f"Missing data in POST period for required geos. Dates: {missing_dates[:5]}...")

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
    
    return y_pre, X_pre, dow_pre, y_post, X_post, dow_post, scaler
