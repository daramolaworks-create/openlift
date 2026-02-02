import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class GeoMatcher:
    def __init__(self, df: pd.DataFrame, date_col: str, geo_col: str, outcome_col: str):
        self.df = df
        self.date_col = date_col
        self.geo_col = geo_col
        self.outcome_col = outcome_col
        
        # Ensure date column is datetime
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df_wide = self.df.pivot(index=self.date_col, columns=self.geo_col, values=self.outcome_col).sort_index()

    def find_controls(
        self, 
        test_geo: str, 
        pool_geos: Optional[List[str]] = None,
        lookback_days: int = 90,
        n_controls: int = 5,
        method: str = "dtw" # "dtw" or "correlation" or "euclidean"
    ) -> List[Tuple[str, float]]:
        """
        Find best matching control geos for a given test geo used time-series similarity.
        """
        if pool_geos is None:
            pool_geos = [g for g in self.df_wide.columns if g != test_geo]
            
        # Filter data for lookback period
        max_date = self.df_wide.index.max()
        start_date = max_date - timedelta(days=lookback_days)
        data = self.df_wide.loc[start_date:max_date]
        
        target_series = data[test_geo].fillna(0).values
        
        scores = []
        for geo in pool_geos:
            if geo not in data.columns:
                continue
                
            control_series = data[geo].fillna(0).values
            
            if method == "dtw":
                # FastDTW returns (distance, path)
                # For univariate series, dist function receives scalars (floats).
                # scipy.spatial.distance.euclidean expects 1-D arrays, so it fails on scalars.
                # We use simple absolute difference.
                dist, _ = fastdtw(target_series, control_series, dist=lambda x, y: abs(x - y))
            elif method == "euclidean":
                dist = np.linalg.norm(target_series - control_series)
            elif method == "correlation":
                # Correlation is similarity, so we convert to distance
                corr = np.corrcoef(target_series, control_series)[0, 1]
                dist = 1 - corr
            else:
                raise ValueError(f"Unknown method: {method}")
                
            scores.append((geo, dist))
            
        # Sort by distance (ascending)
        scores.sort(key=lambda x: x[1])
        
        return scores[:n_controls]

class PowerAnalysis:
    def __init__(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        geo_col: str, 
        outcome_col: str
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.geo_col = geo_col
        self.outcome_col = outcome_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    def simulate_power(
        self,
        test_geo: str,
        control_geos: List[str],
        effect_size_pct: float,
        lookback_days: int = 60,
        test_duration_days: int = 30,
        simulations: int = 100,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Simulate power by injecting effects into historical data.
        
        This mimics a "Pre-Flight Simulator" by:
        1. Taking historical data.
        2. Simulating a treatment effect on the test geo for 'test_duration_days'.
        3. Running a simplified inference (e.g. diff-in-diff or synth-control-like) to see if effect is detected.
        
        Note: Full PyMC sampling is too slow for 100+ simulations in a loop.
        We will use a faster proxy for power estimation, or a very lightweight model.
        For rigorous results, we might accept fewer simulations (e.g. 20) and use PyMC.
        
        For this implementation, we'll use a simplified approach:
        - Fit a model on pre-period (standard OLS/Ridge on controls).
        - Predict counterfactual on post-period.
        - Add effect.
        - Check if (Observed - Counterfactual) is significant.
        """
        from sklearn.linear_model import Ridge
        
        # Prepare data
        df_wide = self.df.pivot(index=self.date_col, columns=self.geo_col, values=self.outcome_col).sort_index()
        
        # User request: Treat missing days as 0 (e.g. no conversions/spend that day)
        df_wide = df_wide.fillna(0)
        
        # We pick a random start date for the "experiment" within the available history
        # allowing for lookback_days before it.
        valid_starts = df_wide.index[lookback_days : -test_duration_days]
        if len(valid_starts) == 0:
            total_days = len(df_wide)
            req_days = lookback_days + test_duration_days
            raise ValueError(f"Not enough history. Data has {total_days} days. Need > {req_days} days (Lookback {lookback_days} + Duration {test_duration_days}).")
            
        detection_count = 0
        
        for i in range(simulations):
            # Pick random start
            sim_start_date = np.random.choice(valid_starts)
            sim_start_idx = df_wide.index.get_loc(sim_start_date)
            
            # Pre/Post split
            pre_start_idx = sim_start_idx - lookback_days
            pre_data = df_wide.iloc[pre_start_idx:sim_start_idx]
            post_data = df_wide.iloc[sim_start_idx:sim_start_idx + test_duration_days]
            
            # 1. Fit model on Pre
            X_pre = pre_data[control_geos].values
            y_pre = pre_data[test_geo].values
            
            model = Ridge(alpha=1.0)
            model.fit(X_pre, y_pre)
            
            # 2. Predict Counterfactual on Post
            X_post = post_data[control_geos].values
            y_base_post = post_data[test_geo].values # The actual historical values (acting as latent outcome)
            
            # 3. Inject Effect
            # Lift is added to the "base" outcome
            lift_amount = y_base_post * effect_size_pct
            y_treated_post = y_base_post + lift_amount
            
            # 4. Estimate Effect
            y_pred_post = model.predict(X_post)
            
            # Residuals in pre-period to estimate noise/variance
            residuals_pre = y_pre - model.predict(X_pre)
            std_error = np.std(residuals_pre)
            
            # Estimated Lift
            estimated_lift = y_treated_post - y_pred_post
            avg_estimated_lift = np.mean(estimated_lift)
            
            # Simple Z-test / T-test logic
            # SE of the mean difference ≈ std_error / sqrt(n) (simplified)
            # A more robust check might consider auto-correlation
            se_mean = std_error # conservative, assuming prediction error similar to daily noise
            
            # If the confidence interval (95%) excludes 0
            # Lower bound > 0
            # 1.96 * std_error of the SUM?
            # Let's look at total uplift.
            total_lift = np.sum(estimated_lift)
            # Std dev of sum of N independent residuals = sqrt(N) * std_error
            se_total = np.sqrt(len(y_treated_post)) * std_error
            
            z_score = total_lift / se_total
            
            # One-sided test (we expect positive lift)
            if z_score > 1.645: # 95% confidence one-sided
                detection_count += 1
                
        power = detection_count / simulations
        return {
            "power": power,
            "simulations": simulations,
            "effect_size": effect_size_pct,
            "duration": test_duration_days
        }

    def recommend_duration(
        self,
        test_geo: str,
        control_geos: List[str],
        effect_size_pct: float,
        target_power: float = 0.8,
        max_duration: int = 60
    ) -> int:
        """
        Find minimum duration to achieve target power.
        """
        for duration in [14, 21, 28, 35, 42, 49, 56, 60]:
            if duration > max_duration:
                break
            
            res = self.simulate_power(
                test_geo, 
                control_geos, 
                effect_size_pct, 
                test_duration_days=duration,
                simulations=50 # fewer simulations for speed during search
            )
            
            if res["power"] >= target_power:
                return duration
                
        return -1 # Not achievable
