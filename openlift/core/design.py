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
        
    def estimate_required_input(
        self,
        test_geo: str,
        input_col: str,
        effect_size_pct: float,
        test_duration_days: int,
        lookback_days: int = 60
    ) -> Dict[str, float]:
        """
        Estimate the amount of Input (X) required to generate the target Lift (Y).
        Based on historical efficiency (Sum X / Sum Y) in the test geo.
        """
        if input_col not in self.df.columns or input_col == "None":
            return {}

        # 1. Get Historical Data for Test Geo
        # We use the most recent 'lookback_days'
        df_geo = self.df[self.df[self.geo_col] == test_geo].sort_values(self.date_col).tail(lookback_days)
        
        if len(df_geo) == 0:
            return {}
            
        # 2. Calculate Efficiency (Cost Per Result)
        total_input = df_geo[input_col].sum()
        total_outcome = df_geo[self.outcome_col].sum()
        
        if total_outcome == 0:
            return {"cpr": 0, "required_input": 0, "baseline_outcome": 0}
            
        cpr = total_input / total_outcome # e.g. Spend / Conversions
        
        # 3. Estimate Baseline Outcome for Test Duration
        # Avg daily outcome * Duration
        avg_daily_outcome = total_outcome / len(df_geo)
        estimated_baseline = avg_daily_outcome * test_duration_days
        
        # 4. Calculate Absolute Lift Needed
        # effect_size_pct is e.g. 0.10 for 10%
        target_lift_abs = estimated_baseline * effect_size_pct
        
        # 5. Calculate Required Input
        # We assume marginal cost = average cost (conservative linear assumption)
        required_input = target_lift_abs * cpr
        
        return {
            "cpr": cpr, # Cost Per Result
            "baseline_outcome": estimated_baseline,
            "target_lift_abs": target_lift_abs,
            "required_input": required_input
        }

    def find_mde(
        self,
        test_geo: str,
        control_geos: List[str],
        test_duration_days: int,
        target_power: float = 0.8,
        lookback_days: int = 60,
        max_lift: float = 0.5
    ) -> float:
        """
        Find the Minimum Detectable Effect (Lift %) required to achieve target_power.
        """
        # Search grid for lift %: 1% to 50%
        search_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        
        for lift in search_grid:
            if lift > max_lift:
                break
                
            res = self.simulate_power(
                test_geo,
                control_geos,
                effect_size_pct=lift,
                test_duration_days=test_duration_days,
                simulations=40, # Faster check
                lookback_days=lookback_days
            )
            
            if res['power'] >= target_power:
                return lift
                
        return -1.0 # Not achievable within max_lift
