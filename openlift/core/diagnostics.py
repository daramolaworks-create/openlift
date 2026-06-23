import pandas as pd
from typing import Dict, Any, Optional

from .schema import validate_growth_schema

class DataValidator:
    """
    Validates uploaded data and provides a Data Quality Score.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        geo_col: str,
        outcome_col: str,
        cost_col: str = None,
        pre_start: Optional[str] = None,
        pre_end: Optional[str] = None,
        post_start: Optional[str] = None,
        post_end: Optional[str] = None,
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.geo_col = geo_col
        self.outcome_col = outcome_col
        self.cost_col = cost_col
        self.pre_start = pd.Timestamp(pre_start) if pre_start else None
        self.pre_end = pd.Timestamp(pre_end) if pre_end else None
        self.post_start = pd.Timestamp(post_start) if post_start else None
        self.post_end = pd.Timestamp(post_end) if post_end else None

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

    def evaluate(self) -> Dict[str, Any]:
        warnings = []
        score = 100

        schema = validate_growth_schema(self.df)

        # Check 1: Missing values in core columns
        missing_outcome = self.df[self.outcome_col].isna().sum()
        if missing_outcome > 0:
            warnings.append(f"Found {missing_outcome} missing values in '{self.outcome_col}'.")
            score -= (missing_outcome / len(self.df)) * 50

        # Check 2: Sparsity / Zero outcomes
        zero_outcomes = (self.df[self.outcome_col] == 0).sum()
        sparsity_ratio = zero_outcomes / len(self.df)
        if sparsity_ratio > 0.2:
            warnings.append(f"High sparsity: {sparsity_ratio:.1%} of outcome rows are zero.")
            score -= 10
        elif sparsity_ratio > 0.05:
            warnings.append(f"Moderate sparsity: {sparsity_ratio:.1%} of outcome rows are zero.")
            score -= 5

        # Check 3: Date Gaps per Geo
        geos = self.df[self.geo_col].unique()
        has_gaps = False
        for g in geos:
            geo_dates = self.df[self.df[self.geo_col] == g][self.date_col].sort_values()
            expected_days = (geo_dates.max() - geo_dates.min()).days + 1
            actual_days = len(geo_dates)
            if actual_days < expected_days:
                has_gaps = True
                break
        
        if has_gaps:
            warnings.append("Date gaps detected in one or more geos. Models might interpolate data.")
            score -= 15

        # Check 4: Period length and pre/post variance if the experiment window is known
        if self.pre_start is not None and self.pre_end is not None:
            pre_days = (self.pre_end - self.pre_start).days + 1
            if pre_days < 14:
                warnings.append(f"Pre-period is too short ({pre_days} days). Use at least 14 days.")
                score -= 20
            elif pre_days < 28:
                warnings.append(f"Pre-period is short ({pre_days} days). Four or more weeks is preferred.")
                score -= 8

        if self.post_start is not None and self.post_end is not None:
            post_days = (self.post_end - self.post_start).days + 1
            if post_days < 7:
                warnings.append(f"Treatment period is very short ({post_days} days).")
                score -= 12
            elif post_days < 14:
                warnings.append(f"Treatment period is short ({post_days} days).")
                score -= 6

        if all(v is not None for v in [self.pre_start, self.pre_end, self.post_start, self.post_end]):
            pre = self.df[
                (self.df[self.date_col] >= self.pre_start)
                & (self.df[self.date_col] <= self.pre_end)
            ]
            post = self.df[
                (self.df[self.date_col] >= self.post_start)
                & (self.df[self.date_col] <= self.post_end)
            ]
            pre_var = pd.to_numeric(pre[self.outcome_col], errors="coerce").var()
            post_var = pd.to_numeric(post[self.outcome_col], errors="coerce").var()
            if pre_var and post_var and pre_var > 0 and post_var / pre_var > 3:
                warnings.append("Post-period outcome variance is much higher than pre-period variance.")
                score -= 10

        # Check 5: Outcome volatility and outliers
        outcome = pd.to_numeric(self.df[self.outcome_col], errors="coerce").dropna()
        if len(outcome) > 2:
            mean = outcome.mean()
            std = outcome.std()
            if mean > 0 and std / mean > 1.0:
                warnings.append("High outcome volatility detected. Wider uncertainty is likely.")
                score -= 10

            q1 = outcome.quantile(0.25)
            q3 = outcome.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((outcome < q1 - 3 * iqr) | (outcome > q3 + 3 * iqr)).sum()
                if outliers:
                    warnings.append(f"Detected {int(outliers)} potential outcome outliers.")
                    score -= min(10, int(outliers))

        # Check 6: Cost/Input quality
        if self.cost_col and self.cost_col != "None" and self.cost_col in self.df.columns:
            missing_cost = self.df[self.cost_col].isna().sum()
            if missing_cost > 0:
                warnings.append(f"Found {missing_cost} missing values in input col '{self.cost_col}'.")
                score -= 10

            zero_spend = (pd.to_numeric(self.df[self.cost_col], errors="coerce").fillna(0) == 0).mean()
            if zero_spend > 0.3:
                warnings.append(f"High zero-input periods: {zero_spend:.1%} of rows have zero {self.cost_col}.")
                score -= 8
                
        # Limit score
        score = max(0, min(100, int(score)))

        if score >= 90:
            status = "Excellent"
        elif score >= 70:
            status = "Good"
        elif score >= 50:
            status = "Fair"
        else:
            status = "Poor"

        return {
            "score": score,
            "status": status,
            "warnings": warnings,
            "schema": {
                "missing_required": schema.missing_required,
                "missing_recommended": schema.missing_recommended,
            },
        }
