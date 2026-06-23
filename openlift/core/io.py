import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    path: str, 
    date_col: str, 
    geo_col: str, 
    outcome_col: str
) -> pd.DataFrame:
    """
    Load data from CSV, validate, and pivot to wide format.
    
    Returns:
        pd.DataFrame: A wide dataframe with dates as index and geos as columns.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")
    
    # Check columns
    missing_cols = [c for c in [date_col, geo_col, outcome_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
        
    # Parse dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        raise ValueError(f"Could not parse date column '{date_col}': {e}")
        
    # Warn on missing outcomes — missing days are later filled with 0 in make_features.
    n_missing = int(df[outcome_col].isnull().sum())
    if n_missing:
        logger.warning(
            "Found %d missing values in outcome column '%s'. "
            "They will be treated as 0 during modelling.",
            n_missing,
            outcome_col,
        )
        
    # Pivot to wide format (date x geo)
    # Aggregate duplicate (date, geo) pairs by summing before pivoting
    df = df.groupby([date_col, geo_col], as_index=False)[outcome_col].sum()
    df_wide = df.pivot(index=date_col, columns=geo_col, values=outcome_col)
    df_wide = df_wide.sort_index()
    
    return df_wide
