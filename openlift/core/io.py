import pandas as pd

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
        
    # Check for missing outcomes
    if df[outcome_col].isnull().any():
        raise ValueError(f"Found missing values in outcome column '{outcome_col}'")
        
    # Pivot to wide format (date x geo)
    # This automatically checks for duplicate (date, geo) pairs because pivot raises error on duplicates
    # unless aggfunc is specified. We want it to raise error.
    try:
        df_wide = df.pivot(index=date_col, columns=geo_col, values=outcome_col)
    except ValueError as e:
        raise ValueError(f"Duplicate entries found for same date and geo. Each geo must have exactly one row per date. Details: {e}")
        
    df_wide = df_wide.sort_index()
    
    return df_wide
