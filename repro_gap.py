import pandas as pd
import numpy as np
from openlift.core.features import make_features

def test_missing_data_interpolation():
    # Create data with a gap
    # Dates: 1, 2, (gap), 4, 5
    dates = [
        "2024-01-01", 
        "2024-01-02", 
        # Missing 2024-01-03
        "2024-01-04", 
        "2024-01-05"
    ]
    df = pd.DataFrame({
        "date": pd.to_datetime(dates * 2), # 4 dates * 2 geos = 8 rows
        "geo": ["A"]*4 + ["B"]*4,
        "outcome": [10.0, 12.0, 16.0, 18.0] * 2
    })
    
    # Wide format 
    # Index: 2024-01-01, 02, 04, 05
    df_wide = df.pivot(index="date", columns="geo", values="outcome")
    
    # We insist on a range that INCLUDES the missing date
    pre_start = pd.Timestamp("2024-01-01")
    pre_end = pd.Timestamp("2024-01-05")
    
    # Post period (dummy, but must exist in data or we get valid missing error)
    post_start = pd.Timestamp("2024-01-05")
    post_end = pd.Timestamp("2024-01-05")
    
    # This should FAIL without interpolation, but PASS with interpolation
    try:
        make_features(
            df_wide, 
            test_geo="A", 
            control_geos=["B"], 
            pre_start=pre_start, 
            pre_end=pre_end,
            post_start=post_start, 
            post_end=post_end
        )
        print("✅ SUCCESS: Interpolation handled the gap!")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_missing_data_interpolation()
