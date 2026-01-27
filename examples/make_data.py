import pandas as pd
import numpy as np
from pathlib import Path

def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-10-01", end="2025-01-21", freq="D")
    geos = ["Lagos", "Ibadan", "Abeokuta", "Benin"]
    
    data = []
    
    # Base trends
    t = np.arange(len(dates))
    trend = 100 + 0.1 * t
    
    # Seasonality (weekly)
    seasonality = 10 * np.sin(2 * np.pi * t / 7)
    
    # Post period start index
    post_start_date = pd.Timestamp("2025-01-01")
    
    for geo in geos:
        # Geo modification
        geo_effect = np.random.normal(0, 5)
        
        # Noise
        noise = np.random.normal(0, 2, size=len(dates))
        
        # Outcome
        y = trend + seasonality + geo_effect + noise
        
        # Add Lift to Lagos in Post period
        if geo == "Lagos":
            is_post = dates >= post_start_date
            lift = 10  # constant lift
            y[is_post] += lift
            
        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "geo": geo,
                "outcome": y[i]
            })
            
    df = pd.DataFrame(data)
    
    output_dir = Path("examples/geo_lift_basic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "data.csv", index=False)
    print(f"Generated {output_dir}/data.csv")

if __name__ == "__main__":
    generate_data()
