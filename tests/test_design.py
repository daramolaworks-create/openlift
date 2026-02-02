import pytest
import pandas as pd
import numpy as np
from openlift.core.design import GeoMatcher, PowerAnalysis
from datetime import date, timedelta

# Create synthetic data for testing
@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    
    # Target: slight trend + noise
    # Control 1: similar to Target
    # Control 2: different
    
    np.random.seed(42)
    t = np.arange(100)
    
    target = 100 + t * 0.5 + np.random.normal(0, 5, 100)
    control_perfect = 100 + t * 0.5 + np.random.normal(0, 5, 100) # Similar
    control_bad = 200 - t * 0.2 + np.random.normal(0, 10, 100) # Different trend
    
    df = pd.DataFrame({
        "date": np.tile(dates, 3),
        "geo": ["Target"]*100 + ["Control_Good"]*100 + ["Control_Bad"]*100,
        "outcome": np.concatenate([target, control_perfect, control_bad])
    })
    
    return df

def test_geo_matcher_dtw(sample_data):
    matcher = GeoMatcher(sample_data, "date", "geo", "outcome")
    matches = matcher.find_controls("Target", method="dtw", lookback_days=50)
    
    # Control_Good should be closer (lower distance) than Control_Bad
    geos = [m[0] for m in matches]
    scores = {m[0]: m[1] for m in matches}
    
    assert "Control_Good" in geos
    assert "Control_Bad" in geos
    assert scores["Control_Good"] < scores["Control_Bad"]

def test_geo_matcher_euclidean(sample_data):
    matcher = GeoMatcher(sample_data, "date", "geo", "outcome")
    matches = matcher.find_controls("Target", method="euclidean", lookback_days=50)
    
    scores = {m[0]: m[1] for m in matches}
    assert scores["Control_Good"] < scores["Control_Bad"]

def test_power_analysis_simulation(sample_data):
    # We need enough history. Data has 100 days.
    # Lookback 30, Duration 10.
    pa = PowerAnalysis(sample_data, "date", "geo", "outcome")
    
    # 1. High lift (20%) -> Should be detected
    res_high = pa.simulate_power(
        "Target", 
        ["Control_Good"], 
        effect_size_pct=0.20, 
        lookback_days=30,
        test_duration_days=10,
        simulations=20
    )
    assert res_high["power"] > 0.0 # Should have some power
    
    # 2. Tiny lift (0.1%) -> Should NOT be detected easily
    res_low = pa.simulate_power(
        "Target", 
        ["Control_Good"], 
        effect_size_pct=0.001, 
        lookback_days=30,
        test_duration_days=10,
        simulations=20
    )
    assert res_low["power"] < res_high["power"]

def test_recommend_duration(sample_data):
    pa = PowerAnalysis(sample_data, "date", "geo", "outcome")
    
    # Try to find duration for moderate lift
    # Note: data only has 100 days, so we can't test huge durations
    duration = pa.recommend_duration(
        "Target", 
        ["Control_Good"], 
        effect_size_pct=0.20,
        target_power=0.5, # Lower target for small sample test
        max_duration=30
    )
    
    assert duration > 0 # Should find a duration
