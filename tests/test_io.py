import pytest
import pandas as pd
from openlift.core.io import load_data

def test_load_data_valid(tmp_path):
    d = tmp_path / "test.csv"
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "geo": ["A", "B", "A", "B"],
        "outcome": [1, 2, 3, 4]
    })
    df.to_csv(d, index=False)
    
    res = load_data(str(d), "date", "geo", "outcome")
    assert res.shape == (2, 2)
    assert "A" in res.columns
    assert "B" in res.columns

def test_load_data_missing_col(tmp_path):
    d = tmp_path / "test_missing.csv"
    df = pd.DataFrame({"date": [], "geo": []})
    df.to_csv(d, index=False)
    
    with pytest.raises(ValueError, match="Missing columns"):
        load_data(str(d), "date", "geo", "outcome")

def test_load_data_duplicates(tmp_path):
    d = tmp_path / "test_dupe.csv"
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "geo": ["A", "A"],
        "outcome": [1, 2]
    })
    df.to_csv(d, index=False)
    
    with pytest.raises(ValueError, match="Duplicate entries"):
        load_data(str(d), "date", "geo", "outcome")
