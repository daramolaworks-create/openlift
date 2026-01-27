import numpy as np
from openlift.core.model_pymc import build_model, fit_model

def test_model_smoke():
    # Synthetic small data
    T = 50
    K = 3
    np.random.seed(42)
    X_pre = np.random.randn(T, K)
    dow_pre = np.random.randint(0, 7, size=T)
    # Simple explicit relation
    y_pre = 2 + X_pre @ np.array([1.0, -0.5, 0.2]) + np.random.randn(T)*0.1
    
    model = build_model(y_pre, X_pre, dow_pre)
    
    # Fast sampling
    idata = fit_model(model, draws=10, tune=10, chains=1, target_accept=0.8)
    assert idata is not None
    assert "posterior" in idata
