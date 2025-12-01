import numpy as np
from sklearn.covariance import LedoitWolf

def shrink_cov(returns_window):
    R = returns_window.dropna(how='any', axis=1)
    if R.shape[1] < 2:
        return None, None
    
    lw = LedoitWolf().fit(R.values)
    Sigma = lw.covariance_
    cols = list(R.columns)
    return Sigma, cols

def vectorize_mu(mu_map, cols, fallback=0.0):
    return np.array([mu_map.get(c, fallback) for c in cols], dtype=float)
