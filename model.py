import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _align_cols(df, cols):
    X = df.copy()
    if 'y' in X.columns:
        X = X.drop(columns=['y'])
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[cols]

def fit_asset_models(train_panel, ridge_alpha=2.0):
    models = {}
    for asset in train_panel.index.get_level_values(1).unique():
        A = train_panel.xs(asset, level=1)
        if len(A) < 150 or 'y' not in A.columns:
            continue
        
        Xdf = A.drop(columns=['y'])
        y = A['y'].values
        feat_cols = list(Xdf.columns)
        
        pipe = Pipeline([('sc', StandardScaler()), ('rd', Ridge(alpha=ridge_alpha))])
        pipe.fit(Xdf.values, y)
        pipe.feat_cols_ = feat_cols
        models[asset] = pipe
    
    return models

def predict_mu(models, X_t_panel):
    mu = {}
    for a in X_t_panel.index.get_level_values(1).unique():
        if a not in models:
            continue
        xdf = X_t_panel.xs(a, level=1)
        if xdf.empty:
            continue
        
        pipe = models[a]
        feat_cols = getattr(pipe, 'feat_cols_', list(xdf.columns))
        xdf_aligned = _align_cols(xdf, feat_cols)
        
        yhat = pipe.predict(xdf_aligned.values)
        mu[a] = float(yhat[-1])
    
    return mu
