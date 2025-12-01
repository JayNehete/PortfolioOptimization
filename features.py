import pandas as pd

def rolling_features(prices, window=60):
    r = prices.pct_change()
    vol = r.rolling(window).std()
    mom = (prices / prices.rolling(window).mean() - 1.0)
    
    dfs = []
    for name, df in [('ret_1', r), ('vol_60', vol), ('mom_60', mom)]:
        dfs.append(df.stack().rename(name))
    
    X = pd.concat(dfs, axis=1).dropna()
    return X

def make_xy(features_panel, returns_panel, horizon=1):
    y = returns_panel.shift(-horizon).stack().rename('y')
    Z = features_panel.join(y, how='inner').dropna()
    return Z
