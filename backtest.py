import numpy as np
import pandas as pd
from features import rolling_features, make_xy
from model import fit_asset_models, predict_mu
from risk import shrink_cov, vectorize_mu
from opti import solve_mv

def walk_forward(prices, start_ix=250, step=5, lookback=200, risk_aversion=5.0, turnover=0.50):
    dates = prices.index.tolist()
    tickers = list(prices.columns)
    n_assets = len(tickers)
    
    rets = prices.pct_change().dropna(how='all')
    feats = rolling_features(prices, window=60)
    panel = make_xy(feats, rets, horizon=1)
    df_map = panel.reset_index().rename(columns={'level_0':'Date', 'level_1':'asset'})
    print(df_map)
    
    equity = [1.0]
    w_current = {t: 1.0/n_assets for t in tickers}
    w_hist = []
    date_hist = []
    rebal_count = 0
    
    for t in range(start_ix, len(dates) - step, step):
        d_end = dates[t]
        d_next_idx = min(t + step, len(dates) - 1)
        d_next = dates[d_next_idx]
        d_train_start = dates[max(0, t - lookback)]
        
        train = df_map[(df_map['Date'] >= d_train_start) & (df_map['Date'] < d_end)]
        test_t = df_map[df_map['Date'] == d_end]
        
        if train.empty or test_t.empty:
            continue
        
        train_panel = train.set_index(['Date','asset'])
        test_panel = test_t.set_index(['Date','asset'])
        
        models = fit_asset_models(train_panel)
        if not models:
            continue
        
        mu_hat = predict_mu(models, test_panel)
        if not mu_hat:
            continue
        
        R_window = rets.loc[d_train_start:d_end]
        Sigma, cols = shrink_cov(R_window)
        if Sigma is None or len(cols) < 2:
            continue
        
        mu_vec = vectorize_mu(mu_hat, cols, fallback=0.0)
        x0_vec = np.array([w_current.get(c, 0.0) for c in cols])
        
        w = solve_mv(mu_vec, Sigma, x0=x0_vec, risk_aversion=risk_aversion, 
                     lb=0.0, ub=0.30, turnover=turnover)
        
        r_slice = rets.loc[d_end:d_next, cols]
        if len(r_slice) > 1:
            r_slice = r_slice.iloc[1:]
        
        if r_slice.empty:
            continue
        
        rp = (r_slice @ w).fillna(0.0)
        cumulative_return = (1.0 + rp).prod()
        equity.append(equity[-1] * cumulative_return)
        
        w_current = {cols[i]: w[i] for i in range(len(cols))}
        w_hist.append(pd.Series(w, index=cols, name=d_end))
        date_hist.append(d_end)
        rebal_count += 1
    
    print('Total rebalances: ' + str(rebal_count))
    history = pd.DataFrame(w_hist) if w_hist else pd.DataFrame()
    curve = pd.Series(equity, name='equity')
    return curve, history
