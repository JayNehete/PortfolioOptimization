import os
import pandas as pd
import yfinance as yf

def fetch_prices(tickers, start, end, cache_dir='data'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'prices_' + str(len(tickers)) + '_' + start + '_' + end + '.csv')
    
    if os.path.exists(cache_file):
        print('Loading cached prices from ' + cache_file)
        df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
        return df
    
    print('Downloading ' + str(len(tickers)) + ' tickers from ' + start + ' to ' + end)
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    df = df.dropna(how='all')
    
    df.to_csv(cache_file)
    print('Saved to ' + cache_file)
    return df

def compute_returns(prices):
    return prices.pct_change().dropna(how='all')

def align_universe(rets, min_obs=400):
    valid = rets.count() >= min_obs
    keep = list(valid[valid].index)
    print('Keeping ' + str(len(keep)) + ' assets with >= ' + str(min_obs) + ' observations')
    return rets[keep]