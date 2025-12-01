import os
from data import fetch_prices, compute_returns, align_universe
from backtest import walk_forward

def main():
    tickers = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','JPM','UNH','HD','PG']
    start, end = '2016-01-01', '2024-12-31'
    
    prices = fetch_prices(tickers, start, end, cache_dir='data')
    print('Loaded prices: ' + str(prices.shape) + ', ' + str(prices.index[0]) + ' to ' + str(prices.index[-1]))
    
    rets = compute_returns(prices)
    rets = align_universe(rets, min_obs=400)
    prices = prices[rets.columns]
    
    print('Running backtest on ' + str(len(prices.columns)) + ' assets over ' + str(len(prices)) + ' days')
    
    curve, hist = walk_forward(prices, start_ix=500, step=5, lookback=250, 
                               risk_aversion=6.0, turnover=0.50)
    
    os.makedirs('data/outputs', exist_ok=True)
    curve.to_csv('data/outputs/equity_curve.csv')
    hist.to_csv('data/outputs/weights_history.csv')
    
    print('\nFinal equity: ' + str(round(curve.iloc[-1], 4)))
    print('Saved results to data/outputs/')
    
    if len(curve) > 1:
        total_return = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
        print('Total return: ' + str(round(total_return, 2)) + '%')

if __name__ == '__main__':
    main()
