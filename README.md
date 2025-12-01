# ML Portfolio Optimization

This project builds a simple end-to-end system that:
- Uses machine learning to predict next period returns for a set of large cap stocks from rolling features like return, volatility, momentum.
- Uses those predictions and a Ledoit–Wolf covariance matrix in a mean–variance portfolio optimization model solved with Gurobi.
- Enforces realistic constraints: weights sum to 1, each stock is capped, and turnover per rebalance is limited.

The code runs a walk-forward backtest: at each rebalance date it retrains the ML models, re-estimates risk, solves the optimization, and updates the portfolio.
