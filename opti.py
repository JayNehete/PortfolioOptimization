import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_mv(mu, Sigma, x0=None, risk_aversion=5.0, lb=0.0, ub=0.30, turnover=0.50):
    n = len(mu)
    
    mu = np.asarray(mu, dtype=float).flatten()
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = np.where(np.isfinite(Sigma), Sigma, 0.0)
    Sigma = 0.5 * (Sigma + Sigma.T)
    
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(x0, dtype=float).flatten()
    
    with gp.Env(params={'OutputFlag': 0}) as env, gp.Model('portfolio', env=env) as m:
        x = m.addVars(n, lb=lb, ub=ub, name='x')
        m.addConstr(sum(x[i] for i in range(n)) == 1.0, 'budget')
        
        buy = m.addVars(n, lb=0.0, name='buy')
        sell = m.addVars(n, lb=0.0, name='sell')
        
        for i in range(n):
            m.addConstr(x[i] - x0[i] == buy[i] - sell[i], 'trade_' + str(i))
        
        m.addConstr(sum(buy[i] + sell[i] for i in range(n)) <= turnover, 'turnover')
        
        portfolio_risk = gp.QuadExpr()
        for i in range(n):
            for j in range(n):
                portfolio_risk += Sigma[i, j] * x[i] * x[j]
        
        portfolio_return = gp.LinExpr()
        for i in range(n):
            portfolio_return += mu[i] * x[i]
        
        m.setObjective(portfolio_risk - (1.0 / max(0.01, risk_aversion)) * portfolio_return, GRB.MINIMIZE)
        m.optimize()
        
        if m.Status != GRB.OPTIMAL:
            return x0
        
        return np.array([x[i].X for i in range(n)])
