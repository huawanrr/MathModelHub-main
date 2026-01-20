import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
#Weight Determination (AHP + Entropy)
def calculate_combined_weights(A_matrix, data_matrix):
    n = A_matrix.shape[0]
    g_i = np.power(np.prod(A_matrix, axis=1), 1/n) 
    w_ahp = g_i / np.sum(g_i)
    Aw = np.dot(A_matrix, w_ahp)
    lambda_max = np.mean(Aw / w_ahp)
    CI = (lambda_max - n) / (n - 1)  # RI = 1.12 for n=5 (Standard Lookup)
    range_val = data_matrix.max() - data_matrix.min()
    p_ij = ((data_matrix - data_matrix.min()) / range_val)
    p_ij = p_ij / p_ij.sum()
    k = 1 / np.log(len(data_matrix))
    e_j = -k * np.nansum(p_ij * np.log(p_ij + 1e-10), axis=0)
    w_entropy = (1 - e_j) / np.sum(1 - e_j)
    w_combined = w_ahp * w_entropy 
    return w_combined / w_combined.sum()
#Time Series Prediction (GM(1,1) & Logistic)
def gm11_predict(history, future_steps=10):
    x0 = np.array(history)
    x1 = np.cumsum(x0) 
    z1 = (x1[:-1] + x1[1:]) / 2.0
    B = np.vstack([-z1, np.ones(len(x0)-1)]).T
    Y = x0[1:].reshape((-1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    def f(k): return (x0[0] - b/a) * np.exp(-a * k) + b/a
    preds_x1 = np.array([f(k) for k in range(len(x0) + future_steps)])
    preds_x0 = np.diff(np.insert(preds_x1, 0, 0)) 
    return preds_x0
def logistic_predict(history, future_steps=10):
    t = np.arange(1, len(history) + 1)
    func = lambda t, K, r, t0: K / (1 + np.exp(-r * (t - t0)))
    try:
        p0 = [max(history)*1.5, 0.5, len(history)/2]
        popt, _ = curve_fit(func, t, history, p0=p0, maxfev=5000)
        return func(np.arange(1, len(history) + future_steps + 1), *popt)
    except: return gm11_predict(history, future_steps) # Fallback
#Investment Optimization (Non-Linear Programming)
def optimize_budget(Sigma_cov, weights, betas, total_budget=1.0, lam=1.0):
    def objective(I):
        ret = np.sum(weights * betas * np.log(1 + I))
        risk = (lam / 2) * np.dot(I.T, np.dot(Sigma_cov, I))
        return -(ret - risk) 
    constraints = ({'type': 'eq', 'fun': lambda I: np.sum(I) - total_budget})
    bounds = tuple((0, total_budget) for _ in range(len(weights)))
    res = minimize(objective, x0=[total_budget/len(weights)]*len(weights), 
                   method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else None