import numpy as np
import pandas as pd


def estimate_covariance(data, window=None):
    if window is not None:
        data = data[-window:]  # Consider only the last 'window' observations
    return data.cov()


def sensitivity_analysis(data, periods, include_stressed=False):
    results = {}
    for period in periods:
        if include_stressed:
            # Manipulate data to include stressed conditions
            stressed_data = apply_stress_test(data, stress_factors={'SP500': -0.2, 'Gold USD': 0.15})
            cov_matrix = estimate_covariance(stressed_data, window=period)
        else:
            cov_matrix = estimate_covariance(data, window=period)

        portfolio_var, _ = compute_portfolio_var(data, weights, cov_matrix)
        VaR, ES = compute_VAR_ES_normal(portfolio_var, confidence_level=0.975)
        results[period] = (VaR, ES)
    return results


def plot_normal_fit(returns):
    mean, std = np.mean(returns), np.std(returns)
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=False, stat="density", linewidth=0)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mean, std)
    plt.title(title)
    plt.show()


def plot_qq(returns):
    sm.qqplot(returns, line='45')
    plt.show()
