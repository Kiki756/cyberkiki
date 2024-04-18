import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.stats import norm, t
from arch import arch_model

# Full path to the 'repo' directory
repo_dir = ''
data_file = 'data/portfolio.csv'

def load_data(data_file):
    OPEN_FILE = os.path.join(repo_dir, 'portfolio.csv')
    data = pd.read_csv(OPEN_FILE, index_col=0, header=0)
    data.columns = data.columns.str.strip()     # Remove leading and trailing spaces from column names
    for col in data.columns:     # Remove commas and convert all columns to numeric format
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
    data.index = pd.to_datetime(data.index)    # Convert dates to datetime format

    return data


def log_returns(data):
    # Select columns
    selected_columns = ['Gold USD', 'Gold EUR', 'JPM Close', 'SP500', 'Siemens Close', 'XR', 'EUROSTOXX Close']
    data_selected = data[selected_columns].copy()
    returns = np.log(data_selected / data_selected.shift(1)).dropna()
    returns = 100 *  returns
    return returns


def compute_risk_parity_weights(returns):
    # Calculate the annualized volatility for each asset
    volatility = returns.std() * np.sqrt(252)  # Assuming 252 trading days per year
    inverse_volatility = 1 / volatility    # Inverse of volatility
    weights = inverse_volatility / inverse_volatility.sum()    # Normalize weights
    return weights


def plot_returns(returns):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=returns, dashes=False)
    plt.title('Log returns')
    plt.savefig(os.path.join(output_dir, 'log_returns.png'))
    # plt.show()


def compute_portfolio_var(returns, weights):
    cov_matrix = returns.cov()    # Calculate covariance matrix
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))    # Calculate portfolio variance

    return portfolio_var, cov_matrix


def compute_VAR_ES_normal(portfolio_var, confidence_level):
    VaR = norm.ppf(1 - confidence_level) * np.sqrt(portfolio_var)
    ES = norm.expect(lambda x: x, loc=0, scale=np.sqrt(portfolio_var), lb=norm.ppf(1 - confidence_level))
    return VaR, ES


def compute_VAR_ES_t(portfolio_var, confidence_level, df):
    # VaR using the Student-t distribution
    VaR_t = t.ppf(1 - confidence_level, df) * np.sqrt(portfolio_var)

    # ES using the Student-t distribution (requires integration or a more explicit form)
    # We need to compute the expected shortfall as the expected return on the tail of the distribution.
    x = np.linspace(t.ppf(0.001, df), t.ppf(0.999, df), 1000)
    pdf = t.pdf(x, df)
    cdf = t.cdf(x, df)
    conditional_loss = x * pdf / (1 - t.cdf(t.ppf(1 - confidence_level, df), df))
    ES_t = (conditional_loss.sum() / len(x)) * np.sqrt(portfolio_var)

    return VaR_t, ES_t

def historical_simulation(returns, confidence_level):
    aggregated_returns = returns.sum(axis=1)  # Summing returns across all columns for each row
    sorted_returns = np.sort(aggregated_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = -sorted_returns[index]
    ES = -sorted_returns[:index].mean()
    return VaR, ES

def ccc_garch(returns, weights):
    scaled_returns = returns
    models = {}
    for column in weights.keys():
        model = arch_model(weights[column], mean='Zero', vol='GARCH', p=1, q=1)
        res = model.fit(update_freq=10, disp='off')
        models[column] = res
    return models

def compute_var_es_ccc(models, correlation_matrix, confidence_level):
    portfolio_var = 0
    for i, (name1, model1) in enumerate(models.items()):
        sigma1 = model1.conditional_volatility[-1] ** 2
        for j, (name2, model2) in enumerate(models.items()):
            sigma2 = model2.conditional_volatility[-1] ** 2
            portfolio_var += correlation_matrix[i, j] * sigma1 * sigma2
    VaR = norm.ppf(1 - confidence_level) * np.sqrt(portfolio_var)
    ES = norm.expect(lambda x: x, loc=0, scale=np.sqrt(portfolio_var), lb=norm.ppf(1 - confidence_level))
    return VaR, ES


def filtered_historical_simulation(returns, lambda_, confidence_level):
    n = len(returns)
    weights = np.full(n, lambda_)
    weights = np.power(weights, np.arange(n)[::-1])
    weights /= weights.sum()

    # Applying weights across each column (asset)
    weighted_returns = returns.multiply(weights, axis=0)  # Use DataFrame.multiply to apply weights per row

    sorted_returns = weighted_returns.sum(
        axis=1).sort_values()  # Sum weighted returns across assets for each time point and sort
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = -sorted_returns.iloc[index]
    ES = -sorted_returns.iloc[:index].mean()
    return VaR, ES

def count_var_breaches(returns, VaR):
    losses = -returns
    breaches = losses > VaR
    return np.sum(breaches), breaches

def plot_breach_clustering(breaches, dates):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, breaches, label='VaR Breach (1 if breach, 0 otherwise)')
    plt.xlabel('Date')
    plt.ylabel('VaR Breach')
    plt.title('Clustering of VaR Breaches Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_breach_independence(breaches):
    """Calculate the autocorrelation of breaches to test for independence."""
    from statsmodels.tsa.stattools import acf
    autocorr = acf(breaches, fft=True, nlags=1)[1]  # Only look at the first lag
    return {'autocorrelation': autocorr}

def apply_stress_test(returns, stress_factors):
    stressed_returns = returns.copy()
    for asset, factor in stress_factors.items():
        # Convert stress factor from percentage to the scale of the returns
        adjusted_factor = factor / 100  # Convert from percentage if necessary
        if asset in stressed_returns.columns:
            stressed_returns[asset] *= (1 + adjusted_factor)
    return stressed_returns

def calculate_stressed_var_es(returns, weights, confidence_level):
    portfolio_var, _ = compute_portfolio_var(returns, weights)
    VaR, ES = compute_VAR_ES_normal(portfolio_var, confidence_level)
    return VaR, ES

def compute_multi_day_var(returns, days, confidence_level):
    # Calculate daily VaR and scale it by square root of days
    daily_var = historical_simulation(returns, confidence_level)[0]
    multi_day_var = daily_var * np.sqrt(days)
    return multi_day_var

def compare_shortfalls(returns, VaR, ES):
    # Function to compare actual losses with expected shortfall
    actual_losses = -returns[returns < -VaR]
    average_actual_shortfall = actual_losses.mean() if not actual_losses.empty else 0
    print(f"Expected Shortfall: {ES}, Average Actual Shortfall: {average_actual_shortfall}")
    return average_actual_shortfall

def plot_distribution_fits(returns):
    sns.histplot(returns, kde=True, stat="density", linewidth=0)
    plt.show()
    sm.qqplot(returns, line ='45')
    plt.show()


def estimate_covariance(data, window=None):
    if window is not None:
        data = data[-window:]  # Consider only the last 'window' observations
    return data.cov()


def sensitivity_analysis(data, periods, weights, confidence_level=0.975):
    results = {}
    for period in periods:
        # Subset data for the specified period
        period_data = data.tail(period)  # Ensure this subsetting is correct

        returns = log_returns(period_data)  # Recalculate returns on normal data

        # Ensure weights are recalculated or correctly aligned with the current returns data
        current_weights = compute_risk_parity_weights(returns)

        portfolio_var, _ = compute_portfolio_var(returns, current_weights)  # Use recalculated returns and weights
        VaR, ES = compute_VAR_ES_normal(portfolio_var, confidence_level)
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


if __name__ == '__main__':
    # Define the output directory
    output_dir = os.path.join(repo_dir, 'output')

    # Load and prepare data
    data = load_data(data_file)
    returns = log_returns(data)
    weights = compute_risk_parity_weights(returns)

    # Plot and save log returns
    plot_returns(returns)

    periods = [252, 756, 1260]  # Example periods in trading days (1 year, 2 years, 3 years)
    sensitivity_results = sensitivity_analysis(data, periods, weights, include_stressed=False)
    sensitivity_results_stressed = sensitivity_analysis(data, periods, weights, include_stressed=True)

    # Print results to see how they differ
    print("Normal Conditions:")
    for period, (VaR, ES) in sensitivity_results.items():
        print(f"For {period} days: VaR = {VaR}, ES = {ES}")
    print("Stressed Conditions:")
    for period, (VaR, ES) in sensitivity_results_stressed.items():
        print(f"For {period} days: VaR = {VaR}, ES = {ES}")

    # Distribution Fits and QQ Plots
    for asset in returns.columns:  # Example assets
        plot_normal_fit(returns[asset])
        plot_qq(returns[asset])

    # Historical Simulation
    VaR_hist, ES_hist = historical_simulation(returns, 0.975)
    print(f"Historical Simulation VaR: {VaR_hist}, ES: {ES_hist}")

    # GARCH and Constant Correlation
    models = ccc_garch(returns, weights)
    VaR_ccc, ES_ccc = compute_var_es_ccc(models, np.eye(len(returns.columns)), 0.975)
    print(f"CCC GARCH VaR: {VaR_ccc}, ES: {ES_ccc}")

    # Filtered Historical Simulation
    VaR_fhs, ES_fhs = filtered_historical_simulation(returns, 0.94, 0.975)  # Example lambda=0.94
    print(f"Filtered Historical Simulation VaR: {VaR_fhs}, ES: {ES_fhs}")

    # Backtesting
    num_breaches, breaches = count_var_breaches(returns.iloc[:, 0], VaR_normal)
    print(f"Number of VaR breaches: {num_breaches}")
    plot_breach_clustering(breaches, returns.index)

    # Independence Test
    independence_results = test_breach_independence(breaches)
    print(f"Autocorrelation of breaches: {independence_results['autocorrelation']}")

    # Stress Testing
    stress_factors = {
        'SP500': -20,  # 20% market drop
        'Gold USD': 15  # 15% increase in gold price
    }

    stressed_returns = apply_stress_test(returns, stress_factors)
    stressed_VaR, stressed_ES = calculate_stressed_var_es(stressed_returns, weights, 0.975)
    print(f"Stressed VaR: {stressed_VaR}, Stressed ES: {stressed_ES}")

    # Multi-day VaR
    VaR_5day = compute_multi_day_var(returns, 5, 0.975)
    VaR_10day = compute_multi_day_var(returns, 10, 0.975)
    print(f"5-Day VaR: {VaR_5day}, 10-Day VaR: {VaR_10day}")

    # Shortfall Comparison
    compare_shortfalls(returns.iloc[:, 0], VaR_normal, ES_normal)

    # Distribution Fits
    plot_distribution_fits(returns.iloc[:, 0])

