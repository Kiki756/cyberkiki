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
output_dir = '/output'

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
    from statsmodels.tsa.stattools import acf
    autocorr_results = {}
    for asset in breaches.columns:
        autocorr = acf(breaches[asset], fft=True, nlags=1)[1]  # Only look at the first lag
        autocorr_results[asset] = autocorr
    return autocorr_results

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

def compute_multi_day_var_check(returns, days, confidence_level):
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


def estimate_covariance(data, window=None):
    if window is not None:
        data = data[-window:]  # Consider only the last 'window' observations
    return data.cov()


def calculate_covariance_matrices(data, periods):
    covariance_results = {}

    for period in periods:
        # Subset data for the specified period
        period_data = data.tail(period)  # Ensure this subsetting is correct
        returns = log_returns(period_data)        # Calculate log returns for the period data
        cov_matrix = returns.cov()        # Calculate covariance matrix from the returns
        covariance_results[period] = cov_matrix   # Store the covariance matrix in the dictionary with the period as the key

    return covariance_results


def calculate_var_es_for_periods(data, periods, weights, confidence_level=0.975):
    results = {}
    normal_results = {}

    for period in periods:
        # Subset data for the specified period
        period_data = data.tail(period)  # Ensure this subsetting is correct

        returns = log_returns(period_data)  # Recalculate returns on normal data

        # Ensure weights are recalculated or correctly aligned with the current returns data
        current_weights = compute_risk_parity_weights(returns)

        portfolio_var, _ = compute_portfolio_var(returns, current_weights)  # Use recalculated returns and weights
        VaR, ES = compute_VAR_ES_normal(portfolio_var, confidence_level)
        results[period] = (VaR, ES)
        normal_results[period] = (VaR, ES)

    return results, normal_results


def plot_data_analysis(returns, plot_type='hist', asset=None):
    plt.figure(figsize=(10, 6))

    if plot_type == 'hist':
        sns.histplot(returns, kde=False, stat="density", linewidth=0, color='blue', label='Histogram')
        mean, std = np.mean(returns), np.std(returns)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
        plt.plot(x, p, 'k', linewidth=2, label='Normal Fit')
        plt.title(f'Histogram and Normal Fit for {asset}')
        plt.legend()

    elif plot_type == 'qq':
        sm.qqplot(returns, line='45', fit=True)
        plt.title(f'QQ Plot for {asset}')

    if asset:
        filename = f'{asset}_{plot_type}_plot.png'
    else:
        filename = f'default_{plot_type}_plot.png'

    plt.savefig(os.path.join(output_dir, filename))

    # Close the figure to free up memory and avoid overlap on subsequent plots
    plt.close()


if __name__ == '__main__':
    # Define the output directory
    output_dir = os.path.join(repo_dir, 'output')

    ### Bullet 1 ### Data prep

    # Load and prepare data
    data = load_data(data_file)
    returns = log_returns(data)
    weights = compute_risk_parity_weights(returns)

    ### Bullet 2.1 ### Normal Distribution
    # Plot and save log returns
    plot_returns(returns)

    periods = [252, 756, 1260]  # Example periods in trading days (1 year, 3 years, 5 years)
    results, normal_results = calculate_var_es_for_periods(data, periods, weights)
    covariance_results = calculate_covariance_matrices(data, periods)

    # Print results to see how they differ
    print("Normal Conditions:")
    for period, (VaR, ES) in results.items():
        print(f"For {period} days: VaR = {VaR}, ES = {ES}")

    # Distribution Fits and QQ Plots
    for asset in returns.columns:  # Example assets
        plot_data_analysis(returns[asset], plot_type='qq', asset=asset)


    ### Bullet 2.2 ### student-t distribution
    portfolio_var, _ = compute_portfolio_var(returns, weights)

    dof_values = [3, 4, 5, 6]    # Define the degrees of freedom
    var_es_results = {}     # Initialize a dictionary to store the results

    # Loop over the degrees of freedom
    for dof in dof_values:
        # Compute VaR and ES for the current degree of freedom
        VaR, ES = compute_VAR_ES_t(portfolio_var, 0.975, dof)

        # Store the results in the dictionary
        var_es_results[dof] = (VaR, ES)

    # Print the results
    for dof, (VaR, ES) in var_es_results.items():
        print(f"For dof = {dof}: VaR = {VaR}, ES = {ES}")


    ### Bullet 2.3 ### Historical Simulation
    VaR_hist, ES_hist = historical_simulation(returns, 0.975)
    print(f"Historical Simulation VaR: {VaR_hist}, ES: {ES_hist}")

    ### Bullet 2.4 ### Constant Conditional Correlation method (GARCH(1,1), normal innovations for risk factor

    # GARCH and Constant Correlation
    models = ccc_garch(returns, weights)
    VaR_ccc, ES_ccc = compute_var_es_ccc(models, np.eye(len(returns.columns)), 0.975)
    print(f"CCC GARCH VaR: {VaR_ccc}, ES: {ES_ccc}")

    ### Bullet 2.5 ### Filtered Historical Simulation method with EWMA for each risk factor.
    # Filtered Historical Simulation
    VaR_fhs, ES_fhs = filtered_historical_simulation(returns, 0.94, 0.975)  # Example lambda=0.94
    print(f"Filtered Historical Simulation VaR: {VaR_fhs}, ES: {ES_fhs}")


    ### Bullet 3 ### backtesting VaR and ES. Plots.
    # Backtesting
    breaches = count_var_breaches(returns, VaR_hist)[1]

    # Independence Test
    independence_results = test_breach_independence(breaches)
    for asset, autocorr in independence_results.items():
        print(f"Autocorrelation of breaches for {asset}: {autocorr}")

    ### Bullet 4 ### Empirical 5 day VaR and 10 day VaR, using historical simulation method. compare with sq root rule

    ##################### multi day var here!!!!!!!!!!!!!!!!

    # Multi-day VaR Check
    VaR_5day = compute_multi_day_var_check(returns, 5, 0.975)
    VaR_10day = compute_multi_day_var_check(returns, 10, 0.975)
    print(f"5-Day VaR: {VaR_5day}, 10-Day VaR: {VaR_10day}")


    ### Bullet 5 ### Stress testing

    # Stress Testing
    stress_factors = {
        'SP500': -20,  # 20% market drop
        'Gold USD': 15  # 15% increase in gold price
    }

    stressed_returns = apply_stress_test(returns, stress_factors)
    stressed_VaR, stressed_ES = calculate_stressed_var_es(stressed_returns, weights, 0.975)
    print(f"Stressed VaR: {stressed_VaR}, Stressed ES: {stressed_ES}")

    # Shortfall Comparison
    #compare_shortfalls(returns.iloc[:, 0], VaR_normal, ES_normal)

    # Distribution Fits for all assets using the new function
    for asset in returns.columns:
        plot_data_analysis(returns[asset], plot_type='hist', asset=asset)


    ### Bullet 6 ### check canvas for announcement extra shit

