import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.stats import norm, t, probplot
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
    returns = returns
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

def compute_portfolio_variance(returns, weights):
    covariance_matrix = np.cov(returns, rowvar=False)
    portfolio_variance = weights.T @ covariance_matrix @ weights
    return portfolio_variance

def plot_qq(variance):
    sm.qqplot(variance, line='45', fit=True)
    plt.title('QQ Plot for Portfolio Variance')
    plt.show()

def calculate_var_es_for_periods(data, periods, weights, confidence_levels=[0.975, 0.99]):
    results = {}

    for period in periods:
        period_data = data.tail(period)  # Subset data for the specified period
        returns = log_returns(period_data)  # Calculate log returns

        portfolio_variance = compute_portfolio_variance(returns, weights)  # Assume weights are constant

        period_results = {}
        for confidence_level in confidence_levels:
            VaR, ES = compute_VAR_ES_normal(portfolio_variance, confidence_level)
            period_results[confidence_level] = {'VaR': VaR, 'ES': ES}

        results[period] = period_results

    return results


def compute_VAR_ES_normal(portfolio_variance, confidence_level):
    std_dev = np.sqrt(portfolio_variance)
    VaR = -norm.ppf(1 - confidence_level) * std_dev
    ES = -std_dev * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
    return VaR, ES


def compute_VAR_ES_t(portfolio_variance, confidence_level, dof):
    std_dev = np.sqrt(portfolio_variance)
    VaR = -t.ppf(1 - confidence_level, df=dof) * std_dev
    ES = -std_dev * (t.pdf(t.ppf(1 - confidence_level, df=dof), df=dof) / (1 - confidence_level)) * ((dof + (t.ppf(1 - confidence_level, df=dof))**2) / (dof - 1))
    return VaR, ES


def historical_simulation(returns, confidence_level):
    aggregated_returns = returns.sum(axis=1)  # Summing returns across all columns for each row
    sorted_returns = np.sort(aggregated_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = -sorted_returns[index]
    ES = -sorted_returns[:index].mean()
    return VaR, ES


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


def plot_data_analysis(data, weights, returns, plot_type='hist', asset=None):
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
        if asset is None:
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            sm.qqplot(portfolio_returns, line='45', fit=True)
            plt.title('QQ Plot for normal Portfolio Returns')
        else:
            sm.qqplot(returns[asset], line='45', fit=True)
            plt.title(f'QQ Plot for normal {asset}')

    if asset:
        filename = f'{asset}_{plot_type}_normal_plot.png'
    else:
        filename = 'portfolio_qq_normal_plot.png'

    plt.savefig(os.path.join(output_dir, filename))

    # Close the figure to free up memory and avoid overlap on subsequent plots
    plt.close()
def plot_portfolio_qq_t(data, weights, dof=5):
    plt.figure(figsize=(10, 6))
    # Calculate portfolio returns
    returns = log_returns(data)
    portfolio_returns = returns.dot(weights)
    # QQ plot for t-distribution
    probplot(portfolio_returns, dist=t(dof), plot=plt)
    plt.title(f'QQ Plot for Portfolio Returns (Student\'s t-distribution, dof={dof})')
    # Save the plot in the output directory
    plt.savefig(os.path.join(output_dir, f'portfolio_qq_t_dof_{dof}.png'))
    plt.show()

def rolling_var(data, confidence_level, dof, method, window=252):

    rolling_vars = pd.Series(index=data.index, dtype=float)

    if method == 'historical':
        for i in range(window, len(data)):
            historical_data = data.iloc[i-window:i]
            rolling_vars[i] = historical_simulation(historical_data, confidence_level)[0]
    elif method == 'normal':
        weights = compute_risk_parity_weights(data)  # assuming weights are pre-defined
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            variance = compute_portfolio_variance(window_data, weights)
            rolling_vars[i] = compute_VAR_ES_normal(variance, confidence_level)[0]
    elif method == 't':
        weights = compute_risk_parity_weights(data)
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            variance = compute_portfolio_variance(window_data, weights)
            rolling_vars[i] = compute_VAR_ES_t(variance, confidence_level, dof)[0]

    return rolling_vars


def rolling_var_normal(returns, window_size, confidence_level):
    # Calculate the rolling standard deviation for the portfolio returns
    rolling_std = returns.rolling(window=window_size).std()

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Calculate the rolling VaR
    rolling_var = -rolling_std * z_score

    # VaR is typically a positive number when representing a loss
    return rolling_var.abs()  # use abs() to convert to positive if needed
def plot_returns_with_var(returns, var_series):

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=returns.sum(axis=1), label='Portfolio Returns')
    sns.lineplot(data=var_series, color='red', label='Rolling VaR')
    plt.title(f'Portfolio Returns and Rolling VaR, % VaR')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'returns_with_var.png'))
    plt.show()




if __name__ == '__main__':
    # Define the output directory
    output_dir = os.path.join(repo_dir, 'output')

    ### Bullet 1 ### Data prep

    # Load and prepare data
    data = load_data(data_file)
    returns = log_returns(data)
    weights = compute_risk_parity_weights(returns)

    ### Bullet 2.1 ### multivariate normal distribution
    # Plot and save log returns
    #plot_returns(returns)

    periods = [252, 756, 1260]  # Example periods in trading days (1 year, 3 years, 5 years)
    VaR_ES_periods = calculate_var_es_for_periods(data, periods, weights)
    covariance_results = calculate_covariance_matrices(data, periods)

    # print results for periods
    for period, period_results in VaR_ES_periods.items():
        print(f"For period = {period} days: ")
        for confidence_level, results in period_results.items():
            VaR, ES = results['VaR'], results['ES']
            print(f"Confidence Level = {confidence_level}: VaR = {VaR}, ES = {ES}")
    # Distribution Fits and QQ Plots
    #for asset in returns.columns:  # Example assets
    #    plot_data_analysis(returns[asset], weights, returns, plot_type='qq', asset=asset)

    # compare variance of portfolio to normal distribution
    #plot_data_analysis(data, weights, returns, plot_type='qq')


    # Assuming 'returns' is a DataFrame of log returns for the portfolio
    window_size = 252  # 252 trading days is typically used for a 1-year rolling window
    confidence_level = 0.99  # confidence level

    # Call the function with your returns DataFrame
    portfolio_returns = returns.sum(axis=1)  # Sum across assets for portfolio returns if needed
    rolling_var_values = rolling_var_normal(portfolio_returns, window_size, confidence_level)
    plot_returns_with_var(returns, rolling_var_values)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    ### Bullet 2.2 ### student-t distribution

    var_series = rolling_var(returns, confidence_level=0.975, dof=10, method='normal')
    plot_returns_with_var(returns, var_series)

    #plot_portfolio_qq_t(data, weights, dof=3)
    #plot_portfolio_qq_t(data, weights, dof=4)
    #plot_portfolio_qq_t(data, weights, dof=5)
    #plot_portfolio_qq_t(data, weights, dof=6)



    ### Bullet 2.3 ### Historical Simulation
    #var_series = 100/(np.sqrt(252))*rolling_var(returns, confidence_level=0.975, method='normal')
    #plot_returns_with_var(returns, var_series)


    ### Bullet 2.4 ### Constant Conditional Correlation method (GARCH(1,1), normal innovations for risk factor


    ### Bullet 2.5 ### Filtered Historical Simulation method with EWMA for each risk factor


    ### Bullet 3 ### backtesting VaR and ES. Plots.


    ### Bullet 4 ### Empirical 5 day VaR and 10 day VaR, using historical simulation method. compare with sq root rule


    ### Bullet 5 ### Stress testing


    ### Bullet 6 ### check canvas for announcement extra shit

