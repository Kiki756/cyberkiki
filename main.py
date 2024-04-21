import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.stats import norm, t, probplot
import matplotlib.lines as mlines
from arch import arch_model

# Full path to the 'repo' directory
repo_dir = ''
data_file = 'portfolio.csv'
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
    # Columns for which to calculate log returns
    selected_columns = ['Gold USD', 'Gold EUR', 'JPM Close', 'SP500', 'Siemens Close', 'XR', 'EUROSTOXX Close']
    log_return_data = np.log(data[selected_columns] / data[selected_columns].shift(1))
    log_return_data = log_return_data.dropna()
    #returns_with_euribor = log_return_data.join(data['EURIBOR'])

    return log_return_data


def compute_risk_parity_weights(returns):
    # Calculate the annualized volatility for each asset
    volatility = returns.std() * np.sqrt(252)  # Assuming 252 trading days per year
    inverse_volatility = 1 / volatility    # Inverse of volatility
    weights = inverse_volatility / inverse_volatility.sum()    # Normalize weights
    return weights


def plot_returns(returns):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=returns, dashes=False, alpha=0.5)
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

def calculate_var_es_for_periods(data, periods, weights, confidence_levels):
    results = {}
    for period in periods:
        period_data = data.tail(period)
        period_results = {}
        for confidence_level in confidence_levels:
            portfolio_variance = compute_portfolio_variance(log_returns(period_data), weights)
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


def historical_simulation(returns, window_size, confidence_level):
    # Calculate the rolling window of returns
    rolling_returns = returns.rolling(window=window_size)

    # Initialize lists to store the VaR and ES values
    var_values = []
    es_values = []

    # For each window of returns
    for window in rolling_returns:
        # Sort the returns in ascending order
        sorted_returns = np.sort(window)

        # Calculate the VaR as the quantile of the sorted returns at the (1 - confidence level) percentile
        var = -np.percentile(sorted_returns, 100 * (1 - confidence_level))

        # Calculate the ES as the mean of the returns that are less than or equal to the VaR
        es = -np.mean(sorted_returns[sorted_returns <= -var])

        # Append the VaR and ES values to their respective lists
        var_values.append(var)
        es_values.append(es)

    # Convert the lists to pandas Series for easier manipulation
    var_values = pd.Series(var_values, index=returns.index)
    es_values = pd.Series(es_values, index=returns.index)

    return var_values, es_values


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

def plot_var_es_separated(returns, window_size, dfs, confidence_levels):
    # Create separate plots for each combination of VaR and ES and confidence levels
    for confidence_level in confidence_levels:
        plt.figure(figsize=(12, 8))
        for df in dfs:
            var_values, es_values = rolling_var_es_student_t(returns, window_size, confidence_level, df)
            plt.plot(var_values.index, var_values, label=f'VaR (dof={df})')
        plt.title(f'Rolling Window VaR for Confidence Level: {confidence_level*100}%')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'rolling_var_{int(confidence_level*100)}.png'))
        plt.close()

        plt.figure(figsize=(12, 8))
        for df in dfs:
            var_values, es_values = rolling_var_es_student_t(returns, window_size, confidence_level, df)
            plt.plot(es_values.index, es_values, linestyle='--', label=f'ES (dof={df})')
        plt.title(f'Rolling Window ES for Confidence Level: {confidence_level*100}%')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'rolling_es_{int(confidence_level*100)}.png'))
        plt.close()
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


def rolling_var_normal(returns, window_size, confidence_level):
    # Calculate the rolling standard deviation for the portfolio returns
    rolling_std = returns.rolling(window=window_size).std()

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Calculate the rolling VaR
    rolling_var = -rolling_std * z_score

    # Calculate the rolling ES
    rolling_es = rolling_std * norm.pdf(z_score) / (1 - confidence_level)

    # VaR and ES are typically positive numbers when representing a loss
    return rolling_var.abs()

def rolling_es_normal(returns, window_size, confidence_level):
    # Calculate the rolling standard deviation for the portfolio returns
    rolling_std = returns.rolling(window_size).std()

    # Calculate the z-score for the specified confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Calculate the rolling ES
    rolling_es = rolling_std * norm.pdf(z_score) / (1 - confidence_level)

    # ES is typically a positive number when representing a loss
    return rolling_es.abs()  # use abs() to convert to positive if needed

def rolling_var_es_student_t(returns, window_size, confidence_level, df):
    # Calculate the rolling standard deviation for the portfolio returns
    rolling_std = returns.rolling(window_size).std()

    # Calculate the t-score for the specified confidence level and degrees of freedom
    t_score = t.ppf(1 - confidence_level, df)

    # Calculate the rolling VaR
    rolling_var = -rolling_std * t_score

    # Calculate the rolling ES
    #rolling_es = rolling_std * (t.pdf(t.ppf(1 - confidence_level, df), df=df) / (1 - confidence_level)) * ((df + (t.ppf(1 - confidence_level, df=df))**2) / (df - 1))


    # VaR and ES are typically positive numbers when representing a loss
    return rolling_var.abs()#, rolling_es.abs()  # use abs() to convert to positive if needed

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

def plot_historical_sim(returns, var_99, var_975):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=returns, label='Portfolio Returns', color='blue')
    sns.lineplot(data=var_99, label='VaR 99%', color='red')
    sns.lineplot(data=var_975, label='VaR 97.5%', color='green')
    plt.title('Portfolio Returns and VaR')
    plt.xlabel('Date')
    plt.ylabel('Returns / VaR')
    plt.legend()
    plt.show()

def plot_var_es_with_returns(returns, window_size, dfs, confidence_levels):
    # Create separate plots for each combination of VaR and ES and confidence levels
    for confidence_level in confidence_levels:
        plt.figure(figsize=(12, 8))
        plt.plot(returns.index, returns, label='Portfolio Returns', color='blue', alpha=0.75)

        # Plot portfolio returns
        #plt.plot(returns.index, returns, label='Portfolio Returns', color='blue', alpha=0.75)

        for df in dfs:
            rolling_var = rolling_var_es_student_t(returns, window_size, confidence_level, df)
            plt.plot(rolling_var.index, rolling_var, label=f'VaR (dof={df})')
        plt.title(f'Rolling Window VaR for Confidence Level: {confidence_level*100}%')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'rolling_var_{int(confidence_level*100)}.png'))
        plt.close()

        plt.figure(figsize=(12, 8))
        # Plot portfolio returns again for ES plot
        plt.plot(returns.index, returns, label='Portfolio Returns', color='blue', alpha=0.75)

        #for df in dfs:
        #    _, rolling_es = rolling_var_es_student_t(returns, window_size, confidence_level, df)
        #    plt.plot(rolling_es.index, rolling_es, label=f'ES (dof={df})')
#
        #plt.title(f'Rolling Window ES for Confidence Level: {confidence_level*100}%')
        #plt.xlabel('Date')
        #plt.ylabel('Value')
        #plt.legend()
        #plt.grid()
        #plt.savefig(os.path.join(output_dir, f'rolling_es_{int(confidence_level*100)}.png'))



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
    # Plot and save log returns
    plot_returns(returns)
    confidence_levels = [0.975, 0.99]

    periods = [252, 756, 1260]  # Example periods in trading days (1 year, 3 years, 5 years)
    VaR_ES_periods = calculate_var_es_for_periods(data, periods, weights, confidence_levels)
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


    def plot_returns_with_var_and_es(returns, var_series, es_series, title='Portfolio Returns Multivariate Normal, Rolling VaR and ES'):
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=returns.rename("Portfolio Returns"), label='Portfolio Returns', color='blue')
        sns.lineplot(data=var_series.rename("Rolling VaR"), color='red', label='Rolling VaR')
        sns.lineplot(data=es_series.rename("Rolling ES"), color='green', label='Rolling ES')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, title.replace(' ', '_').lower() + '.png'))
        plt.show()


    # Call the function with your returns DataFrame
    portfolio_returns = returns.dot(weights)
    rolling_var_values = rolling_var_normal(portfolio_returns, window_size, 0.99)
    rolling_es_values = rolling_es_normal(portfolio_returns, window_size, 0.99)
    plot_returns_with_var_and_es(portfolio_returns, rolling_var_values, rolling_es_values,
                                 title='Portfolio Returns, Rolling VaR and ES, confidence level 0.99')
    portfolio_returns = returns.dot(weights)
    rolling_var_values = rolling_var_normal(portfolio_returns, window_size, 0.975)
    rolling_es_values = rolling_es_normal(portfolio_returns, window_size, 0.975)
    plot_returns_with_var_and_es(portfolio_returns, rolling_var_values, rolling_es_values,
                                 title='Portfolio Returns, Rolling VaR and ES, confidence level 0.975')
    ###################################################################################################################


    ### Bullet 2.2 ### student-t distribution

    window_size = 252
    dfs = [3, 4, 5, 6]
    confidence_levels = [0.975, 0.99]

    # Plotting rolling VaR and ES for normal and t-distributions
    plot_var_es_with_returns(portfolio_returns, window_size, dfs, confidence_levels)


    #plot_portfolio_qq_t(data, weights, dof=3)
    #plot_portfolio_qq_t(data, weights, dof=4)
    #plot_portfolio_qq_t(data, weights, dof=5)
    #plot_portfolio_qq_t(data, weights, dof=6)

    ### Bullet 2.3 ### Historical Simulation
    # Define the confidence levels
    confidence_levels = [0.975, 0.99]

    # Calculate VaR and ES for each confidence level
    var_results = {}
    es_results = {}
    for confidence_level in confidence_levels:
        var_results[confidence_level], es_results[confidence_level] = historical_simulation(portfolio_returns,
                                                                                            window_size,
                                                                                            confidence_level)

    # Plotting
    for confidence_level in confidence_levels:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=portfolio_returns, label='Portfolio Returns', color='blue')
        sns.lineplot(data=var_results[confidence_level], label=f'VaR (confidence level={confidence_level})',
                     color='red')
        sns.lineplot(data=es_results[confidence_level], label=f'ES (confidence level={confidence_level})',
                     color='green')

        # Add labels and a legend
        plt.title(f'Portfolio Returns Historical Simulations, VaR and ES (confidence level={confidence_level})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        # Adjust output_dir to your specific directory for saving the plot
        plt.savefig(os.path.join(output_dir, f'portfolio_returns_var_es_confidence_{confidence_level}.png'))
        plt.show()

    ### Bullet 2.4 ### Constant Conditional Correlation method (GARCH(1,1), normal innovations for risk factor

    def estimate_garch(returns):
        # Rescale the returns
        returns_rescaled = returns
        sample_variance = portfolio_returns.var()

        alpha_guess = 0.05
        beta_guess = 0.92
        omega_guess = sample_variance * (1 - alpha_guess - beta_guess)

        # Fit a GARCH(1, 1) model with normal innovations to the rescaled returns
        model = arch_model(returns_rescaled, vol='Garch', p=1, q=1, dist='Normal',
                           mean='constant')
        fitted_model = model.fit(disp='off', starting_values=np.array([0.0, omega_guess, alpha_guess, beta_guess]))


        return fitted_model

    # Estimate GARCH models for each risk factor
    garch_models = {col: estimate_garch(returns[col]) for col in returns.columns}

    # Calculate the constant correlation matrix
    residuals = pd.DataFrame({col: model.resid for col, model in garch_models.items()})
    correlation_matrix = residuals.corr()

    # Calculate the portfolio variance
    volatilities = pd.DataFrame({col: model.conditional_volatility for col, model in garch_models.items()})
    portfolio_variance = weights.T @ correlation_matrix @ weights * volatilities.var(axis=1)


    def compute_VAR_ES_normal(portfolio_variance, confidence_level):
        std_dev = np.sqrt(portfolio_variance)
        VaR = -norm.ppf(1 - confidence_level) * std_dev
        ES = std_dev * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
        return VaR, ES

    # Define the confidence levels
    confidence_levels = [0.975, 0.99]

    # Calculate VaR and ES for each confidence level
    var_results = {}
    es_results = {}
    for confidence_level in confidence_levels:
        var_results[confidence_level], es_results[confidence_level] = compute_VAR_ES_normal(portfolio_variance,
                                                                                            confidence_level)

    # Plotting
    for confidence_level in confidence_levels:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=portfolio_returns, label='Portfolio Returns', color='blue', alpha=0.5)
        sns.lineplot(data=var_results[confidence_level], label=f'VaR (confidence level={confidence_level})',
                     color='red', alpha = 0.5)
        sns.lineplot(data=es_results[confidence_level], label=f'ES (confidence level={confidence_level})',
                     color='green', alpha = 0.5)

        # Add labels and a legend
        plt.title(f'Portfolio Returns, ccc GARCH(1,1) VaR and ES (confidence level={confidence_level})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        #save the plot
        plt.savefig(os.path.join(output_dir, f'portfolio_returns_var_es_confidence_{confidence_level}.png'))


    ### Bullet 2.5 ### Filtered Historical Simulation method with EWMA for each risk factor
    # Define the lambda parameter for the EWMA model
    # Parameters
    lambda_ = 0.94  # Decay factor for EWMA
    confidence_levels = [0.975, 0.99]
    window_size = 252 # Typically one trading year

    # Initialize the EWMA volatility estimates
    sigma_squared = np.zeros(len(portfolio_returns))
    sigma_squared[0] = np.var(portfolio_variance)  # Initial variance based on a sample

    # Compute EWMA for each point in time
    for t in range(1, len(portfolio_returns)):
        sigma_squared[t] = lambda_ * sigma_squared[t - 1] + (1 - lambda_) * portfolio_returns.iloc[t - 1] ** 2

    volatility = np.sqrt(sigma_squared)

    # Calculate rolling VaR and ES
    var_results = {level: [] for level in confidence_levels}
    es_results = {level: [] for level in confidence_levels}

    for t in range(window_size, len(portfolio_returns)):
        window_returns = portfolio_returns.iloc[t - window_size:t]
        window_volatility = volatility[t - window_size:t]

        # Scale returns by the ratio of current volatility to historical volatilities
        scaled_returns = window_returns * (volatility[t] / window_volatility)
        sorted_returns = np.sort(scaled_returns)

        for level in confidence_levels:
            var_index = int((1 - level) * window_size)
            var = -sorted_returns[var_index]
            es = -sorted_returns[:var_index].mean()

            var_results[level].append(var)
            es_results[level].append(es)

    dates = portfolio_returns.index[window_size:]

    for confidence_level in confidence_levels:
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=dates, y=portfolio_returns.iloc[window_size:], label='Portfolio Returns', color='blue')
        sns.lineplot(x=dates, y=var_results[confidence_level], label=f'VaR (confidence level={confidence_level})',
                     color='red')
        sns.lineplot(x=dates, y=es_results[confidence_level], label=f'ES (confidence level={confidence_level})',
                     color='green')

        # Add labels and a legend
        plt.title(f'Portfolio Returns EWMA model, VaR and ES (confidence level={confidence_level})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        plt.show()
        #save the plot
        plt.savefig(os.path.join(output_dir, f'portfolio_returns_ccc_{confidence_level}.png'))


    ### Bullet 3 ### backtesting VaR and ES. Plots.
    def calculate_var_violations(portfolio_returns, rolling_var_values):
        # Calculate VaR violations; a violation occurs where the returns are less than the negative VaR
        portfolio_returns_aligned = portfolio_returns[rolling_var_values.index]
        var_violations = portfolio_returns < -rolling_var_values
        return var_violations

    def analyze_var_violations(var_violations, confidence_level):
        expected_violations = len(var_violations) * (1 - confidence_level)        # Expected number of violations
        actual_violations = var_violations.sum()        # Actual number of violations
        print(f"Expected number of VaR violations: {expected_violations}")
        print(f"Actual number of VaR violations: {actual_violations}")
        discrepancy = actual_violations - expected_violations  # Calculating the discrepancy
        print(f"Discrepancy between expected and actual VaR violations: {discrepancy}")

        return expected_violations, actual_violations, discrepancy
    def plot_var_violations(var_violations, title='VaR Violations Over Time'):
        plt.figure(figsize=(12, 6))
        plt.plot(var_violations.index, var_violations.astype(int), 'o', color='red', markersize=5)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('VaR Violation (True/False)')
        plt.grid(True)
        plt.show()

    confidence_level = 0.975  # Set the confidence level used in VaR calculation
    confidence_levels = [0.975, 0.99]  # Define the confidence levels
    # multivariate normal
    for confidence_level in confidence_levels:
        var_results_normal = rolling_var_normal(portfolio_returns, window_size, confidence_level)
        var_violations = calculate_var_violations(portfolio_returns, var_results_normal)
        expected_violations, actual_violations, discrepancy = analyze_var_violations(var_violations, confidence_level)
        plot_var_violations(var_violations, title=f'VaR Violations Over Time (confidence level={confidence_level})')


    #student t
    var_values = {}
    for confidence_level in confidence_levels:
        for df in dfs:
            var_values[(confidence_level, df)] = rolling_var_es_student_t(portfolio_returns, window_size, confidence_level, df)
    for df in dfs:
        for confidence_level in confidence_levels:
            key = f"df={df}, CL={confidence_level}"
            results[key] = rolling_var_es_student_t(portfolio_returns, window_size, confidence_level, df)

    for confidence_level in confidence_levels:
        for df in dfs:
            var_violations = calculate_var_violations(portfolio_returns, var_results_t[confidence_level])
            expected_violations, actual_violations, discrepancy = analyze_var_violations(var_violations, confidence_level)
            plot_var_violations(var_violations, title=f'VaR Violations Over Time (Student-t, df={df})')

    #historical simulation backtesting
    window_size = 252
    confidence_levels = [0.975, 0.99]
    var_results = {}
    es_results = {}
    for confidence_level in confidence_levels:
        var_results[confidence_level], es_results[confidence_level] = historical_simulation(portfolio_returns,
                                                                                            window_size,
                                                                                            confidence_level)
    for confidence_level in confidence_levels:
        var_violations = calculate_var_violations(portfolio_returns, var_results[confidence_level])
        expected_violations, actual_violations, discrepancy = analyze_var_violations(var_violations, confidence_level)
        plot_var_violations(var_violations)


    # ccc garch backtesting
    def backtest_ccc_garch(returns, weights, confidence_levels=[0.975, 0.99]):
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)

        # Estimate GARCH models for each risk factor
        garch_models = {col: estimate_garch(returns[col]) for col in returns.columns}

        # Calculate the constant correlation matrix
        residuals = pd.DataFrame({col: model.resid for col, model in garch_models.items()})
        correlation_matrix = residuals.corr()

        # Calculate the portfolio variance
        volatilities = pd.DataFrame({col: model.conditional_volatility for col, model in garch_models.items()})
        portfolio_variance = weights.T @ correlation_matrix @ weights * volatilities.var(axis=1)

        var_results = {}
        es_results = {}
        var_violations = {}
        for confidence_level in confidence_levels:
            # Compute VaR and ES
            var_results[confidence_level], es_results[confidence_level] = compute_VAR_ES_normal(portfolio_variance,
                                                                                                confidence_level)
            var_results[confidence_level], es_results[confidence_level] = var_results[confidence_level] / 10, es_results[confidence_level] / 10
            # Calculate and analyze VaR violations
            var_violations[confidence_level] = portfolio_returns < -var_results[confidence_level]
            analyze_var_violations(var_violations[confidence_level], confidence_level)

            # Plot VaR violations
            plot_var_violations(var_violations[confidence_level],
                                title=f'VaR Violations Over Time at {confidence_level * 100}% Confidence Level')

        return var_results, es_results, var_violations

    var_results, es_results, var_violations = backtest_ccc_garch(returns, weights)

    ### Bullet 5 ### Stress testing


    ### Bullet 6 ### check canvas for announcement extra shit

