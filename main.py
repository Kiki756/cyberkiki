import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm


# Full path to the 'repo' directory
repo_dir = 'c:/Users/kikin.DESKTOP-G4LR6A7/OneDrive/Documenten/Kiki/VU Econometrics/Quantitative Financial Risk Management/Assignment 1 - QFRM/Data/data/repo'

def load_data(repo_dir):

    # Open data files 
    OPEN_FILE = os.path.join(repo_dir, 'portfolio.csv')
    data = pd.read_csv(OPEN_FILE, index_col=0, header=0)
    # Remove leading and trailing spaces from column names
    data.columns = data.columns.str.strip()
    # Remove commas and convert all columns to numeric format
    for col in data.columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

    # Convert dates to datetime format
    data.index = pd.to_datetime(data.index)

    return data

def log_returns(data):
    # Select columns
    selected_columns = ['Gold USD', 'Gold EUR', 'JPM Close', 'SP500', 'Siemens Close', 'XR', 'EUROSTOXX Close']
    data_selected = data[selected_columns].copy()

    # Remove commas and convert columns to numeric format
    for col in data_selected.columns:
        data_selected[col] = pd.to_numeric(data_selected[col].astype(str).str.replace(',', ''), errors='coerce')

    # Calculate log returns
    returns = np.log(data_selected / data_selected.shift(1)).dropna()
    
    return returns

def plot_returns(returns):
    # Plot log returns
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=returns, dashes=False)
    plt.title('Log returns')
    plt.savefig(os.path.join(output_dir, 'log_returns.png'))
    # plt.show()

def compute_portfolio_var(returns, weights):
    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Calculate portfolio variance
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))

    return portfolio_var, cov_matrix

def compute_VAR_ES(portfolio_var, confidence_level):
    # Calculate the VaR
    VaR = norm.ppf((1 - confidence_level), 0, np.sqrt(portfolio_var))

    # Calculate the Expected Shortfall
    ES = (1 - confidence_level) ** -1 * norm.pdf(norm.ppf(1 - confidence_level)) * np.sqrt(portfolio_var)

    return VaR, ES


if __name__ == '__main__':
    # Define the output directory
    output_dir = os.path.join(repo_dir, 'output')

    # Call function load_data
    data = load_data(repo_dir)
    print(data.head())
    print(data.columns)

    # Call function log_returns
    returns = log_returns(data)
    print(returns.head())

    # Plot log returns 
    plot_returns(returns)

    

    # Define the weights of the portfolio
    weights = np.array([0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1])

    # Calculate vovariance matrix and portfolio variance
    print('')
    portfolio_var, cov_matrix = compute_portfolio_var(returns, weights)

    # Print the covariance matrix
    print('')
    print('Covariance Matrix:')
    print(cov_matrix)
    # print the portfolio variance
    print('')
    print(f'Portfolio Variance: {portfolio_var}')

    # Calculate VaR and ES
    print('')
    confidence_level = 0.975
    VaR, ES = compute_VAR_ES(portfolio_var, confidence_level)
    print(f'VaR at {confidence_level} Confidence Interval: {VaR}')
    print(f'ES at {confidence_level} Confidnece Interval: {ES}')







