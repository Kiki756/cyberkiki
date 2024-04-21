import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_file = 'data/portfolio.csv'

OPEN_FILE = 'combined_portfolio2.csv'

data = pd.read_csv(OPEN_FILE, index_col=0)

print(data.head())

# Backward fill the EURIBOR column
data['EURIBOR'] = data['EURIBOR'].fillna(method='bfill')

print(data.head())
print(data.tail())

# Fill other missing values 
# Forward fill NaNs with last observed value
data.fillna(method='ffill', inplace=True)
# Backward fill to fill any remaining NaNs at the start of the columns
data.fillna(method='bfill', inplace=True)

print(data.head())
print(data.tail())

# Save dataframe to a new csv file
data.to_csv('portfolio.csv')