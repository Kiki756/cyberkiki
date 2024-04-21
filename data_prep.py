import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

directory = r'C:\Users\kikin.DESKTOP-G4LR6A7\OneDrive\Documenten\Kiki\VU Econometrics\Quantitative Financial Risk Management\Assignment 1 - QFRM\Data\data'

csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

# Dictionary to store the dataframes
dfs = {}

# Loop through csv files 
for file in csv_files:
    # Read the csv file
    df = pd.read_csv(file)
    # Rename the first column to Date
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    # Convert the Date column to datetime (if it's not already)
    df['Date'] = pd.to_datetime(df['Date'])
    # Group by date and take the value of each group
    df = df.groupby('Date').first()
    
    # Add dataframe to dictionary with filename as key
    dfs[file] = df

# Concatenate all dataframes into one   
combined_df = pd.concat(dfs, axis=1)

# Save the combined dataframe to a csv file
combined_df.to_csv('combined_portfolio.csv')

