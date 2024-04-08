import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Open data files 
OPEN_FILE = 'data/portfolio.csv'
data = pd.read_csv(OPEN_FILE, index_col=0)

print(data.head())

