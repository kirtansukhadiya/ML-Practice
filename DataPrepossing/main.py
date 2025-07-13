# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('/Users/kirtansukhadiya/Desktop/Hub/ML-Practice/DataPrepossing/Data.csv')
X = dataset.iloc[:, :-1].values  # Features
Y = dataset.iloc[:, -1].values  # Target variable

print(X)
print(Y)