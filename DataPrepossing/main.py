# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer # For handling missing data
from sklearn.compose import ColumnTransformer # For encoding categorical data
from sklearn.preprocessing import OneHotEncoder # For encoding categorical data

#importing the dataset
dataset = pd.read_csv('/Users/kirtansukhadiya/Desktop/Hub/ML-Practice/DataPrepossing/Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable

#print(X)
#print(y)

# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#print(X)

# Encoding categorical data

# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

#print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)