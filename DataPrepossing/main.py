# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer # For handling missing data
from sklearn.compose import ColumnTransformer # For encoding categorical data
from sklearn.preprocessing import OneHotEncoder # For encoding categorical data
from sklearn.preprocessing import LabelEncoder # For encoding the dependent variable
from sklearn.model_selection import train_test_split # For splitting the dataset into training and test sets
from sklearn.preprocessing import StandardScaler # For feature scaling

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
catorical_columns = ['Country']
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

#print(X)

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)

#print(y)

# Splitting the dataset into the Training set and Test set
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
#print(x_train)
#print(X_test)
#print(y_train)
#print(y_test)

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
#print(x_train)
#print(X_test)