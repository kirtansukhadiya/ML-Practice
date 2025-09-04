import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('/Users/kirtansukhadiya/Desktop/Hub/ML-Practice/Regrassion/Support Vector Regression (SVR)/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Feature matrix (independent variable)
y = dataset.iloc[:, -1].values  # Target variable (dependent variable)

y = y.reshape(len(y),1)  # Reshape y to be a 2D array
#print(y)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # Standardize the feature matrix
y = sc_y.fit_transform(y)  # Standardize the feature matrix
#print(X)   
#print(y)

regressor = SVR(kernel = 'rbf') # Create an SVR model with RBF kernel
regressor.fit(X, y)  # Fit the model to the whole data

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))  # Predict a new result (after standardizing the input)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Psition level')
plt.ylabel('Salary')
plt.show()