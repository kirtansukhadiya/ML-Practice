import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

dataset = pd.read_csv('Regrassion/Regression modal selection/Data.csv')
X = dataset.iloc[:, :-1].values  # Feature matrix (independent variable)
y = dataset.iloc[:, -1].values  # Target variable (dependent variable)

y = y.reshape(len(y),1)  # Reshape y to be a 2D array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X_train) 
y = sc_y.fit_transform(y_train) 


regressor = SVR(kernel = 'rbf') # Create an SVR model with RBF kernel
regressor.fit(X_train, y_train)  # Fit the model to the whole data

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform((X_test))).reshape(-1,1))  
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))