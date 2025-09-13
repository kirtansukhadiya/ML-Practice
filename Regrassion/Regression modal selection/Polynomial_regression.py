import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Regrassion/Regression modal selection/Data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)
X_l_prad = lin_reg.predict(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
X_poly_prad = lin_reg_2.predict(X_poly)
y_pred = lin_reg_2.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))