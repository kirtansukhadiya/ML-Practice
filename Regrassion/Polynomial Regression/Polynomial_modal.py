import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('/Users/kirtansukhadiya/Desktop/Hub/ML-Practice/Regrassion/Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)
X_l_prad = lin_reg.predict(X)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
X_poly_prad = lin_reg_2.predict(X_poly)

plt.scatter(X, y, color = 'red')
plt.plot(X, X_l_prad, color = 'blue')
plt.plot(X, X_poly_prad, color = 'green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend(['Actual' ,'Linear Regression', 'Polynomial Regression'])
plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))