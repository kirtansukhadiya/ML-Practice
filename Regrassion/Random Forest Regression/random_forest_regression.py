import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Regrassion/Random Forest Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

# Visualising the Random forest regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()