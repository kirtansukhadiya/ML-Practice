import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Classification/Nearest Neighbors (K-NN)/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(x_train, y_train)

print(classifier.predict([[30,87000]]))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(cm, ac)