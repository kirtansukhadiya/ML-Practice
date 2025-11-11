import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train_enc)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", cm)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train_enc, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train_enc, cv=10)
print(accuracies)
accuracies_mean = accuracies.mean()
accuracies_std = accuracies.std()
print(accuracies_mean)
print(accuracies_std)