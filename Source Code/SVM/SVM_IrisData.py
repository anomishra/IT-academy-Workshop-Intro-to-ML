import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)
print('Dataset size: {}'.format(irisdata.shape))
print('Dataset (first 5 rows): {}'.format(irisdata.head()))
X = irisdata.drop('Class', axis=1)  
y = irisdata['Class']

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Polynomial Kernel
svclassifier = SVC(kernel='poly', degree=8)  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# Gaussian Kernel
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

# Sigmoid Kernel
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

