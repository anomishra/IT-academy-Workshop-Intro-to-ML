import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset from local file in current directory
bankdata = pd.read_csv("bill_authentication.csv")

print('Dataset shape = {}'.format(bankdata.shape))
print('Dataset (first 5 rows):: \n {}'.format(bankdata.head()))

# Remove the label column from X data, Y data = label column
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

# 80-20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print('Confusion Matrix: \n {}'.format(confusion_matrix(y_test,y_pred)))
print('Classification Report: \n {}'.format(classification_report(y_test,y_pred)))
