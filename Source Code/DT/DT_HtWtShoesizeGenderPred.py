import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

'''
Data Set Information: Toy data set with 11 rows. Features
are height, weight, shoe-size; label is gender.

Attribute Information:

1. Height
2. Weight
3. Shoe-size

No. of examples: 11

Local filename: htwtshoesize_data.csv
'''


hws_data = pd.read_csv('htwtshoesize_data.csv', sep= ',', header= 'infer')

print("Dataset Length:: {} ".format(len(hws_data)))
print("Dataset Shape:: {}".format(hws_data.shape))

# print the first five rows of data
print("Dataset:: \n{}".format(hws_data))

X = hws_data.values[:, 0:3]
Y = hws_data.values[:,3]

# split data set into 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split( X, Y,
                                                     test_size = 0.2)


# Prediction using gini
clf_gini = DecisionTreeClassifier(criterion = "gini")
clf_gini.fit(X_train, y_train)

test_pred = clf_gini.predict([[181, 81, 44]])
print('test_pred (181, 81, 44) = {}'.format(test_pred))

y_pred = clf_gini.predict(X_test)
print("ypred: {}".format(y_pred))
print("Accuracy gini = {}".format(accuracy_score(y_test,y_pred)*100))      


# Prediction using entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(X_train, y_train)

test_pred_en = clf_gini.predict([[181, 81, 44]])
print('test_pred_en (181, 81, 44) = {}'.format(test_pred_en))

y_pred_en = clf_entropy.predict(X_test)
print("ypred_en: {}".format(y_pred_en))
print("Accuracy entropy = {}".format(accuracy_score(y_test,y_pred_en)*100))      

      

