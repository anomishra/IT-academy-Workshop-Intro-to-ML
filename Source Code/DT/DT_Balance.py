import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

'''
Data Set Information: This data set was generated to model
psychological experimental results. Each example is classified
as having the balance scale tip to the right, tip to the left,
or be balanced. The attributes are the left weight, the left
distance, the right weight, and the right distance. The correct
way to find the class is the greater of (left-distance * left-weight)
and (right-distance * right-weight). If they are equal, it is
balanced.

Attribute Information:

1. Class Name: 3 (L, B, R)
2. Left-Weight: 5 (1, 2, 3, 4, 5)
3. Left-Distance: 5 (1, 2, 3, 4, 5)
4. Right-Weight: 5 (1, 2, 3, 4, 5)
5. Right-Distance: 5 (1, 2, 3, 4, 5)

No. of examples: 625

Web: http://archive.ics.uci.edu/ml/datasets/Balance+Scale?ref=datanews.io
'''


balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
sep= ',', header= None)

print("Dataset Length:: {} ".format(len(balance_data)))
print("Dataset Shape:: {}".format(balance_data.shape))

# print the first five rows of data
print("Dataset:: {}".format(balance_data.head()))

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

# split data set into 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split( X, Y,
                                                     test_size = 0.3,
                                                     random_state = 100)


# Prediction using gini
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

test_pred = clf_gini.predict([[4, 4, 3, 3]])
print('test_pred (4, 4, 3, 3) = {}'.format(test_pred))

y_pred = clf_gini.predict(X_test)
print("ypred: {}".format(y_pred))
print("Accuracy gini = {}".format(accuracy_score(y_test,y_pred)*100))      

# Prediction using entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy",
                                     random_state = 100,
                                     max_depth=3,
                                     min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
print("ypred_en: {}".format(y_pred_en))
print("Accuracy entropy = {}".format(accuracy_score(y_test,y_pred_en)*100))      



      

