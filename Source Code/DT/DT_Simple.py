from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

label = clf.predict([[2., 2.]])
print('Label for (2, 2): {}'.format(label))

conf = clf.predict_proba([[2., 2.]])
print('Conf for (2, 2): {}'.format(conf))
