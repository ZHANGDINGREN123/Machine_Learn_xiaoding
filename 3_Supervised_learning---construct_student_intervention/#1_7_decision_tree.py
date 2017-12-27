from sklearn import tree
from sympy import *
dsolve()
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print clf.predict([[2., 2.]])
print clf.predict_proba([[2., 2.]])
