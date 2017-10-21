from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm
iris = datasets.load_iris()
print iris.data.shape, iris.target.shape

#random_state:split dataset to traing set and test set
X_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=42)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print scores