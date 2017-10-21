import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(20).reshape((10, 2)), range(10)
# print X, y
X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)#random_state:
print X_train, y_train
print "############"
print x_test, y_test