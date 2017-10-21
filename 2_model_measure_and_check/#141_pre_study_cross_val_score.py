from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabets = datasets.load_diabetes()#return X and y
X = diabets.data[:150]
y = diabets.data[:150]
lasso = linear_model.Lasso()#Lasso:it is a model of linear
print cross_val_score(lasso, X, y)