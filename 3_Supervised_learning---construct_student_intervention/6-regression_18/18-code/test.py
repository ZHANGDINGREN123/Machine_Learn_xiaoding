######################################
# coding=utf-8
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [10, 4]], [0, 1, 7])
# 线性模型x的幂数都是一次的 -1.33226762955e-15 + 10*0.5 + 10*0.5 = 7
print reg.coef_
print reg.intercept_
######################################