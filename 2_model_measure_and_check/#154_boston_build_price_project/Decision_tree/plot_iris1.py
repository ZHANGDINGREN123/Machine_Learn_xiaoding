#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
# Load data
iris = load_iris()
print iris.target # 标签
# print iris.data.shape # 数据集


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]#选取data中的两列
    y = iris.target#选取target
    # print X.shape,y.shape
    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    # print pairidx
    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #print y_min,y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    #print xx.shape,yy.shape

    # 调整各个分块图像之间的文字布局
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    # print xx.ravel().shape,yy.ravel().shape 对xx,yy降维为列表
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 恢复xx的形状
    # print  Z.shape
    Z = Z.reshape(xx.shape)
    # print y,Z
    # print Z.shape
    # print xx.shape,yy.shape,Z.shape,Z
    # 画出预测决策树图
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)# 绘制等高线图，前三个参数为x,y,z;x,y表示网格坐标表,z表示高度

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        # print np.where(y==i)#不懂
        # print "####################"
        idx = np.where(y == i)
        # print idx
        # print X[idx, 0].shape
        # print "#################"
        # cmap=plt.cm.RdYlBu:绘制三种颜色 edgecolor='black'：散点框颜色 s=15：散点大小
        # 画出真实散点图
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                   cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
#plt.show()
