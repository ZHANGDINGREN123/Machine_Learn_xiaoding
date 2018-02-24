# coding=utf-8
#################################################
# kmeans: k-means cluster  
# Author : stu_why
# Date   : 2016-12-06
# HomePage : http://blog.csdn.net/zpp1994
# Email  : 1620009136@qq.com
#################################################  
from sklearn.cluster import KMeans
import numpy
import matplotlib.pyplot as plt  

# step 1: load data
print('step 1: load data...') 

#读取testSet.txt数据并存储到dataSet中
dataSet = []
fileIn = open('julei.txt')
for line in fileIn.readlines():  
    lineArr = line.strip().split()
    dataSet.append('%0.6f' % float(lineArr[0]))
    dataSet.append('%0.6f' % float(lineArr[1]))


# step 2: clustering...
print('step 2: clustering...')

#调用sklearn.cluster中的KMeans类
dataSet = numpy.array(dataSet).reshape(80,2)
kmeans = KMeans(n_clusters=4, random_state=0).fit(dataSet)

#求出聚类中心
center=kmeans.cluster_centers_
center_x=[]
center_y=[]
for i in range(len(center)):
    center_x.append('%0.6f' % center[i][0])
    center_y.append('%0.6f' % center[i][1])

#标注每个点的聚类结果
labels=kmeans.labels_
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []
for i in range(len(labels)):
    if labels[i] == 0:
        type1_x.append(dataSet[i][0])
        type1_y.append(dataSet[i][1])
    if labels[i] == 1:
        type2_x.append(dataSet[i][0])
        type2_y.append(dataSet[i][1])
    if labels[i] == 2:
        type3_x.append(dataSet[i][0])
        type3_y.append(dataSet[i][1])
    if labels[i] == 3:
        type4_x.append(dataSet[i][0])
        type4_y.append(dataSet[i][1])

#画出四类数据点及聚类中心
plt.figure(figsize=(10,8), dpi=80)
axes = plt.subplot(111)
type1 = axes.scatter(type1_x, type1_y, s=40, c='red')
type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
type3 = axes.scatter(type3_x, type3_y,s=40, c='pink' )
type4 = axes.scatter(type4_x, type4_y, s=40, c='yellow')
type_center = axes.scatter(center_x, center_y, s=40, c='blue')
plt.xlabel('x')
plt.ylabel('y')

axes.legend((type1, type2, type3, type4,type_center), ('0','1','2','3','center'),loc=1)
plt.show()  