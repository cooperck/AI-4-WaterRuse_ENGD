from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics


##定义一个函数从excel读取数据
def import_excel_matrix(path,n):
  table = xlrd.open_workbook(path).sheets()[n-1] # 获取第n个sheet表
  row = table.nrows # 行数
  col = table.ncols # 列数
  datamatrix = np.zeros((row, col)) # 生成一个nrows行*ncols列的初始矩阵
  for i in range(col): # 对列进行遍历
    cols = np.mat(table.col_values(i)) # 把list转换为矩阵进行矩阵操作
    datamatrix[:, i] = cols # 按列把数据存进矩阵中
  return datamatrix


##测试数据导入情况
data_file = u'D:\博士论文\聚类\聚类测试数据.xls'
a_1 = import_excel_matrix(data_file,5)# 获取测试数据第5个sheet表
#print(a_1)
##测试标准化基础数据导入情况
b = import_excel_matrix(data_file,2)# 获取测试数据第2个sheet表
#print(b)

##数据预处理
a_2=a_1/b[0,:]#除以上限
for i in range(len(a_2)):
  a_2[i,13]=abs(a_2[i,13]-1)
print(a_2)
a=a_2

##定义一个kmeans计算过程，程序结束后运行
def kmeans():
  data=a
  #开始分类
  km =KMeans(n_clusters=2,init='k-means++',algorithm='full')#full-EM算法 'k-means++':启发式选择一个初始聚类中心
  predict = km.fit_predict(data)
  print(predict)#打印分类结果
  #定义一个颜色列表
  colors = ["red","green","yellow","blue"]
  #替换predict内容为颜色
  predict = [colors[i] for i in predict]
  #可视化结果
  plt.scatter(data[:,0],data[:,1],c=predict)
  # 可视化聚类中心
  plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker="*",c="red",label="cluster center")
  plt.show()
  #计算CH系数 轮廓系数
  labels = km.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致


##定义一个DBSCAN计算过程，程序结束后运行
def dbscan():
  data=a
  #开始分类
  db = DBSCAN(eps = 50, min_samples=2,metric='euclidean')#esp聚类半径，min半径内最少点数，metric距离
  predict = db.fit_predict(data)
  print(predict)#打印分类结果
  #定义一个颜色列表
  colors = ["red","blue", "green","yellow"]
  #替换predict内容为颜色
  predict = [colors[i] for i in predict]
  #可视化结果
  plt.scatter(data[:,0],data[:,1],c=predict)
  plt.show()
  #计算CH系数 轮廓系数
  labels = db.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致


##定义一个AGNES从下向上层次聚类计算过程，程序结束后运行
def hac():
  data = a
  # 开始分类
  ag = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='single')  # affinity距离,linkage连接算法
  predict = ag.fit_predict(data)
  print(predict)  # 打印分类结果
  # 定义一个颜色列表
  colors = ["blue", "yellow","red", "green" ]
  # 替换predict内容为颜色
  predict = [colors[i] for i in predict]
  # 可视化结果
  plt.scatter(data[:, 0], data[:, 1], c=predict)
  plt.show()
  #绘制嵌套聚类树状图 纵坐标为distance
  tree = linkage(data,method='single',metric='euclidean')#此处算法一定要与上面ag中的算法一致
  dendrogram(tree)
  plt.show()
  #计算CH系数 轮廓系数
  labels = ag.labels_
  print(metrics.calinski_harabasz_score(data, labels))
  print(metrics.silhouette_score(data, labels, metric='euclidean'))#此处算法一定要与上面ag中的算法一致


if __name__ == '__main__':
  #kmeans()#程序结束后运行kmeans
  #dbscan()#程序结束后运行dbscan
  hac()#程序结束后运行hac