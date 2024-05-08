from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
from scipy.optimize import linear_sum_assignment

#定义准确率计算函数
def purity_calculator(cluster, label):
  # 比较两个list元素多少，元素少的放第一个
  cluster_=list(cluster)#先强行转为list
  label_=list(label)
  a = {}
  for i in cluster_:
    if cluster_.count(i) >= 1:
      a[i] = cluster_.count(i)
  print(a)
  x = len(a)

  b = {}
  for i in label_:
    if label_.count(i) >= 1:
      b[i] = label_.count(i)
  print(b)
  y = len(b)

  if x >= y:
    cluster, label = label, cluster

  cluster = np.array(cluster)
  label = np.array(label)
  indedata1 = {}
  for p in np.unique(label):  # np.unique用于提取label中的重复数字并排序输出
    indedata1[p] = np.argwhere(label == p)  # 返回等于p的数组元组的索引
  indedata2 = {}
  for q in np.unique(cluster):
    indedata2[q] = np.argwhere(cluster == q)

  count_all = []
  for i in indedata1.values():
    count = []
    for j in indedata2.values():
      a = np.intersect1d(i, j).shape[0]
      count.append(a)
    count_all.append(count)

  return sum(np.max(count_all, axis=0)) / len(cluster)

def ACC(y_true, y_pred):
  """
  Calculate clustering accuracy. Require scikit-learn installed
  # Arguments
      y: true labels, numpy.array with shape `(n_samples,)`
      y_pred: predicted labels, numpy.array with shape `(n_samples,)`
  # Return
      accuracy, in [0,1]
  """
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  y_true = y_true.astype(np.int64)
  assert y_pred.size == y_true.size
  D = max(y_pred.max(), y_true.max()) + 1
  w = np.zeros((D, D), dtype=np.int64)
  for i in range(y_pred.size):
    w[y_pred[i], y_true[i]] += 1
  ind = linear_sum_assignment(w.max() - w)
  ind = np.array(ind).T
  return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

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
data_file = u'G:\博士论文\聚类\聚类测试数据.xls'
a_1 = import_excel_matrix(data_file,6)# 获取测试数据第3个sheet表
#print(a_1)
##测试标准化基础数据导入情况
b = import_excel_matrix(data_file,2)# 获取测试数据第2个sheet表
#print(b)


##数据预处理
def logistic(x):  ##定义一个逻辑函数
  return 1 / (0.5 + np.exp(-x))

#def exponential(x):  ##定义一个函数,因为考虑4倍为RO通常的回收率
  #return np.exp(x-0.25)
#  return x*4
for i in range(len(a_1)):
  a_1[i,13]=abs(a_1[i,13]-7) #pH调整为14-1，0-1，7-0的格式
a_2=a_1/b[0,:]#除以上限

 #  for j in range(0, 11):
 #    a_2[i, j] = exponential(a_2[i, j]) #对一类污染物超标倍数进行指数倍变换
 # for k in range(14,21):
 #   a_2[i, k]=logistic(a_2[i, k]) #色度、物化距离指标进行logistic变换，超标后有一个上限值
#print(a_2)
a=a_2
print(a)

##定义一个kmeans计算过程，程序结束后运行
def kmeans():
  data=a
  sse=[]
  ch=[]
  sc=[]
  purity = []
  nmi = []
  ari = []


  labels_true = [1, 1, 1, 0, 0, 0, 1, 1]  # 实际分类
  clusterNum = 8

  #开始分类
  for k in range(1,clusterNum):
      km =KMeans(n_clusters=k,init='k-means++',algorithm='full')#full-EM算法 'k-means++':启发式选择一个初始聚类中心
      predict = km.fit_predict(data)
      print(predict)  # 打印分类结果
      #print(predict)#打印分类结果
      #定义一个颜色列表
      #colors = ["red","green","yellow","blue"]
      #替换predict内容为颜色
     # predict_1 = [colors[i] for i in predict]
      #可视化结果
      #plt.scatter(data[:,0],data[:,1],c=predict_1)
      # 可视化聚类中心
     # plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker="*",c="red",label="cluster center")
      #plt.show()
      print(km.inertia_)

      # 计算purity
      purity.append(ACC(labels_true, predict))
      # 设计算NMI
      nmi.append(metrics.normalized_mutual_info_score(labels_true, predict))
      # 设计算ARI
      ari.append(metrics.adjusted_rand_score(labels_true, predict))


      sse.append(km.inertia_)

      #计算CH系数 轮廓系数
      labels = km.labels_
      print(labels)
      if k == 1:
          ch.append(0)
          sc.append(0)
      if k!=1:
          ch.append(metrics.calinski_harabasz_score(data, labels))
          sc.append(metrics.silhouette_score(data, labels, metric='euclidean'))  # 此处算法一定要与上面ag中的算法一致
      print(ch)
      print(sc)

  print(sse)
  ch=ch [1:clusterNum-1]
  print(ch)
  sc = sc[1:clusterNum-1]
  print(sc)





# 画SSE曲线
  plt.plot(range(1, clusterNum),sse,marker='o',mfc='g',markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('SSE')
  plt.title('SSE of Kmeans for JDD')
  for x, y in zip(range(1, clusterNum),sse):
      plt.text(x + 0.1, y + 20, str(np.around(y,2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()

# 画CH曲线
  plt.plot(range(2, clusterNum), ch, marker='*',mfc='g',markersize=15)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Calinski-Harabasz Index')
  plt.title('Calinski-Harabasz Index of Kmeans for JDD')
  for x, y in zip(range(2, clusterNum),ch):
      plt.text(x + 0.1, y + 0.2, str(np.around(y,2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()

# 画SC曲线
  plt.plot(range(2, clusterNum), sc, marker='X',mfc='g',markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Silhouette Coefficient')
  plt.title('Silhouette Coefficient of Kmeans for JDD')
  for x, y in zip(range(2, clusterNum), sc):
      plt.text(x , y + 0.01, str(np.around(y, 2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()

# 画Purity曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), purity, marker='o', mfc='g', markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Purity')
  plt.title('Purity of Kmeans for JDD')
  for x, y in zip(range(1, clusterNum), purity):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画NMI曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), nmi, marker='*',mfc='g',markersize=15)
  plt.xlabel('Number of Clusters')
  plt.ylabel('NMI')
  plt.title('NMI of Kmeans for JDD')
  for x, y in zip(range(1, clusterNum), nmi):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 4)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画ARI曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), ari, marker='X',mfc='g',markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('ARI')
  plt.title('ARI of Kmeans for JDD')
  for x, y in zip(range(1, clusterNum), ari):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 4)), ha='center', va='bottom', fontsize=10.5)
  plt.show()

def dbscan():
  data = a
  sse = []
  ch = []
  sc = []
  purity = []
  nmi = []
  ari = []
  xs = []
  ys = []
  zs = []

  labels_true = [1, 1, 1, 0, 0, 0, 1, 1]  # 实际分类
  epsmax = 13
  minspmax = 4

  for ms in range(1,minspmax):
    for e in range(2,epsmax):
      xs.append(ms) #建立x数列
      ys.append(e)  #建立y数列
      #开始分类
      db = DBSCAN(eps = e, min_samples=ms,metric='euclidean')#esp聚类半径，min半径内最少点数，metric距离
      predict = db.fit_predict(data)
      print(predict)#打印分类结果
      labels = db.labels_
      print('labels=',labels)
      #可视化结果
      purity.append(ACC(labels_true, predict))
      # 设计算NMI
      nmi.append(metrics.normalized_mutual_info_score(labels_true, predict))
      # 计算ARI
      ari.append(metrics.adjusted_rand_score(labels_true, predict))
      # 计算CH系数 轮廓系数
      ch.append(metrics.calinski_harabasz_score(data, labels))
      sc.append(metrics.silhouette_score(data, labels, metric='euclidean'))  # 此处算法一定要与上面ag中的算法一致



  print('min_samples:',xs)
  print('eps:',ys)
  print('SC:', sc)
  print('CH:',ch)
  print('purity:',purity)
  print('NMI:',nmi)
  print('ARI:',ari)


#开始绘图
  #绘制min_samples-esp-SC图
  fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
  ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
  # 基于ax变量绘制三维图
  # xs表示x方向的变量
  # ys表示y方向的变量
  # zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
  # m表示点的形式，o是圆形的点，^是三角形（marker)
  # c表示颜色（color for short）
  ax.plot(xs, ys, sc, c='r', marker='X', label='Silhouette Score')  # 点为红色三角形

  # 显示图例
  ax.legend()
  # 设置坐标轴
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  ax.set_xlabel('min samples')
  ax.set_ylabel('eps')
  ax.set_zlabel('Silhouette Score')
  ax.set_title('min_samples-eps-SC for JDD')
  for x, y, z in zip(xs,ys,sc):
    ax.text(x, y, z, str(np.around(z,2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
  plt.show()


  # 绘制min_samples-esp-ch图
  fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
  ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
  ax.plot(xs, ys, ch, c='r', marker='o', label='Calinski-Harabasz')
  # 显示图例
  ax.legend()
  # 设置坐标轴
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  ax.set_xlabel('min samples')
  ax.set_ylabel('eps')
  ax.set_zlabel('Calinski-Harabasz')
  ax.set_title('min_samples-eps-ch for JDD')
  for x, y, z in zip(xs, ys, ch):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
  plt.show()

  # 绘制min_samples-eps-purity图
  fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
  ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
  ax.plot(xs, ys, purity, c='r', marker='o', label='purity')  # 点为红色圆形
  # 显示图例
  ax.legend()
  # 设置坐标轴
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  ax.set_xlabel('min samples')
  ax.set_ylabel('eps')
  ax.set_zlabel('purity')
  ax.set_title('min_samples-eps-purity for JDD')
  for x, y, z in zip(xs, ys, purity):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
  plt.show()

  # 绘制min_samples-eps-NMI图
  fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
  ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
  ax.plot(xs, ys, nmi, c='r', marker='*', label='NMI')  # 点为红色*形
  # 显示图例
  ax.legend()
  # 设置坐标轴
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  ax.set_xlabel('min samples')
  ax.set_ylabel('eps')
  ax.set_zlabel('NMI')
  ax.set_title('min_samples-eps-NMI for JDD')
  for x, y, z in zip(xs, ys, nmi):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
  plt.show()

  # 绘制min_samples-eps-ARI图
  fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
  ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。
  ax.plot(xs, ys, ari, c='r', marker='X', label='ARI')  # 点为红色X形
  # 显示图例
  ax.legend()
  # 设置坐标轴
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  ax.set_xlabel('min samples')
  ax.set_ylabel('eps')
  ax.set_zlabel('ARI')
  ax.set_title('min_samples-eps-ARI for JDD')
  for x, y, z in zip(xs, ys, ari):
    ax.text(x, y, z, str(np.around(z, 2)), ha='center', va='bottom', fontsize=8.5)
  # 显示图形
  plt.show()





def hac():
  data = a
  sse = []
  ch = []
  sc = []
  purity = []
  nmi = []
  ari = []
  xs = []
  ys = []
  zs = []
  # 开始分类
  labels_true = [1, 1, 1, 0, 0, 0, 1, 1]  # 实际分类
  clusterNum = 8

  # 开始分类
  for k in range(1, clusterNum):
    ag = AgglomerativeClustering(n_clusters=k, affinity='euclidean',linkage='ward')  # affinity距离,linkage连接算法
    predict = ag.fit_predict(data)
    print(predict)  # 打印分类结果k
    # 定义一个颜色列表
    #colors = ["blue", "yellow","red", "green" ]
    # 替换predict内容为颜色
    #predict = [colors[i] for i in predict]
    # 可视化结果
    #plt.scatter(data[:, 0], data[:, 1], c=predict)
    #plt.show()


    labels = ag.labels_

 # 计算purity
    purity.append(ACC(labels_true, predict))
 # 设计算NMI
    nmi.append(metrics.normalized_mutual_info_score(labels_true, predict))
 #计算ARI
    ari.append(metrics.adjusted_rand_score(labels_true, predict))



  #计算CH系数 轮廓系数
    print(labels)
    if k == 1:
        ch.append(0)
        sc.append(0)
    if k != 1:
        ch.append(metrics.calinski_harabasz_score(data, labels))
        sc.append(metrics.silhouette_score(data, labels, metric='euclidean'))  # 此处算法一定要与上面ag中的算法一致
    print(ch)
    print(sc)


  ch=ch [1:clusterNum-1]
  print('CH:',ch)
  sc = sc[1:clusterNum-1]
  print('SC:',sc)

# 绘制嵌套聚类树状图 纵坐标为distance
  tree = linkage(data,method='ward',metric='euclidean')#此处算法一定要与上面ag中的算法一致
  dendrogram(tree)
  plt.show()


# 画CH曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(2, clusterNum), ch, marker='*',mfc='b',markersize=15)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Calinski-Harabasz Index')
  plt.title('Calinski-Harabasz Index of AGNES for JDD')
  for x, y in zip(range(2, clusterNum),ch):
      plt.text(x + 0.1, y + 0.2, str(np.around(y,2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画SC曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(2, clusterNum), sc, marker='X',mfc='b',markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Silhouette Coefficient')
  plt.title('Silhouette Coefficient of AGNES for JDD')
  for x, y in zip(range(2, clusterNum), sc):
      plt.text(x , y + 0.01, str(np.around(y, 2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画Purity曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), purity, marker='o', mfc='b', markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Purity')
  plt.title('Purity of AGNES for JDD')
  for x, y in zip(range(1, clusterNum), purity):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 2)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画NMI曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), nmi, marker='*',mfc='b',markersize=15)
  plt.xlabel('Number of Clusters')
  plt.ylabel('NMI')
  plt.title('NMI of AGNES for JDD')
  for x, y in zip(range(1, clusterNum), nmi):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 4)), ha='center', va='bottom', fontsize=10.5)
  plt.show()


# 画ARI曲线
  ax = plt.gca()
  # ax为两条坐标轴的实例
  x_major_locator = MultipleLocator(1)
  # 把x轴的刻度间隔设置为1，并存在变量里
  ax.xaxis.set_major_locator(x_major_locator)
  # 把x轴的主刻度设置为1的倍数
  plt.plot(range(1, clusterNum), ari, marker='X',mfc='b',markersize=10)
  plt.xlabel('Number of Clusters')
  plt.ylabel('ARI')
  plt.title('ARI of AGNES for JDD')
  for x, y in zip(range(1, clusterNum), ari):
    plt.text(x + 0.1, y + 0.007, str(np.around(y, 4)), ha='center', va='bottom', fontsize=10.5)
  plt.show()





if __name__ == '__main__':
  #kmeans()#程序结束后运行kmeans
  #dbscan()#程序结束后运行dbscan
  hac()#程序结束后运行hac