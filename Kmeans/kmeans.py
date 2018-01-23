import numpy as np
from sklearn.cluster import KMeans
def getkdata(label):
	number, n_counts = np.unique(label, return_counts=True)
	index = np.argsort(-n_counts)
	rbestindex = number[index[:3]]
	res = False
	for i in rbestindex:
		res=res|(label==i)
	return res

#用kmeans来做是不是更简单？
def main():
	data = np.random.rand(100, 3)  # 生成一个随机数据，样本大小为100, 特征数为3
	# 假如我要构造一个聚类数为3的聚类器
	estimator = KMeans(n_clusters=5)  # 构造聚类器
	estimator.fit(data)  # 聚类
	label_pred = estimator.labels_  # 获取聚类标签
	centroids = estimator.cluster_centers_  # 获取聚类中心
	inertia = estimator.inertia_  # 获取聚类准则的总和
	resindex=getkdata(label_pred)
	res=data[resindex]
	
	print(__name__)
if __name__ == '__main__':
    main()