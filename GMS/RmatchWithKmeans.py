import cv2
import time
import numpy as np
from sklearn.cluster import KMeans
# import scipy.signal as ss
import Func
import math
import sys

src_folder="./images/"
res_folder="./matches/"

class MatchKmeans:
	def __init__(self, img1, img2, kptnumber=10000, resizeflag=False, width=640, height=480,savename='Mcluster'):
		self.savename=savename
		self.DEBUG=True
		self.img1 = img1.copy()
		self.img2 = img2.copy()
		if resizeflag:
			ddsize = (width, height)
			self.img1 = cv2.resize(self.img1, ddsize)
			self.img2 = cv2.resize(self.img2, ddsize)
		self.kptnumber = kptnumber
		# self.initgrid()
		self.init()
	
	def init(self):
		self.TreshFactor = 6
		# 最大特征点数
		self.orb = cv2.ORB_create(self.kptnumber)
		# self.orb.setFastThreshold(0)
		self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
		self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
		# 提取并计算特征点
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
		self.createlist()
	def getlabel(self,index,shape):
		# 生成标签矩阵
		return index[0]*shape[1]+index[1]+1
	def createlist(self):
		# 统计一下坐标
		lens = len(self.matches)
		lshape=self.img1.shape[:2]
		rshape=self.img2.shape[:2]
		featurenum=3
		# 用于卷积计算阈值
		self.llabel=np.zeros((lens,featurenum))
		for i in range(lens):
			pt1 = self.kp1[self.matches[i].queryIdx].pt
			pt2 = self.kp2[self.matches[i].trainIdx].pt
			self.llabel[i]=[pt1[1],pt1[0],self.getlabel(pt1[::-1],lshape)-self.getlabel(pt2[::-1], rshape)]
			# self.llabel[i]=[(pt1[1]**2+pt1[0]**2),((pt1[1]-pt2[1])**2+(pt1[0]-pt2[0])**2)]
		#个个维度做0均值单位方差
		self.llabel=(self.llabel-self.llabel.mean(0))/self.llabel.std(0)
	def getkindex(self):
		number, n_counts = np.unique(self.label_pred, return_counts=True)
		index = np.argsort(-n_counts)
		rbestindex = number[index[:self.k]]
		resindex = False
		for i in rbestindex:
			resindex=resindex|(self.label_pred==i)
		return resindex
	# 聚类，取数量最多的前k个
	def kmeans(self,k=10,max_k=50):
		self.createlist()
		self.k=k
		self.max_k=max_k
		# 假如我要构造一个聚类数为3的聚类器
		estimator = KMeans(n_clusters=self.max_k)  # 构造聚类器
		estimator.fit(self.llabel)  # 聚类
		self.label_pred = estimator.labels_  # 获取聚类标签
		#取前k类
		self.TrueMatch=self.getkindex()
		
	# 同样返回Match对象，用于其他用途
	def getTrueMatch(self, thre=1):
		self.gridmatches = []
		self.leftkeypoint = []
		self.rightkeypoint = []
		number = 0
		for i in range(len(self.matches)):
			if self.TrueMatch[i]:
				self.gridmatches.append(cv2.DMatch(number, number, 1))
				self.leftkeypoint.append(self.kp1[self.matches[i].queryIdx])
				self.rightkeypoint.append(self.kp2[self.matches[i].trainIdx])
				number += 1
		return self.gridmatches, self.leftkeypoint, self.rightkeypoint
	
	# 画出TrueMatch的点
	def drawTrueMatch(self):
		self.getTrueMatch()
		gmsmatchimg = cv2.drawMatches(self.img1, self.leftkeypoint, self.img2, self.rightkeypoint, self.gridmatches, None)
		filename=res_folder+self.savename+"_k("+str(self.k)+")_max_k("+str(self.max_k)+").png"
		cv2.imwrite(filename, gmsmatchimg)
		return gmsmatchimg
	
	# 统计
	def run(self,k=10,max_k=50):
		self.kmeans(k,max_k)
		self.drawTrueMatch()

def main(argv):
	if len(argv)<2:
		print("input arguments like this:\n\timg1 img2 k(optional) max_k(optional) savename(optional)")
	img1path=argv[1]
	img2path=argv[2]
	args=[2,50,"MCluser"]
	for i in range(3):
		if len(argv)>i+3:
			args[i]=argv[i+3]
	k,max_k,savename=args
	# img1path=src_folder + '000.png'
	# img2path = src_folder + '020.png'
	# img1path = src_folder + 'img1.jpg'
	# img2path = src_folder + 'img2.jpg'
	# img1path='./images/img.jpg'
	# img2path = './images/img2.jpg'
	img1 = cv2.imread(img1path)
	img2 = cv2.imread(img2path)
	ddsize = (640, 480)
	img1 = cv2.resize(img1, ddsize)
	img2 = cv2.resize(img2, ddsize)
	time_start = time.time()
	mk = MatchKmeans(img1, img2, savename=savename)
	mk.run(k=k,max_k=max_k)
	# gmf.run(gridnum=20, ktype='s', sigma=1, neiborwidth=1)
	# gmf.run(gridnum=40,ktype='g', sigma=1.2, neiborwidth=5)
	time_end = time.time();  # time.time()为1970.1.1到当前时间的毫秒数
	print('cost time is %fs' % (time_end - time_start))

if __name__ == '__main__':
	main(sys.argv)

