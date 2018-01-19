import cv2
import time
import numpy as np
import scipy.signal as ss
import Func
import math
class GridMatchFilter:
	def __init__(self, img1, img2, kptnumber=10000, resizeflag=False, width=640, height=480):
		self.img1 = img1.copy()
		self.img2 = img2.copy()
		if resizeflag:
			ddsize = (width, height)
			self.img1 = cv2.resize(self.img1, ddsize)
			self.img2 = cv2.resize(self.img2, ddsize)
		self.kptnumber = kptnumber
		self.initgrid()
		self.init()
	
	def init(self):
		self.TreshFactor = 6
		# 最大特征点数
		self.orb = cv2.ORB_create(self.kptnumber)
		self.orb.setFastThreshold(0)
		self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
		self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
		# 提取并计算特征点
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
		# 显示
		rawmatchimg = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches, None)
		# cv2.imshow('rawmatchimg', rawmatchimg)
		# cv2.waitKey()
		
		self.gridmatchesindex = np.zeros([len(self.matches)])
		
		self.initgrid()  # 初始化网格
		self.gridmatches = []
		self.multiplemap()  # 将matches转化为矩阵
	
	def initgrid(self, leftgridnum=20, rightgridnum=20):
		self.lgn = leftgridnum
		self.rgn = rightgridnum
		# 计算划分后网格的高和宽
		self.leftgridsize = np.array([
		math.ceil(self.img1.shape[0] / self.lgn), math.ceil(self.img1.shape[1] / self.lgn)])  # [r,c]
		self.rightgridsize = np.array([
		math.ceil(self.img2.shape[0] / self.rgn), math.ceil(self.img2.shape[1] / self.rgn)])  # [r,c]
		# 生成标签矩阵
		self.leftlabel = (np.arange(1, self.lgn ** 2 + 1).reshape(self.lgn, self.lgn)) \
			.repeat(self.leftgridsize[0], 0).repeat(self.leftgridsize[1], 1).astype(np.int32)
		self.rightlabel = (np.arange(1, self.rgn ** 2 + 1).reshape(self.rgn, self.rgn)) \
			.repeat(self.rightgridsize[0], 0).repeat(self.rightgridsize[1], 1).astype(np.int32)
	
	# 在
	def multiplemap(self):
		# 统计一下坐标
		lens = len(self.matches)
		kp1r = np.zeros([lens]).astype(np.int32)
		kp1c = np.zeros([lens]).astype(np.int32)
		kp2r = np.zeros([lens]).astype(np.int32)
		kp2c = np.zeros([lens]).astype(np.int32)
		leftsize = self.img1.shape[:2]
		rightsize = self.img2.shape[:2]
		# 用于卷积计算阈值
		self.leftimg = np.zeros(leftsize).astype(np.int32)
		self.leftbiasr = np.zeros(leftsize).astype(np.int32)
		self.leftbiasc = np.zeros(leftsize).astype(np.int32)
		# 设定最大可接受重复映射邻域宽度
		max_neibor_width = 1
		MultipleMap = True  # 是否对重复映射做邻域替换映射，为False则不管重复映射，只取最后一个映射值
		for i in range(lens):
			pt1 = np.array(self.kp1[self.matches[i].queryIdx].pt)
			p1 = np.array([pt1[1], pt1[0]], np.int32)
			breakflag = False  # 跳出循环参数
			if MultipleMap and self.leftimg[p1[0], p1[1]] > 0:  # 说明是重复的点
				for j in range(p1[0] - max_neibor_width, p1[0] + max_neibor_width + 1):  # 因为左右宽度，再加上1个中心点
					for k in range(p1[1] - max_neibor_width, p1[1] + max_neibor_width + 1):  # 上下宽度再加一个中心点
						if self.leftimg[j, k] == 0:  # 说明有空位
							# print("find space area!")
							self.leftimg[j, k] += 1
							self.leftbiasr[j, k] = p1[0] - j
							self.leftbiasc[j, k] = p1[1] - k
							kp1r[i] = j
							kp1c[i] = k
							breakflag = True
							break
					if breakflag:
						break
				if not breakflag:
					print("neibor overflow!")
			else:
				kp1r[i], kp1c[i] = p1
				self.leftimg[p1[0], p1[1]] += 1
			# 对右图不做处理
			pt2 = np.array(self.kp2[self.matches[i].trainIdx].pt)
			kp2r[i] = int(pt2[1])
			kp2c[i] = int(pt2[0])
		self.kp1list = (np.array(kp1r, np.int32), np.array(kp1c, np.int32))
		self.kp2list = (np.array(kp2r, np.int32), np.array(kp2c, np.int32))
		# Func.matchesshow(self.img1,self.img2,[kp1list,kp2list],"B matches")
		# print("img lens:%d == kplist lens:%d == match number:%d" % ((self.leftimg > 0).sum(), len(kp1r), lens))
		# 用于计算是否为正确匹配
		# 假设初始时全部为假匹配
		self.TrueMatches = np.zeros(leftsize)
		# 用于卷积计算打分,并获取匹配点
		self.leftmatchr = np.zeros(leftsize)
		self.leftmatchc = np.zeros(leftsize)
		self.leftimglabel = np.zeros(leftsize)
		# self.leftmatch = np.zeros(leftsize)
		self.leftmatchgrid = np.zeros(leftsize)
		# self.rightimglabel[kp2list] = self.rightlabel[kp2list]
		# Func.imagesc(self.leftimglabel, 'leftimglabel')
		# Func.imagesc(self.leftmatchgrid,'leftmatchgrid')
		# Func.imagesc(self.rightimglabel,'rightimglabel')
		# self.TrueMatches[kp1list]=1
		# 只保存匹配图片的匹配点坐标[r,c]
		self.leftmatchr[self.kp1list] = kp2r
		self.leftmatchc[self.kp1list] = kp2c
	#获取卷积核和步长，
	# type='g' 输出的高斯核是和为1的
	# type='m' 输出的均值核是和为1的
	# type='s' 输出的求和核
	def getKernel(self,type='g',sigma=1,ksize=(24,32)):
		r,c=ksize
		if type=='g':#高斯核
			rr=cv2.getGaussianKernel(r, sigma)
			cc=cv2.getGaussianKernel(c, sigma)
			kernel=rr.reshape(-1,1).dot(cc.reshape(1,-1))
		elif type=='m':#均值核
			kernel=np.ones(ksize)/(r*c)
		elif type=='s':#求和核
			kernel = np.ones(ksize)
		return kernel
	#获取左右图标签矩阵，是否平移,shift=1表示移动半个网格
	def createlabel(self,shift=(0,0),show=False):
		# 生成标签矩阵
		self.leftlabel = (np.arange(1, self.lgn ** 2 + 1).reshape(self.lgn, self.lgn)) \
			.repeat(self.leftgridsize[0], 0).repeat(self.leftgridsize[1], 1).astype(np.int32)
		self.rightlabel = (np.arange(1, self.rgn ** 2 + 1).reshape(self.rgn, self.rgn)) \
			.repeat(self.rightgridsize[0], 0).repeat(self.rightgridsize[1], 1).astype(np.int32)
		if shift[0]==1:
			self.leftlabel=np.roll(self.leftlabel,-int(self.leftgridsize[0]/2),axis=0)
			self.rightlabel=np.roll(self.rightlabel, -int(self.rightgridsize[0] / 2), axis=0)
		if shift[1]==1:
			self.leftlabel=np.roll(self.leftlabel, -int(self.leftgridsize[1] / 2), axis=1)
			self.rightlabel=np.roll(self.rightlabel, -int(self.rightgridsize[1] / 2), axis=1)
		# 标签
		self.leftimglabel[self.kp1list] = self.leftlabel[self.kp1list]
		# 只保存匹配特征所在的网格，反正也不会计算其实际坐标
		self.leftmatchgrid[self.kp1list] = self.rightlabel[self.kp2list]  # index2value(rightkpt, shape)
		if show:
			Func.imagesc(self.leftlabel,"leftLabel")
	#返回三个核
	def createKernel(self,shift=(0,0)):
		self.neiborwidth = 1  # 多远 的算邻居
		self.kernel=self.getKernel(type='s',ksize=self.leftgridsize*(2*self.neiborwidth+1))
		self.stride=self.leftgridsize
		self.start = (self.leftgridsize / 2).astype(np.int32)
		if shift[0]==1:
			self.start[0]=0
		if shift[1]==1:
			self.start[1]=0
	# 计算阈值和得分
	def computescoreandthre(self):
		# 计算阈值
		# filter = np.ones(self.leftgridsize)
		
		print("self.leftimg:max:%f,min:%f" % (self.leftimg.max(), self.leftimg.min()))
		# self.leftgridkpoints = Func.conv2withstride(self.leftimg, filter, stride=self.leftgridsize, start=None,
		#                                             gridnum=self.lgn)
		# print("self.leftgridkpoints:max:%f,min:%f" % (self.leftgridkpoints.max(), self.leftgridkpoints.min()))
		# 显示计数
		# Func.imagesc(tmp, '左图计数')
		# threfilter = np.ones((3, 3)) / 9  # 计算均值
		# threfilter = np.ones((3, 3))
		self.thre = Func.conv2withstride(self.leftimg, self.kernel, stride=self.stride, start=self.start,
		                                            gridnum=self.lgn)
		print("self.thre:max:%f,min:%f" % (self.thre.max(), self.thre.min()))
		self.thre = self.TreshFactor * np.sqrt(self.thre/9)  # 阈值计算公式
		print("self.thre:max:%f,min:%f" % (self.thre.max(), self.thre.min()))
		# 显示阈值
		# Func.imagesc(self.thre, 'thre')
		# 计算打分
		self.score = np.zeros((self.lgn, self.lgn))
		self.lgshape = (self.lgn, self.lgn)

		# filtershape = (self.leftgridsize[0] * (2 * neiborwidth + 1), self.leftgridsize[1] * (2 * neiborwidth + 1))
		# neiborfilter = np.ones(filtershape)
		for i in range(self.lgn):  # r
			for j in range(self.lgn):  # c
				if self.thre[i, j] == 0:
					continue
				leftvalue = Func.index2value((i, j), self.lgshape) + 1
				bestmatchgrid = self.leftmatchgrid[self.leftimglabel == leftvalue]
				if bestmatchgrid.size < 1:
					continue  # 点数小于阈值则不计算，默认为不匹配
				number, n_counts = np.unique(bestmatchgrid, return_counts=True)
				rbestindex = number[np.argsort(n_counts)[-1]]
				index = (self.leftimglabel == leftvalue) & (self.leftmatchgrid == rbestindex)
				neiborsindex = (((self.leftimglabel - self.leftmatchgrid) == (leftvalue - rbestindex)) &
						self.leftimg).astype(np.int32)
				neiborsindexconv = Func.conv2withstride(neiborsindex, self.kernel,
				                                        stride=self.stride, start=self.start, gridnum=self.lgn)
				self.score[i, j] = neiborsindexconv[i, j]
				# print("calc grid(%d,%d)\n thre=%.f,index.sum=%d,neiborsindex.sum=%d,score=%d"
				#       % (i, j,self.thre[i,j],index.sum(),neiborsindex.sum(),self.score[i, j]))
				if neiborsindexconv[i, j] < self.thre[i, j]:
					continue
				self.TrueMatches += index
	
	# 同样返回Match对象，用于其他用途
	def getTrueMatch(self, thre=1):
		Truelistindex = self.TrueMatches >= thre
		# 获取img1的特征点列表
		leftkeypointarr = np.where(self.TrueMatches >= thre)
		# 因为x,y对应的是[c,r]，所以需要反过来
		self.leftkeypoint = [cv2.KeyPoint(y, x, 1) for (x, y) in zip(leftkeypointarr[0], leftkeypointarr[1])]
		# 获取对应的img2的坐标
		rightkeypointarr = [self.leftmatchc[Truelistindex], self.leftmatchr[Truelistindex]]
		self.rightkeypoint = [cv2.KeyPoint(x, y, 1) for (x, y) in zip(rightkeypointarr[0], rightkeypointarr[1])]
		# 生成match
		lens = len(self.leftkeypoint)
		self.truematch = [cv2.DMatch(x, y, 1) for (x, y) in zip(np.arange(0, lens), np.arange(0, lens))]
		# 获取
		return self.leftkeypoint, self.rightkeypoint, self.truematch
	
	# 画出TrueMatch的点
	def drawTrueMatch(self):
		self.getTrueMatch()
		gmsmatchimg = cv2.drawMatches(self.img1, self.leftkeypoint, self.img2, self.rightkeypoint, self.truematch, None)
		cv2.imwrite('GMSwithGridFilter.png', gmsmatchimg)
		return gmsmatchimg
	
	# 统计
	def run(self, type=0):
		for i in range(2):
			shift=(i,i)
			self.createlabel(shift=shift,show=False)
			self.createKernel(shift=shift)
			self.computescoreandthre()  # 计算出TrueMatcher
			# self.TrueMatches[np.arange(1,100,2),np.arange(1,100,2)]=1
			# return self.getTrueMatch()
		ssds = self.drawTrueMatch()
		# Func.imshow(ssds)


if __name__ == '__main__':
	root = './images/'
	img1path='./images/000.png'
	img2path = './images/020.png'
	# img1path = root + 'img1.jpg'
	# img2path = root + 'img2.jpg'
	# img1path='./images/img.jpg'
	# img2path = './images/img2.jpg'
	img1 = cv2.imread(img1path)
	img2 = cv2.imread(img2path)
	ddsize = (640, 480)
	img1 = cv2.resize(img1, ddsize)
	img2 = cv2.resize(img2, ddsize)
	time_start = time.time()
	gmf = GridMatchFilter(img1, img2)
	gmf.run()
	time_end = time.time();  # time.time()为1970.1.1到当前时间的毫秒数
	print('cost time is %fs' % (time_end - time_start))