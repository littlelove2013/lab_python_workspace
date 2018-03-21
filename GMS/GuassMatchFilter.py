import cv2
import time
import numpy as np
# import scipy.signal as ss
import Func
import math
import sys

src_folder="./images/"
res_folder="./matches/"
eps=1e-4
class GuassMatchFilter:
	def __init__(self, img1, img2, kptnumber=10000, resizeflag=False, width=640, height=480,savename='GMF'):
		self.savename=savename
		self.DEBUG=False
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
		self.TreshFactor = 6.0
		# 最大特征点数
		self.orb = cv2.ORB_create(self.kptnumber)
		self.orb.setFastThreshold(0)
		self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
		self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
		# 提取并计算特征点
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
		# 显示
		# rawmatchimg = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches, None)
		# cv2.imshow('rawmatchimg', rawmatchimg)
		# cv2.waitKey()
	
	def initgrid(self, leftgridnum=20, rightgridnum=20):
		self.lgn = leftgridnum
		self.rgn = rightgridnum
		# 计算划分后网格的高和宽
		self.leftgridsize = np.array([
		math.ceil(self.img1.shape[0] / self.lgn), math.ceil(self.img1.shape[1] / self.lgn)])  # [r,c]
		self.rightgridsize = np.array([
		math.ceil(self.img2.shape[0] / self.rgn), math.ceil(self.img2.shape[1] / self.rgn)])  # [r,c]
		# 生成标签矩阵
		# self.leftlabel = (np.arange(1, self.lgn ** 2 + 1).reshape(self.lgn, self.lgn)) \
		# 	.repeat(self.leftgridsize[0], 0).repeat(self.leftgridsize[1], 1).astype(np.int32)
		# self.rightlabel = (np.arange(1, self.rgn ** 2 + 1).reshape(self.rgn, self.rgn)) \
		# 	.repeat(self.rightgridsize[0], 0).repeat(self.rightgridsize[1], 1).astype(np.int32)

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
		self.leftimg = np.zeros(leftsize).astype(np.float32)
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
		# self.TrueMatches = np.zeros(leftsize)
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
		else:
			print("error:ktype not support!")
			return 0
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
		if self.DEBUG or show:
			Func.imagesc(self.leftlabel,"leftlabel shift=(%d,%d)"%(shift[0],shift[1]))
			Func.imagesc(self.leftimglabel, "leftimglabel shift=(%d,%d)" % (shift[0], shift[1]))
			Func.imagesc(self.leftmatchgrid, "leftmatchgrid shift=(%d,%d)" % (shift[0], shift[1]))
	#返回三个核
	def createKernel(self,shift=(0,0),ktype='s',sigma=1,neiborwidth=1):
		self.neiborwidth = neiborwidth  # 多远 的算邻居
		self.neibor=(2*self.neiborwidth+1)
		self.ktype=ktype
		self.sigma = sigma
		size=self.leftgridsize[0]*self.leftgridsize[1]*self.neibor
		# self.kernel=self.getKernel(type='g',sigma=1,ksize=self.leftgridsize*neibor)*size
		# self.kernel = self.getKernel(type='g', sigma=1, ksize=[self.neibor,self.neibor]).repeat(self.leftgridsize[0],0).repeat(self.leftgridsize[1],1)
		# self.kernel = self.getKernel(type='s', sigma=1, ksize=self.leftgridsize * neibor)
		self.kernel = self.getKernel(type=self.ktype, sigma=self.sigma, ksize=[self.neibor, self.neibor])\
			.repeat(self.leftgridsize[0],0)\
			.repeat(self.leftgridsize[1], 1)
		if self.DEBUG:
			print("kernel sum=%f"%(self.kernel.sum()))
		self.stride=self.leftgridsize
		self.start = (self.leftgridsize / 2).astype(np.int32)
		if shift[0]==1:
			self.start[0]=0
		if shift[1]==1:
			self.start[1]=0
	#获取邻域
	def getneighbor(self,lgrid,rgrid):
		# tmp = np.zeros(self.leftgridsize*self.neibor)
		#取出self.leftgridsize*self.neibor这么一小块的部分
		# rgrid=rgrid.reshape(-1)
		start=self.leftgridsize*(lgrid-self.neiborwidth)
		start[start<=0]=0
		end=start+self.leftgridsize*self.neibor
		if end[0]>self.img1.shape[0]:
			end[0]=self.img1.shape[0]
		if end[1] > self.img1.shape[1]:
			end[1] = self.img1.shape[1]
		mask=(self.leftimglabel[start[0]:end[0],start[1]:end[1]]>eps).reshape(-1)
		lr,lc=Func.value2index(self.leftimglabel[start[0]:end[0],start[1]:end[1]]-1,shape=(self.lgn,self.lgn))
		lr=np.abs(lr-lgrid[0])
		lc = np.abs(lc - lgrid[1])
		rr,rc=Func.value2index(self.leftmatchgrid[start[0]:end[0],start[1]:end[1]]-1,shape=(self.lgn,self.lgn))
		rr = np.abs(rr - rgrid[0])
		rc = np.abs(rc - rgrid[1])
		#取横轴坐标中大的
		lr[lr<=lc]=lc[lr<=lc]
		rr[rr<=rc]=rc[rr<=rc]
		num=((lr==rr)&mask).sum()
		if self.DEBUG:
			print(lgrid, rgrid,start,end,rr.shape,lr.shape,num)
		klen=end-start
		#过滤网格不匹配的
		# return ((lr==rr).reshape(klen)*self.kernel[0:klen[0],0:klen[1]]).sum()
		return num
	# 计算阈值和得分
	def computescoreandthre(self):
		# 计算阈值
		self.thre = Func.conv2withstride(self.leftimg, self.kernel, stride=self.stride, start=self.start,
		                                            gridnum=self.lgn)
		# De=self.neibor**2
		self.thre = self.TreshFactor * np.sqrt(self.thre)  # 阈值计算公式
		if self.DEBUG:
			print("self.leftimg:max:%f,min:%f" % (self.leftimg.max(), self.leftimg.min()))
			print("self.thre:max:%f,min:%f" % (self.thre.max(), self.thre.min()))
			print("self.thre:max:%f,min:%f" % (self.thre.max(), self.thre.min()))
		# 计算打分
		self.score = np.zeros((self.lgn, self.lgn))
		self.lgshape = (self.lgn, self.lgn)
		# nnb=1
		for i in range(self.lgn):  # r
			for j in range(self.lgn):  # c
				if self.thre[i, j] == 0:
					continue
				leftvalue = Func.index2value((i, j), self.lgshape) + 1
				bestmatchgrid = self.leftmatchgrid[self.leftimglabel == leftvalue]
				if bestmatchgrid.size < 3:
					continue  # 点数小于阈值则不计算，默认为不匹配
				number, n_counts = np.unique(bestmatchgrid, return_counts=True)
				rbestindex = number[np.argsort(n_counts)[-1]]
				r2dindex=Func.value2index(rbestindex-1, shape=(self.lgn, self.lgn))
				index = (self.leftimglabel == leftvalue) & (self.leftmatchgrid == rbestindex)
				# neiborsindex = (((self.leftimglabel - self.leftmatchgrid) == (leftvalue - rbestindex)) & self.leftimg.astype(np.bool)).astype(np.float32)
				# neiborsindexconv = Func.conv2withstride(neiborsindex, self.kernel,
				#                                         stride=self.stride, start=self.start, gridnum=self.lgn)
				# self.score[i, j] = neiborsindexconv[i, j]
				self.score[i, j] = self.getneighbor(np.array([i,j]),np.array(r2dindex).reshape(-1))
				if self.DEBUG:
					print("calc grid(%d,%d)\n thre=%f,index.sum=%d,neiborsindex.sum=%d,score=%f"
				      % (i, j,self.thre[i,j],index.sum(),neiborsindex.sum(),self.score[i, j]))
				if self.score[i, j] < self.thre[i, j]:
					continue
				# nnb += 1
				self.TrueMatches += index
		# print("batchsize %d"%(nnb))
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
		if not self.savename=='':
			filename = res_folder + self.savename + "_gridnum(" + str(
				self.lgn) + ")_ktype(" + self.ktype + ")_sigma(" + str(self.sigma) + ")_neibor(" + str(
				self.neiborwidth) + ").png"
			cv2.imwrite(filename, gmsmatchimg)
		return gmsmatchimg
	
	# 统计
	def run(self,gridnum=20,ktype='s',sigma=1,neiborwidth=1):
		self.TrueMatches = np.zeros(self.img1.shape[:2])
		self.initgrid(gridnum,gridnum)  # 初始化网格
		self.multiplemap()  # 将matches转化为矩阵
		for i in range(4):
			shift=(i%2,int(i/2))
			# shift=(1,1)
			self.createlabel(shift=shift,show=False)
			self.createKernel(shift=shift,ktype=ktype,sigma=sigma,neiborwidth=neiborwidth)
			self.computescoreandthre()  # 计算出TrueMatcher
			# self.TrueMatches[np.arange(1,100,2),np.arange(1,100,2)]=1
			# return self.getTrueMatch()
		return self.drawTrueMatch()
		# Func.imshow(ssds)

def main(argv):
	if len(argv)<2:
		print("input arguments like this:\n\timg1 img2 gridnum(optional) ktype(optional) sigma(optional) neiborwidth(optional) savename(optional)")
	img1path=argv[1]
	img2path=argv[2]
	args=[20,'s',1,1,"GAF_beer_k4"]
	for i in range(5):
		if len(argv)>i+3:
			args[i]=argv[i+3]
	args[0]=int(args[0])
	args[2] = float(args[2])
	args[3] = int(args[3])
	# print(args)
	print("arglist:gridnum:%d,ktype:%s,sigma:%f,neiborwidth:%d,savename:%s_"%(args[0],args[1],args[2],args[3],args[4]))
	gridnum,ktype,sigma,neiborwidth,savename=args
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
	#对img2做旋转
	# M=cv2.getRotationMatrix2D((ddsize[0]/2,ddsize[1]/2),90,1)
	# dst=cv2.warpAffine(img2,M,ddsize)
	# cv2.imshow("rot 90",dst)
	# cv2.waitKey()
	time_start = time.time()
	gmf = GuassMatchFilter(img1, img2, savename=savename)
	gmf.run(gridnum=gridnum, ktype=ktype, sigma=sigma, neiborwidth=neiborwidth)
	# gmf.run(gridnum=20, ktype='s', sigma=1, neiborwidth=1)
	# gmf.run(gridnum=40,ktype='g', sigma=1.2, neiborwidth=5)
	time_end = time.time() # time.time()为1970.1.1到当前时间的毫秒数
	print('cost time is %fs' % (time_end - time_start))

if __name__ == '__main__':
	main(sys.argv)

