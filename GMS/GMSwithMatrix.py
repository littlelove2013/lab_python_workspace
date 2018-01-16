import cv2
import time
import numpy as np
import scipy.signal as ss
import Func
import math

eps=1e-4
class GMSwithMatrix:
    def __init__(self, img1, img2, kptnumber=10000, resizeflag=False, width=640, height=480):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        if resizeflag:
            ddsize = (width, height)
            self.img1=cv2.resize(self.img1,ddsize)
            self.img2 = cv2.resize(self.img2, ddsize)
        self.kptnumber=kptnumber
        self.init()
    
    def init(self):
        self.TreshFactor=6
        # 最大特征点数
        self.orb = cv2.ORB_create(self.kptnumber)
        self.orb.setFastThreshold(0)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
        # 提取并计算特征点
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
        #显示
        rawmatchimg = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches, None)
        cv2.imshow('rawmatchimg', rawmatchimg)
        # cv2.waitKey()

        self.gridmatchesindex=np.zeros([len(self.matches)])

        self.initgrid()#初始化网格
        self.gridmatches=[]
        # 统计一下坐标
        lens=len(self.matches)
        kp1r=np.zeros([lens])
        kp1c=np.zeros([lens])
        kp2r=np.zeros([lens])
        kp2c=np.zeros([lens])
        for i in range(lens):
            pt1=np.array(self.kp1[self.matches[i].queryIdx].pt)
            pt2 = np.array(self.kp2[self.matches[i].trainIdx].pt)
            kp1r[i] =pt1[1]
            kp1c[i] =pt1[0]
            kp2r[i] =pt2[1]
            kp2c[i] =pt2[0]
        kp1list=(np.array(kp1r,np.int32),np.array(kp1c,np.int32))
        kp2list=(kp2r,kp2c)
        leftsize=self.img1.shape[:2]
        rightsize=self.img2.shape[:2]
        #用于卷积计算阈值
        self.leftimg=np.zeros(leftsize)
        #用于计算是否为正确匹配
        # 假设初始时全部为假匹配
        self.TrueMatches= np.zeros(leftsize)

        #用于卷积计算打分,并获取匹配点
        self.leftmatchr = np.zeros(leftsize)
        self.leftmatchc = np.zeros(leftsize)

        # rightimg=np.zeros(rightsize)
        self.leftimg[kp1list]=1
        # self.TrueMatches[kp1list]=1
        #只保存匹配图片的匹配点坐标[r,c]
        self.leftmatchr[kp1list]=kp2r
        self.leftmatchc[kp1list]=kp2c
    def initgrid(self,leftgridnum=20,rightgridnum=20):
        self.lgn=leftgridnum
        self.rgn=rightgridnum
        #计算划分后网格的高和宽
        self.leftgridsize=(int(self.img1.shape[0]/self.lgn),int(self.img1.shape[1]/self.lgn))#[r,c]
        self.rightgridsize=(int(self.img2.shape[0]/self.rgn),int(self.img2.shape[1]/self.rgn))#[r,c]
    #根据给定gridid，返回在原图的上下限
    def getblock(self,gridid,gridsize):
        k,m=gridid
        start = (k * gridsize[0], m * gridsize[1])
        end = ((k + 1) * gridsize[0], (m + 1) *gridsize[1])
        return start,end
    #获取位于（leftgridid,rightid）一对网格内的左右特征点对，并返回
    def kindexingridpair(self,leftgridid,rightid):
        #获取左图坐标对应块
        lstart,lend=self.getblock(leftgridid,self.leftgridsize)
        rstart, rend = self.getblock(rightid, self.rightgridsize)
        #获取左图的块
        # leftblock=self.leftimg[lstart[0]:lend[0], lstart[1]:lend[1]]
        leftblock=np.zeros(shape=self.leftgridsize)
        # 行列号
        tmpleftmatchrgrid = self.leftmatchr[lstart[0]:lend[0], lstart[1]:lend[1]]
        tmpleftmatchcgrid = self.leftmatchc[lstart[0]:lend[0], lstart[1]:lend[1]]

        index = (tmpleftmatchrgrid >= rstart[0]) & (tmpleftmatchcgrid >= rstart[1]) & (
                tmpleftmatchrgrid <= rend[0]) & (tmpleftmatchcgrid <= rend[1])
        leftblock[index]=1
        rightkpoints=np.array(tmpleftmatchrgrid[index].reshape(-1),tmpleftmatchcgrid[index].reshape(-1))
        return leftblock,rightkpoints
    #计算阈值和得分
    def computescoreandthre(self):
        #计算阈值
        filter=np.ones(self.leftgridsize)
        tmp=Func.conv2withstride(self.leftimg,filter,stride=self.leftgridsize,start=None,gridnum=self.lgn)
        #显示计数
        Func.imagesc(tmp,'左图计数')
        threfilter=np.ones((3,3))/9#计算均值
        self.thre=ss.convolve2d(tmp,threfilter,'same',boundary='wrap')
        self.thre=self.TreshFactor*np.sqrt(self.thre)#阈值计算公式
        # 显示阈值
        Func.imagesc(self.thre,'阈值图')
        #计算打分
        self.socre=np.zeros((self.lgn,self.lgn))

        for i in range(self.lgn):#r
            for j in range(self.lgn):#c
                print("calc grid(%d,%d)"%(i,j))
                showdebug=False
                #对于img1中的每个网格区域和9邻域，计算其匹配的右img2的值
                bestareastart=(i*self.leftgridsize[0],j*self.leftgridsize[1])
                bestareaend = ((i+1) * self.leftgridsize[0], (j+1) * self.leftgridsize[1])
                leftkpoints=self.leftimg[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]
                # 建一个rightbetgrid,用于统计最匹配网格
                rightbestimg = np.zeros(self.img2.shape[:2])
                #行列号
                tmpleftmatchrgrid = self.leftmatchr[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]
                tmpleftmatchcgrid=self.leftmatchc[bestareastart[0]:bestareaend[0],bestareastart[1]:bestareaend[1]]
                #取索引并展平
                tmpleftmatchr=np.array(tmpleftmatchrgrid[tmpleftmatchrgrid!=0],np.int32)
                tmpleftmatchc=np.array(tmpleftmatchcgrid[tmpleftmatchcgrid!=0],np.int32)
                if tmpleftmatchr.size<10:
                    continue#点数小于阈值则不计算，默认为不匹配
                    showdebug=True
                #将展平的横纵坐标撒在图像上
                rightbestimg[(tmpleftmatchr,tmpleftmatchc)]=1
                Func.imagesc(rightbestimg, 'rightbestimg',ShowDebug=showdebug)
                #统计网格特征数
                filter = np.ones(self.rightgridsize)
                #测试不用卷积速度
                rightbestgrid=Func.conv2withstride(rightbestimg,filter,stride=self.rightgridsize,start=None,gridnum=self.rgn)
                # rightbestgrid=np.random.rand(self.rgn,self.rgn)
                # 显示得分
                Func.imagesc(rightbestgrid,'rightbestgrid',ShowDebug=showdebug)
                #取得分最大的网格[m,n]作为[i,j]对应的最匹配网格,因为可能有多个最大值，所以只取第一个
                rightbestgridindex=np.where(rightbestgrid==rightbestgrid.max())
                rightbestgridindex=(rightbestgridindex[0][0],rightbestgridindex[1][0])

                # 9邻域
                #有一个想法：对所有网格编号，若想取某一区域内网格点数不被其余网格污染，则用当前网格编号M，减去匹配网格编号N，
                # 如果网格内的值==|M-N|则对应的正好是匹配网格的点，否则是混乱网格匹配点，直接置0
                # 比如：L:（3，4,64）match R:(9,10,190),则将R特征点撒到R网格上，设撒的值=R.grid-pointl.grid,只有值==|64-190|的才是对应网格匹配
                # 乱点排序法
                neiborareastart = [(i - 1) * self.leftgridsize[0], (j - 1) * self.leftgridsize[1]]
                neiborareaend = [(i + 2) * self.leftgridsize[0], (j + 2) * self.leftgridsize[1]]
                if neiborareastart[0]<0:
                    neiborareastart[0]=0
                if neiborareastart[1] < 0:
                    neiborareastart[1] = 0
                if neiborareaend[0]>self.img1.shape[0]:
                    neiborareaend[0]=self.img1.shape[0]
                if neiborareaend[1]>self.img1.shape[1]:
                    neiborareaend[1] = self.img1.shape[1]
                #建一个right9neiborgrid用于统计9邻域得分
                right9neiborimg = np.zeros(self.img2.shape[:2])
                # 行列号
                tmpneibormatchr = self.leftmatchr[neiborareastart[0]:neiborareaend[0], neiborareastart[1]:neiborareaend[1]]
                tmpneibormatchc = self.leftmatchc[neiborareastart[0]:neiborareaend[0], neiborareastart[1]:neiborareaend[1]]
                # 取索引并展平
                tmpneibormatchr = np.array(tmpneibormatchr[tmpneibormatchr != 0],np.int32)
                tmpneibormatchc = np.array(tmpneibormatchc[tmpneibormatchc != 0],np.int32)
                # 将展平的横纵坐标撒在图像上
                right9neiborimg[(tmpneibormatchr, tmpneibormatchc)] = 1
                # 统计网格特征数
                # filter = np.ones(self.rightgridsize)
                right9neiborgrid = Func.conv2withstride(right9neiborimg, filter, stride=self.rightgridsize, start=None,gridnum=self.rgn)
                # right9neiborgrid=np.random.rand(self.rgn,self.rgn)
                # 显示计数
                Func.imagesc(right9neiborgrid,'right9neiborgrid',ShowDebug=showdebug)
                # 取得分最大的网格[m,n]作为[i,j]对应的最匹配网格
                # rightbestgridindex = np.where(rightbestgrid == rightbestgrid.max())
                #对9邻域卷积打分
                neiborfilter = np.ones((3, 3))
                neiborgird = ss.convolve2d(right9neiborgrid, neiborfilter, 'same', boundary='wrap')
                # neiborgird = np.random.rand(self.rgn,self.rgn)
                Func.imagesc(neiborgird, 'neiborgird',ShowDebug=showdebug)
                self.socre[i,j]=neiborgird[rightbestgridindex]
                #计算得分是否超过阈值，超过则accept矩阵取1,该网格内所有点为匹配点为匹配点
                #则可以用img保存所有的匹配点对，对所有特征点，若匹配则为1，然后去找对应的匹配坐标，若不匹配则为0
                if self.socre[i,j]>self.thre[i,j]:#则匹配度量加上匹配值，最后最匹配的点，其匹配值应该最大，取出其对应匹配的r,c坐标即可
                    #必须只保留匹配点位于最佳匹配格的的匹配点
                    k,m=rightbestgridindex
                    # m=rightbestgridindex[1]
                    bestrangestart=(k*self.rightgridsize[0],m*self.rightgridsize[1])
                    bestrangeend = ((k+1) * self.rightgridsize[0], (m+1) * self.rightgridsize[1])
                    # tmp=self.leftimg[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]].copy()
                    tmp=leftkpoints.copy()
                    index=(tmpleftmatchrgrid<bestrangestart[0])|(tmpleftmatchcgrid<bestrangestart[1])|(tmpleftmatchrgrid>bestrangeend[0])|(tmpleftmatchcgrid>bestrangeend[1])
                    tmp[index]=0
                    self.TrueMatches[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]+=tmp
        self.accept=self.socre>self.thre
        # 显示accept
        Func.imagesc(self.thre, 'accept')
        Func.imagesc(self.TrueMatches, 'TrueMatches')
    # 同样返回Match对象，用于其他用途
    def getTrueMatch(self,thre=1):
        Truelistindex=self.TrueMatches>=thre
        #获取img1的特征点列表
        leftkeypointarr=np.where(self.TrueMatches>=thre)
        #因为x,y对应的是[c,r]，所以需要反过来
        self.leftkeypoint = [cv2.KeyPoint(y, x,1) for (x, y) in zip(leftkeypointarr[0], leftkeypointarr[1])]
        #获取对应的img2的坐标
        rightkeypointarr=[self.leftmatchc[Truelistindex],self.leftmatchr[Truelistindex]]
        self.rightkeypoint=[cv2.KeyPoint(x, y,1) for (x, y) in zip(rightkeypointarr[0], rightkeypointarr[1])]
        #生成match
        lens=len(self.leftkeypoint)
        self.truematch=[cv2.DMatch(x,y,1) for (x,y) in zip(np.arange(0,lens),np.arange(0,lens))]
        #获取
        return self.leftkeypoint,self.rightkeypoint,self.truematch
    #画出TrueMatch的点
    def drawTrueMatch(self):
        self.getTrueMatch()
        gmsmatchimg=cv2.drawMatches(self.img1,self.leftkeypoint,self.img2,self.rightkeypoint,self.truematch,None)
        cv2.imwrite('GMSwithMatrix.png',gmsmatchimg)
        return gmsmatchimg

    #统计
    def run(self,type=0):
        if type==0:
            self.computescoreandthre()#计算出TrueMatcher
            # self.TrueMatches[np.arange(1,100,2),np.arange(1,100,2)]=1
            # return self.getTrueMatch()
            ssds=self.drawTrueMatch()
            cv2.imshow('ssds', ssds)
            cv2.waitKey()


class GMSwithGridFilter:
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
		# self.orb.setFastThreshold(0)
		self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
		self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
		# 提取并计算特征点
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
		# 显示
		rawmatchimg = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches, None)
		cv2.imshow('rawmatchimg', rawmatchimg)
		# cv2.waitKey()
		
		self.gridmatchesindex = np.zeros([len(self.matches)])
		
		self.initgrid()  # 初始化网格
		self.gridmatches = []
		# 统计一下坐标
		lens = len(self.matches)
		kp1r = np.zeros([lens])
		kp1c = np.zeros([lens])
		kp2r = np.zeros([lens])
		kp2c = np.zeros([lens])
		for i in range(lens):
			pt1 = np.array(self.kp1[self.matches[i].queryIdx].pt)
			pt2 = np.array(self.kp2[self.matches[i].trainIdx].pt)
			kp1r[i] = pt1[1]
			kp1c[i] = pt1[0]
			kp2r[i] = pt2[1]
			kp2c[i] = pt2[0]
		kp1list = (np.array(kp1r, np.int32), np.array(kp1c, np.int32))
		kp2list = (np.array(kp2r, np.int32), np.array(kp2c, np.int32))
		leftsize = self.img1.shape[:2]
		rightsize = self.img2.shape[:2]
		# 用于卷积计算阈值
		self.leftimg = np.zeros(leftsize)
		# 用于计算是否为正确匹配
		# 假设初始时全部为假匹配
		self.TrueMatches = np.zeros(leftsize)
		
		# 用于卷积计算打分,并获取匹配点
		self.leftmatchr = np.zeros(leftsize)
		self.leftmatchc = np.zeros(leftsize)
		self.leftimglabel = np.zeros(leftsize)
		# self.leftmatch = np.zeros(leftsize)
		self.leftmatchgrid = np.zeros(leftsize)
		# self.rightimglabel = np.zeros(rightsize)
		
		# 生成标签
		self.leftlabel = (np.arange(1, self.lgn ** 2 + 1).reshape(self.lgn, self.lgn)).repeat(self.leftgridsize[0], 0).repeat(self.leftgridsize[1], 1).astype(np.int32)
		self.rightlabel = (np.arange(1, self.rgn ** 2 + 1).reshape(self.rgn, self.rgn)).repeat(self.rightgridsize[0], 0).repeat(self.rightgridsize[1], 1).astype(np.int32)
		# Func.imagesc(self.leftlabel==20, 'leftlabel==20')
		# Func.imagesc(self.leftlabel == 21, 'leftlabel==21')
		# Func.imagesc(self.rightlabel, 'rightlabel')
		# rightimg=np.zeros(rightsize)
		self.leftimg[kp1list]=1
		self.leftimglabel[kp1list] = self.leftlabel[kp1list]
		# self.leftmatch[kp1list] = Func.index2value(kp2list, self.img2.shape[:2])
		# 只保存匹配特征所在的网格，反正也不会计算其实际坐标
		self.leftmatchgrid[kp1list] = self.rightlabel[kp2list]  # index2value(rightkpt, shape)
		# self.rightimglabel[kp2list] = self.rightlabel[kp2list]
		# Func.imagesc(self.leftimglabel, 'leftimglabel')
		# Func.imagesc(self.leftmatchgrid,'leftmatchgrid')
		# Func.imagesc(self.rightimglabel,'rightimglabel')
		# self.TrueMatches[kp1list]=1
		# 只保存匹配图片的匹配点坐标[r,c]
		self.leftmatchr[kp1list] = kp2r
		self.leftmatchc[kp1list] = kp2c
	
	def initgrid(self, leftgridnum=20, rightgridnum=20):
		self.lgn = leftgridnum
		self.rgn = rightgridnum
		# 计算划分后网格的高和宽
		self.leftgridsize = (math.ceil(self.img1.shape[0] / self.lgn), math.ceil(self.img1.shape[1] / self.lgn))  # [r,c]
		self.rightgridsize = (math.ceil(self.img2.shape[0] / self.rgn), math.ceil(self.img2.shape[1] / self.rgn))  # [r,c]
	
	# 根据给定gridid，返回在原图的上下限
	def getblock(self, gridid, gridsize):
		k, m = gridid
		start = (k * gridsize[0], m * gridsize[1])
		end = ((k + 1) * gridsize[0], (m + 1) * gridsize[1])
		return start, end
	
	# 获取位于（leftgridid,rightid）一对网格内的左右特征点对，并返回
	def kindexingridpair(self, leftgridid, rightid):
		# 获取左图坐标对应块
		lstart, lend = self.getblock(leftgridid, self.leftgridsize)
		rstart, rend = self.getblock(rightid, self.rightgridsize)
		# 获取左图的块
		# leftblock=self.leftimg[lstart[0]:lend[0], lstart[1]:lend[1]]
		leftblock = np.zeros(shape=self.leftgridsize)
		# 行列号
		tmpleftmatchrgrid = self.leftmatchr[lstart[0]:lend[0], lstart[1]:lend[1]]
		tmpleftmatchcgrid = self.leftmatchc[lstart[0]:lend[0], lstart[1]:lend[1]]
		
		index = (tmpleftmatchrgrid >= rstart[0]) & (tmpleftmatchcgrid >= rstart[1]) & (
				tmpleftmatchrgrid <= rend[0]) & (tmpleftmatchcgrid <= rend[1])
		leftblock[index] = 1
		rightkpoints = np.array(tmpleftmatchrgrid[index].reshape(-1), tmpleftmatchcgrid[index].reshape(-1))
		return leftblock, rightkpoints
	
	# 计算阈值和得分
	def computescoreandthre(self):
		# 计算阈值
		filter = np.ones(self.leftgridsize)
		self.leftgridkpoints = Func.conv2withstride(self.leftimg, filter, stride=self.leftgridsize, start=None, gridnum=self.lgn)
		# 显示计数
		# Func.imagesc(tmp, '左图计数')
		# threfilter = np.ones((3, 3)) / 9  # 计算均值
		threfilter = np.ones((3, 3))
		Q=self.leftgridkpoints>0
		Q=ss.convolve2d(Q, threfilter, 'same')
		self.thre = ss.convolve2d(self.leftgridkpoints, threfilter, 'same')
		self.thre = self.TreshFactor * np.sqrt(self.thre/(Q+eps))  # 阈值计算公式
		# 显示阈值
		Func.imagesc(self.thre, 'thre')
		# 计算打分
		self.socre = np.zeros((self.lgn, self.lgn))
		self.threnew = np.zeros((self.lgn, self.lgn))
		
		self.lgshape=(self.lgn,self.lgn)
		for i in range(self.lgn):  # r
			for j in range(self.lgn):  # c
				# print("calc grid(%d,%d)" % (i, j))
				if self.thre[i,j]==0:
					continue
				leftvalue=Func.index2value((i,j),self.lgshape)+1
				bestmatchgrid=self.leftmatchgrid[self.leftimglabel==leftvalue]
				if bestmatchgrid.size <10:
					continue  # 点数小于阈值则不计算，默认为不匹配
				# Func.imagesc(self.leftimglabel==leftvalue, 'leftimglabel')
				# Func.imagesc(self.leftmatchgrid, 'leftmatchgrid')
				number,n_counts=np.unique(bestmatchgrid,return_counts=True)
				rbestindex=number[np.argsort(n_counts)[-1]]
				index=(self.leftimglabel==leftvalue)&(self.leftmatchgrid==rbestindex)
				# if index.sum()<self.thre[i,j]:
				# 	continue
				#对于匹配的i,j网格，求其邻域，只需要取一个patch：比如（3，3）
				#则取以i为中心的label_patchi，取以j为中心的label_patch
				#label_patchi-label_patchj,取==0的像素，即为正确匹配的网格
				#而实际上，即为(self.leftimglabel-self.leftmatchgrid)==(leftvalue-rbestindex)
				#所以我搞定了多维度的问题？
				#如果最后再对其做一个高斯加权平均就更好了
				neiborwidth=1#多远 的算邻居
				neiborsindex=(self.leftimglabel-self.leftmatchgrid)==(leftvalue-rbestindex)
				lsize=np.array(self.leftgridsize)
				p=np.array([[i-neiborwidth,i+neiborwidth],[j-neiborwidth,j+neiborwidth]])*(lsize.reshape(-1,1))
				if neiborsindex[p[0,0]:p[0,1],p[1,0]:p[1,1]].sum()<self.thre[i,j]:
					continue
				# print(rbestindex, index.sum(), self.thre[i, j],neiborsindex[p[0,0]:p[0,1],p[1,0]:p[1,1]].sum())
				self.socre[i,j]=index.sum()
				self.TrueMatches+= index
				'''
				showdebug = False
				# 对于img1中的每个网格区域和9邻域，计算其匹配的右img2的值
				bestareastart = (i * self.leftgridsize[0], j * self.leftgridsize[1])
				bestareaend = ((i + 1) * self.leftgridsize[0], (j + 1) * self.leftgridsize[1])
				leftkpoints = self.leftimg[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]
				# 建一个rightbetgrid,用于统计最匹配网格
				rightbestimg = np.zeros(self.img2.shape[:2])
				# 行列号
				tmpleftmatchrgrid = self.leftmatchr[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]
				tmpleftmatchcgrid = self.leftmatchc[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]]
				# 取索引并展平
				tmpleftmatchr = np.array(tmpleftmatchrgrid[tmpleftmatchrgrid != 0], np.int32)
				tmpleftmatchc = np.array(tmpleftmatchcgrid[tmpleftmatchcgrid != 0], np.int32)
				if tmpleftmatchr.size < 10:
					continue  # 点数小于阈值则不计算，默认为不匹配
					showdebug = True
				# 将展平的横纵坐标撒在图像上
				rightbestimg[(tmpleftmatchr, tmpleftmatchc)] = 1
				Func.imagesc(rightbestimg, 'rightbestimg', ShowDebug=showdebug)
				# 统计网格特征数
				filter = np.ones(self.rightgridsize)
				# 测试不用卷积速度
				rightbestgrid = Func.conv2withstride(rightbestimg, filter, stride=self.rightgridsize, start=None,
				                                     gridnum=self.rgn)
				# rightbestgrid=np.random.rand(self.rgn,self.rgn)
				# 显示得分
				Func.imagesc(rightbestgrid, 'rightbestgrid', ShowDebug=showdebug)
				# 取得分最大的网格[m,n]作为[i,j]对应的最匹配网格,因为可能有多个最大值，所以只取第一个
				rightbestgridindex = np.where(rightbestgrid == rightbestgrid.max())
				rightbestgridindex = (rightbestgridindex[0][0], rightbestgridindex[1][0])
				
				# 9邻域
				# 有一个想法：对所有网格编号，若想取某一区域内网格点数不被其余网格污染，则用当前网格编号M，减去匹配网格编号N，
				# 如果网格内的值==|M-N|则对应的正好是匹配网格的点，否则是混乱网格匹配点，直接置0
				# 比如：L:（3，4,64）match R:(9,10,190),则将R特征点撒到R网格上，设撒的值=R.grid-pointl.grid,只有值==|64-190|的才是对应网格匹配
				# 乱点排序法
				neiborareastart = [(i - 1) * self.leftgridsize[0], (j - 1) * self.leftgridsize[1]]
				neiborareaend = [(i + 2) * self.leftgridsize[0], (j + 2) * self.leftgridsize[1]]
				if neiborareastart[0] < 0:
					neiborareastart[0] = 0
				if neiborareastart[1] < 0:
					neiborareastart[1] = 0
				if neiborareaend[0] > self.img1.shape[0]:
					neiborareaend[0] = self.img1.shape[0]
				if neiborareaend[1] > self.img1.shape[1]:
					neiborareaend[1] = self.img1.shape[1]
				# 建一个right9neiborgrid用于统计9邻域得分
				right9neiborimg = np.zeros(self.img2.shape[:2])
				# 行列号
				tmpneibormatchr = self.leftmatchr[neiborareastart[0]:neiborareaend[0],
				                  neiborareastart[1]:neiborareaend[1]]
				tmpneibormatchc = self.leftmatchc[neiborareastart[0]:neiborareaend[0],
				                  neiborareastart[1]:neiborareaend[1]]
				# 取索引并展平
				tmpneibormatchr = np.array(tmpneibormatchr[tmpneibormatchr != 0], np.int32)
				tmpneibormatchc = np.array(tmpneibormatchc[tmpneibormatchc != 0], np.int32)
				# 将展平的横纵坐标撒在图像上
				right9neiborimg[(tmpneibormatchr, tmpneibormatchc)] = 1
				# 统计网格特征数
				# filter = np.ones(self.rightgridsize)
				right9neiborgrid = Func.conv2withstride(right9neiborimg, filter, stride=self.rightgridsize, start=None,
				                                        gridnum=self.rgn)
				# right9neiborgrid=np.random.rand(self.rgn,self.rgn)
				# 显示计数
				Func.imagesc(right9neiborgrid, 'right9neiborgrid', ShowDebug=showdebug)
				# 取得分最大的网格[m,n]作为[i,j]对应的最匹配网格
				# rightbestgridindex = np.where(rightbestgrid == rightbestgrid.max())
				# 对9邻域卷积打分
				neiborfilter = np.ones((3, 3))
				neiborgird = ss.convolve2d(right9neiborgrid, neiborfilter, 'same', boundary='wrap')
				# neiborgird = np.random.rand(self.rgn,self.rgn)
				Func.imagesc(neiborgird, 'neiborgird', ShowDebug=showdebug)
				self.socre[i, j] = neiborgird[rightbestgridindex]
				# 计算得分是否超过阈值，超过则accept矩阵取1,该网格内所有点为匹配点为匹配点
				# 则可以用img保存所有的匹配点对，对所有特征点，若匹配则为1，然后去找对应的匹配坐标，若不匹配则为0
				if self.socre[i, j] > self.thre[i, j]:  # 则匹配度量加上匹配值，最后最匹配的点，其匹配值应该最大，取出其对应匹配的r,c坐标即可
					# 必须只保留匹配点位于最佳匹配格的的匹配点
					k, m = rightbestgridindex
					# m=rightbestgridindex[1]
					bestrangestart = (k * self.rightgridsize[0], m * self.rightgridsize[1])
					bestrangeend = ((k + 1) * self.rightgridsize[0], (m + 1) * self.rightgridsize[1])
					# tmp=self.leftimg[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]].copy()
					tmp = leftkpoints.copy()
					index = (tmpleftmatchrgrid < bestrangestart[0]) | (tmpleftmatchcgrid < bestrangestart[1]) | (
							tmpleftmatchrgrid > bestrangeend[0]) | (tmpleftmatchcgrid > bestrangeend[1])
					tmp[index] = 0
					self.TrueMatches[bestareastart[0]:bestareaend[0], bestareastart[1]:bestareaend[1]] += tmp
				'''
		self.accept = self.socre > self.thre
		Func.imagesc(self.socre, 'socre')
		# Func.imagesc(self.thre, 'oldthre')
		# 显示accept
		# Func.imagesc(self.TrueMatches, 'TrueMatches')
	
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
		cv2.imwrite('GMSwithGridFilter.png',gmsmatchimg)
		return gmsmatchimg
	
	# 统计
	def run(self, type=0):
		if type == 0:
			self.computescoreandthre()  # 计算出TrueMatcher
			# self.TrueMatches[np.arange(1,100,2),np.arange(1,100,2)]=1
			# return self.getTrueMatch()
			# ssds = self.drawTrueMatch()
			# cv2.imshow('ssds', ssds)
			# cv2.waitKey()

cv2.filter
class GMS:
    def __init__(self, img1, img2, kptnumber=10000, resizeflag=False, width=640, height=480):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        if resizeflag:
            ddsize = (width, height)
            self.img1=cv2.resize(self.img1,ddsize)
            self.img2 = cv2.resize(self.img2, ddsize)
        self.kptnumber=kptnumber
        self.init()
    def init(self):
        self.TreshFactor=6
        # 最大特征点数
        self.orb = cv2.ORB_create(self.kptnumber)
        # self.orb.setFastThreshold(0)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
        # 提取并计算特征点
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
        self.gridmatchesindex=np.zeros([len(self.matches)])
        self.gridmatches=[]
        # self.test()
    def test(self):
	    self.setparam()
	    lens = len(self.matches)
	    kp1r = np.zeros([lens]).astype(np.int32)
	    kp1c = np.zeros([lens]).astype(np.int32)
	    kp2r = np.zeros([lens]).astype(np.int32)
	    kp2c = np.zeros([lens]).astype(np.int32)
	    self.leftgridsize = np.array([self.grid1h, self.grid1w])
	    self.rightgridsize = np.array([self.grid2h, self.grid2w])
	    self.leftlabel = (np.arange(1, self.rows1 ** 2 + 1).reshape(self.rows1, self.rows1))\
		    .repeat(self.leftgridsize[0],0)\
		    .repeat(self.leftgridsize[1], 1).astype(np.int32)
	    self.rightlabel = (np.arange(1, self.rows2 ** 2 + 1).reshape(self.rows2, self.rows2))\
		    .repeat(self.rightgridsize[0],0)\
		    .repeat(self.rightgridsize[1], 1).astype(np.int32)
	    grid1=np.zeros(self.rows1**2)
	    # 用于卷积计算阈值
	    leftsize = self.img1.shape[:2]
	    rightsize = self.img2.shape[:2]
	    self.leftimg = np.zeros(leftsize)
	    self.rightimg = np.zeros(rightsize)
	    for i in range(lens):
		    pt1 = np.array(self.kp1[self.matches[i].queryIdx].pt)
		    pt2 = np.array(self.kp2[self.matches[i].trainIdx].pt)
		    kp1r[i] = int(pt1[1])
		    kp1c[i] = int(pt1[0])
		    kp2r[i] = int(pt2[1])
		    kp2c[i] = int(pt2[0])
		    id1, id2 = self.getRegion(self.matches[i], 1)
		    id11=self.leftlabel[kp1r[i], kp1c[i]]-1
		    id22=self.rightlabel[kp2r[i], kp2c[i]]-1
		    if id1!=id11 or id2!=id22:
			    print("error:\nid1=%d but id11=%d \nid2=%d but id22=%d"%(id1,id11,id2,id22))
		    grid1[id1]+=1
		    self.leftimg[(kp1r[i],kp1c[i])] =self.leftimg[(kp1r[i],kp1c[i])]+ 1
		    self.rightimg[(kp2r[i], kp2c[i])] = self.rightimg[(kp2r[i], kp2c[i])] + 1
	    kp1list = (kp1r,kp1c)
	    kp2list = (kp2r,kp2c)
	    if (self.leftimg>1).sum()>0:
		    print('it is one to more!')
	    if(self.rightimg > 1).sum() > 0:
	        print('it is more to one!')
	    
	    filter=np.ones(shape=self.leftgridsize)
	    self.leftgridkpoints= Func.conv2withstride(self.leftimg, filter, stride=self.leftgridsize, start=None, gridnum=self.rows1)
	    self.testimg=np.zeros(shape=(self.img1h,self.img1w))
	    for i in range(self.img1h):
		    for j in range(self.img1w):
			    id1 = (math.floor((j % self.img1w) / self.grid1w) + math.floor(
				    (i % self.img1h) / self.grid1h) * self.rows1)
			    self.testimg[i,j]=id1
	    Func.imagesc(grid1.reshape(self.rows1, self.rows1) - self.leftgridkpoints, 'GMS score')
	    Func.imagesc(self.testimg - self.leftlabel, 'GMS-GGF label testimg')
    #设置分成得网格得行宽
    def setparam(self,rows1=20,rows2=20):
        self.rows1=rows1
        self.rows2=rows2
        #img1 img2的网格宽和高
        self.img1w=self.img1.shape[1]
        self.img1h=self.img1.shape[0]
        self.img2w = self.img2.shape[1]
        self.img2h = self.img2.shape[0]
        self.grid1h=math.ceil(self.img1.shape[0]/rows1)
        self.grid1w = math.ceil(self.img1.shape[1] / rows1)
        self.grid2h = math.ceil(self.img2.shape[0] / rows2)
        self.grid2w = math.ceil(self.img2.shape[1] / rows2)
    #获取match点在左边的网格ID
    #type表示网格位移
    def getRegion(self,match,type=1):
        pt1=np.array(self.kp1[match.queryIdx].pt)
        pt2 = np.array(self.kp2[match.trainIdx].pt)
        if type==2:
            pt1[0]+= math.floor(self.grid1w/2)
            pt2[0] += math.floor(self.grid2w / 2)
        elif type==3:
            pt1[1]+=math.floor(self.grid1h/2)
            pt2[1] += math.floor(self.grid2h / 2)
        elif type==4:
            pt1[0] += math.floor(self.grid1w / 2)
            pt1[1] += math.floor(self.grid1h / 2)
            pt2[0] += math.floor(self.grid2w / 2)
            pt2[1] += math.floor(self.grid2h / 2)
        id1=(math.floor((pt1[0]%self.img1w)/self.grid1w) + math.floor((pt1[1]%self.img1h)/self.grid1h)*self.rows1)
        id2=(math.floor((pt2[0] % self.img2w) / self.grid2w) + math.floor((pt2[1] % self.img2h) / self.grid2h) * self.rows2)
        return id1,id2
    #self.bestmatch,记录最匹配的grid
    def getBestMatchGrid(self):
        self.bestmatch=np.ones([self.rows1**2])*-1
        self.validGrid=0
        for i in range(self.rows1**2):
            if self.listgrid1[i]<=0:
                continue
            a=self.listgrid2[i]
            match_max=a.max()
            max_index=np.where(self.listgrid2[i] == match_max)[0]
            self.bestmatch[i]=max_index[0]#取匹配相应最大的第一个值的下标索引
            self.validGrid=self.validGrid+1
    def getneibor(self,gridnum,num):
        neibor=np.zeros([9])
        for i in range(3):
            for j in range(3):
                neibor[j+3*i]=gridnum+(i-1)*num+(j-1)
        return neibor
    def getsocreandthre(self,gridnum):
        score = 0.0
        thresh = 0.0
        leftneibor = self.getneibor(gridnum,self.rows1)
        rightneibor = self.getneibor(self.bestmatch[gridnum],self.rows2)
        num=0
        for i in range(9):
            pos_left = int(leftneibor[i])
            pos_right=int(rightneibor[i])
            #若果grid内点数==0则不算
            if(pos_left<0 or pos_left>=self.rows1**2 or pos_right<0 or pos_right>=self.rows2**2 or self.listgrid1[gridnum]<=0):
                continue
            # print(pos_left,int(rightneibor[i]))
            positive_points=self.listgrid2[pos_left][pos_right]
            score=score+positive_points
            thresh=thresh+self.listgrid1[pos_left]
            num=num+1
        if num==0:
            return 0,0
        thresh=math.sqrt(thresh/(num))*self.TreshFactor
        return score,thresh
    def run(self,type=1):
        #设置参数
        self.setparam()
        grid1=np.zeros([self.rows1**2])
        grid2 = np.zeros([self.rows1**2,self.rows2 ** 2])
        #每次获取网格匹配点数
        for match in self.matches:
            id1,id2=self.getRegion(match,type)
            grid1[id1]=grid1[id1]+1
            grid2[id1,id2]=grid2[id1,id2]+1
        self.listgrid1=grid1
        self.listgrid2=grid2
        #计算最匹配的网格
        self.getBestMatchGrid()
        #计算每个网格的得分，并判断对应网格是否通过
        localthreshold=0
        ACCEPTSCORE=0
        globalthreshold = math.sqrt(len(self.matches) / self.validGrid) * self.TreshFactor
        if not localthreshold:
            ACCEPTSCORE=globalthreshold
        for i in range(self.rows1 ** 2):
            if self.listgrid1[i]<=0:
                continue;
            score,localthreshold=self.getsocreandthre(i)
            if localthreshold>0:
                ACCEPTSCORE=localthreshold
            if score<ACCEPTSCORE:
                self.bestmatch[i]=-2
        #计算出网格里的match点数
        for i in range(len(self.matches)):
            match=self.matches[i]
            id1,id2=self.getRegion(match,type)
            #只有处于两个最匹配的网格才保留
            if self.bestmatch[id1]==id2:
                self.gridmatchesindex[i]=1

    def getGmsMatches(self):
        for i in range(1):
            self.run(i+1)
        for i in range(len(self.matches)):
            
            if(self.gridmatchesindex[i]==1):
                self.gridmatches.append(self.matches[i])
                #记录新的kpt并返回
        return self.gridmatches,self.kp1,self.kp2

    def getGmsMatchesImg(self):
        gmsmatches = self.getGmsMatches()
        return cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, gmsmatches,None)
    def show(self):
        cv2.drawKeypoints(image=self.img1,keypoints=self.kp1,outImage=self.img1)
        cv2.drawKeypoints(image=self.img2, keypoints=self.kp2,outImage=self.img2)
        matchimg=cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.matches,None)
        gmsmatches,kp1,kp2=self.getGmsMatches()
        gmsmatchimg=cv2.drawMatches(self.img1,kp1,self.img2,kp2,gmsmatches,None)
        cv2.imshow('img1', self.img1)
        cv2.imshow('img2', self.img2)
        cv2.imshow('matchimg', matchimg)
        cv2.imshow('gmsmatchimg', gmsmatchimg)
        cv2.imwrite('gms.png',gmsmatchimg)
        cv2.waitKey()


def main():
    print(__name__)
    root='./images/'
    img1path='./images/000.png'
    img2path = './images/020.png'
    # img1path=root+'img1.jpg'
    # img2path = root+'img2.jpg'
    # img1path='./images/img.jpg'
    # img2path = './images/img2.jpg'
    img1=cv2.imread(img1path)
    img2=cv2.imread(img2path)
    ddsize=(640,480)
    img1 = cv2.resize(img1, ddsize)
    img2 = cv2.resize(img2, ddsize)
    #gms
    time_start = time.time()
    gms = GMS(img1, img2)
    gms.run()
    time_end = time.time();  # time.time()为1970.1.1到当前时间的毫秒数
    print('cost time is %fs' % (time_end - time_start))
    # #GMSwithMatrix
    # time_start=time.time()
    # gms=GMSwithMatrix(img1,img2)
    # gms.run()
    # time_end=time.time();#time.time()为1970.1.1到当前时间的毫秒数
    # print('cost time is %fs'%(time_end-time_start))
    # #GMSwithGridFilter
    # time_start = time.time()
    ggf = GMSwithGridFilter(img1, img2)
    ggf.run()
    time_end = time.time();  # time.time()为1970.1.1到当前时间的毫秒数
    print('cost time is %fs' % (time_end - time_start))
    # print(((ggf.leftgridkpoints.reshape(-1) - gms.listgrid1)**2).sum())
    print('leftimg var:',((ggf.leftimg - gms.leftimg) ** 2).sum())
    print('listgrid1 var:', ((ggf.leftgridkpoints.reshape(-1) - gms.listgrid1) ** 2).sum())
    
    # Func.imagesc(ggf.leftgridkpoints, 'GGF socre')
    
    #gms.show()
    a=np.ones([4,4])
    b=np.ones([3,3])
    
    

if __name__=='__main__':
    main();