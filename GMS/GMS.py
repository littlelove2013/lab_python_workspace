import cv2
import math
import numpy as np
import scipy.signal#卷积库

eps=1e-4

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
        self.orb.setFastThreshold(0)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
        # 提取并计算特征点
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches = self.bf.match(self.des1, trainDescriptors=self.des2)
        self.gridmatchesindex=np.zeros([len(self.matches)])
        self.gridmatches=[]
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
                continue;
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
            if(pos_left<0 or pos_left>=self.rows1**2 or pos_right<0 or pos_right>=self.rows2**2 or self.listgrid1[gridnum]<=0):
                continue;
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
            if self.bestmatch[id1]==id2:
                self.gridmatchesindex[i]=1

    def getGmsMatches(self):
        for i in range(4):
            self.run(i+1)
        for i in range(len(self.matches)):
            if(self.gridmatchesindex[i]==1):
                self.gridmatches.append(self.matches[i])
        return self.gridmatches
    def getGmsMatchesImg(self):
        gmsmatches = self.getGmsMatches()
        return cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, gmsmatches, None)
    def show(self):
        cv2.drawKeypoints(image=self.img1,keypoints=self.kp1,outImage=self.img1)
        cv2.drawKeypoints(image=self.img2, keypoints=self.kp2,outImage=self.img2)
        matchimg=cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.matches,None)
        gmsmatches=self.getGmsMatches()
        gmsmatchimg=cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,gmsmatches,None)
        cv2.imshow('img1', self.img1)
        cv2.imshow('img2', self.img2)
        cv2.imshow('matchimg', matchimg)
        cv2.imshow('gmsmatchimg', gmsmatchimg)
        cv2.waitKey()


def main():
    print(__name__)
    img1path='./images/000.png'
    img2path = './images/020.png'
    # img1path='./images/img.jpg'
    # img2path = './images/img2.jpg'
    img1=cv2.imread(img1path)
    img2=cv2.imread(img2path)
    gms=GMS(img1,img2)
    gms.show()

if __name__ == '__main__':
    main()