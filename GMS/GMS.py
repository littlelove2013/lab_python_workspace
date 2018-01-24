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
        if self.listgrid1[gridnum]<=0:
            return 0,0
        score = 0.0
        thresh = 0.0
        leftneibor = self.getneibor(gridnum,self.rows1)
        rightneibor = self.getneibor(self.bestmatch[gridnum],self.rows2)
        num=0
        for i in range(9):
            pos_left = int(leftneibor[i])
            pos_right=int(rightneibor[i])
            if(pos_left<0 or pos_left>=self.rows1**2 or pos_right<0 or pos_right>=self.rows2**2):
                continue;
            # print(pos_left,int(rightneibor[i]))
            positive_points=self.listgrid2[pos_left][pos_right]
            score=score+positive_points
            thresh=thresh+self.listgrid1[pos_left]
            num=num+1
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
        # nnd=1
        for i in range(self.rows1 ** 2):
            if self.listgrid1[i]<=0:
                continue;
            # nnd+=1
            score,localthreshold=self.getsocreandthre(i)
            if localthreshold>0:
                ACCEPTSCORE=localthreshold
            if score<ACCEPTSCORE:
                self.bestmatch[i]=-2
        # print("batchsize %d"%(nnd))
        #计算出网格里的match点数
        for i in range(len(self.matches)):
            match=self.matches[i]
            id1,id2=self.getRegion(match,type)
            if self.bestmatch[id1]==id2:
                self.gridmatchesindex[i]=1

    def getGmsMatches(self):
        for i in range(4):
            self.run(i+1)
        # self.gridmatches=[self.matches[i] for i in range(len(self.matches)) if self.gridmatchesindex[i]==1]
        self.gridmatches=[]
        self.leftkeypoint=[]
        self.rightkeypoint=[]
        number=0
        for i in range(len(self.matches)):
	        if self.gridmatchesindex[i] == 1:
		        self.gridmatches.append(cv2.DMatch(number,number,1))
		        self.leftkeypoint.append(self.kp1[self.matches[i].queryIdx])
		        self.rightkeypoint.append(self.kp2[self.matches[i].trainIdx])
		        number+=1
        return self.gridmatches,self.leftkeypoint, self.rightkeypoint
    def getGmsMatchesImg(self):
        return cv2.drawMatches(self.img1, self.leftkeypoint, self.img2, self.rightkeypoint, self.gridmatches,None)
    def show(self,justwritetofile=False):
        # cv2.drawKeypoints(image=self.img1,keypoints=self.kp1[:100],outImage=self.img1)
        # cv2.drawKeypoints(image=self.img2, keypoints=self.kp2[:200],outImage=self.img2)
        matchimg=cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.matches,None)
        gmsmatches,kp1,kp2=self.getGmsMatches()
        gmsmatchimg=cv2.drawMatches(self.img1, kp1, self.img2, kp2, gmsmatches,None)
        cv2.imwrite('gmsmatchimg.png', gmsmatchimg)
        if not justwritetofile:
            cv2.imshow('img1', self.img1)
            cv2.imshow('img2', self.img2)
            cv2.imshow('matchimg', matchimg)
            cv2.imshow('gmsmatchimg', gmsmatchimg)
            cv2.waitKey()

def getTransformPoint(oripoint,tran):
    oriP=np.array([[oripoint[0]],[oripoint[1]],[1]])
    targetP=tran.dot(oriP)
    return np.array([targetP[0]/targetP[2],targetP[1]/targetP[2]])

def main():
    print(__name__)
    root='./images/'
    # img1path='./images/000.png'
    # img2path = './images/020.png'
    img1path=root+'img1.jpg'
    img2path = root+'img2.jpg'
    # img1path='./images/img.jpg'
    # img2path = './images/img2.jpg'
    img1=cv2.imread(img1path)
    img2=cv2.imread(img2path)
    ddsize=(640,480)
    img1 = cv2.resize(img1, ddsize)
    img2 = cv2.resize(img2, ddsize)
    gms=GMS(img1,img2)
    gms.show()
    gridmatch,kp1,kp2=gms.getGmsMatches()
    #做变换矩阵
    #1、把match点保存到两个字符矩阵
    point1=[]
    point2=[]
    for match in gridmatch:
        point1.append(kp1[match.queryIdx].pt)
        point2.append(kp2[match.trainIdx].pt)
    #2、求得变换矩阵
    homo=cv2.findHomography(np.float32(point2),np.float32(point1),cv2.RANSAC)
    # pts1=np.float32(point1)
    # pts2=np.float32(point2)
    # tran = cv2.getPerspectiveTransform(pts1[0:4], pts2[0:4])
    tran=np.array(homo[0])
    adjustMat=np.array([[1,0,img1.shape[1]],[0,1,0],[0,0,1]])
    adjustHomo = adjustMat.dot(tran)
    oripoint=point1[0]
    targetpoint=getTransformPoint(oripoint,adjustHomo)
    basepoint = point2[0]
    print('homo:\n',tran)
    #图像配准
    img2transfom=cv2.warpPerspective(img2,tran,(2*img1.shape[1],img1.shape[0]))
    #cv2.imshow('转换后图像',img2transfom)
    tmpimg = img2transfom[:,0:img1.shape[1]]
    index=np.where(tmpimg>0)
    #将img1拷贝到img2transfom上去：
    catimg=img2transfom.copy()
    catimg[:,0:img1.shape[1],:]=img1.copy()
    cv2.imshow('catimage',catimg)
    cv2.imwrite('catimg.jpg',catimg)
    #优化拼接地方的亮度不平衡
    left = np.min(index[1])#取得最左边界
    for i in range(len(index[1])):
        alpha=(img1.shape[1]-index[1][i])/(img1.shape[1]-left)
        r=index[0][i]
        c=index[1][i]
        chanel=index[2][i]
        catimg[r,c,chanel]=img1[r,c,chanel]*alpha+img2transfom[r,c,chanel]*(1-alpha)
    cv2.imshow('catimage',catimg)
    cv2.imwrite('smoothborder.jpg',catimg)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()