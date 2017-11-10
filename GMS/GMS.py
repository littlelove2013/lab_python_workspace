import cv2
import numpy as np

class GMS:


    def __init__(self,img1path,img2path,kptnumber=10000,resizeflag=True,width = 640,height = 480):

        self.img1=cv2.imread(img1path)
        self.img2=cv2.imread(img2path)
        if resizeflag:
            ddsize = (width, height)
            self.img1=cv2.resize(self.img1,ddsize)
            self.img2 = cv2.resize(self.img2, ddsize)
        # 最大特征点数,需要修改，5000太大。
        self.orb = cv2.ORB_create(kptnumber)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.img2, None)
        # 提取并计算特征点
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches = self.bf.match(self.des1, trainDescriptors = self.des2)


    def show(self):
        cv2.drawKeypoints(image=self.img1,keypoints=self.kp1,outImage=self.img1)
        cv2.drawKeypoints(image=self.img2, keypoints=self.kp2,outImage=self.img2)

        matchimg=cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.matches,None)
        cv2.imshow('img1', self.img1)
        cv2.imshow('img2', self.img2)
        cv2.imshow('matchimg', matchimg)
        cv2.waitKey()


def main():
    print(__name__)
    img1path='./images/img.jpg'
    img2path = './images/img2.jpg'
    gms=GMS(img1path,img2path)
    gms.show()

if __name__ == '__main__':
    main()