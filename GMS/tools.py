import GMS
import cv2
import math
import numpy as np

def rotateimage(img,degree,center,panning):
    M=cv2.getRotationMatrix2D(center,degree,1)
    r,c,_=img.shape
    angle=degree*math.pi/180

    new_r=r*math.fabs(math.cos(angle))+c*math.fabs(math.sin(angle))
    new_c=r*math.fabs(math.sin(angle))+c*math.fabs(math.cos(angle))+(r-panning[0])*math.fabs(math.sin(angle))+(c-panning[1])*math.fabs(math.cos(angle))
    ddsize=(math.floor(new_c),math.floor(new_r))
    #旋转后平移到原位置
    # M[0][2] += center[0]*math.sin(angle)
    # M[1][2] += new_c-c
    #旋转后做平移
    lens=r*math.fabs(math.cos(angle))
    M[0][2]+=panning[0]
    M[1][2]+=panning[1]
    rotated=cv2.warpAffine(img,M,ddsize)
    return rotated


def main():
    print(__name__)
    img1=cv2.imread('./images/img1.jpg')
    img2 = cv2.imread('./images/img2.jpg')
    img1=cv2.resize(img1,(640,480))
    img2 = cv2.resize(img2, (640, 480))

    cv2.imshow('src img', img1)
    gms = GMS.GMS(img1, img2)
    # gms.show()
    gridmatch, kp1, kp2 = gms.getGmsMatches()
    # 做变换矩阵
    # 1、把match点保存到两个字符矩阵
    point1 = []
    point2 = []
    for match in gridmatch:
        point1.append(kp1[match.queryIdx].pt)
        point2.append(kp2[match.trainIdx].pt)
    point1=np.array(point1)
    point2=np.array(point2)

    a1=point1[1]
    a2=point2[1]
    b1=point1[-1]
    b2=point2[-1]
    panning=a1-a2
    # 计算角度
    ab1 = b1 - a1
    ab2 = b2 - a2
    # 缩放
    scale = math.sqrt(ab1.dot(ab1) / ab2.dot(ab2))
    print("scale:",scale)
    r, c, _ = img2.shape
    ddsize=(math.floor(scale*c),math.floor(scale*r))
    # img2=cv2.resize(img2,ddsize)
    # 做旋转，找出旋转角度，假设用的是第一个点

    #求角度
    degree=math.acos(ab1.dot(ab2)/math.sqrt(ab1.dot(ab1)*ab2.dot(ab2)))*180/math.pi
    center=(a2[0],a2[1])
    rimg=rotateimage(img2,degree,center,panning)
    cv2.imshow('dst img', img2)
    cv2.imshow('rotetaed img',rimg)
    cv2.waitKey()
    #平移




if __name__ == '__main__':
    main()