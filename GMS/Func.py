import numpy as np
import random
import scipy.signal as ss
import math
import matplotlib.pyplot as py

def conv2withstride(m,filter,stride=(1,1),start=None,gridnum=20):
    tmp=ss.convolve2d(m,filter,'same',boundary='wrap')
    if start==None:
        start=(int(stride[0]/2),int(stride[1]/2))
    r=np.arange(0,gridnum)*stride[0]+start[0]
    c=np.arange(0,gridnum)*stride[0]+start[1]
    grid=np.zeros([gridnum,gridnum])
    for i in range(gridnum):
        grid[i,:]=tmp[r[i],c]
    return grid
#值转换为二维索引,列优先,value应该是一个向量或矩阵，返回的应该是二维矩阵
def value2index(value,shape):
    [h,w]=shape
    #展平数组
    v=np.array(value).ravel()
    r=np.array(v%h,np.int32)
    c=np.array(np.floor(v/h),np.int32)
    res= (r,c)
    return res
#二维索引转为value
def index2value(index,shape):
    [h, w] = shape
    index=np.array(index)
    value=index[1]*h+index[0]
    return value

#imagesc
def imagesc(matrix,title=None,savefile=True,savepath='./dataset/',ShowDebug=True):
    if ShowDebug:
        r,c=matrix.shape
        py.imshow(matrix,extent=[0,r,0,c])
        if title!=None:
            py.title(title)
        py.show()
#按匹配网格分匹配点
def test():
    kpnum=10000
    shape=np.array([640,480])
    [r,c]=shape
    gridnum=20
    lgn=(shape/20).astype(np.int32)
    rgn=(shape/20).astype(np.int32)
    imagesize=640*480
    leftimg=np.zeros(shape=(r,c))
    leftmatch=np.zeros(shape=(r,c))
    leftmatchgrid=np.zeros(shape=(r,c))
    rightimg = np.zeros(shape=(r, c))
    #生成标签
    leftlabel=(np.arange(1,gridnum**2+1).reshape(gridnum,gridnum)).repeat(lgn[0],0).repeat(lgn[1],1)
    rightlabel = (np.arange(1, gridnum**2+1).reshape(gridnum,gridnum)).repeat(rgn[0],0).repeat(rgn[1],1)
    # imagesc(leftlabel,'leftlabel')
    # imagesc(rightlabel,'rightlabel')
    #随机生成匹配点
    leftkpt=value2index(random.sample(range(0,imagesize),kpnum),shape)
    rightkpt=value2index(random.sample(range(0,imagesize),kpnum),shape)
    leftimg[leftkpt]=leftlabel[leftkpt]
    leftmatch[leftkpt]=index2value(rightkpt,shape)
    #只保存匹配特征所在的网格，反正也不会计算其实际坐标
    leftmatchgrid[leftkpt] = rightlabel[rightkpt]#index2value(rightkpt, shape)
    rightimg[rightkpt]=rightlabel[rightkpt]
    # imagesc(leftmatchgrid,'leftmatchgrid')
    # imagesc(rightimg,'rightimg')
    #生成以(i,j)为匹配网格的对应网格匹配点
    i,j=[9,300]
    lindex=(leftlabel==i)&(leftmatchgrid==j)
    imagesc(leftlabel==i, 'leftlabel==i')
    imagesc(leftmatchgrid==j, 'leftmatchgrid==j')
    print(leftmatchgrid[lindex])
    imagesc(rightlabel * (rightlabel==j), 'rightlabelgrid')

    rightmatchimg = np.zeros(shape=(r, c))
    rightmatchimg[rightkpt] = rightlabel[rightkpt]
    rightmatchimg=rightmatchimg-leftimg
    # imagesc(rightmatchimg,'rightmatchimg')

def main():
    a=np.ones([6,6])
    b=np.ones([2,2])
    c=conv2withstride(a,b,gridnum=2)
    print(a,b,c)

    c=np.array([[1,2,3],[4,5,6],[7,8,9]])
    d=np.array([[2,4],[7,8]])
    index=value2index(d,c.shape)
    value=index2value(index,c.shape)

    print(c,value,c[index])

if __name__ == '__main__':
    # main()
    test()