import numpy as np
import scipy.signal as ss
import math
# import matplotlib.pyplot as py

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
    v=value.ravel()
    r=np.array(v%h,np.int32)
    c=np.array(np.floor(v/h),np.int32)
    res= (r,c)
    return res
#二维索引转为value
def index2value(index,shape):
    [h, w] = shape
    value=index[1]*h+index[0]
    return value

#imagesc
def imagesc(matrix,title=None,savefile=True,savepath='./dataset/',ShowDebug=True):
    if 0:
        r,c=matrix.shape
        py.imshow(matrix,extent=[0,r,0,c])
        if title!=None:
            py.title(title)
        py.show()
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
    main()