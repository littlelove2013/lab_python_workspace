import numpy as np
import random
import scipy.signal as ss
import cv2
import math
import matplotlib.pyplot as py
import tensorflow as tf
import time

eps=1e-8
def conv2withstride(m,filter,stride=(1,1),start=None,gridnum=20):
    # s=cv2.getTickCount()
    tmp=cv2.filter2D(m,-1,filter)+eps
    # tmp=ss.convolve2d(m,filter,'same')
    # e=cv2.getTickCount()
    # print("conv cost time %f",(e-s)/cv2.getTickFrequency())
    if start==None:
        start=(math.floor(stride[0]/2),math.floor(stride[1]/2))
    r = np.arange(start[0], (gridnum) * stride[0], stride[0]).repeat(gridnum)
    c=list(range(start[1],(gridnum)*stride[1],stride[1]))*gridnum
    
    # r=np.arange(0,gridnum)*stride[0]+start[0]
    # c=np.arange(0,gridnum)*stride[1]+start[1]
    # grid=np.zeros([gridnum,gridnum])
    # for i in range(gridnum):
    #     grid[i,:]=tmp[r[i],c]
    # print(r,c,tmp[(r,c)].shape)
    return tmp[(r,c)].reshape(gridnum,gridnum)
#值转换为二维索引,列优先,value应该是一个向量或矩阵，返回的应该是二维矩阵
#axis==0表示行优先
def value2index(value,shape,axis=0):
    [h,w]=shape
    #展平数组
    v=np.array(value).ravel()
    res=(-1,-1)
    #列
    if axis==1:
        r=np.array(v%h,np.int32)
        c=np.array(np.floor(v/h),np.int32)
        res= (r,c)
	#行
    if axis==0:
        c = np.array(v % w, np.int32)
        r = np.array(np.floor(v / w), np.int32)
        res = (r, c)
    return res
#二维索引转为value,axis==0表示行优先，
def index2value(index,shape,axis=0):
    [h, w] = shape
    index=np.array(index)
    value=0
    #列优先
    if axis==1:
        value=index[1]*h+index[0]
    #行优先
    if axis==0:
        value=index[0]*w+index[1]
    return value

def getlabelM(lgn,rgn,leftgridsize,rightgridsize,shift=(0,0)):
	leftlabel = (np.arange(1, lgn ** 2 + 1).reshape(lgn, lgn))\
		.repeat(leftgridsize[0],0).\
		repeat(leftgridsize[1], 1).astype(np.int32)
	rightlabel = (np.arange(1, rgn ** 2 + 1).reshape(rgn, rgn))\
		.repeat(rightgridsize[0],0)\
		.repeat(rightgridsize[1], 1).astype(np.int32)
	#移位
	if shift[0]==1:
		leftlabel=np.roll(leftlabel,int(leftgridsize[0]/2),axis=0)
		rightlabel=np.roll(rightlabel, int(rightgridsize[0]/2), axis=0)
	if shift[1]==1:
		leftlabel=np.roll(leftlabel,int(leftgridsize[1]/2), axis=1)
		rightlabel=np.roll(rightlabel, int(rightgridsize[1]/2), axis=0)
	return leftlabel,rightlabel

#imagesc
def imagesc(matrix,title=None,savefile=True,savepath='./dataset/',ShowDebug=True):
    if ShowDebug:
        r,c=matrix.shape
        py.imshow(matrix,extent=[0,c,0,r])
        if title!=None:
            py.title(title)
        py.show()
#matches应该包含两个数组，每个分别对应一个图像上的行列坐标值列表，
#mathes:kep1list,kep2list:[rlist,clist]
def matchesshow(img1,img2,matches,title=None):
    r1,c1=img1.shape[:2]
    r2,c2=img2.shape[:2]
    r = max(r1, r2)
    c = c1 + c2
    if img1.ndim>2:
	    d=img1.shape[2]
	    showimg = np.zeros((r, c,d))
	    showimg[0:r1, 0:c1,0:d] = img1
	    showimg[0:r2, c1:c,0:d] = img2
    else:
        showimg=np.zeros((r,c))
        showimg[0:r1,0:c1]=img1
        showimg[0:r2,c1:c]=img2
    #画出匹配线
    kp1list=matches[0]
    kp2list = matches[1]
    kp2c=kp2list[1]+c1
    py.imshow(showimg)
    lens=len(kp1list[0])
    for i in range(lens):
        py.plot([kp1list[1][i],kp2c[i]],[kp1list[0][i],kp2list[0][i]],linewidth=0.5)
    if title!=None:
        py.title(title)
    py.show()
def imshow(img):
	py.imshow(img)
	py.show()
#按匹配网格分匹配点
def test():
    kpnum=10000
    shape=np.array([480,640])
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

    img1h,img1w=[480,640]
    grid1h,grid1w=[24,32]
    rows1=20
    grid1 = np.zeros([rows1 ** 2])
    leftkptvalue=index2value(leftkpt,shape)
    for i in range(kpnum):
	    i,j=value2index(leftkptvalue[i],shape=shape)
	    id1 = (math.floor((j % img1w) / grid1w) + math.floor((i % img1h) / grid1h) * rows1)
	    grid1[id1]+=1
    leftimg=np.zeros(shape=shape)
    leftimg[leftkpt]=1
    filter=np.ones(shape=(grid1h,grid1w))
    grid2=conv2withstride(leftimg, filter, stride=(grid1h,grid1w), start=None, gridnum=rows1)
    
    imagesc(grid1.reshape(rows1,rows1),'grid1')
    imagesc(grid2, 'grid2')
    imagesc(grid1.reshape(rows1,rows1)-grid2, 'grid1-grid2')
    
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
#按批次计算卷积
#x:n,h,w,c的四维输入
#W:h,w,c,o的四维卷积核
def batchconv4d(input,kernel,stride=[1,1,1,1]):
    x=tf.Variable(input,tf.float32)
    W=tf.Variable(kernel,tf.float32)
    res = tf.nn.conv2d(x, W, strides=stride, padding='VALID')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result=sess.run(res)
    return result
def testbatchconv4d():
    s=time.time()
    input=np.random.randint(0,10,size=10*640*480).reshape(10,480,640,1).astype(np.float32)
    kernel=np.ones((24,32,1,1)).astype(np.float32)
    res=batchconv4d(input,kernel,stride=[1,24,32,1])
    print("cost tim is %fs"%(time.time()-s),res.shape)
def main():
    a=np.random.randint(0,10,size=54).reshape(9,6)
    b=np.ones([3,2])
    c=conv2withstride(a,b,stride=(3,2),gridnum=3)
    print(a)
    print(b)
    print(c)

    c=np.array([[1,2,3],[4,5,6],[7,8,9]])
    d=np.array([[2,4],[7,8]])
    index=value2index(d,c.shape)
    value=index2value(index,c.shape)

    print(c,value,c[index])

if __name__ == '__main__':
    # main()
    # test()
    testbatchconv4d()