# -*- coding: utf-8 -*-
import math
import os
import pandas as pd #数据分析
import numpy as np #科学计算
import scipy.io as sio
from scipy.misc import imread, imresize
import savelabeltocvs as sl
# import matplotlib.pyplot as plt

root='../../../include_data/dogbreed/'
batchsize=50
lens=math.ceil(10222/batchsize)
testlens=math.ceil(10357/batchsize)

#从给定的csv文件里读出图片并resize，返回更新图片及其标签
def read_img_to_mat(csvpath,imgpath,size=(224,224),savefile=True,savepath='./dataset/',batchsize=100):
    #载入数据
    labels_pd = pd.read_csv(csvpath)
    #获取种类数名字列表，按字典序排序。将其映射到整数域
    bread_name=labels_pd.copy().breed.unique()
    bread_name.sort()
    one_hot_len=len(bread_name)

    lens=len(labels_pd)
    #取十张看看
    # lens=200
    ktime=math.floor((lens-1)/batchsize)+1
    showtime=50
    #保存img及其labels
    for k in range(ktime):
        start=k*batchsize
        end = (k+1)*batchsize
        if end>lens:
            end=lens
        matfilename = 'dogbreed_' + str(k) +'_'+str(batchsize)+ '.mat'
        realpath = savepath + matfilename
        images = []
        labels = []
        if (os.path.isfile(realpath)):
            print('there has a saved mat file : %s' % (realpath))
            #file = sio.loadmat(realpath)
            #return file
            continue
        for i in np.arange(start,end):
            #每一个id
            idx=labels_pd.id[i]
            label_name=labels_pd.breed[i]
            id_label=np.where(bread_name==label_name)[0][0]
            one_hot=np.zeros([one_hot_len])
            one_hot[id_label]=1
            file_path=imgpath+idx+'.jpg'
            img=imread(file_path, mode='RGB')#获取0,1之间的数
            img=imresize(img,size)
            # img=np.asarray(img, np.float32)
            #对img做0均值单位方差
            # mu = np.array([img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()])
            # stds = np.array([img[:, :, 0].std(), img[:, :, 1].std(), img[:, :, 2].std()])
            # img = (img - mu) / stds
            # if i % showtime == 0:
            #     print("img mean:(%.4f,%.4f,%.4f) and std:(%.4f,%.4f,%.4f)"
            #           % (img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(), img[:, :, 0].std(),
            #              img[:, :, 1].std(), img[:, :, 2].std()))

            images.append(img)
            labels.append(one_hot)
        images=np.asarray(images)
        #对images的三通道做0均值单位方差归一化

        labels=np.asarray(labels,np.int32)
        train={'images':images,'labels':labels}
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sio.savemat(realpath,train)
        print('save file to %s'%(realpath))

    return lens,batchsize,ktime

def getdata(batchnumber,savepath=root+'dataset/',batchsize=100,size=(224,224)):
    imgpath = '../../../include_data/train/'
    # imgpath='I:/学习/研一/机器视觉/课程设计-狗类别判定/train/train/'
    csvpath = 'labels.csv'
    matfilename = 'dogbreed_' + str(batchnumber) + '_' + str(batchsize) + '.mat'
    realpath = savepath + matfilename
    if (os.path.isfile(realpath)==False):
        lens, batchsize, ktime = read_img_to_mat(csvpath, imgpath,savepath=savepath,batchsize=batchsize,size=size)
        if batchnumber>ktime:
            print('the batch number called is out of index!')
            return None
    file = sio.loadmat(realpath)
    return file

def read_testimg_to_mat(imgpath,size=(224,224),savefile=True,savepath='./testdataset/',batchsize=100):
    Idlist, _ = sl.gettestname(sl.test)
    lens = len(Idlist)
    # 取十张看看
    # lens=200
    ktime = math.floor((lens - 1) / batchsize) + 1
    showtime = 50
    # 保存img及其labels
    for k in range(ktime):
        start = k * batchsize
        end = (k + 1) * batchsize
        if end > lens:
            end = lens
        matfilename = 'dogbreed_test_' + str(k) + '_' + str(batchsize) + '.mat'
        realpath = savepath + matfilename
        images = []
        # labels = []
        if (os.path.isfile(realpath)):
            print('there has a saved mat file : %s' % (realpath))
            # file = sio.loadmat(realpath)
            # return file
            continue
        for i in np.arange(start, end):
            file_path = imgpath+Idlist[i]+ '.jpg'
            img = imread(file_path, mode='RGB')  # 获取0,1之间的数
            img = imresize(img, size)
            images.append(img)
        images = np.asarray(images)
        test = {'images': images}
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sio.savemat(realpath, test)
        print('save file to %s' % (realpath))

    return lens, batchsize, ktime

def gettest(batchnumber,savepath=root+'testdataset/',batchsize=100,size=(224,224)):
    imgpath = '../../../include_data/test/'
    # imgpath='I:\学习\研一\机器视觉\课程设计-狗类别判定/test/test/'
    # 获取测试图片名
    #
    matfilename = 'dogbreed_test_' + str(batchnumber) + '_' + str(batchsize) + '.mat'
    realpath = savepath + matfilename
    if (os.path.isfile(realpath) == False):
        lens, batchsize, ktime = read_testimg_to_mat(imgpath, savepath=savepath, batchsize=batchsize, size=size)
        if batchnumber > ktime:
            print('the batch number called is out of index!')
            return None
    file = sio.loadmat(realpath)
    return file

    #return train

#data为4维的数据，(n,h,w,c):n:数量，c:通道
def addgaussandrot90(data,debug=False):
    #添加噪声
    [n,h,w,c]=data.shape
    g=np.random.randn(n,h,w,c)
    k=16
    data=data+k*g
    data[data>255]=255
    data[data<0]=0
    #随机数，选择是否旋转90度,向左1还是向右-1,或者不旋转
    randnum=np.random.randint(0,3,3)
    #90度旋转
    rot=[-1,0,1]
    axes=(1,2)#对1，2维度进行旋转
    data=np.rot90(data,k=rot[randnum[0]],axes=axes)
    #水平垂直翻转
    fliper=[0,1,2]
    if randnum[1]!=0:
        data=np.flip(data,axis=fliper[randnum[1]])
    #做随机循环移位
    roller=[0,1,2]#表示选择的维度，0为不操作
    shiftnum=0
    if False:#randnum[2]!=0:
        axis=roller[randnum[2]]
        wid=data.shape[axis]
        shiftnum=np.random.randint(int(wid/8),7*int(wid/8))
        data=np.roll(data,shift=shiftnum,axis=axis)
    if debug:#显示第一张图片查看是否正确转换
        print('(rot,fliper,roller)=((k=%d),(axis=%d),(axis=%d,shift=%d))'%(rot[randnum[0]],fliper[randnum[1]],roller[randnum[2]],shiftnum))
        imgindex=np.random.randint(0,n)
        img=data[imgindex].astype(np.uint8)
        # plt.imshow(img)
        # plt.show()
    return data





def main():
    # file0=getdata(0,batchsize=batchsize)
    # imgs = file0['images']
    # labels = file0['labels']
    test1 = gettest(1, batchsize=batchsize)
    images=addgaussandrot90(test1['images'])
    print('main')

if __name__ == '__main__':
    main()
