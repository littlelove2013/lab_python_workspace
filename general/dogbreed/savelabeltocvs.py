import math
import os
import pandas as pd #数据分析
import numpy as np #科学计算
import scipy.io as sio
import pandas as pd #数据分析

root="../../../include_data/"
test=root+'test/'

#获取文件id列表
def listdir(path, extension='jpg'):
    filename=[]
    for file in os.listdir(path):
        fileexten=file.split('.')
        if(fileexten[-1]==extension):
            filename.append(fileexten[0])
    #返回排好序的
    filename.sort()
    return filename

def save2vsc(Idlist,dogbreedlist,labels,savepath='./dataset/',savename='test'):
    result = pd.DataFrame({'PassengerId': Idlist})
    for i in range(len(dogbreedlist)):
        result[dogbreedlist[i]]=labels[:,i]
    filesavename = savepath+'pre_' + savename + '.csv'
    result.to_csv(filesavename, index=False)
    # print(result)
    print("save csv file to %s" % filesavename)
#获取所有test文件的name列表和id列表
def gettestname(testpath):
    #获取testid
    testid = listdir(test, 'jpg')
    # print(testid)
    lens=len(testid)
    #获取dogbreed种类
    # 载入数据
    csvpath='labels.csv'
    labels_pd = pd.read_csv(csvpath)
    # 获取种类数名字列表，按字典序排序。将其映射到整数域
    bread_name = labels_pd.copy().breed.unique()
    bread_name.sort()
    Idlist=testid
    dogbreedlist=bread_name
    r=lens
    c=len(dogbreedlist)
    labels=np.random.rand(r,c)
    summ=labels.sum(1)
    labels=labels/summ.reshape(summ.size,1)
    save2vsc(Idlist,dogbreedlist,labels)



if __name__ == '__main__':
    #假设label值已知
    print("hello")
    gettestname(test)

