import re
import scipy.io as sio
import os
import numpy as np
import math
import time
root ='../../../webbigdata/'
#极小量防除零错
eps=1e-4
maxnum=21398
#定义名字：
useridvectorname='useridvector'
itemscorematname='itemscoremat'
itemidmatname='itemidmat'
ksimilarmatname='ksimilarmat'

#将数据存为稀疏的矩阵形式
def savetrain(savepath='./dataset/'):

    savename=savepath+'train.mat'
    if (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        # file = sio.loadmat(realpath)
        # return file
        train = sio.loadmat(savename)
        #print(train['userrateitemscore'], train['userrateitemscore'].shape)
        return train

    file = open(root + 'train.txt')
    train = (file.read())
    b = train.split('\n')
    # user对item打分的矩阵，每一行维打分的item
    useritemratevector = []
    # user对item打分的id，每一行为打过分的item的id，对应于上面的打分矩阵
    useritemrateidvector = []
    # userid向量，对应于上面的行号
    useridvector = []
    itemlen = 624960
    print('process is working....')
    # 定义隔多少次交互显示一次
    timeshow = 100000
    lens = len(b)
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            userid = int(res[0])
            userrate = []
            userrateid = []
            # 循环查找每一项
            for j in range(num):
                # 交互显示
                if i%timeshow==0:
                   print("process userid:%d,it has %d item score!\thas processed %.2f%%(%d,%d)"%(userid,num,100*(i/lens),i,lens))
                i += 1
                pair = b[i].split()
                userrate.append(int(pair[1]))
                # 过滤掉得分为0的值
                # if int(pair[1])==0:
                #     continue;
                userrateid.append(int(pair[0]))
            # 对得分进行0均值单位方差处理
            ratevector = np.asarray(userrate, np.float32)
            ratevector = (ratevector - ratevector.mean()) / (ratevector.std() + eps)
            useritemratevector.append(ratevector)
            # 存4B整数
            useritemrateidvector.append(np.asarray(userrateid, np.int32))
            useridvector.append(userid)
    # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    useritemratevector = np.asarray(useritemratevector)
    useritemrateidvector = np.asarray(useritemrateidvector)
    useridvector = np.asarray(useridvector)
    savefile={'userrateitemscore':useritemratevector,"userrateitemid":useritemrateidvector,"useridvector":useridvector}
    sio.savemat(savename, savefile)
    print('save file to %s' % (savename))
    return savefile

#filter为选择过滤方式，默认为None，可能值为，'item':过滤item数少于thre的条目，'std':过滤方差为0的user
#过滤掉user买的item少于thre的user
#gettest:是否获取测试集，如果为True，则每个user前6个item取为test集（注：train集和里也包含该6个item）
def savetrainwithfilter(filename='train',savepath='./dataset/',filter=('item','std'),thre=100,gettest=False,testfilname='gettest'):
    savename=savepath+'thre_'+str(thre)+'_filter'+str(filter)+'_'+filename+'.mat'
    testname=savepath+testfilname+'.mat'
    testitemlen=6
    #测试集
    if  gettest and os.path.isfile(testname) and (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        print('there has a saved mat file : %s' % (testname))
        # file = sio.loadmat(realpath)
        # return file
        train = sio.loadmat(savename)
        test = sio.loadmat(testname)
        #print(train['userrateitemscore'], train['userrateitemscore'].shape)
        return [train,test]
    elif (not gettest) and (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        train = sio.loadmat(savename)
        return train
    file = open(root + filename+'.txt')
    train = (file.read())
    b = train.split('\n')
    # user对item打分的矩阵，每一行维打分的item
    useritemratevector = []
    # user对item打分的id，每一行为打过分的item的id，对应于上面的打分矩阵
    useritemrateidvector = []
    # userid向量，对应于上面的行号
    useridvector = []
    #存user对应打分的均值和标准差X*2
    usermeanstd=[]

    #测试集
    testuseridvector=[]
    testitemidmat=[]
    testitemscoremat=[]
    itemlen = 624960
    print('process is working....')
    # 定义隔多少次交互显示一次
    timeshow = 100000
    lens = len(b)
    totaluser=0
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            userid = int(res[0])
            userrate = []
            userrateid = []

            testuseridvector.append(userid)#记录测试id
            testitemscorevector=[]#itemscorevector
            testitemidvector=[]#itemidvector
            totaluser+=1
            for j in range(num):
                # 交互显示
                if i%timeshow==0:
                   print("process userid:%d,\tit has %d item score!\thas processed %.2f%%(%d,%d)"%(userid,num,100*(i/lens),i,lens))
                i += 1
                pair = b[i].split()
                itemid=int(pair[0])
                itemscore=float(pair[1])
                if j < testitemlen:
                    testitemscorevector.append(itemscore)
                    testitemidvector.append(itemid)
                userrate.append(itemscore)
                # 过滤掉得分为0的值
                # if int(pair[1])==0:
                #     continue;
                userrateid.append(itemid)
            # userinfo={userid:uservector}
            # 对得分进行0均值单位方差处理

            #test文件处理
            testitemidmat.append(testitemidvector)
            testitemscoremat.append(testitemscorevector)
            #train文件处理
            ratevector = np.asarray(userrate, np.float32)
            ratevectormean=ratevector.mean()
            ratevectorstd=ratevector.std()
            usermeanstd.append([ratevectormean,ratevectorstd])
            if (filter[0] == 'item' and num < thre) or (filter[1]=='std' and ratevectorstd==0):#如果标准差为0，则过滤此用户,num<thre过滤
                continue
            ratevector = (ratevector - ratevectormean) / (ratevectorstd + eps)
            useritemratevector.append(ratevector)
            # 存4B整数
            useritemrateidvector.append(np.asarray(userrateid, np.int32))
            useridvector.append(userid)
    # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    useritemratevector = np.asarray(useritemratevector)
    useritemrateidvector = np.asarray(useritemrateidvector)
    useridvector = np.asarray(useridvector)
    usermeanstd = np.asarray(usermeanstd,np.float32)
    savefile={'userrateitemscore':useritemratevector,"userrateitemid":useritemrateidvector,"useridvector":np.reshape(useridvector,[1,len(useridvector)]),'usermeanstd':usermeanstd}
    sio.savemat(savename, savefile)
    print('save trainfile to %s' % (savename))
    print('total user:\t%d,filter user:\t%d,save user:\t%d' % (totaluser,totaluser-len(useridvector),len(useridvector)))
    #处理test
    if gettest:
        testuseridvector=np.array(testuseridvector,np.int32)
        testitemidmat=np.array(testitemidmat,np.int32)
        testitemscoremat=np.array(testitemscoremat,np.float32)
        testfile={useridvectorname:np.reshape(testuseridvector,[1,len(testuseridvector)]),itemidmatname:testitemidmat,itemscorematname:testitemscoremat}
        sio.savemat(testname, testfile)
        print('save testfile to %s' % (testname))
        return [savefile,testfile]
    return savefile

#每次获取一个矩阵的值，batchnum为第几个包，batchsize为一包最多行数
def getbatchtrainwithfilter(batchnum,batchsize=100):

    savename=savepath+'thre_'+str(thre)+'_train.mat'
    if (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        # file = sio.loadmat(realpath)
        # return file
        train = sio.loadmat(savename)
        #print(train['userrateitemscore'], train['userrateitemscore'].shape)
        return train

    file = open(root + 'train.txt')
    train = (file.read())
    b = train.split('\n')
    # user对item打分的矩阵，每一行维打分的item
    useritemratevector = []
    # user对item打分的id，每一行为打过分的item的id，对应于上面的打分矩阵
    useritemrateidvector = []
    # userid向量，对应于上面的行号
    useridvector = []
    itemlen = 624960
    print('process is working....')
    # 定义隔多少次交互显示一次
    timeshow = 100000
    lens = len(b)
    totaluser=0
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            userid = int(res[0])
            userrate = []
            userrateid = []
            totaluser+=1
            # 如果满足阈值循环查找每一项
            if num<thre:
                i+=num
                continue
            for j in range(num):
                # 交互显示
                if i%timeshow==0:
                   print("process userid:%d,\tit has %d item score!\thas processed %.2f%%(%d,%d)"%(userid,num,100*(i/lens),i,lens))
                i += 1
                pair = b[i].split()
                userrate.append(int(pair[1]))
                # 过滤掉得分为0的值
                # if int(pair[1])==0:
                #     continue;
                userrateid.append(int(pair[0]))
            # userinfo={userid:uservector}
            # 对得分进行0均值单位方差处理
            ratevector = np.asarray(userrate, np.float32)
            ratevector = (ratevector - ratevector.mean()) / (ratevector.std() + eps)
            useritemratevector.append(ratevector)
            # 存4B整数
            useritemrateidvector.append(np.asarray(userrateid, np.int32))
            useridvector.append(userid)
    # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    useritemratevector = np.asarray(useritemratevector)
    useritemrateidvector = np.asarray(useritemrateidvector)
    useridvector = np.asarray(useridvector)
    savefile={'userrateitemscore':useritemratevector,"userrateitemid":useritemrateidvector,"useridvector":useridvector}
    sio.savemat(savename, savefile)
    print('save file to %s' % (savename))
    print('total user:\t%d,filter user:\t%d,save user:\t%d' % (totaluser,totaluser-len(useridvector),len(useridvector)))
    return savefile

#similarknumber:k的取值
#函数思路：循环计算每一个user对应的user的相似度，算法复杂度为nlogn
def calcusersimilarity(savepath='./dataset/',similarknumber=100):
    #相似得分以及最相似用户列表
    savename = savepath + 'k_' + str(similarknumber) + '_usersimilar.mat'
    if os.path.isfile(savename) :
        print('there has a saved mat file : %s' % (savename))
        # file = sio.loadmat(realpath)
        # return file
        file = sio.loadmat(savename)
        # print(train['userrateitemscore'], train['userrateitemscore'].shape)
        return file

    itemlen=624961#读取的item长度
    # batchsize=1000
    train=savetrainwithfilter(filter=('item','std'))#获取train
    useritemratevector=train['userrateitemscore'][0]
    useritemrateidvector = train['userrateitemid'][0]
    useridvector = train['useridvector'][0]
    userlen = len(useridvector)
    # userlen=100
    #取前k个最相似的
    # similarknumber=100
    ksimilar=np.zeros([userlen,similarknumber],np.float32)
    kuserid = np.zeros([userlen, similarknumber],np.int32)
    # userlen=len(useridvector)
    tmpx=np.zeros([itemlen])
    tmpy = np.zeros([itemlen])
    showtime=2
    for i in range(userlen):
        if (i+1)%showtime==0:
            time_end = time.time()
            print('processed  user id:%d(total:%.2f%%(%d/%d)),cost time:%.4fs'%(useridvector[i],100*((i+1)/userlen),(i+1),userlen,(time_end-time_start)))
        time_start = time.time()
        #第index行的所有坐标值应该等于打分值
        #index=useridvector[i]
        index=i
        useriid=useridvector[index]
        useri=useritemrateidvector[index][0]
        userirate = useritemratevector[index][0]
        # 方差为0的没有相似度
        if userirate.std() == 0:
            continue
        useriset=set(useri)
        #tmpx[useritemrateidvector[index]]=useritemratevector[index]
        for j in range(i+1,userlen):
            # jndex=useridvector[j]
            jndex=j
            userj = useritemrateidvector[jndex][0]
            userjrate = useritemratevector[jndex][0]
            #方差为0的没有相似度
            if userjrate.std()==0:
                continue
            userjid = useridvector[jndex]
            userjset = set(userj)
            cross = useriset & userjset
            #只计算交集元素
            similiar=0
            #如果没有cross，跳过，相似度为0
            if len(cross)<0:
                continue
            for item in cross:
                indexi=np.where(useri==item)[0]
                indexj = np.where(userj == item)[0]
                similiar+=userirate[indexi]*userjrate[indexj]
            #将similiar保存到前k个
            if ksimilar[index,similarknumber-1]<similiar:
                #如果similar大于原来的值则替换
                ksimilar[index, similarknumber - 1]=similiar
                kuserid[index, similarknumber - 1]=userjid
                sortid=np.argsort(-ksimilar[index])
                kuserid[index]=kuserid[index,sortid]
                ksimilar[index] = ksimilar[index, sortid]
            #j对应的i不用再计算
            if ksimilar[jndex,similarknumber-1]<similiar:
                #如果similar大于原来的值则替换
                ksimilar[jndex, similarknumber - 1]=similiar
                kuserid[jndex, similarknumber - 1]=useriid
                sortid=np.argsort(-ksimilar[jndex])
                kuserid[jndex]=ksimilar[jndex,sortid]
            #tmpy[useritemrateidvector[jndex]] = useritemratevector[jndex]
            #similiar=tmpx.dot(tmpy)
            #重核利用内存
            #tmpy[useritemrateidvector[j]] = 0
        #tmpx[useritemrateidvector[index]] = 0
    #保存用户相似度矩阵
    savedict={'useridvector':np.reshape(useridvector,[1,len(useridvector)]),'userksimilarscore':ksimilar,'userksimilaruser':kuserid}
    sio.savemat(savename,savedict)
    print('save file to %s' % (savename))
    return savedict

#获取itemattr,两个属性看作坐标，计算knn，similarknumber=100
#过滤掉为None的item
def getitemattr(savepath='./dataset/',filter=True):
    savename = savepath + 'itemAttr.mat'
    if (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        # file = sio.loadmat(realpath)
        # return file
        train = sio.loadmat(savename)
        # print(train['userrateitemscore'], train['userrateitemscore'].shape)
        return train
    #从文件读取
    file = open(root + 'itemAttribute.txt')
    train = (file.read())
    b = train.split('\n')
    totalitem=len(b)
    #记录保留的itemid，600000*1
    itemidvector=[]
    #记录item的属性值，600000*2
    itemattrmatrix=[]
    timeshow=10000
    for i in range(totalitem):
        obj=b[i]
        itemattrlist=obj.split('|')
        if (len(itemattrlist)<3) or (itemattrlist[1]=='None') or (itemattrlist[2]=='None'):
            #如果有为None，则不计算
            continue
        itemid=int(itemattrlist[0])
        itemattr1=int(itemattrlist[1])
        itemattr2=int(itemattrlist[2])
        itemidvector.append(itemid)
        itemattrmatrix.append([itemattr1,itemattr2])
        if i % timeshow == 0:
            print("process itemid:%d,\t has processed %.2f%%(%d,%d)" % (itemid, 100 * (i / totalitem), i, totalitem))
        # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    saveitme=len(itemidvector)
    itemidvector = np.asarray(itemidvector,np.int32)
    itemattrmatrix = np.asarray(itemattrmatrix,np.int32)
    savefile = {'itemidvector': itemidvector, "itemattrmatrix": itemattrmatrix}
    sio.savemat(savename, savefile)
    print('save file to %s' % (savename))
    print('total item number:\t%d,filter item:\t%d,save item:\t%d' % (totalitem, totalitem - saveitme, saveitme))
    return savefile

#从txt获取test，hasscore为是否带得分
def gettest(filename='test',savepath='./dataset/',hasscore=False):
    savename=savepath+filename+'.mat'
    if (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        train = sio.loadmat(savename)
        return train

    file = open(root + filename+'.txt')
    test = (file.read())
    b = test.split('\n')
    # user对item打分的id，每一行为打过分的item的id，对应于上面的打分矩阵
    useritemrateidvector = []
    useritemratevector=[]
    # userid向量，对应于上面的行号
    useridvector = []
    itemlen = 624960
    print('process is working....')
    # 定义隔多少次交互显示一次
    timeshow = 10000
    lens = len(b)
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            userid = int(res[0])
            userrateid = []
            userrate=[]
            # 循环查找每一项
            for j in range(num):
                # 交互显示
                if i%timeshow==0:
                   print("process userid:%d,it has %d item to pre!\thas processed %.2f%%(%d,%d)"%(userid,num,100*(i/lens),i,lens))
                i += 1
                # 过滤掉得分为0的值
                # if int(pair[1])==0:
                #     continue;
                if hasscore:
                    itemattr=b[i].split()
                    userrateid.append(int(itemattr[0]))
                    userrate.append(float(itemattr[1]))
                else:
                    userrateid.append(int(b[i]))
            # 存4B整数
            useritemrateidvector.append(np.asarray(userrateid, np.int32))
            if hasscore:
                useritemratevector.append(np.asarray(userrate, np.float32))
            useridvector.append(userid)
    # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    useritemrateidvector = np.asarray(useritemrateidvector)
    useridvector = np.asarray(useridvector)
    if hasscore:
        useritemratevector=np.asarray(useritemratevector,np.float32)
        savefile = {itemidmatname: useritemrateidvector, useridvectorname:np.reshape(useridvector,[1,len(useridvector)]),itemscorematname:useritemratevector}
    else:
        savefile={itemidmatname:useritemrateidvector,useridvectorname:np.reshape(useridvector,[1,len(useridvector)])}
    sio.savemat(savename, savefile)
    print('save file to %s' % (savename))
    return savefile
# 预测用户得分，首先，如果用户已经对其打分，则就是该得分。
# 如果没有，则取得用户相似的且对item打分且与user相似度大于thre的N个user的打分按相似度加权求和，
# 如果没有相似用户，则使用用户打分平均值求得
def prescore(savepath='./dataset/',filename='prescore',test=None):
    savename=savepath+'prescore.mat'
    if (os.path.isfile(savename)):
        print('there has a saved mat file : %s' % (savename))
        train = sio.loadmat(savename)
        return train
    #获取测试mat,默认处理自带test
    if test==None:
        test=gettest()
    # 预测每个user的每个item
    testuseridvector = test[useridvectorname][0]
    testuserrateitemid = test[itemidmatname]
    testuserratematrix=[]#得分矩阵
    #获取相似mat
    similarity=calcusersimilarity()
    userksimilarscore=similarity['userksimilarscore']
    userksimilaruser=similarity['userksimilaruser']
    #h获取训练mat
    train=savetrainwithfilter()
    useritemratevector = train['userrateitemscore'][0]
    useritemrateidvector = train['userrateitemid'][0]
    useridvector = train['useridvector'][0]
    usermeanstd = train['usermeanstd']

    #注销一些内存
    # test=[]
    # train=[]
    # similarity=[]
    userlen=len(testuseridvector)
    showtime=10
    for i in range(userlen):
        ratevector=[]
        testitemlen=len(testuserrateitemid[i])
        userid = testuseridvector[i]
        #先查找userid是否在相似度的矩阵里，如果不在则说明其没有相似用户，则直接用其平均分代替，或者用item-item协同
        userindex = (np.where(useridvector==userid))[0]
        if len(userindex)==0:
            #不存在相似用户,则直接返回该用户打分均值
            ratevector=np.ones(testitemlen)*usermeanstd[userid,0]#[mean,std]
            testuserratematrix.append(ratevector)
            continue
        #找到与useri最相似的k个用户
        ksimilaruser=userksimilaruser[userindex[0]]
        ksimilarscore=userksimilarscore[userindex[0]]
        #对每个item打分
        for itemid in testuserrateitemid[i]:
            N=0#记录有几个相似用户对item打分
            S=[]#记录得分
            W=[]#记录权值
            #1、检测user有没有对该item打分
            for j in range(len(ksimilaruser)):
                if ksimilarscore[j]==0:
                    #没有相似user了
                    break
                userj = ksimilaruser[j]
                #k个最相似user，查找其item列表是否存在该item的打分
                userjndex = (np.where(useridvector == userj))[0]
                #如果
                if len(userjndex)==0:
                    print(ksimilaruser,useridvector)
                userjitemvecter=useritemrateidvector[userjndex[0]][0]
                # userjitemvecter=userjitemvecter[0]
                #找是否有item
                jndexitem=(np.where(userjitemvecter == itemid))[0]
                #如果没找到，则过滤
                if len(jndexitem)==0:
                    continue
                #找到了，则记录得分和权值
                scoreobj=useritemratevector[userjndex[0]][0]
                score=scoreobj[jndexitem[0]]
                weight=ksimilarscore[j]
                S.append(score)
                W.append(weight)
            if len(S)==0:
                #没有找到同样的分的user，则用0代替
                ratevector.append(0)
                continue
            #如果存在，则按如下公式计算打分
            s=np.array(S)
            w=np.array(W)
            calcscore=(s*w).sum()/w.sum()
            ratevector.append(calcscore)
        #记录对每个item的打分
        ratevector=np.array(ratevector)
        #还需要乘以标准差再加均值[mean,std]
        ratevector=ratevector*usermeanstd[userid,1] +usermeanstd[userid,0]
        #小于0的取零
        ratevector[ratevector<0]=0
        testuserratematrix.append(ratevector)
        if i%(showtime)==0:
            print('calc user:%d\'s rate,rate vector is:\n'%(userid),ratevector)

    # 保存到文件
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #将得到的打分表存到mat
    savefile = {useridvectorname:np.reshape(testuseridvector,[1,len(testuseridvector)]),itemidmatname:testuserrateitemid,itemscorematname: np.array(testuserratematrix)}
    sio.savemat(savename, savefile)
    print('save file to %s' % (savename))
    return savefile

#写入文本，如果没有打分矩阵则输出测试文本
def write2txt(useridvector,itemidmat,itemscoremat,filename,savepath='./dataset/'):
    f=open(savepath+filename,'w')
    format='  '
    userlen=len(useridvector)
    for i in range(userlen):
        userid=useridvector[i]
        itemlen=len(itemidmat[i])
        f.write(str(userid)+'|'+str(itemlen)+'\n')
        for j in range(itemlen):
            itemid=itemidmat[i][j]
            if itemscoremat==None:
                f.write(str(itemid)+'\n')
                continue
            itemscore=itemscoremat[i][j]
            f.write(str(itemid)+format+str(itemscore)+format+'\n')
    f.close()

def calcRMSE(mat1,mat2):
    RMSE=np.sqrt(((mat1-mat2)**2).mean())
    return RMSE

def main():
    start = time.time()
    #savetrain()
    train,test=savetrainwithfilter(gettest=True,testfilname='gettext')
    #calcusersimilarity()
    # gettest()
    pre=prescore(filename='traintest',test=test)
    write2txt(pre[useridvectorname][0],pre[itemidmatname],pre[itemscorematname],'firstprescore.txt')
    write2txt(test[useridvectorname][0],test[itemidmatname],test[itemscorematname],'traintest.txt')
    mat1=gettest(filename='traintest',hasscore=True,savepath='./dataset/')
    mat2=gettest(filename='firstprescore',hasscore=True,savepath='./dataset/')
    RMSE=calcRMSE(mat1[itemscorematname],mat2[itemscorematname])
    print('RMSE:%.4f'%(RMSE))
    # getitemattr()
    costtime=time.time()-start
    print('cost total time is %.4fmin(%.4fs)'%(costtime/60,costtime))
if __name__ == '__main__':
    main()