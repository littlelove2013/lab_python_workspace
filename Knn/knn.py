import numpy as np
import readiris
import time
# xr=1000#x行数
# yr=60000#y的行数
# c=100#特征列数
# #模拟输入数据
# X=np.random.rand(xr,c)
# Y=np.random.rand(yr,c)
def knn(X,Y):
    if(len(X[0])!=len(Y[0])):
        print("X,Y should be same cols!")
        return False
    lenX=len(X)
    lenY=len(Y)
    # Xdouble=np.diag(X.dot(X.T))#行向量
    Xdouble=(X*X).sum(1)
    Xdouble=Xdouble.reshape(-1,1).dot(np.ones((1,lenY)))
    # Ydouble=np.diag(Y.dot(Y.T))
    Ydouble=(Y*Y).sum(1)
    Ydouble = np.ones((lenX,1)).dot(Ydouble.reshape(1,-1))
    XY=X.dot(Y.T)
    dist=Xdouble+Ydouble-2*XY
    dist.sort(1)
    #排序，计算前k个最好的
    return dist
def crossvalidation(likehoodfunc):
    #分为十份
    k=10
    tail=4#小数点后位数
    score=np.zeros([k])
    datalen=len(shufflelabels)
    perlens=math.floor(datalen/k)
    if likehoodfunc.__name__=='kneibor' or likehoodfunc.__name__=='knn':
        #knn则需要验证集合
        #neibork=3->20
        #验证最佳的k值
        global knnk
        maxmeanscore=0
        maxstd=0
        bestknnk=1
        neibork=np.arange(1,20,2)#k值列表
        dic={}
        for ks in neibork:#寻找最合适的k值
            knnk=ks
            for i in range(k):
                trainf=np.arange(0,i*perlens)
                traint=np.arange((i+1)*perlens,datalen)
                validateind=np.arange((i)*perlens,(i+1)*perlens)
                #获取训练集和测试集
                trainind=np.concatenate((trainf,traint),axis=0)
                #初始化数据
                setglobalvalue(trainind,validateind,[])
                labelss=np.zeros([len(validate)])
                for j in range(len(validate)):
                    labelss[j]=knn(validate[j])
                res = labelss==validatelabels
                acc = res.sum() / len(res)
                score[i]=round(acc,tail)
            meanscore=round(score.sum()/len(score),tail)
            dic[knnk]=meanscore
            if maxmeanscore<meanscore:
                maxmeanscore=meanscore
                bestknnk=ks
                maxstd=score.std()
        print(likehoodfunc.__name__,"method's cross validation:")
        print('各个key值验证得到的正确率为：\n',dic,'\n验证得到最好的k值是:',bestknnk\
              ,'\n最优的正确率是%.4f'%(maxmeanscore)\
            ,'\n标准差%.4f'%(maxstd))
        knnk=bestknnk
        #测试

    else:
        #其他三种只需要训练和测试集合
        for i in range(k):
            trainf=np.arange(0,i*perlens)
            traint=np.arange((i+1)*perlens,datalen)
            testind=np.arange(i*perlens,(i+1)*perlens)
            #获取训练集和测试集
            trainind=np.concatenate((trainf,traint),axis=0)
            #初始化数据
            setglobalvalue(trainind,[],testind)
            #训练并测试数据
            [r, c] = test.shape  # r为特征数量，r为维数
            labelss = np.zeros([r])
            for j in range(r):
                labelss[j] = likehoodtest(test[j],likehoodfunc)
            res = labelss==testlabels
            acc = res.sum() / len(res)
            score[i]=round(acc,tail)
        print(likehoodfunc.__name__,"method's cross validation:")
        print('cross validation %d score:\n'%(k),score)
        meanscore=score.sum()/len(score)
        print('mean score is %.4f'%(meanscore))
        print('std score is %.4f'%(score.std()))
if __name__ == '__main__':
    start=time.time()()
    print("cost time is %fs"%(time.time()-start))