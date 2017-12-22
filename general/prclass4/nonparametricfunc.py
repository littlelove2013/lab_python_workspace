import numpy as np #科学计算
import matplotlib.pyplot as plt
import csv
import math

# 获取数据
cvfile = csv.reader(open('./HWData3.csv', 'r'))
srcdata = []
srclabels = []
for data in cvfile:
    srcdata.append(data[0:4])
    srclabels.append(data[4])
#原始数据
srcdata = np.array(srcdata, dtype=float)
srclabels = np.array(srclabels, dtype=int)
#将原始数据打乱
datalen=len(srclabels)
shuffleindex=np.arange(0,datalen)
#打乱顺序,定义为global变量
np.random.shuffle(shuffleindex)
shuffledata=srcdata[shuffleindex]
shufflelabels=srclabels[shuffleindex]
#print(shuffledata,shufflelabels)

labelspace = [1, 2, 3]

# 计算先验概率
trainsizes = []
Pprior = []
#训练集
train=[]
trainlabels=[]
#
test=[]
testlabels=[]
#
validate=[]
validatelabels=[]
#定义为global变量
#参数估计的均值和方差矩阵
mu = []
S = []
#均一化方差的训练集
trainnormaldata=[]
trainnormalstd=[]
#平滑核函数的h
h = []
# print(Pprior)
# 估计每一类的均值mu和协方差矩阵
def getmuandsigma(X):
    mu = X.mean(0)
    tmp = X - mu
    Sigma = tmp.T.dot(tmp) / (len(X) - 1)
    return mu, Sigma

def setglobalvalue(trainind,validateind,testind):
    global train,trainlabels,validate,validatelabels,test,testlabels
    train=shuffledata[trainind]
    trainlabels=shufflelabels[trainind]
    test=shuffledata[testind]
    testlabels=shufflelabels[testind]
    if len(validateind)>0:
        validate=shuffledata[validateind]
        validatelabels=shufflelabels[validateind]
    #计算一些参数：先验概率，均值方差
    global trainsizes,Pprior,mu,S
    p = trainlabels.size
    size1 = (trainlabels == labelspace[0]).sum()
    p1 = size1 / p
    size2 = (trainlabels == labelspace[1]).sum()
    p2 = size2 / p
    size3 = (trainlabels == labelspace[2]).sum()
    p3 = size3 / p
    #size
    trainsizes = [size1, size2, size3]
    #先验概率
    Pprior = [p1, p2, p3]
    #均值方差
    mu1, S1 = getmuandsigma(train[trainlabels == labelspace[0]])
    mu2, S2 = getmuandsigma(train[trainlabels == labelspace[1]])
    mu3, S3 = getmuandsigma(train[trainlabels == labelspace[2]])
    mu = np.array([mu1, mu2, mu3])
    S = np.array([S1, S2, S3])
    #计算四维均一化方差
    global trainnormaldata,trainnormalstd,h
    trainnormaldata,trainnormalstd=normaldata(train,trainlabels)
    h1 = 1.06 * 1 * trainsizes[0] ** (-1 / 5)
    h2 = 1.06 * 1 * trainsizes[1] ** (-1 / 5)
    h3 = 1.06 * 1 * trainsizes[2] ** (-1 / 5)
    h = [h1, h2, h3]

# 非参数估计
# 将原始数据的各个维度归一化为单位方差数据
# 非0均值1方差归一化
def normalvector(data, check=False):
    if data.size != data.shape[0]:
        print('please input vector')
    s = data.std()
    norm = (data) / s
    if (check):
        print('mean:', norm.mean(), '  std:', norm.std())
    return norm,s
def normaldata(traindata,trainlabels):
    normdata = traindata.copy()
    normstd=[]
    for label in labelspace:
        tmp = normdata[trainlabels == label]
        tmpstd=[]
        for i in range(tmp.shape[1]):
            normed,s = normalvector(tmp[:, i], False)
            normdata[trainlabels == label, i] = normed
            tmpstd.append(s)
        normstd.append(tmpstd)
    return normdata,np.array(normstd)
'''
normdata = normaldata()
# 计算h，即边长，但是因为已经归一化单位方差，所有维上草书相同，各向同性，所以只需要估计一维的h
h1 = 1.06 * 1 * trainsizes[0] ** (-1 / 5)
h2 = 1.06 * 1 * trainsizes[1] ** (-1 / 5)
h3 = 1.06 * 1 * trainsizes[2] ** (-1 / 5)
h = [h1, h2, h3]
'''


# 正态分布密度函数
# sigma为协方差
def normpdf(mu, sigma, x):
    d = len(mu)
    mu = np.mat(mu)
    x = np.mat(x)
    sigma = np.mat(sigma)
    factor1 = 1 / (math.pow(math.sqrt(2 * math.pi), d) * math.sqrt(np.linalg.det(sigma)))
    factor2 = math.exp((-1 / 2) * ((x - mu) * sigma.I * (x - mu).T))
    return factor1 * factor2

# P(x|w):似然函数
def Plikehood(x, w):
    global mu,S
    mu1 = mu[w - 1]
    sigma = S[w - 1]
    return normpdf(mu1, sigma, x)


# 似然率测试
# Plikehood:为似然函数
def likehoodtest(x, likehoodfunc=Plikehood):
    lab = labelspace[0]
    max_score = likehoodfunc(x, lab) * Pprior[lab - 1]
    for i in labelspace[1:]:
        tmp = 0
        tmp = likehoodfunc(x, i) * Pprior[i - 1]
        if tmp > max_score:
            max_score = tmp
            lab = i
    return lab

def likehoodtestforvector(x, likehoodfunc=Plikehood):
    [r, c] = x.shape  # r为特征数量，r为维数
    labelss = np.zeros([r])
    for i in range(r):
        labelss[i] = likehoodtest(x[i],likehoodfunc)
    return labelss




# 非参数估计,平滑核函数
# 平滑核函数使用方差为h/2，均值为0的高斯函数
# w为类别，估计的似然函数
# 因为x不知道类别，所以x用对应训练集类别方差做缩放
def guasskenel(x, w):
    d = 4
    tmp = trainnormaldata[trainlabels == w]
    #对x的各维按照训练类别的维度进行归一化
    normlizestd=x/trainnormalstd[w-1]
    k = 0
    for i in tmp:
        # 计算欧式距离
        dist = np.sqrt(((normlizestd - i) ** 2).sum())
        # 计算均值=0，标准差=h/2的概率，加起来
        k += normpdf([0], h[w - 1]/2, dist)
    # 计算似然函数
    p = (1 / (trainsizes[w - 1] * h[w - 1] ** d)) * k
    return p
#非参数估计：朴素贝叶斯：
def naivebayes(x,w):
    #对x的每一维进行单变量的平滑核函数估计，得到p
    p=1
    data = train[trainlabels==w]
    for i in range(len(x)):
        val=data[:,i]
        sig=val.std()
        hopt=1.06*sig*trainsizes[w-1] ** (-1 / 5)
        k=0
        for value in val:
            k+=normpdf([0], hopt/2, x[i]-value)
        p*=(1 / (trainsizes[w - 1] * hopt)) * k
    return p
#最近邻密度估计
def kneibor(x,w):
    #维数
    d=4
    #k
    k=10
    data=train[trainlabels==w]
    CD= math.pi**(d/2)/np.math.factorial(d/2)
    #计算第k个最近邻的距离Rk
    dist=np.sqrt(((x-data)**2).sum(1))
    ind=dist.argsort()
    Rk=dist[ind[k-1]]
    p=k/(trainsizes[w-1]*CD*Rk**d)
    return p

knnk=1
def knn(x):
    #计算与所有训练集的距离
    dist=np.sqrt(((x-train)**2).sum(1))
    #print(dist.shape)
    ind=dist.argsort()
    #取前k个
    Rk=trainlabels[ind[0:knnk]]
    #print((Rk==labelspace[1]).sum())
    arr=np.array([(Rk==labelspace[0]).sum(),(Rk==labelspace[1]).sum(),(Rk==labelspace[2]).sum()])
    label = labelspace[np.argmax(arr)]
    #print(Rk,"label:",label)
    return label
def showacc(prelabels):
    res = prelabels == labels
    acc = res.sum() / len(labels)
    print('acc:%.4f ,error:%0.4f'%(acc,1-acc))
    return acc,res

#交叉验证函数
def crossvalidation(likehoodfunc=Plikehood):
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

def main():
    '''
    prelabels1 = likehoodtestforvector(train)
    showacc(prelabels1)
    # fei参数估计
    prelabels2 = likehoodtestforvector(normdata,guasskenel)
    showacc(prelabels2)
    prelabels3 = likehoodtestforvector(train,naivebayes)
    showacc(prelabels3)
    prelabels4 = likehoodtestforvector(train,kneibor)
    showacc(prelabels4)
    '''

    crossvalidation(likehoodfunc=Plikehood)
    crossvalidation(likehoodfunc=guasskenel)
    crossvalidation(likehoodfunc=naivebayes)
    crossvalidation(likehoodfunc=knn)
    
    
if __name__ == '__main__':
    main()
