import numpy as np
import math
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

#根据N，P的值，获取X的值
#label:0,1,2
labels=np.array([0,1,2])
#生成number个随机二维高斯分布矩阵
def generate2Drandnormal(mu,sigma,number):
    #np.random.seed(0)
    #开方，即求标准差
    R = cholesky(sigma)
    s = np.dot(np.random.randn(number, 2), R) + mu
    return s
def generateXwithNP(N,P):
    X = np.zeros([N,2])
    label = np.zeros(N)
    index_rand = np.random.rand(N)
    for i in range(len(P)):
        #找出在该概率区间的下标
        index = np.where((index_rand>=sum(P[:i])) & (index_rand<sum(P[:i+1])))
        len_ind = len(index[0])
        data = generate2Drandnormal(mu[i],S,len_ind)
        X[index]=data
        label[index]=labels[i]
    return label,X
#正态分布密度函数
def normpdf(mu,sigma,x):
    d=len(mu)
    mu=np.mat(mu)
    x=np.mat(x)
    sigma=np.mat(sigma)
    factor1=1/(math.pow(math.sqrt(2*math.pi),d)*math.sqrt(np.linalg.det(sigma)))
    factor2 = math.exp((-1 / 2) *((x - mu) * sigma.I * (x - mu).T))
    return factor1*factor2

def showinpic(x,label,name,color=['orange','green','blue'],test_label=[],test_color='black'):
    #画出点图
    plt.figure(0)
    #S_X_=np.zeros([3,2,2])
    if test_label!=[]:
        right = (label != test_label)
        model1 = len(np.where(label[right]==labels[0])[0])
        model2 = len(np.where(label[right] ==labels[1])[0])
        model3 = len(np.where(label[right] == labels[2])[0])
        for c, i, target_names in zip(color, labels, ['model1:'+str(model1), 'model2:'+str(model2), 'model3:'+str(model3)]):
            data = x[np.where(label == i)]
            plt.scatter(data[:, 0], data[:, 1], c=c, label=target_names)
        #取错误的点
        res=(label!=test_label)
        data = x[res]
        plt.scatter(data[:, 0], data[:, 1], c=test_color, label='error:'+str(len(data)),marker='.')
    else:
        for c, i, target_names in zip(color, labels, ['model1', 'model2', 'model3']):
            data = x[np.where(label == i)]
            plt.scatter(data[:, 0], data[:, 1], c=c, label=target_names)
    plt.grid()
    plt.legend()
    plt.title(name)
    plt.savefig(name+'.png')
    #plt.show()
    plt.close()

#均值
mu=np.array([[1,1],[4,4],[8,1]])
#协方差
S=2*np.array([[1,0],[0,1]])
#1000个数据点集
N = 1000
#X1的数据集，先验概率相同
P1=[1/3,1/3,1/3]
#X2的数据集，先验概率
P2=[0.6,0.3,0.1]
#test = generate2Drandnormal(mu[0],S,1000)
[label1,X1] = generateXwithNP(N,P1)
[label2,X2] = generateXwithNP(N,P2)
showinpic(X1,label1,'X1')
showinpic(X2,label2,'X2')

#P(w)先验概率
#setnumber:数据集标签
def Pprior(w,setnumber):
    # X1的数据集，先验概率相同
    P1 = [1 / 3, 1 / 3, 1 / 3]
    # X2的数据集，先验概率
    P2 = [0.6, 0.3, 0.1]
    if setnumber==1:
        return P1[w]
    if setnumber==2:
        return P2[w]
    return 0
#P(x|w):似然函数
def Plikehood(x,w):
    # 均值
    mu = np.array([[1, 1], [4, 4], [8, 1]])
    # 协方差
    S = 2 * np.array([[1, 0], [0, 1]])
    for i in range(3):
        if(w==i):
            mu= mu[i]
            sigma=S
            return normpdf(mu, sigma, x)
    return 0
#P(w|x):最大后验概率
def Posterior(w,x,setnumber):
    #xx
    xx=0
    for i in labels:
        tmp = 0
        tmp=Plikehood(x,i)*Pprior(i,setnumber)
        xx=xx+tmp
    pwx=(Plikehood(x,w)*Pprior(w,setnumber))/xx
    return pwx

#似然率测试
#x:待测试样本
#label:数据集label，用于计算先验概率
def likehoodtest(x,setnumber):
    lab=0
    max_score = Plikehood(x,labels[0])*Pprior(labels[0],setnumber)
    for i in labels[1:]:
        tmp = 0
        tmp=Plikehood(x,i)*Pprior(i,setnumber)
        if tmp>max_score:
            max_score=tmp
            lab=i
    return lab
def likehoodtestforvector(x,setnumber):
    [r,c]=x.shape#r为特征数量，r为维数
    labelss=np.zeros([r])
    for i in range(r):
        labelss[i]=likehoodtest(x[i],setnumber)
    return labelss

#贝叶斯风险测试
#代价矩阵
def getcost(i,j):
    C=np.array([[0,2,3],[1,0,2.5],[1,1,0]])
    return C[i,j]
#x关于某一类w的条件风险
def getR(w,x,setnumber):
    C = np.array([[0, 2, 3], [1, 0, 2.5], [1, 1, 0]])
    i=w
    sumcost=0
    for j in labels:
        pwx=Posterior(j,x,setnumber)
        sumcost=sumcost+C[i,j]*pwx
    return sumcost
#贝叶斯多累条件风险测试
def Rtest(x,setnumber):
    lab = 0
    #最小得分取第一个label的风险值
    min_score = getR(labels[0], x, setnumber)
    for i in labels[1:]:
        tmp = getR(i, x, setnumber)
        if tmp < min_score:
            min_score = tmp
            lab = i
    return lab
def Rtestforvector(x,setnumber):
    [r, c] = x.shape  # r为特征数量，r为维数
    labelss = np.zeros([r])
    for i in range(r):
        labelss[i] = Rtest(x[i], setnumber)
    return labelss

#最大后验概率
def posteriortest(x,setnumber):
    lab = 0
    max_score = Posterior(labels[0],x,setnumber)
    for i in labels[1:]:
        tmp = Posterior(i,x,setnumber)
        if tmp > max_score:
            max_score = tmp
            lab = i
    return lab
def posteriortestforvector(x,setnumber):
    [r,c]=x.shape#r为特征数量，r为维数
    labelss=np.zeros([r])
    for i in range(r):
        labelss[i]=posteriortest(x[i],setnumber)
    return labelss

#最小欧氏距离
#计算两点的欧式距离
def distant(x,y):
    delt=y-x
    return math.sqrt(delt.dot(delt))
def distanttest(x):
    lab=0
    min_dis=distant(x,mu[0])
    for i in labels[1:]:
        tmp=distant(x,mu[i])
        if tmp<min_dis:
            lab=i
            min_dis=tmp
    return lab
#return [labelss]
#每次取一个点，然后取不包含该点得其余所有点集
def distanttestforvector(x):
    [r,c]=x.shape#r为特征数量，r为维数
    labelss = np.zeros([r])
    for i in range(r):
        labelss[i]=distanttest(x[i])
    return labelss
#得准确率
#计算10次，得到似然率测试的平均准确率
#flag:表示选择哪种规则：1、似然率测试，2、贝叶斯风险测试，3、最大后验概率，4、最小欧氏距离测试
def gettestaccuracy(set,label,setnumber,flag=1):
    num=1
    score=np.zeros([num])
    reallabel=label
    for i in range(num):
        testlabel=None
        res = None
        if flag==1:
            testlabel= likehoodtestforvector(set,setnumber)
            res = (testlabel == label)
        if flag==2:
            testlabel= Rtestforvector(set,setnumber)
            res = (testlabel == label)
        if flag==3:
            testlabel = posteriortestforvector(set, setnumber)
            res = (testlabel == label)
        if flag==4:
            testlabel = distanttestforvector(set)
            res = (testlabel == label)
        score[i]= res.sum()/len(res)
    acc=score.sum()/num
    err=1-acc
    return acc,err,testlabel
#显示结果函数
def showresult(x,label,name,setnumber,flag=1):
    acc, err, test_label = gettestaccuracy(x, label,setnumber,flag)
    showinpic(x, label, name,['orange','green','blue'], test_label,test_color='red')
    print('\n'+name+'：\n accracy:%.5f,error:%.5f' % (acc, err))


#似然率测试画图
showresult(X1,label1,'X1 likehoodtest',1)
showresult(X2,label2,'X2 likehoodtest',2)

#贝叶斯风险测试画图
showresult(X1,label1,'X1 BayesRisk',1,2)
showresult(X2,label2,'X2 BayesRisk',2,2)

#最大后验概率
showresult(X1,label1,'X1 BestPosterior',1,3)
showresult(X2,label2,'X2 BestPosterior',2,3)

#最小欧氏距离测试
showresult(X1,label1,'X1 MinimumEuclideanDistance',1,4)
showresult(X2,label2,'X2 MinimumEuclideanDistance',2,4)
