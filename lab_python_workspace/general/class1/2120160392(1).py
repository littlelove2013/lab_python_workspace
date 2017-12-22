import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

#生成number个随机二维高斯分布矩阵
def generate2Drandnormal(mu,sigma,number):
    #np.random.seed(0)
    #开方，即求标准差
    R = cholesky(sigma)
    s = np.dot(np.random.randn(number, 2), R) + mu
    return s
#根据N，P的值，获取X的值
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
        label[index]=i
    return label,X
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
len11=len(np.where(label1==0)[0])
len12=len(np.where(label1==1)[0])
len13=len(np.where(label1==2)[0])
#print(len11,len12,len13)
plt.figure(0)
mu_X=np.zeros([3,2])
S_X=np.zeros([3,2,2])
#S_X_=np.zeros([3,2,2])
for c ,i , target_names in zip('ryb', [0, 1,2], ['model1:'+str(len11),'model2:'+str(len12),'model3:'+str(len13)]):
    data = X1[np.where(label1 == i)]
    #均值
    mu_X[i]=np.sum(data,0)/len(data)
    tmp=data-mu_X[i]
    #协方差
    S_X[i]=tmp.T.dot(tmp)/(len(data)-1)
    #S_X_[i]=np.cov(data.T)
    plt.scatter(data[:,0], data[:,1], c=c, label=target_names)
plt.grid()
plt.legend()
plt.title('X')
print("\nmu_X:\n",mu_X,"\nS_X:\n",S_X)

[label2,X2] = generateXwithNP(N,P2)
len21=len(np.where(label2==0)[0])
len22=len(np.where(label2==1)[0])
len23=len(np.where(label2==2)[0])
plt.figure(1)
for c ,i , target_names in zip('ryb', [0, 1,2], ['model1:'+str(len21),'model2:'+str(len22),'model3:'+str(len23)]):
    plt.scatter(X2[np.where(label2 == i),0], X2[np.where(label2 == i),1], c=c, label=target_names)
plt.grid()
plt.legend()
plt.title("X'")
plt.show()