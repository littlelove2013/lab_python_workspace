import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import random
import math
N = 1000
i=0
j=0
num1=0
num2=0
num3=0
mu1 = np.array([[1, 1]])  # 定义均值
mu2 = np.array([[4, 4]])
mu3 = np.array([[8, 1]])
sigma = np.array([[2, 0], [0, 2]])  # 定义方差
modelNo = [1, 2, 3]#模型标签
p1=[1/3,1/3,1/3]
p2=[0.6,0.3,0.1]

#根据先验概率得出从模型中所取数据数目
def getNumber(p1,p2):
 global i
 global num1
 global num2
 global num3
 array=random.sample(range(0,1000),1000)
 for array[i] in array:
  if array[i] <=p1:
     num1+=1
     i+=1
  elif p1<array[i]<=p2:
         num2+=1
         i+=1
  else:
      num3+=1
      i+=1
def generateX (num1,num2,num3):
   r = cholesky(sigma)  # 开平方
#生成点集合
   s1 = np.dot(np.random.randn(num1, 2), r) + mu1
   s2 = np.dot(np.random.randn(num2, 2), r) + mu2
   s3 = np.dot(np.random.randn(num3, 2), r) + mu3
#画出散点图
   plt.plot(s1[:, 0], s1[:, 1], '+',label='s1: '+ str(len(s1)))
   plt.plot(s2[:, 0], s2[:, 1], '+',label='s2: '+ str(len(s2)))
   plt.plot(s3[:, 0], s3[:, 1], '+',label='s3: '+ str(len(s3)))
   plt.legend()
   plt.show()

   Label = [ ]   #生成数据集的类标签
   for i in range (0, num1):
        Label.append(1)
   for i in range (num1, num1 + num2):
        Label.append(2)
   for i in range ( num1 + num2, N):
        Label.append(3)
   X = np.vstack((s1, s2, s3)) #合并三个集合
   return X, Label

def normpdf (x,mu):#似然函数
    m = 1 / (2 * math.pi * math.sqrt(sigma[0][0] *sigma[1][1] - sigma[0][1] * sigma[1][0]))
    dif = [0 for i in range(0, 2)]
    dif[0] = x[0] - mu[0]
    dif[1] = x[1] - mu[1] #这是什么？
    n = np.mat(dif) * (np.mat(sigma)).I * (np.mat(dif)).T  # a3为array类型
    p = n.tolist()[0][0]
    pdf = m * (np.exp(-1/2 * p))
    return pdf
def LRT (x,mu,p):#似然率测试规则
    score=[0 for i in range (0,3)]
    score[0] = normpdf(x,mu1) * p[0]
    score[1] = normpdf(x,mu2) * p[1]
    score[2] = normpdf(x,mu3) * p[2]
    Max=max(score)
    return  score.index(Max)              #返回最大的那个model
def show(X,Lable,predict ):

def bayes_risk(x,mu,p):#贝叶斯风险测试
    c=np.array([[0, 2, 3], [1, 0, 2.5], [1, 1, 0]])
    risk=[0 for i in range (0,3)]
    ri=[0 for i in range (0,3)]
    ri[0] = normpdf(x, mu1) * p[0]
    ri[1] = normpdf(x, mu2) * p[1]
    ri[2] = normpdf(x, mu3) * p[2]
    for i in range(0,3):
        for j in range(0,3):
            risk[i] += c[i][j] * ri[i]  # 计算风险值
    Min=min(risk)       #找出最小的风险值，
    return risk.index(Min)

def MAP(x,mu,p): #最大后验概率判据
    score=[0 for i in range (0,3)]
    for i in range (0,3):
        score[i] = normpdf(x,mu)* p[i]
        sum+=score[i]
    score[0] = score[0] / sum
    score[1] = score[1] / sum
    score[2] = score[2] / sum
    Max = max(score)
    return score.index(Max)

def distance(x): #最短欧式距离
    distance = [0 for i in range (0,3)]
    distance[0] = (x[0] - mu1[0]) * (x[0] - mu1[0]) + (x[1] -mu1[1]) * (x[1] - mu1[1])
    distance[1] = (x[0] - mu2[0]) * (x[0] - mu2[0]) + (x[1] - mu2[1]) * (x[1] - mu2[1])
    distance[2] = (x[0] - mu3[0]) * (x[0] - mu3[0]) + (x[1] - mu3[1]) * (x[1] - mu3[1])
    Min = min(distance)
    return distance.index(Min)

#getNumber(1000/3,2000/3)#产生X的数目
getNumber(600,900)#产生X'的数目
generateX (num1,num2,num3 )

