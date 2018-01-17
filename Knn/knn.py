import numpy as np
import time
xr=1000#x行数
yr=60000#y的行数
c=100#特征列数
#模拟输入数据
X=np.random.rand(xr,c)
Y=np.random.rand(yr,c)
def knn():
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
    return 0
if __name__ == '__main__':
    start=time.time()()
    print("cost time is %fs"%(time.time()-start))