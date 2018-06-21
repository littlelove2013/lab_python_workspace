import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time

def loaddata(filename):
	data= scio.loadmat(filename)
	return data['Xtrain'],data['ytrain'].reshape(-1),data['Xtest'],data['ytest'].reshape(-1)
#Stnd0均值单位方差
def Stnd(data):
	mu=data.mean(0)
	s = data.std(0)
	new_data = (data-mu)/s
	# print("mu:",new_data.mean(),"  std:",new_data.std())
	return new_data
def Log(data):
	return np.log(data+0.1)
def Binary(data):
	return np.array(data>0,np.float)
#sigmoid
def sigmoid(inX):  #sigmoid函数
    return 1.0/(1+np.exp(-inX))
#h函数：X每行一条特征，theta为列向量
def h_theta(theta,X):
	h = X.dot(theta)
	return sigmoid(h)
#梯度函数
def GD(lam,theta,X,Y):
	h= h_theta(theta,X)
	h = (h-Y.reshape(-1,1))
	h = h*X
	g= h.mean(0)
	g=g+lam*(theta.reshape(-1))
	return g.reshape(-1,1)
#梯度下降函数
def sgd(lam,X,Y,alpha= 1.0,iter = 500):
	sig=1e-4
	m,n = X.shape
	theta = np.random.randn(n, 1)  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
	# theta = np.ones((n, 1))  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
	for i in range(iter):
		new_theta=theta-alpha*GD(lam,theta,X,Y)
		if np.abs(new_theta-theta).mean()<sig:
			return new_theta
		# print("iter = %d, delta theta %.10f" %(i, alpha*np.abs(new_theta - theta).mean()))
		theta=new_theta
		
	return theta

def gradAscent(dataMat, labelMat): #梯度上升求最优参数
    dataMatrix=np.mat(dataMat) #将读取的数据转换为矩阵
    classLabels=np.mat(labelMat).transpose() #将读取的数据转换为矩阵
    m,n = np.shape(dataMatrix)
    alpha = 0.001  #设置梯度的阀值，该值越大梯度上升幅度越大
    maxCycles = 500 #设置迭代的次数，一般看实际数据进行设定，有些可能200次就够了
    weights = np.ones((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (classLabels - h)     #求导后差值
        new_weights = weights + alpha * dataMatrix.transpose()* error #迭代更新权重
        # print("iter = %d, delta theta %.10f" % (k, np.abs(new_weights - weights).mean()))
        weights=new_weights
    return weights

#测试平均错误率
def getres(d,theta):
	ll = h_theta(theta,d)
	res = np.array(ll>0.5,np.uint8).reshape(-1)
	# l=np.array(l,np.uint8).reshape(-1)
	# res = np.array(ll2!=l,np.uint8).reshape(-1)
	return res

def LR(test_d, train_d, train_l, lam):
	theta = sgd(lam, train_d, train_l)
	res = getres(test_d, theta)
	return res

# 交叉验证
def crossvalidation(Data, Label, k_cross=10):
	# k_cross=10#10折
	datalen = len(Data)
	perlens = int(datalen / k_cross)
	# 混淆数据
	arglen = np.arange(0, datalen)
	np.random.shuffle(arglen)
	Data = Data[arglen]
	Label = Label[arglen]
	start_k = 0.0
	end_k = 0.01
	step=0.0002
	neibork = np.arange(start_k, end_k+step,step)  # k值列表
	err_arr = np.zeros(neibork.size)
	for i in range(k_cross):
		trainf = np.arange(0, i * perlens)
		traint = np.arange((i + 1) * perlens, datalen)
		validateind = np.arange((i) * perlens, (i + 1) * perlens)
		# 获取训练集和测试集
		trainind = np.concatenate((trainf, traint), axis=0)
		test_d = Data[validateind]
		test_l = Label[validateind]
		train_d = Data[trainind]
		train_l = Label[trainind]
		Kerrrate = geterrarr(test_d, test_l, train_d, train_l, neibork)
		err_arr += Kerrrate
	# 求平均错误
	err_arr = err_arr / k_cross
	min_ind = err_arr.argmin()
	min_err = err_arr[min_ind]
	min_k = neibork[min_ind]
	plt.plot(neibork, err_arr)
	plt.title("%d cross validation: Optimal K =%f,Error rate=%.4f" % (k_cross, min_k, min_err))
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.grid()
	plt.show()
	print("method's cross validation:")
	print("Optimal K :%f \nOptimal Error rate:%.4f" % (min_k, min_err))
	return min_k


# 测试一种数据集划分在所有k值内结果
def geterrarr(test_d, test_l, train_d, train_l, validalist,func=LR):
	# 测试
	Kerrrate = []
	X_l = validalist
	for i in validalist:
		k = i
		start = time.time()
		prelabel = func(test_d, train_d, train_l, k)
		res = prelabel == test_l
		acc = res.sum() / len(res)
		err = 1 - acc
		# print("k=%.4f,acc=%.4f,err=%.4f,cost time is %fs" % (k, acc, err, time.time() - start))
		Kerrrate.append(err)
	return np.asarray(Kerrrate)

#不同预处理并交叉验证逻辑回归
def preLR(Train,Trainl,Test,Testl,prefun):
	TrainS = prefun(Train)
	TestS = prefun(Test)
	min_lab = crossvalidation(TrainS, Trainl)
	ress = LR(TestS, TrainS, Trainl, min_lab)
	error = ress != (Testl.reshape(-1))
	print("%s pre process function on lambda = %f , error = %f" % (prefun.__name__,min_lab, error.sum() / error.size))

# @错误率曲线
def err_rate(Data, Label, rate=0.1):
	length = len(Data)
	test_len = int(length * rate)
	arglen = np.arange(0, length)
	np.random.shuffle(arglen)
	Data = Data[arglen]
	Label = Label[arglen]
	# 划分测试机和训练集
	test_d = Data[:test_len]
	test_l = Label[:test_len]
	train_d = Data[test_len:]
	train_l = Label[test_len:]
	neibork = np.arange(1, 121)
	Kerrrate = geterrarr(test_d, test_l, train_d, train_l, neibork)
	plt.plot(neibork, Kerrrate)
	min_ind = Kerrrate.argmin()
	min_err = Kerrrate[min_ind]
	min_k = neibork[min_ind]
	plt.title("Error rate curve: Optimal K=%f,Error rate=%.4f" % (min_k, min_err))
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.grid()
	plt.show()
if __name__ == '__main__':
    Train,Trainl,Test,Testl=loaddata(filename="spamData.mat")
    preLR(Train, Trainl, Test, Testl, prefun=Stnd)
    preLR(Train, Trainl, Test, Testl, prefun=Log)
    preLR(Train, Trainl, Test, Testl, prefun=Binary)
    #test error
    # Train=Binary(Train)
    # theta = sgd(0.0,Train,Trainl)
    # # theta = gradAscent(Train, Trainl)
    # res = getres(Train, theta)
    # res= res!=(Trainl.reshape(-1))
    # print("sgd error:", res.sum() / res.size)
    
  
    # #测试平均错误率
    # res = geterror(Train,Trainl,theta)
    # print(" gradAscent error:",res.sum()/res.size)
    # Test= Binary(Test)
    print("main")