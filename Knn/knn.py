import numpy as np
import readiris
import matplotlib.pyplot as plt
import time
#基于矩阵乘法的快速欧氏距离计算的KNN算法
def knn(X,Y,label,k=5):
	if(len(X[0])!=len(Y[0])):
		print("X,Y should be same cols!")
		return False
	lenX=len(X)
	lenY=len(Y)
	labelspace=np.unique(label)
	# Xdouble=np.diag(X.dot(X.T))#行向量
	Xdouble=(X*X).sum(1)
	Xdouble=Xdouble.reshape(-1,1).dot(np.ones((1,lenY)))
	# Ydouble=np.diag(Y.dot(Y.T))
	Ydouble=(Y*Y).sum(1)
	Ydouble = np.ones((lenX,1)).dot(Ydouble.reshape(1,-1))
	XY=X.dot(Y.T)
	dist=Xdouble+Ydouble-2*XY
	ind=dist.argsort(1)
	testlabel=[]
	for i in range(len(X)):
		Rk = label[ind[i,0:k]]
		arr = np.array([(Rk == labelspace[0]).sum(), (Rk == labelspace[1]).sum(), (Rk == labelspace[2]).sum()])
		Rk_label = labelspace[np.argmax(arr)]
		testlabel.append(Rk_label)
	#排序，计算前k个最好的
	return np.asarray(testlabel,np.int32)
#交叉验证
def crossvalidation(Data,Label,k_cross=10):
	# k_cross=10#10折
	datalen=len(Data)
	perlens=int(datalen/k_cross)
	#混淆数据
	arglen = np.arange(0, datalen)
	np.random.shuffle(arglen)
	Data = Data[arglen]
	Label = Label[arglen]
	start_k=1
	end_k=120
	neibork=np.arange(start_k,end_k+1)#k值列表
	err_arr=np.zeros(end_k-start_k+1)
	for i in range(k_cross):
		trainf=np.arange(0,i*perlens)
		traint=np.arange((i+1)*perlens,datalen)
		validateind=np.arange((i)*perlens,(i+1)*perlens)
		#获取训练集和测试集
		trainind=np.concatenate((trainf,traint),axis=0)
		test_d=Data[validateind]
		test_l = Label[validateind]
		train_d=Data[trainind]
		train_l=Label[trainind]
		Kerrrate=geterrarr(test_d,test_l,train_d,train_l,neibork)
		err_arr+=Kerrrate
	#求平均错误
	err_arr=err_arr/k_cross
	min_ind=err_arr.argmin()
	min_err=err_arr[min_ind]
	min_k=neibork[min_ind]
	plt.plot(neibork,err_arr)
	plt.title("%d cross validation: Optimal K =%d,Error rate=%.4f"%(k_cross,min_k,min_err))
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.grid()
	plt.show()
	print("knn method's cross validation:")
	print("Optimal K :%d \nOptimal Error rate%.4f"%(min_k,min_err))
#测试一种数据集划分在所有k值内结果
def geterrarr(test_d,test_l,train_d,train_l,neibork):
	# 测试
	Kerrrate = []
	X_l = neibork
	for i in neibork:
		k = i
		start = time.time()
		prelabel = knn(test_d, train_d, train_l, k)
		res = prelabel == test_l
		acc = res.sum() / len(res)
		err = 1 - acc
		print("k=%d,acc=%.4f,err=%.4f,cost time is %fs" % (k, acc, err, time.time() - start))
		Kerrrate.append(err)
	return np.asarray(Kerrrate)
#@错误率曲线
def err_rate(Data,Label,rate=0.1):
	
	length = len(Data)
	test_len=int(length*rate)
	arglen=np.arange(0,length)
	np.random.shuffle(arglen)
	Data=Data[arglen]
	Label=Label[arglen]
	# 划分测试机和训练集
	test_d=Data[:test_len]
	test_l=Label[:test_len]
	train_d=Data[test_len:]
	train_l=Label[test_len:]
	neibork=np.arange(1,121)
	Kerrrate=geterrarr(test_d,test_l,train_d,train_l,neibork)
	plt.plot(neibork,Kerrrate)
	min_ind = Kerrrate.argmin()
	min_err = Kerrrate[min_ind]
	min_k = neibork[min_ind]
	plt.title("Error rate curve: Optimal K=%d,Error rate=%.4f" % (min_k, min_err))
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.grid()
	plt.show()
if __name__ == '__main__':
	filename = 'iris.txt'
	data = readiris.PRdata(filename)
	Data=data.Data
	Label=data.Label
	err_rate(Data,Label)
	crossvalidation(Data, Label)