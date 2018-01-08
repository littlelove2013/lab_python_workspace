from predata import *
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
#导入Scikit learn库中的KNN算法
from sklearn import svm
#从sklearn库中导入svm
from sklearn import cross_validation
import sparseautoencoder as sn
import Stacking


#回归树预测
def RFC(file_path='./dataset/train.csv',savefile=False,savepath='./data',ratio=0,predataflag=0):
    # te, te_l, tr, tr_l,_ = predata(file_path,savepath=savepath,savefile=savefile,ratio=ratio,predataflag=predataflag)
    _, _, tr, tr_l, _ = predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                predataflag=predataflag)
    rfr = RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
    cross=cross_validation.cross_val_score(rfr, tr, tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))

    # 取训练集
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    rfr.fit(tr, tr_l)
    predictedl = rfr.predict(te)
    return passengerIdlist,predictedl
    '''
    rfr.fit(tr, tr_l)
    predictedl = rfr.predict(te)
    res = te_l == predictedl
    count = res[res == True].shape[0]
    acc = count / len(res)
    print(te_l == predictedl, 'RandomForestClassifier is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
    '''
#逻辑回归预测
def LR(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0,predataflag=1):
    # te, te_l, tr, tr_l, _ = predata(file_path, savepath=savepath, savefile=savefile, ratio=ratio, predataflag=predataflag)
    _, _, tr, tr_l, _ = predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                predataflag=predataflag)
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    cross = cross_validation.cross_val_score(clf, tr, tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))

    # 取训练集
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    clf.fit(tr, tr_l)
    predictedl = clf.predict(te)
    return passengerIdlist,predictedl
    '''
    clf.fit(tr, tr_l)
    predictedl = clf.predict(te)
    preint = predictedl.astype(np.int32)
    res = te_l == preint
    count = res[res == True].shape[0]
    acc = count / len(res)
    print(te_l == predictedl, 'LogisticRegression is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
    '''

def KNN(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0,predataflag=1):
    # te, te_l, tr, tr_l, _ = predata(file_path, savepath=savepath, savefile=savefile, ratio=ratio, predataflag=predataflag)
    _, _, tr, tr_l, _ = predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                predataflag=predataflag)
    knn = KNeighborsClassifier(n_neighbors=5)

    cross=cross_validation.cross_val_score(knn, tr, tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))

    # 取训练集
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    knn.fit(tr, tr_l)
    predictedl = knn.predict(te)
    return passengerIdlist,predictedl

def SVM(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0,predataflag=1):
    _, _, tr, tr_l,_ = predata(file_path, savefile,savename='train', savepath=savepath, ratio=0,predataflag=predataflag)
    #对tr用稀疏矩阵编码：
    # w=np.load('weights1.npy')
    # trsparse=tr.dot(w)
    #取训练集
    test_path='./dataset/test.csv'
    te,_,_,_,passengerIdlist=predata(test_path, savefile,savename='test',savepath=savepath,ratio=1,predataflag=predataflag)
    # 对tr用稀疏矩阵编码：
    # tesparse=te.dot(w)
    svc = svm.SVC(C=1.0, kernel = 'rbf', degree = 3)
    cross=cross_validation.cross_val_score(svc, tr, tr_l, cv=5)
    print(cross,"mean=%.4f"%(cross.mean()))

    svc.fit(tr, tr_l)
    predictedl = svc.predict(te)
    return passengerIdlist,predictedl
    #保存结果
    # savepredictcsv(passengerIdlist,predictedl,'test')
    # res = te_l == predictedl
    # count = res[res == True].shape[0]
    # acc = count / len(res)
    # print(te_l == predictedl, 'RandomForestClassifier is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
#回归树预测
def AdaBoost(file_path='./dataset/train.csv',savefile=False,savepath='./data',ratio=0,predataflag=0):
    # te, te_l, tr, tr_l,_ = predata(file_path,savepath=savepath,savefile=savefile,ratio=ratio,predataflag=predataflag)
    _, _, tr, tr_l, _ = predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                predataflag=predataflag)
    clf = AdaBoostClassifier(n_estimators=100)  # 迭代100次
    cross=cross_validation.cross_val_score(clf, tr, tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))

    # 取训练集
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    clf.fit(tr, tr_l)
    predictedl = clf.predict(te)
    return passengerIdlist,predictedl
    '''
    rfr.fit(tr, tr_l)
    predictedl = rfr.predict(te)
    res = te_l == predictedl
    count = res[res == True].shape[0]
    acc = count / len(res)
    print(te_l == predictedl, 'RandomForestClassifier is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
    '''
#2层稀疏编码网络预测:0.7894
def softmax2layerfc():
    # 取训练集
    savefile = True
    savepath = './data'
    predataflag = 1
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    predictedl=sn.softmaxpredict(te)
    return passengerIdlist, predictedl

def main():
    passengerIdlist, p1 = RFC()
    _,p2=LR()
    _,p3=KNN()
    _,p4=SVM()
    passengerIdlist,p5=AdaBoost()
    passengerIdlist, p6 = softmax2layerfc()
    passengerIdlist,p7 =Stacking.savetest()
    # num=5
    # p=np.array([p1,p2,p3,p4,p5]).sum(0)
    # p[p>num/2]=1
    # p[p<=num/2]=0
    # # 保存结果
    savepredictcsv(passengerIdlist,p7,'stacking_test')
if __name__ == '__main__':
    main()