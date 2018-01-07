import math
import numpy as np
import predata
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#导入Scikit learn库中的KNN算法
from sklearn import svm
#从sklearn库中导入svm
from sklearn import cross_validation

class Stacking:
    def __init__(self,train,train_label,test):
        self.tr=train
        self.tr_l=train_label
        self.te=test
        self.init()
    def init(self):
        #分批次
        self.fold=5
        self.foldtrainlen=math.floor(len(self.tr)/self.fold)#下取整
        self.trainend=self.fold*self.foldtrainlen
        # self.trainlens=len(self.tr)
        self.trainlens=self.trainend
        self.testlens=len(self.te)
        self.index=np.arange(0,self.trainlens)
        #需要训练的模型列表,多搞几个
        self.model=[linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6),
                    # RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1),
                    # KNeighborsClassifier(n_neighbors=5),
                    svm.SVC(C=1.0, kernel='rbf', degree=3),
                    # AdaBoostClassifier(n_estimators=100),
                    DecisionTreeClassifier(),
                    # GradientBoostingClassifier()
                    # GaussianNB(),
                    # MultinomialNB(),
                    # BernoulliNB()
                    ]*15
        #每个分类器针对的特征列不一样,比如KNN应该针对连续的,而逻辑回归针对离散的,svm针对连续的
        self.featurelist=[
            # list(range(3,26)),
            # list(range(3,26)),
            # list(range(3,26)),
            list(range(0, 26)),
            list(range(0, 26)),
            list(range(0, 26))
        ]*15
        # self.featurelist = list(range(26))*7
        # self.featurelist = np.array(self.featurelist).reshape(-1,1)
        # print(self.featurelist)
    def gettrainsplit(self,fold):
        if fold==0:
            np.random.shuffle(self.index)
        start = self.foldtrainlen * fold
        end = self.foldtrainlen * (fold + 1)
        test_index=self.index[start:end]
        train_index = np.concatenate((self.index[0:start], self.index[end:self.trainend]), axis=0)
        # train_index= self.index[0:start]+self.index[end:self.trainend]
        return test_index,train_index
    #训练一趟folder
    def trainfold(self,modelnum):
        model= self.model[modelnum]
        flist=self.featurelist[modelnum]
        oof_train = np.zeros(shape=(self.trainlens))
        # 因为要训练fold次,然后取均值
        oof_test_skf = np.zeros(shape=(self.testlens, self.fold))
        oof_test = np.zeros(shape=(self.testlens))
        # 训练fold次,每次取一条测试,fold-1条训练
        for i in range(self.fold):
            test_index, train_index = self.gettrainsplit(i)
            kf_x_test = self.tr[test_index]
            kf_x_test = kf_x_test[:,flist]
            kf_x_train = self.tr[train_index]
            kf_x_train = kf_x_train[:,flist]
            kf_y_train = self.tr_l[train_index]
            model.fit(kf_x_train, kf_y_train)
            # 对未训练的测试集做预测
            oof_train[test_index] = model.predict(kf_x_test)
            # 对训练集做预测
            x_pre=self.te[:,flist]
            oof_test_skf[:, i] = model.predict(x_pre)
        # 预测的fold个测试机取均值
        oof_test[:] = oof_test_skf.mean(1)
        return oof_train,oof_test
    def stacking(self):
        modellens=len(self.model)
        self.trainfeature=np.zeros(shape=(self.trainlens,modellens))
        self.trainlabels=self.tr_l[0:self.trainlens]
        self.testfeature=np.zeros(shape=(self.testlens,modellens))
        #提取特征的模型列表
        for i in range(modellens):
            oof_train, oof_test=self.trainfold(i)
            self.trainfeature[:,i]=oof_train
            self.testfeature[:,i]=oof_test
    def getstackingfeature(self):
        self.stacking()
        return self.trainfeature,self.trainlabels,self.testfeature

def savetest(save=False):
    # 取训练集
    file_path = './dataset/train.csv'
    savefile = True
    savepath = './data'
    predataflag = 1
    _, _, tr, tr_l, _ = predata.predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                            predataflag=predataflag)
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata.predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
    stack = Stacking(tr, tr_l, te)
    new_tr, new_tr_l, new_te = stack.getstackingfeature()
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    cross = cross_validation.cross_val_score(clf1, new_tr, new_tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))

    clf=linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(new_tr, new_tr_l)
    predictedl = clf.predict(new_te)
    if(save):
        predata.savepredictcsv(passengerIdlist, predictedl, 'stacking_test')
    return passengerIdlist, predictedl

def main():
    file_path = './dataset/train.csv'
    savefile = True
    savepath = './data'
    predataflag = 1
    #stacking
    _, _, tr, tr_l, _ = predata.predata(file_path, savefile, savename='train', savepath=savepath, ratio=0,
                                        predataflag=predataflag)
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata.predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                                   predataflag=predataflag)
    svc = GradientBoostingClassifier()
    cross = cross_validation.cross_val_score(svc, tr, tr_l, cv=5)
    print('no stacking : ', cross, "mean=%.4f" % (cross.mean()))

    stack = Stacking(tr, tr_l, te)
    new_tr, new_tr_l, new_te = stack.getstackingfeature()
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    cross = cross_validation.cross_val_score(clf1, new_tr, new_tr_l, cv=5)
    print(cross, "mean=%.4f" % (cross.mean()))


if __name__ == '__main__':
    main()
    # savetest()