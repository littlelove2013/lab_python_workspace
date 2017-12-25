import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
import scipy.io as sio
import os

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
#导入Scikit learn库中的KNN算法
from sklearn import svm
#从sklearn库中导入svm
from sklearn import cross_validation

def changeDtype(src,dstdtype='int64'):
    a=src.values
    a=np.array(a,dtype=dstdtype)
    return a
#对有无Age分开
def getAgeData(data):
    scaler = preprocessing.StandardScaler()
    data_hasAge = data[pd.notnull(data.Age)]
    age_scale_param = scaler.fit(data_hasAge['Age'])
    data_hasAge['Age_scaled'] = scaler.fit_transform(data_hasAge['Age'], age_scale_param)
    # 没有Age
    data_noAge = data[pd.isnull(data.Age)]
    data_noAge.drop(['Age'], axis=1, inplace=True)
    return data_hasAge,data_noAge

def set_missing_ages(data,featurelist):
    df=data.copy()
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[featurelist]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[df.Age.notnull()].as_matrix()
    unknown_age = age_df[df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = df[df.Age.notnull()].Age.as_matrix()
    # X即特征属性值
    X = known_age

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age)
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    return df

#ratio为取测试样本的比例
#ratio=0则表示只取训练集，带标签
#ratio=1则表示只取测试集，不带标签
def predata(file_path='./dataset/train.csv',savefile=True,savename='',savepath='./data',ratio=0.2,predataflag=1):
    #读取
    # 载入数据,如果存在保存的数据，则直接载入保存数据
    filesavename=savepath + '/'+ savename+ '_Titanic_data' + "_" + str(ratio)+'.mat'
    if savefile and os.path.exists(filesavename) == True:
        #直接载入数据
        print('从已保存数据载入...')
        DataSetMat = sio.loadmat(savepath + '/'+  'Titanic_data' + "_" + str(ratio)+"_predataflag"+str(predataflag)+'.mat')
        if(ratio==0):
            return np.array([]),np.array([]),np.array(DataSetMat['Tr']),np.array(DataSetMat['Tr_l'])[0]
        return np.array(DataSetMat['Te']),np.array(DataSetMat['Te_l'])[0],np.array(DataSetMat['Tr']),np.array(DataSetMat['Tr_l'])[0]
    else:
        print("the dataset you selected dos'nt exist!!! working on data preprocess!")
    #否则，预处理数据
    data_train = pd.read_csv(file_path)
    data_len=len(data_train)
    test_number=int(ratio*(data_len))
    train_number=data_len-test_number
    print('总样本集为%d行，取测试集比例%f，取%d个测试集，%d个训练集'%(data_len,ratio,test_number,train_number))
    data_pre = data_train.copy()
    # 预处理数据，离散的映射到类别空间，连续的映射到[0,1]
    data_pre.drop(['Name', 'Ticket'], axis=1, inplace=True)
    # 离散值映射
    data_pre.loc[(data_pre.Sex == 'male'), 'Sex'] = 0
    data_pre.loc[(data_pre.Sex == 'female'), 'Sex'] = 1
    data_pre.Sex = changeDtype(data_pre.Sex)

    data_pre.loc[(data_pre.Cabin.notnull()), 'Cabin'] = 1
    data_pre.loc[(data_pre.Cabin.isnull()), 'Cabin'] = 0
    data_pre.Cabin = changeDtype(data_pre.Cabin)

    data_pre.loc[(data_pre.Embarked.isnull()), 'Embarked'] = 0
    data_pre.loc[(data_pre.Embarked == 'Q'), 'Embarked'] = 1
    data_pre.loc[(data_pre.Embarked == 'S'), 'Embarked'] = 2
    data_pre.loc[(data_pre.Embarked == 'C'), 'Embarked'] = 3
    data_pre.Embarked = changeDtype(data_pre.Embarked)
    data_pre.loc[(data_pre.Parch != 0), 'Parch'] = 1
    data_pre.loc[(data_pre.SibSp != 0), 'SibSp'] = 1
    # 对实数变量，缩放到0-1
    #fare是票价，如果缺失，则用其所在Pclass的类别的Fare的均值填充
    # 对缺失的Fare，用Pclass对应的mean补上
    Fare_mean = [data_pre[data_pre.Pclass == 1]['Fare'].mean(), data_pre[data_pre.Pclass == 2]['Fare'].mean(),
                 data_pre[data_pre.Pclass == 3]['Fare'].mean()]
    Fare_mean = np.array(Fare_mean)
    data_pre.loc[pd.isnull(data_pre.Fare), 'Fare'] = Fare_mean[data_pre[pd.isnull(data_pre.Fare)]['Pclass'].as_matrix() - 1]

    scaler = preprocessing.StandardScaler()
    fare_scale_param = scaler.fit(data_pre['Fare'])
    data_pre['Fare_scaled'] = scaler.fit_transform(data_pre['Fare'], fare_scale_param)
    #预测缺失的age值
    featurelist=['Pclass','Sex','SibSp','Fare_scaled','Parch','Cabin','Embarked']
    data_pre=set_missing_ages(data_pre,featurelist)
    #并归一化
    fare_scale_param = scaler.fit(data_pre['Age'])
    data_pre['Age_scaled'] = scaler.fit_transform(data_pre['Age'], fare_scale_param)

    if(predataflag==0):#该种预处理方式是直接将类别标号，将离散数据归一化
        #分别获取其特征和标签矩阵
        if ratio!=1:
            featurelist2 = ['Survived','Pclass', 'Sex', 'SibSp', 'Fare_scaled','Age_scaled', 'Parch', 'Cabin', 'Embarked']
            data=data_pre[featurelist2].as_matrix()
            # 分出测试集和训练集
            data_test = np.array(data[:test_number,1:])
            label_test=np.array(data[:test_number,0])
            data_train = np.array(data[test_number:,1:])
            label_train = np.array(data[test_number:, 0])
        elif ratio==1:
            #只取测试集，不带标签
            featurelist2 = ['Pclass', 'Sex', 'SibSp', 'Fare_scaled', 'Age_scaled', 'Parch', 'Cabin',
                            'Embarked']
            data = data_pre[featurelist2].as_matrix()
            # 分出测试集和训练集
            data_test = np.array(data[:test_number,:])
            label_test = []
            data_train = []
            label_train = []
    elif(predataflag==1):#该种预处理，对类别特征离散为二值特征
        #二值化
        dummies_Cabin = pd.get_dummies(data_pre['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(data_pre['Embarked'], prefix='Embarked')
        dummies_Parch = pd.get_dummies(data_pre['Parch'], prefix='Parch')
        dummies_SibSp = pd.get_dummies(data_pre['SibSp'], prefix='SibSp')
        dummies_Pclass = pd.get_dummies(data_pre['Pclass'], prefix='Pclass')
        dummies_Sex = pd.get_dummies(data_pre['Sex'], prefix='Sex')
        #链接
        df = pd.concat([data_pre, dummies_Cabin, dummies_Embarked, dummies_Parch, dummies_SibSp, dummies_Sex, dummies_Pclass],axis=1)
        #去除源类别
        df.drop(['Pclass', 'Sex','SibSp','Parch', 'Cabin', 'Embarked'], axis=1, inplace=True)
        #featurelist2 = ['Survivdfed', 'Pclass', 'Sex', 'SibSp', 'Fare_scaled', 'Age_scaled', 'Parch', 'Cabin', 'Embarked']
        # 分出测试集和训练集
        if ratio != 1:
            # 获取需要的类别
            data = df.filter(
                regex='Survived|Age_.*|SibSp|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()
            data_test = np.array(data[:test_number, 1:])
            label_test = np.array(data[:test_number, 0])
            data_train = np.array(data[test_number:, 1:])
            label_train = np.array(data[test_number:, 0])
        elif ratio == 1:
            # 获取需要的类别
            data = df.filter(
                regex='Age_.*|SibSp|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()
            data_test = np.array(data[:test_number, :])
            label_test = []
            data_train = []
            label_train = []
    passengerIdlist=data_pre['PassengerId'].as_matrix()
    if(savefile):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            # filesavename
        filedict={'Te':data_test,'Te_l':label_test,'Tr':data_train,'Tr_l':label_train,'PassengerId':passengerIdlist}
        sio.savemat(filesavename + '.mat',filedict)
        print('数据已保存到%s'%(filesavename))
    #
    return data_test,label_test,data_train,label_train,passengerIdlist

def savepredictcsv(passengerIdlist,predictedl,savename='test'):
    result = pd.DataFrame(
        {'PassengerId': passengerIdlist, 'Survived': predictedl.astype(np.int32)})
    filesavename = './dataset/pre_'+savename+'.csv'
    result.to_csv(filesavename, index=False)
    print(result)
    print("save csv file to %s" % filesavename)

#回归树预测
def RFC(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0):
    te, te_l, tr, tr_l = predata(file_path,savefile,savepath,ratio,predataflag=0)
    rfr = RandomForestClassifier(random_state=0, n_estimators=2000, n_jobs=-1)
    print(cross_validation.cross_val_score(rfr, tr, tr_l, cv=5))
    '''
    rfr.fit(tr, tr_l)
    predictedl = rfr.predict(te)
    res = te_l == predictedl
    count = res[res == True].shape[0]
    acc = count / len(res)
    print(te_l == predictedl, 'RandomForestClassifier is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
    '''
#逻辑回归预测
def LR(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0):
    te, te_l, tr, tr_l = predata(file_path, savefile, savepath, ratio)
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    print(cross_validation.cross_val_score(clf, tr, tr_l, cv=5))
    '''
    clf.fit(tr, tr_l)
    predictedl = clf.predict(te)
    preint = predictedl.astype(np.int32)
    res = te_l == preint
    count = res[res == True].shape[0]
    acc = count / len(res)
    print(te_l == predictedl, 'LogisticRegression is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))
    '''

def KNN(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0):
    te, te_l, tr, tr_l = predata(file_path, savefile, savepath, ratio)
    knn = KNeighborsClassifier(n_neighbors=5)
    print(cross_validation.cross_val_score(knn, tr, tr_l, cv=5))

def SVM(file_path='./dataset/train.csv',savefile=True,savepath='./data',ratio=0,predataflag=0):
    _, _, tr, tr_l,_ = predata(file_path, savefile,savename='train', savepath=savepath, ratio=0,predataflag=predataflag)
    #取训练集
    test_path='./dataset/test.csv'
    te,_,_,_,passengerIdlist=predata(test_path, savefile,savename='test',savepath=savepath,ratio=1,predataflag=predataflag)
    svc = svm.SVC(C=1.0, kernel = 'rbf', degree = 3)
    print(cross_validation.cross_val_score(svc, tr, tr_l, cv=5))

    svc.fit(tr, tr_l)
    predictedl = svc.predict(te)
    #保存结果
    savepredictcsv(passengerIdlist,predictedl,'test')
    # res = te_l == predictedl
    # count = res[res == True].shape[0]
    # acc = count / len(res)
    # print(te_l == predictedl, 'RandomForestClassifier is acc:%.4f,count:%d,total:%d' % (acc, count, len(res)))

if __name__ == '__main__':
    SVM()
    # test_path = './dataset/test.csv'
    # te, _, _, _ = predata(test_path, savefile=True, savename='test', savepath='./data/', ratio=1, predataflag=0)