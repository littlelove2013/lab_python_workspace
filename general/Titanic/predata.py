import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
import scipy.io as sio
import os
import re

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
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
#称呼栏只保留Miss等称谓
def setName(data):
#     print(data)
    name=[]
    Mr='^.* Mr\. .*$'
    Miss='^.* Miss\. .*$'
    Mrs='^.* Mrs\. .*$'
    Master='^.* Master\. .*$'
    lens = len(data)
    print(lens)
    Name_num=data.copy()
    for i in range(lens):
        if re.match(Mr,data[i]):
            name.append('Mr')
            data.loc[i]='Mr'
            Name_num.loc[i]=0
        elif re.match(Miss,data[i]):
            name.append('Miss')
            data.loc[i]='Miss'
            Name_num.loc[i]=1
        elif re.match(Mrs,data[i]):
            name.append('Mrs')
            data.loc[i]='Mrs'
            Name_num.loc[i]=2
        elif re.match(Master,data[i]):
            name.append('Master')
            data.loc[i]='Master'
            Name_num.loc[i]=3
        else:
            name.append('Norm')
            data.loc[i]='Norm'
            Name_num.loc[i]=4
#     print(len(name),name)
    #转换类型
    Name_num=changeDtype(Name_num)
#     print(data.values)
    return data,Name_num


# 将票分为有字幕和无字母开头的，以及数字两列特征
def setTicket(data):
    #     print(data)

    number = '[\d]+'  # 至少一个数字
    strmatch = '[^\d]+'  # 至少一个非数字
    '''
    A
    P
    S
    C
    F
    W
    L
    '''
    lens = len(data)
    print(lens)
    number_col = []
    hasstr = []
    for i in range(lens):
        num = re.search(number, data[i])
        if num:
            number_col.append(int(num.group()))
        else:
            print("error:exsit null ticket number is '%s' in rows:%d!" % (data[i], i))
            number_col.append(0)
        strs = re.search(strmatch, data[i])
        if strs:
            s = strs.group()
            if re.search('^L', s) or re.search('^W', s) or re.search('^A', s):
                hasstr.append(3)
            elif re.search('^P', s):
                hasstr.append(0)
            elif re.search('^S', s) or re.search('^C', s) or re.search('^F', s):
                hasstr.append(2)
            else:
                print("error:no exsit ticket str is '%s' in rows:%d!" % (s, i))
        else:
            hasstr.append(1)
    number_col = np.array(number_col, np.float32)
    number_col = (number_col - number_col.mean()) / number_col.std()
    hasstr = np.array(hasstr, np.int32)
    return number_col, hasstr
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
    filesavename=savepath + '/'+ str(savename)+ '_Titanic_data' + "_ratio(" + str(ratio)+')_predataflag('+str(predataflag)+').mat'
    if savefile and os.path.exists(filesavename) == True:
        #直接载入数据
        print('从已保存数据载入...')
        DataSetMat = sio.loadmat(filesavename)
        if(ratio==0):
            return np.array([]),np.array([]),np.array(DataSetMat['Tr']),np.array(DataSetMat['Tr_l'])[0],np.array(DataSetMat['PassengerId'])[0]
        if(ratio==1):
            return  np.array(DataSetMat['Te']),np.array([]), np.array([]), np.array([]),np.array(DataSetMat['PassengerId'])[0]
        # elif(ratio==1):
        #     return np.array([]), np.array([]), np.array(DataSetMat['Te']), np.array(DataSetMat['Te_l'])[0]
        return np.array(DataSetMat['Te']),np.array(DataSetMat['Te_l'])[0],np.array(DataSetMat['Tr']),np.array(DataSetMat['Tr_l'])[0],np.array(DataSetMat['PassengerId'])[0]
    else:
        print("the dataset (%s) you selected dos'nt exist!!! working on data preprocess!"%(filesavename))
    #否则，预处理数据
    data_train = pd.read_csv(file_path)
    data_len=len(data_train)
    test_number=int(ratio*(data_len))
    train_number=data_len-test_number
    print('总样本集为%d行，取测试集比例%f，取%d个测试集，%d个训练集'%(data_len,ratio,test_number,train_number))
    data_pre = data_train.copy()
    # 预处理数据，离散的映射到类别空间，连续的映射到[0,1]
    # data_pre.drop(['Name', 'Ticket'], axis=1, inplace=True)
    # data_pre.drop(['Ticket'], axis=1, inplace=True)
    Ticket_num, Ticket_str = setTicket(data_pre['Ticket'])
    # data_pre['Ticket_num'],data_pre['Ticket_str']
    data_pre['Ticket_num'] = Ticket_num
    data_pre['Ticket_str'] = Ticket_str
    # 匹配Name并离散化为变量，Name_num
    data_pre['NameExtens'], data_pre['Name_num'] = setName(data_pre['Name'].copy())
    # 离散值映射
    data_pre.loc[(data_pre.Sex == 'male'), 'Sex'] = 0
    data_pre.loc[(data_pre.Sex == 'female'), 'Sex'] = 1
    data_pre.Sex = changeDtype(data_pre.Sex)

    data_pre.loc[(data_pre.Cabin.notnull()), 'Cabin'] = 1
    data_pre.loc[(data_pre.Cabin.isnull()), 'Cabin'] = 0
    data_pre.Cabin = changeDtype(data_pre.Cabin)
    #train有Embarked缺失，而test没有，所以不能把缺失当作一类，而要补全
    #因为缺失的均为获救的，而C的获救概率最高，所以丢失的用C的值补全
    data_pre.loc[(data_pre.Embarked.isnull()), 'Embarked'] = 2
    data_pre.loc[(data_pre.Embarked == 'Q'), 'Embarked'] = 0
    data_pre.loc[(data_pre.Embarked == 'S'), 'Embarked'] = 1
    data_pre.loc[(data_pre.Embarked == 'C'), 'Embarked'] = 2
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

    # scaler = preprocessing.StandardScaler()
    # fare_scale_param = scaler.fit(data_pre['Fare'])
    # data_pre['Fare_scaled'] = scaler.fit_transform(data_pre['Fare'], fare_scale_param)

    fares = data_pre['Fare'].as_matrix()
    fares = (fares-fares.mean()) / fares.std()
    data_pre['Fare_scaled']=fares
    #预测缺失的age值,使用称呼的平均值来代替
    has_Age = data_pre[pd.notnull(data_pre.Age)]
    # 求称呼对应的年龄均值
    Age_mean = [has_Age[has_Age.Name_num == 0]['Age'].mean(), has_Age[has_Age.Name_num == 1]['Age'].mean(),
                has_Age[has_Age.Name_num == 2]['Age'].mean(), has_Age[has_Age.Name_num == 3]['Age'].mean(),
                has_Age[has_Age.Name_num == 4]['Age'].mean()]
    Age_mean = np.array(Age_mean)
    data_pre.loc[pd.isnull(data_pre.Age), 'Age'] = Age_mean[data_pre[pd.isnull(data_pre.Age)]['Name_num'].as_matrix()]
    data_pre.Age = changeDtype(data_pre.Age,dstdtype=np.int32)
    # featurelist=['Pclass','Sex','SibSp','Fare_scaled','Parch','Cabin','Embarked']
    # data_pre=set_missing_ages(data_pre,featurelist)
    #并归一化
    # fare_scale_param = scaler.fit(data_pre['Age'])
    # data_pre['Age_scaled'] = scaler.fit_transform(data_pre['Age'], fare_scale_param)
    #0均值单位方差
    ages = data_pre['Age'].as_matrix()
    ages = (ages -ages.mean())/ ages.std()
    data_pre['Age_scaled'] = ages
    #将age分为三类：0-15，35-55，15-35&55-80
    # data_pre.loc[(data_pre.Age<=15), 'Age'] = 0
    # data_pre.loc[(data_pre.Age > 15), 'Age'] = 1
    # data_pre.loc[(data_pre.Age>35)&(data_pre.Age<55), 'Age'] = 1
    # data_pre.loc[((data_pre.Age>15)&(data_pre.Age<=35))|((data_pre.Age>=55)), 'Age'] = 2

    if(predataflag==0):#该种预处理方式是直接将类别标号，将离散数据归一化
        #分别获取其特征和标签矩阵
        if ratio!=1:
            # featurelist2 = ['Survived','Pclass', 'Sex', 'SibSp', 'Fare_scaled','Age_scaled', 'Parch', 'Cabin', 'Embarked','Name_num','Ticket_str','Ticket_num']#添加name的标签
            featurelist2 = ['Survived','Sex','Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked','Name_num','Ticket_str','Fare_scaled','Age_scaled','Ticket_num']  # 添加name的标签

            data=data_pre[featurelist2].as_matrix()
            # 分出测试集和训练集
            data_test = np.array(data[:test_number,1:])
            label_test=np.array(data[:test_number,0])
            data_train = np.array(data[test_number:,1:])
            label_train = np.array(data[test_number:, 0])
        elif ratio==1:
            #只取测试集，不带标签
            #featurelist2 =              ['Pclass', 'Sex', 'SibSp', 'Fare_scaled', 'Age_scaled', 'Parch', 'Cabin','Embarked','Name_num','Ticket_str','Ticket_num']
            featurelist2 = [ 'Sex','Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked','Name_num','Ticket_str','Fare_scaled','Age_scaled','Ticket_num']  # 添加name的标签
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

        #添加name的标签
        dummies_Name_num = pd.get_dummies(data_pre['Name_num'], prefix='Name_num')
        dummies_Ticket_str = pd.get_dummies(data_pre['Ticket_str'], prefix='Ticket_str')
        #链接
        df = pd.concat([data_pre,dummies_Cabin, dummies_Embarked, dummies_Parch, dummies_SibSp, dummies_Sex, dummies_Pclass,dummies_Name_num,dummies_Ticket_str],axis=1)
        #去除源类别
        df.drop(['Pclass','Sex','SibSp','Parch', 'Cabin', 'Embarked','NameExtens','Name_num'], axis=1, inplace=True)
        #featurelist2 = ['Survivdfed', 'Pclass', 'Sex', 'SibSp', 'Fare_scaled', 'Age_scaled', 'Parch', 'Cabin', 'Embarked']
        # 分出测试集和训练集
        if ratio != 1:
            # 获取需要的类别
            data = df.filter(
                regex='Survived|Age_.*|SibSp|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Name_num_.*|Ticket_str_.*|Ticket_num').as_matrix()
            data_test = np.array(data[:test_number, 1:])
            label_test = np.array(data[:test_number, 0])
            data_train = np.array(data[test_number:, 1:])
            label_train = np.array(data[test_number:, 0])
        elif ratio == 1:
            # 获取需要的类别
            data = df.filter(
                regex='Age_.*|SibSp|Parch_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Name_num_.*|Ticket_str_.*|Ticket_num').as_matrix()
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
        sio.savemat(filesavename,filedict)
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

if __name__ == '__main__':
    passengerIdlist,p1=RFC()
    # _,p2=LR()
    # _,p3=KNN()
    # _,p4=SVM()
    passengerIdlist,p5=AdaBoost()
    # num=5
    # p=np.array([p1,p2,p3,p4,p5]).sum(0)
    # p[p>num/2]=1
    # p[p<=num/2]=0
    # # 保存结果
    savepredictcsv(passengerIdlist,p1,'test')


    # test_path = './dataset/test.csv'
    # te, _, _, _ = predata(test_path, savefile=True, savename='test', savepath='./data/', ratio=1, predataflag=0)