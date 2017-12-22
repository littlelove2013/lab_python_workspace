import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# 载入数据
data_train = pd.read_csv("./dataset/train.csv")
# 获取年龄不缺失的集合
data_allage = data_train[data_train.Age.notnull()]
# 分别获取获救集合与未获救集合
data_Survived = data_allage[data_allage.Survived == 1]
data_notSurvived = data_allage[data_allage.Survived == 0]


# 分别显示获救和未获救的曲线（数据为连续实数：dataflag=‘real’），或者直方图（数据为离散类别：dataflag='class'）
def showDataFig(surviveddatalist, notsurviveddatalist, flag, title=None):
    if (flag == 'real'):  # 实数
        surviveddatalist.plot(kind='kde')
        notsurviveddatalist.plot(kind='kde')
        plt.legend((u'获救', u'未获救'), loc='best')
    elif (flag == 'class'):
        df = pd.DataFrame({u'未获救': notsurviveddatalist.value_counts(), u'获救': surviveddatalist.value_counts()})
        df.plot(kind='bar', stacked=True)
    else:
        return;
    if title != None:
        plt.title(title)
    #plt.show()


# 分析所有数值特征在获救和未获救数据集上的情况，主要看均值和方差
# print(data_Survived.describe(),data_notSurvived.describe())
sd = data_Survived.describe()
snd = data_notSurvived.describe()
sd.PassengerId.mean()
# 画出两类的年龄情况
showDataFig(data_Survived.Age, data_notSurvived.Age, 'real', '两类的年龄情况')
# Fare	票价
showDataFig(data_Survived.Fare, data_notSurvived.Fare, 'real', 'Fare票价')

# 各登录港口乘客的获救情况
showDataFig(data_Survived.Embarked, data_notSurvived.Embarked, 'class', '各登录港口乘客的获救情况')
# 乘客等级
showDataFig(data_Survived.Pclass, data_notSurvived.Pclass, 'class', '乘客等级')
# 乘客性别
showDataFig(data_Survived.Sex, data_notSurvived.Sex, 'class', '乘客性别')

# 考虑是独生子的人，可能会优先考虑（没有兄弟姐妹）
# 乘客堂兄弟/妹个数
ds = data_Survived.copy()
dns = data_notSurvived.copy()
ds.loc[(data_Survived.SibSp != 0), 'SibSp'] = 1
dns.loc[(data_notSurvived.SibSp != 0), 'SibSp'] = 1
showDataFig(ds.SibSp, dns.SibSp, 'class', '在船上的乘客堂兄弟/妹个数')
# 考虑乘客有父母或者子女的，可能要优先考虑
# 乘客父母与小孩个数
ds.loc[(data_Survived.Parch != 0), 'Parch'] = 1
dns.loc[(data_notSurvived.Parch != 0), 'Parch'] = 1
showDataFig(ds.Parch, dns.Parch, 'class', '船上乘客父母与小孩个数')
# 看Cabin有无对获救的影响
ds.loc[(pd.notnull(data_train.Cabin)), 'Cabin'] = 1
dns.loc[(pd.notnull(data_train.Cabin)), 'Cabin'] = 1
ds.loc[(pd.isnull(data_train.Cabin)), 'Cabin'] = 0
dns.loc[(pd.isnull(data_train.Cabin)), 'Cabin'] = 0
showDataFig(ds.Parch, dns.Parch, 'class', '船上乘客父母与小孩个数')

data_pre=data_train.copy()
#预处理数据，离散的映射到类别空间，连续的映射到[0,1]
data_pre.drop(['Name', 'Ticket'], axis=1, inplace=True)
def changeDtype(src,dstdtype='int64'):
    a=src.values
    a=np.array(a,dtype=dstdtype)
    return a
#离散值映射

data_pre.loc[ (data_pre.Sex=='male'), 'Sex' ] = 0
data_pre.loc[ (data_pre.Sex=='female'), 'Sex'] = 1
data_pre.Sex=changeDtype(data_pre.Sex)
data_pre.loc[ (data_pre.Cabin.notnull()), 'Cabin' ] = 1
data_pre.loc[ (data_pre.Cabin.isnull()), 'Cabin' ] = 0
data_pre.Cabin=changeDtype(data_pre.Cabin)
#有NaN值
data_pre.loc[ (data_pre.Embarked.isnull()), 'Embarked' ] = 0
data_pre.loc[ (data_pre.Embarked=='Q'), 'Embarked' ] = 0
data_pre.loc[ (data_pre.Embarked=='S'), 'Embarked' ] = 1
data_pre.loc[ (data_pre.Embarked=='C'), 'Embarked' ] = 2
data_pre.Embarked=changeDtype(data_pre.Embarked)

data_pre.loc[(data_pre.Parch != 0),'Parch']=1
data_pre.loc[(data_pre.SibSp != 0),'SibSp']=1
#对实数变量，缩放到0-1
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(data_pre['Fare'])
data_pre['Fare_scaled'] = scaler.fit_transform(data_pre['Fare'], fare_scale_param)

#有Age
data_hasAge=data_pre[pd.notnull(data_pre.Age)]
age_scale_param = scaler.fit(data_hasAge['Age'])
data_hasAge['Age_scaled'] = scaler.fit_transform(data_hasAge['Age'], age_scale_param)
#没有Age
data_noAge=data_pre[pd.isnull(data_train.Age)]
data_noAge.drop(['Age'], axis=1, inplace=True)

#随机森林
from sklearn.ensemble import RandomForestRegressor
def getRFmodel(df,feature_list):
    #取出特征
    y=df[['Survived']].as_matrix()
    y=y.ravel()
    feature=df[feature_list].as_matrix()
    #print(feature,np.isnan(feature).any())
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(feature, y)
    return rfr,feature,y

feature_list=['Pclass','Sex','SibSp','Age_scaled','Fare_scaled','Cabin','Embarked']
#feature_list=['Sex','Cabin','Embarked']
rfr,feature,y = getRFmodel(data_hasAge,feature_list)
