from math import log
###计算香农熵(为float类型）
def calShang(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}##创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    #print('labelCount:',labelCounts)
    return shannonEnt

###划分数据集（以指定特征将数据进行划分）
def splitDataSet(dataSet,feature,value):##传入待划分的数据集、划分数据集的特征以及需要返回的特征的值
    newDataSet = []
    for featVec in dataSet:
        if featVec[feature] == value:
            reducedFeatVec = featVec[:feature]
            reducedFeatVec.extend(featVec[feature + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet

##选择最好的划分方式(选取每个特征划分数据集，从中选取信息增益最大的作为最优划分)在这里体现了信息增益的概念
def chooseBest(dataSet):
    featNum = len(dataSet[0]) - 1
    baseEntropy = calShang(dataSet)
    bestInforGain = 0.0
    bestFeat = -1##表示最好划分特征的下标

    for i in range(featNum):
        featList = [example[i] for example in dataSet] #列表,得到每一列
        #print(featList)
        uniqueFeat = set(featList)##得到每个特征中所含的不同元素
        #print(uniqueFeat)
        newEntropy = 0.0
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShang(subDataSet)
        inforGain = baseEntropy - newEntropy
        #print(inforGain)
        if (inforGain > bestInforGain):
            bestInforGain = inforGain
            bestFeature = i#第i个特征是最有利于划分的特征
    return bestFeature

def creatDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

##测试
myData,labels = creatDataSet()
best = chooseBest(myData)
print(best)
