import readiris as rd
import plottree as pt
from math import log


# 计算香农熵
def calcShannonEnt(label):
    numEntries = len(label) #nrows
    #计算香农熵
    shannonEnt=0.0
    for key in set(label):
        prob = float(label.count(key)) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#定义按照某个特征进行划分的函数splitDataSet
#输入三个变量（待划分的数据集，特征，分类值）
def splitDataSet(data,label,axis,value):
    retDataSet=[]
    retLabel=[]
    for i in range(len(data)):
	    featVec=data[i]
	    if featVec[axis]==value:
		    retLabel.append(label[i])
		    reduceFeatVec=featVec[:axis]
		    reduceFeatVec.extend(featVec[axis+1:])
		    retDataSet.append(reduceFeatVec)
    return retDataSet,retLabel #返回不含划分特征的子集

#定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(data,label):
    numFeature=len(data[0])
    baseEntropy=calcShannonEnt(label)#香农熵
    bestInforGain=0
    bestFeature=-1
    for i in range(numFeature):
        featList=[number[i] for number in data] #得到某个特征下所有值（某列）
        uniqualVals=set(featList) #set无重复的属性特征值
        newEntropy=0
        for value in uniqualVals:
            subDataSet,sublabel=splitDataSet(data,label,i,value)
            prob=len(sublabel)/float(len(label)) #即p(t)
            newEntropy+=prob*calcShannonEnt(sublabel)#对各子集香农熵求和
        infoGain=baseEntropy-newEntropy #计算信息增益
        #最大信息增益
        # print(i, ":", infoGain, bestInforGain)
        if (infoGain>bestInforGain):
	        bestInforGain=infoGain
	        bestFeature=i
    return bestFeature #返回特征值

#投票表决代码
def majorityCnt(label):
	best_key=-1
	best_num=0
	for key in set(label):
		num = label.count(key)
		if num>best_num:
			best_num=num
			best_key=key
	return best_key
#创建树
def createTree(dataSet,labels,names):
    classList=labels
    #类别相同，停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #特征长度长度为1，则返回出现次数最多的类别
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #按照信息增益最高选取分类特征属性
    labelnames=names.copy()
    bestFeat=chooseBestFeatureToSplit(dataSet,labels)#返回分类的特征序号
    bestFeatLable=labelnames[bestFeat] #该特征的label
    myTree={bestFeatLable:{}} #构建树的字典
    del(labelnames[bestFeat]) #从labels的list中删除该label
    #找到所有该列特征
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        sublabelnames=labelnames[:] #子集合
        #构建数据的子集合，并进行递归
        retDataSet, retLabel=splitDataSet(dataSet,labels,bestFeat,value)
        myTree[bestFeatLable][value]=createTree( retDataSet, retLabel,sublabelnames)
    return myTree

def createAndShowTree(filename,labelnames):
	# filename = 'data'
	data = rd.PRdata(filename)
	tree1 = createTree(data.Data, data.Label, labelnames)
	print(tree1)
	pt.createPlot(tree1)

if __name__ == '__main__':
	# data, label, dictory, labeldic = rd.getLabelData()
	# print(label,calcShannonEnt(label))
	# chooseBestFeatureToSplit(data, label)
	labelnames = ["Outlook", "Temperature", "Humidity", "Wind"]
	#不添加D14数据的树
	createAndShowTree("data",labelnames)
	#添加D14的树
	createAndShowTree("data2", labelnames)