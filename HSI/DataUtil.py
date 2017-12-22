#!/usr/bin/env python
# coding=utf-8
# dawang jinwan hele pidan doufutang. @ zhoujing @
# author @ jiabing leng @ nankai university @ tadakey@163.com

import time
import scipy.io as sio
import numpy as np
from random import shuffle
from math import ceil
import sys
from sys import argv
import os
# import lmdb
# import caffe
#sys.path.insert(0,'/home/jiabing/caffe/python')

global neighbors 
prompt = '>'
context = '/home/para/caffe/'
datasetName = ''
#neighbors = 4
global train_ratio
neighbors = 8
path_prefix = '../../HSI_DATA/'
sys.path.insert(0,context + '/python')


def loadData(data_set_name, strategy):
    strategy = int(strategy)
    neighbors = strategy
    if strategy == 0:
        print( 'please enter the neighbor pixels strategy, you can choose from 1,4 and 8.')
        neighbors = int(input(prompt))
    #neighbors = neighbors
    print( neighbors)


    data_set_path = path_prefix + data_set_name
    print(data_set_path)
    
    #list all files under this folder
    #TODO should check if the files are correct under this folder to go preprocessing
    print( "指定的数据集文件夹下包含如下文件: ")
    for _file_name in os.listdir(data_set_path):
        print( _file_name)
    
    #load data and index file
    # print( 'validation dataset done, correct.')
    # in case of the disorder of the content of data, should looking for the correct menu of the data index. for example,
    # the labels may be orginazed like['__version__', '__header__', 'ClsID', '__globals__'] rather than
    #['ClsID', all the other contents], so the code should not just use following index to fetch useable data.
    print('正在载入原始频谱数据...')
    DataSetMat = sio.loadmat(data_set_path + '/' + data_set_name + 'Data.mat')
<<<<<<< HEAD
    #记录所有关键字
    key_data_name = DataSetMat.keys()
    data_key = ''
    #找到第一个不是默认属性得关键字
=======
    key_data_name = DataSetMat.keys()
    data_key = ''
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    for temp_key in key_data_name:
        if temp_key != '__version__' and temp_key != '__header__' and temp_key != '__globals__':
            data_key = temp_key
            break
<<<<<<< HEAD
    #获取该属性下得值
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    DataSet = DataSetMat[data_key]
    print("正在载入原始标签数据……")
    LabelsMat = sio.loadmat(data_set_path + '/' + data_set_name + 'Gt.mat')
    key_label_name = LabelsMat.keys()
    label_key = ''
    for temp_key in key_label_name:
        if temp_key != '__version__' and temp_key != '__header__' and temp_key != '__globals__':
            label_key = temp_key
            break
<<<<<<< HEAD
    # 获取该属性下得值
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    Labels = LabelsMat[label_key]
    #原始数据载入完毕
    print("所有原始数据载入完毕！")

    #rows 高光谱图像行数
    #lines 高光谱图像列数
    rows = len(Labels)
    lines = len(Labels[0])
    print( '载入的数据集包含%s行%s列的数据'%(str(rows),str(lines)))
    class_num = np.max(Labels)
    print( '载入的%s数据集中共含有%s种分类'%(str(data_set_name),str(class_num)))
    print( '数据集中每个像元的波段长度为 ' + str(len(DataSet[0][0])))
    
    #define many lists which number equals to maxClass,put it in a list
    #return shuffledDataList, neighbors, shuffledPositionList, rows, lines
    #DataList 用于存放数据和类别的二维List
    #PositionList 用于存储位置信息的向量
    DataList = []
    PositionList = []
    shuffledDataList = []
    shuffledPositionList = []
    for _c in range(class_num):
        DataList.append([])
        PositionList.append([])
        shuffledDataList.append([])
        shuffledPositionList.append([])
    #为每个分类建立一个空列表,以下是一个极其错误的做法，用*号初始化一个二维列表，这样做的话使用下标索引进行的操作将也被重复到左右元素上
    # DataList = [[]]*class_num
    # PositionList = [[]]*class_num
    for indexRow in range(rows):
        for indexLine in range(lines):
            label = Labels[indexRow,indexLine]
            #position = {'row':indexRow, 'line':indexLine}
            #position = str(indexRow) + "|" + str(indexLine)
            position = [indexRow,indexLine]

            #store non-zero data
            if label != 0:
                #for test purpose printing...
                #print '[' + str(indexRow) + ',' + str(indexLine) + ']'
                data = DataSet[indexRow,indexLine]
                if  neighbors> 1:
                    center_data = data
                    ####################################################################################################################
                    # fetching data around the target pixel according to following illustruction:
                    # 
                    #           data1      data2      data3
                    #           data4     center      data5
                    #           data6      data7      data8
                    ####################################################################################################################
                    data1 = []
                    data2 = []
                    data3 = []
                    data4 = []
                    data5 = []
                    data6 = []
                    data7 = []
                    data8 = []

                    # data1
                    if indexRow - 1 >= 0 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data1 = DataSet[indexRow - 1, indexLine - 1]
                    elif indexRow - 1 >= 0 and indexLine + 1 <= lines - 1 and Labels[indexRow - 1, indexLine + 1] > 0:
                        data1 = DataSet[indexRow - 1, indexLine + 1]
                    else:
                        data1 = center_data
                    
                    # data2
                    if indexRow - 1 >= 0 and Labels[indexRow - 1, indexLine] > 0:
                        data2 = DataSet[indexRow - 1, indexLine]
                    elif indexRow + 1 <= rows - 1 and Labels[indexRow + 1, indexLine] > 0:
                        data2 = DataSet[indexRow + 1, indexLine]
                    else:
                        data2 = center_data
                        
                    # data3
                    if indexRow - 1 >= 0 and indexLine + 1 <= lines - 1 and Labels[indexRow - 1, indexLine + 1] > 0:
                        data3 = DataSet[indexRow - 1, indexLine + 1]
                    elif indexRow - 1 >= 0 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data3 = DataSet[indexRow - 1, indexLine - 1]
                    else:
                        data3 = center_data
                        
                    # data4
                    if indexLine - 1 >= 0 and Labels[indexRow, indexLine - 1] > 0:
                        data4 = DataSet[indexRow, indexLine - 1]
                    elif indexLine + 1<= lines - 1 and Labels[indexRow, indexLine + 1] > 0:
                        data4 = DataSet[indexRow, indexLine + 1]
                    else:
                        data4 = center_data
                    
                    # data5
                    if indexLine + 1 <= lines - 1 and Labels[indexRow, indexLine + 1] > 0:
                        data5 = DataSet[indexRow, indexLine + 1]
                    elif indexLine - 1 >= 0 and Labels[indexRow, indexLine - 1] > 0:
                        data5 = DataSet[indexRow, indexLine - 1]
                    else:
                        data5 = center_data
                    
                    # data6
                    if indexRow + 1 <= rows - 1 and indexLine - 1 >= 0 and Labels[indexRow + 1, indexLine - 1] > 0:
                        data6 = DataSet[indexRow + 1, indexLine - 1]
                    elif indexRow + 1 <= rows - 1 and indexLine + 1 <= lines - 1 and Labels[indexRow + 1, indexLine + 1] > 0:
                        data6 = DataSet[indexRow + 1, indexLine + 1]
                    else:
                        data6 = center_data
                        
                        
                    # data7
                    if indexRow + 1 <= rows - 1 and Labels[indexRow + 1, indexLine] > 0:
                        data7 = DataSet[indexRow + 1, indexLine]
                    elif indexRow - 1 >= 0 and Labels[indexRow - 1, indexLine] > 0:
                        data7 = DataSet[indexRow - 1, indexLine]
                    else:
                        data7 = center_data
                        
                    # data8
                    if indexRow + 1 <= rows - 1 and indexLine + 1 <= lines - 1 and Labels[indexRow + 1, indexLine + 1] > 0:
                        data8 = DataSet[indexRow + 1, indexLine + 1]
                    elif indexRow + 1 <= rows - 1 and indexLine - 1 >= 0 and Labels[indexRow - 1, indexLine - 1] > 0:
                        data8 = DataSet[indexRow + 1, indexLine - 1]
                    else:
                        data8 = center_data
                    
                    if neighbors == 4:
                        data_1 = np.append(data2, data4)
                        data_2 = np.append(data5, data7)
                        data_3 = np.append(data_1, data_2)
                        data  = np.append(data, data_3)
                        #data = data + data2 + data4 + data5 + data7
                    elif neighbors == 8:
                        #print data
                        #data = np.append(data, data1, data2, data3, data4, data5, data6, data7, data8)
#                        print( "neighbor startegy is 8")
                        data_1 = np.append(data1, data2)
                        data_2 = np.append(data3, data4)
                        data_3 = np.append(data5, data6)
                        data_4 = np.append(data7, data8)
                        data_5 = np.append(data_1, data_2)
                        data_6 = np.append(data_3, data_4)
                        data_7 = np.append(data_5, data_6)
                        data = np.append(data, data_7)

                        #print( data)
                        #print( 'data1' + str(data1) + 'data2 ' + str(data2) + 'data3' + str(data3))
                        #print( 'data1 + data2:')
                        #print( np.append(data1, data2))
                #    elif neighbors == 1:
                #        data = 

                DataList[label - 1].append(data)
                # the position string includes following informations:
                # row | line | class number.
<<<<<<< HEAD
                position.append(label - 1)
                PositionList[label - 1].append(position)
=======
                PositionList[label - 1].append(position.append(label - 1))
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e

    print("原始数据分类处理完毕")

    #进行shuffle  TODO后期将shuffle抽取出来，删掉之前的几个重复的shuffle，统一成一些函数。
    shuffledDataList = DataList
    shuffledPositionList = PositionList
    print( '对原始数据进行混洗操作')
    
    for class_index in range(len(DataList)):
        #将每一类内的全部数据进行shuffle
        cur_datalist, cur_positionlist = shuffling_tow_list(DataList[class_index], PositionList[class_index])
        print("混洗后第%d类元素数量为%d"%(class_index,len(cur_datalist)))
        print("混洗后第%d类元素位置的数量为%d" % (class_index, len(cur_positionlist)))
        shuffledDataList[class_index] = cur_datalist
        shuffledPositionList[class_index] = cur_positionlist
    
    return shuffledDataList, neighbors, shuffledPositionList, rows, lines

def shuffling_tow_list(dataList, positionList):
    print( '正在调用shuffling_tow_list进行类内混洗...')
    matched_length = len(dataList)
    shuffleMark = list(range(matched_length))
    shuffledA = dataList
    shuffledB = positionList
    #print shuffleMark
    reloadMark = 0
    shuffle(shuffleMark)
    for tempCount in shuffleMark:
        #print len(listB[tempCount])
        shuffledA[reloadMark] = (dataList[tempCount])
        shuffledB[reloadMark] = (positionList[tempCount])
        reloadMark = reloadMark + 1

    print( 'shuffling_tow_list操作完成')
    return shuffledA, shuffledB

def shuffling(dataList,ids, positionList):
    print( 'shuffling data...')
    if (len(dataList) == len(positionList) and len(dataList) == len(ids)) != True:
        print( 'The length of data list and position list does not match.')
        return 0
    shuffleMark = list(range(len(dataList)))
    
    shuffledData = []
    shuffledIds = []
    shuffledPosition = []
    
    shuffle(shuffleMark)
    for tempCount in shuffleMark:
        shuffledData.append(dataList[tempCount])
        shuffledPosition.append(positionList[tempCount])
        shuffledIds.append(ids[tempCount])
#
#    for sub_list in dataList:
#        shuffle(sub_list)
    print( 'shuffled.')
    return shuffledData, shuffledIds, shuffledPosition
# def writeToLMDB(list, name, procedure):
#
#     # prepare the data list
#     #print list[0]
#     new_big_list = []
#     #add_count = 0
#     classCount = 1
#     for sub_list in list:
#         #print 'samples number :' + str(len(sub_list))
#         for sub_list_data in sub_list:
#             print( 'number of samples in this class ' + str(len(sub_list_data)))
#             for to_be_assemblied_data in sub_list_data:
#                 data_dict = {'label': classCount, 'data': to_be_assemblied_data}
#                 new_big_list.append(data_dict)
#             classCount = classCount + 1
#     # now the data format have been transformed into this:
#     # new_big_list = [data_dicts....]
#     # in which data_dict is {'label': a label, 'data': data value}
#     # print new_big_list[0:20]
#     #print 'shuffling data again among different classes....'
#     #shuffle(new_big_list)
#     #print new_big_list[0]['label']
#     #print new_big_list[0]['data']
#     print( 'the number of spectral in this dataset is :' + str(len(new_big_list[0]['data'])))
#
#     map_size = sys.getsizeof(new_big_list) * 100000
#     # prepare the lmdb format file
#     print( 'creating training lmdb ' + procedure + 'format dataset...')
#     env = lmdb.open('HSI' + name + procedure + 'lmdb', map_size = map_size)
#     #count = 0
#     spectralBands = len(new_big_list[0]['data'])
#     print( 'this data set '+ name +' had spectral bands of ' + str(spectralBands))
#     temp_i = 0
#     countingMark = 0
#     sampleCounts = range(len(new_big_list))
#     shuffle(sampleCounts)
#     with env.begin(write = True) as txn:
#         for temp in sampleCounts:
#             sample = new_big_list[temp]
#             datum = caffe.proto.caffe_pb2.Datum()
#             datum.channels = 1
#             datum.height = 1
#             datum.width = spectralBands
#             # print sample
#             datum.data = sample['data'].tostring()
#             datum.label = int(sample['label'])
#             str_id = '{:08}'.format(temp_i)
#             txn.put(str_id.encode('ascii'), datum.SerializeToString())
# 	    temp_i = temp_i + 1
#             countingMark = countingMark + 1
# 	    #print '.'
#     print( 'Done.')
#     print( str(countingMark) + ' samples have successfully writed into lmdb format data file.')
def prepareMatList(list, positions):
    Data = []
    CId = []
    Positions = []
#    DataTe = []
#    CIdTe = []
    classCount = 1
    #positionMark_A = 0
    #PositionMark_B = 0
    #PositionMark_C = 0
    #print positions.shape
    #TODO: put these following two fors into one for.
<<<<<<< HEAD
    #sub_list为每一类别数据集合
    for sub_list in list:
        #sub_list_data为具体一个类别中的所有数据列表
        for sub_list_data in sub_list:
            print( 'number of samples in number ' + str(classCount) + ' class ' + str(len(sub_list_data)))
            #to_be_assemblied_data：每一行数据
=======
    for sub_list in list:

        for sub_list_data in sub_list:

            print( 'number of samples in number ' + str(classCount) + ' class ' + str(len(sub_list_data)))
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
            for to_be_assemblied_data in sub_list_data:

                Data.append(to_be_assemblied_data) 
                CId.append(classCount)
                #Positions.append(positions[positionMark])
                #positionMark = positionMark + 1
                #PositionMark_C = PositionMark_C
                #Positions.append(positions[positionMark_A][PositionMark_B][PositionMark_C])
            classCount = classCount + 1
            #PositionMark_B = PositionMark_B + 1
        #positionMark_A = positionMark_A + 1
    for sub_positions in positions:
        #print str(len(sub_positions)) + '  '
        for sub_sub_positions in sub_positions:
            #print len(sub_sub_positions)
            #print str(len(sub_sub_positions))
            for actual_Position in sub_sub_positions:
            #    print str(len(actual_Positions))
                Positions.append(actual_Position)
    
    newData, newCId, newPositions = shuffling(Data, CId, Positions)
    
    return newData, newCId, newPositions
# write to .mat data format
def writeToMAT(trainList, testList,trainPositions, testPositions, datasetName, train_ratio, neighbors):
    DataTr, CIdTr, PositionsTr = prepareMatList(trainList, trainPositions)
    DataTe, CIdTe, PositionsTe = prepareMatList(testList, testPositions)

    ltime = time.localtime()
    time_stamp = str(ltime[0]) + "_" + str(ltime[1]) + "_" + str(ltime[2]) + "_" + str(ltime[3]) + "_" + str(ltime[4])

    folderPath = "../experiments/" + datasetName + '_' + str(neighbors) + '_' + str(train_ratio) + "_" + time_stamp + "/"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    realPath = folderPath + datasetName + "_" + str(neighbors) + "_" + str(train_ratio)

    sio.savemat(realPath + '.mat',{'DataTr':DataTr, 'CIdTr':CIdTr, 'PositionsTr':PositionsTr,  'DataTe':DataTe, 'CIdTe':CIdTe, 'PositionsTe':PositionsTe})
    return realPath, neighbors

<<<<<<< HEAD
#
#@list:data
#@positionList:
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
def assembleData(list,positionList, datasetName, neighbors, learning_ratio, dataset_format):
    ratio = 0
    if(learning_ratio == 0):
        print( "please enter the ratio of training samples, eg. 80.")
        ratio = int(input(prompt))
        #train_ratio = ratio
    else:
        ratio = learning_ratio
    # prepare the lmdb format dataset
    # allocate the storage space for the dataset
    # TODO: check how to allocate space according to the specific dataset instead of use the following map_size directly.
    #map_size = list.nbytes * 0
    #create the lmdb data
    #envTrain = lmdb.open(datasetName + 'HSITrainlmdb', map_size = map_size)
    #envTest = lmdb.open(datasetName + 'HSITestlmdb', map_size = map_size)

    
    # split the dataset according to the ratio to caffe recognizable datasets
    positionMark = 0
    trainList = []
    testList = []
    trainPositions = []
    testPositions = []
<<<<<<< HEAD
    #数据类别长度，每个list[label]保存当前类别得数据
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    for mark in range(len(list)):
        trainList.append([])
        testList.append([])
        trainPositions.append([])
        testPositions.append([])
    print( 'confirm the number of classes in this dataset is ' + str(len(list)))
    trainingCount = 0
    testingCount = 0
    #for sub_list in list:
    positionMark = 0
    print( '#########################ratioing############################')
<<<<<<< HEAD
    #datalist是每个类别内的数据集合
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    for dataList in list:
         positionNow = positionList[positionMark]
        #trainingNumer = ceil((len(dataList) * float(ratio) / 100.0)
        # print 'the number of samples in this class is :' + str(len(dataList))
         trainingNumber = int(ceil((len(dataList) * int(ratio)) / 100.0))
         testingNumber = int(len(dataList) - trainingNumber)
        # print 'the position of training list is from  0 to ' + str(trainingNumber)  + '.' 
         trainList[positionMark].append(dataList[0:trainingNumber])
         testList[positionMark].append(dataList[trainingNumber:len(dataList)])
         trainPositions[positionMark].append(positionNow[0:trainingNumber])
         testPositions[positionMark].append(positionNow[trainingNumber:len(dataList)])
         trainingCount = trainingCount + trainingNumber
         print( '.............................................................')
         print( 'class ' + str(positionMark))
         print( 'train samples\' count:' + str(trainingNumber))
         testingCount = testingCount + testingNumber
         print( 'test samples\' count:' + str(testingNumber))
         print( str(len(dataList)) + '.')
         positionMark = positionMark + 1
    print( '---------------------------------------------------------------')
    print( 'data splited in to different datasets:')
    print( 'there are ' + str(trainingCount) + ' training samples and ')
    print( 'there are ' + str(testingCount) + ' testing samples.')
    print( 'writing dataset...')

    data_format = 0
    #dataset_format = int(dataset_format)
    print( dataset_format)
    if(dataset_format == "" and dataset_format != 1 and dataset_format != 2):
        print( "choose the data format, enter 1 for lmdb or enter 2 for mat")
        data_format = int(input(prompt))
    elif(dataset_format == 1 or dataset_format == 2):
        data_format = dataset_format
    if(data_format == 1):
        # write the splited data into lmdb format files
        # writeToLMDB(trainList, datasetName, 'training')
        # writeToLMDB(testList, datasetName, 'testing')
        print("写入lmdb数据模块正在完善中……")
    elif(data_format == 2):
        return writeToMAT(trainList, testList, trainPositions, testPositions, datasetName, ratio, neighbors)

#def assembleData(list, datasetName):
#    print "choose the data format, enter 1 for lmdb or enter 2 for mat"
#    data_format = int(input(prompt))
#    if data_format == 1:
#        assembleLMDB(list, datasetName)
#    elif:
#        assembleMAT(list, datasetName)

#processing code segment
#dataset_format: 1 for lmdb;2 for mat
def prepare(training_ratio, data_set_name, neighbor_size=8, save_format=2):
    #print "want to #1:construct a new dataset or #2:use existing dataset?"
    #if_new = int(input(prompt))
    if_new = 1
    # judge if the dataset is exists. to determain if the code will use the existing dataset and the exsiting experiment results.
    if(if_new == 1):
        # path = data_set_name
        while( data_set_name == "NONE"):
            print( "please enter the dataset name you want to transform...")
            data_set_name = input(prompt)
            if os.path.exists(path_prefix + data_set_name) != True:
                print( "ERROR: the dataset you selected dos'nt exist!!!")

        (dataList, inner_neighbors, positionList, rows, lines) \
            = loadData(data_set_name, neighbor_size)
        print("载入数据数量 : ",len(dataList[0]))
        print("载入的位置数量：",len(positionList[0]))

        (realPath, wrong_neighbor) \
            = assembleData(dataList, positionList, data_set_name, inner_neighbors, training_ratio, save_format)
        print( "预处理后的数据保存路径为：" + realPath + ".mat")
        print( "近邻数量：",inner_neighbors)
        return (realPath, inner_neighbors, rows, lines)
    # elif(if_new == 2) :
    #     print( "enter the existing dataset path:")
    #     realPath = input(prompt)
    #     #TODO后期要根据路径名去判断数据集的信息，并且赋给neighbors 变量，暂先用8固定
    #     neighbors = 8
    #     realPath = "../experiments/" + realPath + "/" +realPath
    #     return realPath, neighbors, raws, lines

if __name__ == '__main__':
    prepare(training_ratio=80,data_set_name='KSC')