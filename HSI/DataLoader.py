import os
import numpy as np
import collections
import scipy.io as sio
import matplotlib.pyplot as plt
import xlrd
from numpy.random import shuffle
from functools import reduce
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from functools import reduce

"""
    程序流程：
        load_dataSets()
            |
            load_spectMat()
            load_datasets_switch_avgrat() : 进行随机化和训练与测试数据的划分
            
"""


"""
    原始数据集只有Data/Gt和index，是不变数据集
    我们自己生成了Class数据，含list和sum，是可变数据集，因此其load方法没有使用全局变量的方式
"""
CLSS_MAT_KEY = 'ClsPos'
CLSS_SUM_KEY = 'ClsSum'
base_data_array = None
ground_truth_array = None
<<<<<<< HEAD
path_prefix = '../../HSI_DATA/'
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e

Dataset = collections.namedtuple('Dataset', ['data', 'target','pos'])
# Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])

class DataSet(object):
    def __init__(self, images, labels, positions, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=False, seed=None,
                 channel_ceiling=65535):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0] == positions.shape[0] , (
                'images.shape: %s labels.shape: %s position.shape: %s ' % (images.shape, labels.shape, positons.shape))
            self._num_examples = images.shape[0]
            # print("样本总量：",self._num_examples)
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # if reshape:
            #   # assert images.shape[3] == 1
            #   assert images.shape[-1] != None
            #   images = images.reshape(images.shape[0],
            #                           images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, channel_ceiling] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / channel_ceiling)
        self._images = images
        self._labels = labels
        self._postions = positions
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._sample_length = images.shape[1]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def positions(self):
        return self._postions

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def sample_length(self):
        return self._sample_length

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            self._postions = self.positions[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            positions_rest_part = self._postions[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                self._postions = self.positions[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            positions_new_part = self._postions[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0), np.concatenate((positions_rest_part,positions_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end],self._postions[start:end]


def get_shuffled_2D_index(x):
    temp = list(range(x))
    np.random.shuffle(temp)
    return np.array(temp)

def load_datasets_switch_minest(spec_mat, train_ratio=0.8, validation_ratio=0.2):
    print("使用最小分类数量集的80%来决定采样数量")

"""
    load时核心点是我们的分类目标点，然后四周选用其真实近邻,与先前加冰的若四周存在标签为0的无效数据时则使用核心点数据进行填充不同
    fake_data：仅对边缘处的填充元素进行控制，True则填充0，False则填充中心元素
    存储的mat数组的结构为
    维度     含义
     1      class（0为第一类）
     2      像元信息组012 345 678
     3      各近邻具体的频谱数据
     参数含义：
     fill_flag : cube的填充模式，
                0代表天然填充，除了边缘外其他全部使用真实近邻数据进行填充
                1代表完全fake填充，此时可以将卷积操作视为对于同一个频谱信息的多重计算 《== 默认方式
                2邻域内同类标签最大化方案，类似冰哥的处理方式
    #而且，对于训练集我们使用仿同类频谱信息的方式，对于测试集我们使用复制频谱信息的方式
"""
def load_spectMat(name="KSC",fake_flag=False,file_path=None,fill_flag=1):
    print("function load_spectMat")
    if(file_path==None):        #文件保存名
        file_path = "../DATA/" + name.upper() + "/"
        if(not os.path.exists(file_path)):
            print("不存在数据集文件夹../DATA/",name.upper(),"，请检查后重试！")
            exit(-1)
    file_name = file_path + name.upper() + "Spec_"+str(fill_flag)+".mat"

    index_list = load_classIndexList(name=name)

    mat_data = []
    #如果存在已经保存的mat文件则直接载入
    if(os.path.exists(file_name)):
        print("载入已经存在的spectMat数据")
        mat_data = sio.loadmat(file_name).__getitem__("Data")[0]
    else:#否则重新生成
        print("正在生成新的spectMat数据……")
        base_data_array = load_baseDataArray(name=name)
        (_max_height,_max_width,_fake_length) = base_data_array.shape
        """  数据测试
            # print("最大高度：",_max_height)      #512
            # print("最大宽度：", _max_width)      #614
            # print("_fake_length：",_fake_length)
            # index_list = [[ [0,0]],[ [0,2], [0,4]],[ [2,0], [4,0]],[ [4,4]],[ [2,2]]]     #测试数据，使用(0,0),(0,2),(2,0),(0,4),(4,0),(4,4),(2,2)进行边缘测试

        """


        """
            修改代码逻辑，使用方向数组进行控制
        """
        x_mov = [-1, 0, 1]
        y_mov = [-1, 0, 1]

        # 真实填充模式
        if(fill_flag==0):
            for i in range(1,len(index_list)):# 第i类
                cur_class = []

                for (x,y) in index_list[i]:#第i类中的所有像素点进行采样

                    core_data = list(base_data_array[x][y])
                    # 源数据的类型
                    # print("源数据类型：",type(base_data_array[0][0][0]))      #numpy.uint16
                    if (fake_flag):
                        fake_data = core_data
                    else:
                        fake_data = [0]*_fake_length

                    cur_data = []
                    #对单相素点得到其近邻数据集
                    for _i in x_mov: #行序，左到右
                        for _j in y_mov: #列序，上到下
                            cur_x = x + _i
                            cur_y = y + _j
                            # print("cur x : ", cur_x, "cur_y：", cur_y)
                            if(cur_x<0 or cur_y<0 or cur_x>_max_height-1 or cur_y>_max_height-1):
                                cur_data.append(fake_data)
                                # print("appending : ",fake_data)
                            else:
                                ele = base_data_array[cur_x][cur_y]
                                cur_data.append(ele)
                                # print("appending：",ele)
                    # print("当前元素信息：",cur_data)
                    cur_class.append(cur_data)

                mat_data.append(cur_class)

        # 复制填充模式
        if(fill_flag==1):
            for i in range(1, len(index_list)):  # 第i类
                cur_class = []

                for (x, y) in index_list[i]:  # 第i类中的所有像素点进行采样

                    core_data = list(base_data_array[x][y])
                    # 源数据的类型
                    fake_data = core_data   #强行fake
                    cur_data = []
                    # 对单相素点得到其近邻数据集
                    for _i in x_mov:  # 行序，左到右
                        for _j in y_mov:  # 列序，上到下
                            cur_data.append(fake_data)
                    cur_class.append(cur_data)

                mat_data.append(cur_class)

        if(fill_flag==2): #TODO 邻域最大类内元素
            for i in range(1, len(index_list)):  # 第i类
                cur_class = []
                # print(type(index_list[i]))        #ndarray  761x2
                cur_class_set = set([tuple(x) for x in index_list[i]])
                # print(cur_class_set)
                for (x, y) in index_list[i]:  # 第i类中的所有像素点进行采样

                    core_data = list(base_data_array[x][y])
                    fake_data = core_data   #强行指定fake数据

                    cur_data = []
                    # 对单相素点得到其近邻数据集
                    for _i in x_mov:  # 行序，左到右
                        for _j in y_mov:  # 列序，上到下
                            cur_x = x + _i
                            cur_y = y + _j

                            if ((cur_x,cur_y) not in cur_class_set):
                                cur_data.append(fake_data)
                                # print("appending : ",fake_data)
                            else:
                                ele = base_data_array[cur_x][cur_y]
                                cur_data.append(ele)
                                # print("appending：",ele)
                    # print("当前元素信息：",cur_data)
                    cur_class.append(cur_data)

                mat_data.append(cur_class)


        result_dict = {'Data': mat_data}
        sio.savemat(file_name, result_dict)  # 将数据保存为mat


"""
    处理流程：
    1.先shuffle数据顺序
    2.划分训练与测试集
    3.分别对训练集和测试集进行cube组装
    数据采用先划分，保存到mat文件中，然后在取出来后再进行shuffle到方式进行使用
    训练和测试比例为8:2，有两种实现方式，
    一：每个分类都是8:2；
    二：只使用最小分类的80%为训练集，然后以这个数量为标准划分其他类别到训练集数量，剩余到都为测试集，也就是固定所有分类到训练元素到数量
"""
def generateDatasetMat(name="KSC", train_ratio=0.8, split_strategy=1):
    print("function generateDatasetMat")
    data_name = name.upper()
<<<<<<< HEAD
    data_path = path_prefix+data_name+"/"
=======
    data_path = "../DATA/"+data_name+"/"
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    file_name = data_path+"preparedDataset_"+str(split_strategy)+".mat"
    if(not os.path.exists(data_path)):
        print("不存在您要载入的数据集路径")
        return -1

<<<<<<< HEAD
    #获取数据和真值图
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    base_data_array = load_baseDataArray(data_name)
    ground_truth_array = load_groundTruthData(data_name)
    max_height= len(ground_truth_array)
    max_width = len(ground_truth_array[0])

    # print("最大高度：",max_height,"最大宽度：",max_width)
    (shuffled_class_index,shuffled_labels) = load_shuffledData(data_name)

    class_lens = [len(x) for x in shuffled_class_index]

    train_class = []
    train_label = []
    test_class = []
    test_label=[]

    if(split_strategy==1):
        #所有class等比例划分
        for i in range(len(shuffled_class_index)):  #此时下标为0的代表的是第1类
            cur_class = shuffled_class_index[i]
            cur_labels = shuffled_labels[i]
            cut_point = int(train_ratio * len(cur_class))
            train_class.append(cur_class[:cut_point])
            train_label.append(cur_labels[:cut_point])
            test_class.append(cur_class[cut_point:])
            test_label.append(cur_labels[cut_point:])
    elif(split_strategy==2):
        #按最小class的比例进行划分
        _min = min(class_lens)
        cut_point = int(train_ratio*_min)
        for i in range(len(shuffled_class_index)):
            cur_class = shuffled_class_index[i]
            cur_labels = shuffled_labels[i]
            train_class.append(cur_class[:cut_point])
            train_label.append(cur_labels[:cut_point])
            test_class.append(cur_class[cut_point:])
            test_label.append(cur_labels[cut_point:])

    # --------------------以上已经实现了不同种类的筛选操作，可以整体混洗了----------------------------#
    #---------------------取消种类界限，全部拉直为一条向量
    train_pos = []
    train_labels = []
    for i in range(len(train_class)):
        train_pos=train_pos+list(train_class[i])
        train_labels=train_labels+list(train_label[i])
    # train_class = reduce(lambda x,y:list(x)+list(y),train_class)

    test_pos = []
    test_labels = []
    for i in range(len(test_class)):
        test_pos=test_pos+list(test_class[i])
        test_labels=test_labels+list(test_label[i])
    # print(test_class)

    #---------------------以上操作得到了训练集和测试集的核心元素的坐标信息----------------------------#
    #TODO 分别对训练数据集和测试数据集进行不同的数据组装方式：
    #训练集数据的填充
    x_mov = [-1, 0, 1]
    y_mov = [-1, 0, 1]

    # 生成训练数据集
    # TODO 邻域最大类内元素
    train_data = []
    for (x,y) in train_pos:  # 第i类
        core_data = list(base_data_array[x][y])
        fake_data = core_data  # 强行指定fake数据

        cur_data = []
        # 对单相素点得到其近邻数据集
        for _i in x_mov:  # 行序，左到右
            for _j in y_mov:  # 列序，上到下
                cur_x = x + _i
                cur_y = y + _j

                if (cur_x<0 or cur_x> max_height or cur_y<0 or cur_y>max_width or ground_truth_array[cur_x][cur_y]!=ground_truth_array[x][y] ):
                    cur_data.append(fake_data)
                    # print("appending : ",fake_data)
                else:
                    ele = base_data_array[cur_x][cur_y]
                    cur_data.append(ele)

        train_data.append(cur_data)

    #测试数据集
    # TODO 复制填充模式
    test_data=[]
    for (x,y) in test_pos:  # 第i类
        core_data = list(base_data_array[x][y])
        fake_data = core_data  # 强行指定fake数据

        cur_data = []
        # 对单相素点得到其近邻数据集
        for _i in x_mov:  # 行序，左到右
            for _j in y_mov:  # 列序，上到下
                cur_data.append(fake_data)
        test_data.append(cur_data)

    result_dict = {'TrData': train_data,'TrLabel':train_labels,'TrPos':train_pos,'TeData':test_data,'TeLabel':test_labels,'TePos':test_pos}
    sio.savemat(file_name, result_dict)  # 将数据保存为mat
    print("已生成cube组织的数据信息")

#TODO 下面的方法实现将各类坐标点随机化并生成对应的标签信息
#返回的数据位置信息是 [[[x1,y1],[x1,y1],[x1,y1]],[[c1,d1],[c2,d2]]]
#返回的标签为 [[1,1,1],[2,2]]
def load_shuffledData(name="KSC"):
    print("function load_shuffledClass")
    data_name = name.upper()
<<<<<<< HEAD
    data_path = path_prefix+data_name+"/"
=======
    data_path = "../DATA/"+data_name+"/"
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    if(not os.path.exists(data_path)):
        print("不存在您要载入的数据集路径")
        return -1
    _class_index_array = load_classIndexList(name=name)
    shuffled_class = []
    shuffled_labels = []
    #将数据随机化
    class_num = len(_class_index_array)
    for i in range(1,class_num):
        label_ele = [0]*class_num
        label_ele[i]=1
        _list = _class_index_array[i]
        _ele_num = len(_list)
        index = list(range(_ele_num))
        np.random.shuffle(index)
        shuffled_index = index
        shuffled_class.append(_class_index_array[i][shuffled_index])
        shuffled_labels.append([label_ele]*_ele_num)
    return (shuffled_class,shuffled_labels)

#TODO 下面的方法负责载入基础数据
def load_baseDataArray(name="KSC"):
    print("function load_baseDataArray")
    data_name = name.upper()
<<<<<<< HEAD
    data_path = path_prefix + data_name + "/"
=======
    data_path = "../DATA/" + data_name + "/"
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    if (not os.path.exists(data_path)):
        print("不存在您要载入的数据集路径")
        return -1
    global base_data_array
    base_data = sio.loadmat(data_path + data_name + "Data.mat")
    bsekey = extract_real_key(list(base_data.keys()))[0]    # print(bskey)
    base_data_array = base_data.__getitem__(bsekey)
    """
        print(type(base_data_array))
        print(base_data_array.shape)
        #numpy.ndarray类型,KSC数据维度：(512, 614, 176)
    """
    print("通过load_baseData函数获取到基础数据")
    return base_data_array

#TODO 下面的方法负责载入groundtruth数据
def load_groundTruthData(name="KSC"):
    print("function load_groundTruthData")
    data_name = name.upper()
<<<<<<< HEAD
    data_path = path_prefix+data_name+"/"
=======
    data_path = "../DATA/"+data_name+"/"
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    if(not os.path.exists(data_path)):
        print("不存在您要载入的数据集路径")
        return -1
    groundtruth_mat = sio.loadmat(data_path + data_name + "Gt.mat")
    gthkey = extract_real_key(list(groundtruth_mat.keys()))[0]  # print(gtkey)
    """
        #从groundtruth数据中统计各个类型分别拥有的数量
    """
    ground_truth_array = groundtruth_mat.__getitem__(gthkey)
    # draw_classResult(groundtruth_array)
    # print("标记数组中的维度：",ground_truth_array.shape)#共512x614 = 314368
    return ground_truth_array

#TODO 下面的方法负责将不同分类的坐标信息载入为一个数组索引的列表
#class_index_list结构为[{(x,y),(x,y)},{(x,y),(x,y),(x,y)}],其中list[0]代表第0类分类的所有坐标
def load_classIndexList(name="KSC"):
    # print("function load_classIndexList")
    data_name = name.upper()
<<<<<<< HEAD
    data_path = path_prefix+data_name+"/"
=======
    data_path = "../DATA/"+data_name+"/"
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    if(not os.path.exists(data_path)):
        print("不存在您要载入的数据集路径")
        return -1
    ground_truth_array=load_groundTruthData(name=name)

    """
        获取按类组合的位置列表
    """
    class_matfile_name = data_path + data_name + "Class.mat"
    if(os.path.exists(class_matfile_name)):
        print("正在载入已经保存的Class分类的mat文件……")
        class_index_mat = sio.loadmat(class_matfile_name)
        # print("分类mat中包含的键：",class_index_mat.keys())
        class_index_list = list(class_index_mat.__getitem__(CLSS_MAT_KEY)[0])
        # class_sum_list = list(class_index_mat.__getitem__(CLSS_SUM_KEY)[0])
    else:
        print("生成新的Class分类mat文件……")
        class_index_mat = generate_classMat(ground_truth_array, class_matfile_name, CLSS_MAT_KEY)       #分析并保存分类数据
        class_index_list = class_index_mat.__getitem__(CLSS_MAT_KEY)
        # class_sum_list = class_index_mat.__getitem__(CLSS_SUM_KEY)
    return class_index_list
#下面的方法实现将groundtruth_array中的元素根据类别进行归类，每个数组下保存的都是该类型的坐标数据
def generate_classMat(groundtruth_array, class_matfile_name, dict_key="ClsPos"):
    (height, width) = groundtruth_array.shape
    print("heights:", height, "width:", width)
    class_index_list = []
    class_sum_list = []
    class_num = groundtruth_array.max()+1
    for i in range(class_num):  # 初始化位置与分类列表
        class_index_list.append(set())
        class_sum_list.append(0)
    # 遍历标记数组的元素，分别将不同种类标记的位置计入相应的队列
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += 1
            flag = groundtruth_array[i][j]
            class_index_list[flag].add((i,j))  # 将位置信息记录到相应的分类中,存储的时候使用set方式进行存储，存入mat文件的时候需要再转换成list，因为sio.savemat貌似不支持set格式
            class_sum_list[flag] += 1  # 同时增加该类型元素的总数

    """
    # 输出统计的各个类型的数量
        print("遍历的元素总数：", sum)
        print(class_sum_list)  # for KSC: [761, 243, 256, 252, 161, 229, 105, 431, 520, 404, 419, 503, 927] 共5211个有效数据
    """
    #先将set转化为list
    for i in range(len(class_index_list)):
        class_index_list[i] = list(map(lambda x:list(x),class_index_list[i]))
    #然后将我们分类后的数据存入mat文件的字典中：
    result_dict = {dict_key: class_index_list,CLSS_SUM_KEY:class_sum_list}
    sio.savemat(class_matfile_name, result_dict)
    return result_dict

#以下各方法实现数据统计和展现功能
#2.将我们的各个类型的频率信息绘制到不同的图形上 ==> 多数据绘制曲线图
def draw_classSpect(name="KSC", _spec_max = 400, _max_sample_num = 300):#将分类后的class的频段进行绘图
    print("function draw_classSpect")
    # _spec_max = 400             #TODO 此处将绘图时的频谱强度最大值固定为400，仅针对KSC数据集
    # _max_sample_num = 800

    data_name = name.upper()
    figure_path = "../DATA/" + data_name + "/classStatisticsFigures/"
    if (not os.path.exists(figure_path)): #为保存类型频谱信息创建文件夹
        os.mkdir(figure_path)

    color_list = get_colorList()
    _color_num = len(color_list)
    print("总体的颜色数量为：",_color_num)

    # 载入频谱信息
    if(base_data_array==None):
        load_baseDataArray()
    #首先获取已经分类好的频谱信息
    class_index_list = load_classIndexList(name=name)
    #准备分类种类信息，画图用
    _class_num = len(class_index_list)
    #准备频谱波段长度信息，画图用
    _shape_list = base_data_array.shape
    spec_num = _shape_list[-1]

    for i in range(_class_num): #每一类绘制一张图，一共12类
        # print("类别i: ",i," in ",_class_num)
        plt.figure(figsize=(12, 8))  # figuresize的单位是英寸，宽x高
        _sample_num = len(class_index_list[i])
        # 取色    每种类型使用同一种颜色
        _cur_color = color_list[i]

        _temp_list = class_index_list[i]
        # print("class_index_list[i]的类型为：",type(_temp_list))      #numpy.ndarray
        # print("具体数据为：",_temp_list)                              #[[102 595][ 33 440][342 418]...,[116 548][ 63 255][356  75]]
        index_list = list(_temp_list)

        # 将分类频谱信息随机化，然后取到其中的频谱信息        shuffle(class_index_list)

        for j in range(_sample_num):
            if(j >= _max_sample_num):
                print("j: ",j,"超出最大值",_max_sample_num)
                break
            # print("i,j:",i,",",j)
            #取点
            [x,y] = index_list[j]
            # print("随机点的坐标为：(",x,",",y,")")
            spec = base_data_array[x][y]
            # print("该点的频谱信息：",spec)
            # print("波段应有：", spec_num, "波段实际：", len(spec))
            #作图
            x = np.arange(spec_num)            #使用numpy的arrange可以指定小数，而Python的arrange只能指定整数
            y = spec
            plt.plot(x,y,color=_cur_color)  #x对应的是横坐标，y对应的是纵坐标，linewidth用于指定线的宽度，还可以指定线的颜色、样式(实线-,虚线--，点线.)

        plt.axis([0, spec_num, 0, _spec_max])  # axis可以指定坐标轴的范围
        plt.xlabel("spectrum")
        plt.ylabel("strength")
        plt.title("spectrum feature of class "+str(i))
        plt.grid(True)  #显示网格
        # plt.legend()    #显示图例
        plt.savefig(figure_path+"class_"+str(i)+".png")
        # plt.show()
def get_colorList(shuffle_flag=False):
    color_book = xlrd.open_workbook("./color_tables.xls")
    color_table = color_book.sheet_by_index(0)
    _row_num = color_table.nrows
    color_list = []
    for i in range(1,_row_num):
        _val = color_table.cell_value(i, 4)
        color_list.append(_val)
    # print(color_list)
    # 使用numpy.random.shuffle()进行随机化，该随机化是原地随机的，且只对第一维数据进行随机化
    if(shuffle_flag):
        shuffle(color_list)
    # print("随机后的颜色列表：", color_list)
    return color_list
#根据分类后的ground_truth中提取的class_index_list绘制出正确的分类图
#若存在错误点则使用特殊的标记将错误点也绘制出来
def draw_classResult(name="KSC", error_index_list=None):
    #若输入中没有给出测试分类结果的class_index_list则只绘制ground_truth的图样
    class_index_list = load_classIndexList(name=name)
    color_list = get_colorList()
    #只显示ground_truth的结果
    #第一步，从class_index_list中提取出各个分类了的坐标情况[]
    # print(class_index_list[1][0][0])
    scale = 1
    legend_scale = 8
    plt.figure(figsize=(11,10))                 #宽，高
    for i in range(1,len(class_index_list)) :
        x_list = [p[0] for p in class_index_list[i]]        #可以确定的是这个列表生成式能够得到所有x的坐标
        y_list = [p[1] for p in class_index_list[i]]
        plt.scatter(x_list, y_list, c=color_list[i],s=scale,alpha=1, label="class "+str(i))  #s，散点的大小，alpha为不透明度，, edgecolors='white'

    if(error_index_list!=None):
        #绘制出错误点的坐标
        # print("error_index_list的类型为：",type(error_index_list))
        # print(error_index_list)
        x_list = [p[0] for p in error_index_list]        #可以确定的是这个列表生成式能够得到所有x的坐标
        y_list = [p[1] for p in error_index_list]
        plt.scatter(x_list, y_list, c="red",marker="*", s=4, alpha=1,
                    label="error cases ")  # s，散点的大小，alpha为不透明度，, edgecolors='white'

    plt.title('Scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(markerscale =legend_scale)     #markerscale   The relative size of legend markers compared with the originally drawn ones. Default is None which will take the value from the
    # plt.grid(True)
    plt.savefig(name+"_groundtruth_classifacation.png")
    plt.show()

def extract_real_key(key_list):
    # 提取到所有核心数据列的key值信息
    key = list(filter(lambda x:not x.startswith("_"),key_list))
    return key

def load_Dataset(name="ksc",split_strategy=1):
    print("function loadDataset")
    data_name = name.upper()
    file_name = "../DATA/"+data_name+"/"+"preparedDataset_"+str(split_strategy)+".mat"
    if(not os.path.exists(file_name)):
        print("不存在您要载入的数据集")
        return -1
    matdata = sio.loadmat(file_name)
    train_data = matdata.__getitem__("TrData")
    print("使用的训练数据为：",len(train_data))
    train_label = matdata.__getitem__("TrLabel")
    train_pos = matdata.__getitem__("TrPos")
    training_data = DataSet(train_data,train_label,train_pos)
    test_data = matdata.__getitem__("TeData")
    print("测试数据总量为：",len(test_data))
    test_label = matdata.__getitem__("TeLabel")
    test_pos = matdata.__getitem__("TePos")
    testing_data = DataSet(test_data,test_label,test_pos)
    return Datasets(train=training_data,test=testing_data)


"""
    将已经分类并组织好的数据根据策略的不同载入为train,test,validation的DataSet数据
    参数：
    name：数据的名称，根据不同的名称载入不同的数据对应的频谱组织信息，ksc为默认数据
    strategy：组织策略名称，0代表的是按不同分类相同比例的方式进行数据的划分，1代表的是按不同分类相同数量训练集的方式进行采样，0为默认
"""
def load_dataSets(name="KSC", strategy=0,train_ratio=0.8,validation_ratio=0.2,fill_flag = 2):
    print("function load_dataSets ")
    spec_mat = load_spectMat(name=name,fill_flag=fill_flag)

    #Python中的switch语句：
    strategySwitcher = {
        0:load_datasets_switch_avgrat,
        1:load_datasets_switch_minest
    }
    return strategySwitcher.get(strategy)(spec_mat,train_ratio,validation_ratio)

def label_map(i,l):#i是实际标签值，l是单个标签的最大长度
    _temp = [0.0]*l
    _temp[i] = 1.0
    return _temp

def load_datasets_switch_avgrat(spec_mat, train_ratio=0.8, validation_ratio=0.2):
    print("使用各个种类等概率采样的策略")
    class_length_list = [len(x) for x in spec_mat]      #此时，mat_count中保存的就是每个分类中不同种类的样本数
    # print("得到的specMat的数量分布：", class_length_list)

    #TODO 约定： 1代表第一种分类
    label_num = len(spec_mat)+1
    labels = [[label_map(i,label_num) for _ in range(len)] for i,len in enumerate(class_length_list,start=1)]

    # TODO 在切分前应该先使用shuffle进行随机化
    new_index = [get_shuffled_2D_index(x) for x in class_length_list]
    # 将二维数组逐行随机化
    data = [d[i] for d, i in zip(spec_mat, new_index)]
    # print("new indexed data : ", data)

    #得到训练集合测试集的分界点
    training_points = [ int(x*train_ratio) for x in class_length_list]
    # print("得到的cutting_points : ",training_points)
    #得到训练集中的验证集的分界点
    validation_points = [int(x*(1-validation_ratio)) for x in training_points]
    # print("得到的validation_points : ", validation_points)
    #利用得到的cut point对数据进行切分
    training_data = [x[:i] for x,i in zip(data ,training_points)]
    training_label = [x[:i] for x,i in zip(labels ,training_points)]

    # TODO 以上数据均为二维数据，第一维保存有class属性，第二维（及更深）维度保存的是不同像元的频谱信息
    # 为了方便批量化取数，我们需要把第一维去掉，然后才可以在不同分类间实现随机化,此处我们使用reduce函数来将之降维
    validation_data = [x[i:] for x,i in zip(training_data ,validation_points)]
    validation_data = reduce(lambda x,y:np.array(list(x)+list(y)),validation_data)
    validation_label = [x[i:] for x,i in zip(training_label ,validation_points)]
    validation_label = reduce(lambda x,y:np.array(list(x)+list(y)),validation_label)

    training_data = [x[:i] for x,i in zip(training_data ,validation_points)]
    training_data = reduce(lambda x, y: np.array(list(x) + list(y)), training_data)
    training_label = [x[:i] for x,i in zip(training_label ,validation_points)]
    training_label = reduce(lambda x, y: np.array(list(x) + list(y)), training_label)

    test_data = [x[i:] for x,i in zip(data ,training_points)]
    test_data = reduce(lambda x, y: np.array(list(x) + list(y)), test_data)
    test_label = [x[i:] for x, i in zip(labels, training_points)]
    test_label = reduce(lambda x, y: np.array(list(x) + list(y)), test_label)

    train = DataSet(training_data, training_label)
    validation = DataSet(validation_data, validation_label)
    test = DataSet(test_data, test_label)

    return Datasets(train=train, validation=validation, test=test)



if __name__ == "__main__":
    # loadMat()
    # draw_classSpect()
    # draw_classResult(name="KSC",error_index_list=[[1,2],[2,4],[4,8],[8,16]])
    # load_spectMat()
    # load_dataSets()
    generateDatasetMat()
    # load_Dataset()
    # alpha()
