import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from funcs import *
import os
import NN_inference as infe

filemat=getMat(filename='application', prefix='../input/')
#测试与训练集
Train=filemat['Tr']
Train_labels=filemat['Tr_l']
Train_labels=Train_labels.reshape(-1)
Test=filemat['Te']
Test_id=filemat['Te_id']
Test_id=Test_id.reshape(-1)
#将测试集划分为验证和测试集
Tr_len=len(Train)
Te_len=len(Test)
#随机乱序
shuffle_index=np.arange(0,Tr_len)
np.random.shuffle(shuffle_index)
Train=Train[shuffle_index]
Train_labels=Train_labels[shuffle_index]
Val_len=1000
Validata=Train[0:Val_len]
Validata_labels=Train_labels[0:Val_len]
Train=Train[Val_len:Tr_len]
Train_labels=Train_labels[Val_len:Tr_len]
Tr_len=len(Train)
print("shape is:\n",Train.shape,Train_labels.shape,Validata.shape,Validata_labels.shape,Test.shape)
#one hot变量
Train_labels_hot=np.zeros([Tr_len,2])
Train_labels_hot[np.arange(0,Tr_len),np.array(Train_labels,np.int32)]=1
Val_labels_hot=np.zeros([Tr_len,2])
Val_labels_hot[np.arange(0,Val_len),np.array(Validata_labels,np.int32)]=1

#获取输入节点数,应该为特征列数
feature_len=Train.shape[1]
#获取正样本比例
pos_len=np.sum(Train_labels==1)
alpha = pos_len/Tr_len
print("positive sample ratio is %.4f (%d\/%d)"%(alpha,pos_len,Tr_len))

#对tr每一列做0均值单位方差
eps=1e-4
testnum=0
batchnum=0
batchlist = np.arange(0, Tr_len)
vallist=np.arange(0, Val_len)
#每次调用获取下一批训练数据
def sampleImage():
    global batchnum
    global batchlist
    if batchnum==0:
        # 每次初始化一个batch序列
        np.random.shuffle(batchlist)
    batchsize = 100
    if batchnum+batchsize>Tr_len:
        batchnum=Tr_len-batchsize
    index=batchlist[batchnum:batchsize + batchnum]
    trs=Train[index]
    trls=Train_labels_hot[index]
    batchnum=(batchnum+batchsize)%Tr_len
    return trs,trls
#每次获取验证集
def gettest():
    if Val_len==0:
        return 0,0
    global vallist
    np.random.shuffle(vallist)
    val_data = Validata[vallist]
    val_labels = Val_labels_hot[vallist]
    return val_data,val_labels

# b=sampleImage()
# 通过xvaier初始化第一层的权重值，xvaier初始化详见http://blog.csdn.net/shuzfan/article/details/51338178
def xvaier_init(input_size, output_size):
    low = -np.sqrt(6.0 / (input_nodes + output_nodes))
    high = -low
    return tf.random_uniform((input_size, output_size), low, high, dtype=tf.float32)
#稀疏参数
def KL(p,pjj):
    pj = tf.clip_by_value(pjj, 1e-10, 1 - 1e-10)#截断操作,
    return tf.reduce_sum(p * tf.log(p / pj) + (1 - p) * tf.log((1 - p) / (1 - pj)))
# 计算代价函数，代价函数由三部分组成，均方差项，权重衰减项，以及稀疏因子项
def computecost():
    input_nodes = feature_len  # 输入节点数
    expand_size = 100  # 升维数节点数
    layer1_size = 200
    layer2_size = 100
    output_nodes = 2  # 输出节点数
    # 逐步减小学习率和增大稀疏惩罚
    #p = input_nodes / (layer2_size*4)
    beta = 4
    lamda = 4
    out = 2
    learnrate = 1e-4
    balanced=tf.constant([alpha,1-alpha],name='balance_factor')#平衡因子
    #输入参数
    x = tf.placeholder(tf.float32, shape=[None, input_nodes],name='input_x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, shape=[None, output_nodes], name='input_y')
    with tf.name_scope('expand_layer') as scope:
        # w = tf.Variable(xvaier_init(input_nodes, layer1_size))
        w = tf.Variable(tf.truncated_normal([input_nodes, expand_size], 0, 0.1))
        b = tf.Variable(tf.truncated_normal([expand_size], 0.1))
        res = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b))
        #将扩展的特征与原特征链接，一起作为下一层的输入
        expand_input=tf.concat([x,res],axis=1)
        # droupout
        #fcl1_drop = tf.nn.dropout(fcl1, keep_prob)
        regular0=tf.reduce_sum(w * w)
    with tf.name_scope('layer1') as scope:
        # w = tf.Variable(xvaier_init(input_nodes, layer1_size))
        #输入维扩展维度
        w = tf.Variable(tf.truncated_normal([input_nodes+expand_size, layer1_size], 0, 0.1))
        b = tf.Variable(tf.truncated_normal([layer1_size], 0.1))
        fcl1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b))
        # droupout
        fcl1_drop = tf.nn.dropout(fcl1, keep_prob)
        regular1=tf.reduce_sum(w * w)
    with tf.name_scope('layer2') as scope:
        w = tf.Variable(tf.truncated_normal([layer1_size, layer2_size], 0, 0.1))
        b = tf.Variable(tf.truncated_normal([layer2_size], 0.1))
        fcl2=tf.nn.relu(tf.nn.bias_add(tf.matmul(fcl1_drop, w), b))
        #droupout
        fcl2_drop = tf.nn.dropout(fcl2, keep_prob)
        regular2 = tf.reduce_sum(w * w)
    with tf.name_scope('softmax_layer') as scope:
        w = tf.Variable(tf.truncated_normal([layer2_size, output_nodes], 0, 0.1))
        b = tf.Variable(tf.truncated_normal([output_nodes], 0.1))
        fc3 = tf.nn.bias_add(tf.matmul(fcl2_drop, w), b)
        # softmax多分类器
        y_conv = tf.nn.softmax(fc3)
        #y_conv=tf.nn.sigmoid(fc3)
        regular3 = tf.reduce_sum(w * w)
    with tf.name_scope('loss_layer') as scope:
        #pjj =tf.concat([tf.reduce_mean(fcl1, 0),tf.reduce_mean(fcl2, 0)],0)
        #sparse_cost = KL(p,pjj)
        regular = lamda * (regular0+ regular1+regular2+regular3) / 2
        # cross_entropy = tf.reduce_mean(tf.pow(output - x, 2)) / 2 + sparse_cost * beta + regular  # + regular+sparse_cost*beta
        # 添加平衡因子
        cross_entropy = -tf.reduce_sum(balanced*(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))))\
                        + regular# + sparse_cost * beta # + regular+sparse_cost*beta
        #添加平衡因子
        #cross_entropy = -tf.reduce_sum( \
	     #   positive_balance*y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))\
         #   +(1-positive_balance)*(1-y_) * tf.log(1-tf.clip_by_value(y_conv, 0, 1.0-1e-10)))\
         #   +regular
        train_step = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)  # 调用优化器优化
        pred=y_conv
        tf.add_to_collection('pred_network', pred)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 记录日志
    with tf.name_scope('log') as scope:
        # 首先再源码中加入需要跟踪的变量：
        tf.summary.scalar("loss", cross_entropy)  # 损失函数值
        tf.summary.scalar("accuracy", accuracy)  # 损失函数值
        # )然后定义执行操作：
        merged_summary_op = tf.summary.merge_all()
    return x,keep_prob,y_,cross_entropy, fcl1, y_conv,y_,train_step,accuracy,regular,pred,merged_summary_op

def softmaxpredict(te):
    x, keep_prob, y_, cost, fcl1, y_conv, y_, train_step, accuracy, regular, pred,_ = computecost()
    train_x, train_l = sampleImage()
    sess = tf.Session()
    # 创建saver
    saver = tf.train.Saver()
    # 然后再session执行的时候，保存：
    if os.path.exists('save/model.model.meta'):  # 判断模型是否存在
        print('restore weightes form model!')
        saver.restore(sess, tf.train.latest_checkpoint('save'))  # 存在就从模型中恢复变量
    # te = (te - te.mean(0)) / (te.std(0) + eps)
    # 测试数据
    predict = sess.run(pred, feed_dict={x: Test, keep_prob: 1.0})
    return predict
def main():
    x, keep_prob, y_,cost, fcl1,y_conv,y_ ,train_step,accuracy,sparse_cost,regular,pred,merged_summary_op= computecost()

    train_x,train_l = sampleImage()
    sess = tf.Session()
    # 再session中定义保存路径：
    summary_writer = tf.summary.FileWriter('log', sess.graph)
    # 创建saver
    saver = tf.train.Saver()
    istrian=True
    # 然后再session执行的时候，保存：
    if os.path.exists('save/model.model.meta'):  # 判断模型是否存在
        print('restore weightes form model!')
        saver.restore(sess, tf.train.latest_checkpoint('save'))  # 存在就从模型中恢复变量
        istrian = False
    else:
        # sess.run(tf.global_variables_initializer())
        print('init weightes!')
        #若不存在，则训练数据
        sess.run(tf.global_variables_initializer())
        istrian = True
    if istrian:
        #训练和测试准确率曲线,查看是否过拟合
        trainacc=[]
        valacc=[]
        for i in range(500):
            train_x, train_l = sampleImage()
            if i % 2 == 0:
                #训练数据
                cost_c,y_c,acctr,reg=sess.run([cost,y_conv,accuracy,regular], feed_dict={x: train_x, y_: train_l,keep_prob:1.0})
                trainacc.append(acctr)
                print('time (%d)\n'%(int(i/1000)),'\tloss:%.4f'%(cost_c),' accuracy is %.4f  regular is %.4f'%(acctr,reg))
                # 测试数据
                teshuffle, te_hotshuffle=gettest()
                if teshuffle!=0:
                    accte = sess.run(accuracy, feed_dict={x: teshuffle, y_: te_hotshuffle, keep_prob: 1.0})
                    valacc.append(accte)
                    print("\ttest data acc is %.4f"%(accte))
                #保存参数
                save_path = saver.save(sess, "save/model.model")
            sumarylog,_=sess.run([merged_summary_op,train_step], feed_dict={x: train_x, y_: train_l,keep_prob:0.5})
            summary_writer.add_summary(sumarylog, i)
        if Val_len != 0:
            #训练完之后,画出训练测试的acc图
            x_axis=np.arange(0,len(trainacc))
            plt.plot(x_axis,trainacc)
            plt.plot(x_axis,valacc)
            plt.legend(labels = ['trainacc','valacc'])
            plt.show()

if __name__ == '__main__':
    main()
    infe.inference(Test,Test_id,'NN_inference')