import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import predata
import os

file_path = './dataset/train.csv'
savefile = True
savepath = './data'
predataflag=1
_, _, tr, tr_l, _ = predata.predata(file_path, savefile, savename='train', savepath=savepath, ratio=0, predataflag=predataflag)

#对tr每一列做0均值单位方差
eps=1e-4
testnum=0

te=tr[0:testnum]
te_l=tr_l[0:testnum]
tr=tr[testnum:]
tr_l=tr_l[testnum:]
# tr=(tr-tr.mean(0))/(tr.std(0)+eps)
# te=(te-te.mean(0))/(te.std(0)+eps)
te_lens=te_l.shape[0]
te_hot=np.zeros([te_lens,2])
te_hot[np.arange(0,te_lens),np.array(te_l,np.int32)]=1

lens=tr_l.shape[0]
one_hot=np.zeros([lens,2])
one_hot[np.arange(0,lens),np.array(tr_l,np.int32)]=1
lens = len(tr)


batchnum=0
batchlist = np.arange(0, lens)
testlist=np.arange(0, te_lens)
def sampleImage():
    global batchnum
    global batchlist
    if batchnum==0:
        # 每次初始化一个batch序列
        np.random.shuffle(batchlist)
    batchsize = 100
    if batchnum+batchsize>lens:
        batchnum=lens-batchsize
    index=batchlist[batchnum:batchsize + batchnum]
    trs=tr[index]
    trls=one_hot[index]
    batchnum=(batchnum+batchsize)%lens
    return trs,trls

def gettest():
    if te_lens==0:
        return 0,0
    global testlist
    np.random.shuffle(testlist)
    teshuffle = te[testlist]
    te_hotshuffle = te_hot[testlist]
    return teshuffle,te_hotshuffle

# b=sampleImage()
# 通过xvaier初始化第一层的权重值，xvaier初始化详见http://blog.csdn.net/shuzfan/article/details/51338178
def xvaier_init(input_size, output_size):
    low = -np.sqrt(6.0 / (input_nodes + output_nodes))
    high = -low
    return tf.random_uniform((input_size, output_size), low, high, dtype=tf.float32)

def KL(p,pjj):
    pj = tf.clip_by_value(pjj, 1e-10, 1 - 1e-10)#截断操作,
    return tf.reduce_sum(p * tf.log(p / pj) + (1 - p) * tf.log((1 - p) / (1 - pj)))
# 计算代价函数，代价函数由三部分组成，均方差项，权重衰减项，以及稀疏因子项
def computecost():
    input_nodes = 26  # 输入节点数
    layer1_size = 100  # 隐藏节点数
    layer2_size = 150
    output_nodes = 2  # 输出节点数
    # 逐步减小学习率和增大稀疏惩罚
    p = input_nodes / (layer2_size*4)
    beta = 4
    lamda = 4
    out = 2
    learnrate = 1e-4
    #输入参数
    x = tf.placeholder(tf.float32, shape=[None, input_nodes])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, shape=[None, output_nodes], name='input_y')
    with tf.name_scope('layer1') as scope:
        # w = tf.Variable(xvaier_init(input_nodes, layer1_size))
        w = tf.Variable(tf.truncated_normal([input_nodes, layer1_size], 0, 0.1))
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
        regular3 = tf.reduce_sum(w * w)
    with tf.name_scope('loss_layer') as scope:
        pjj =tf.concat([tf.reduce_mean(fcl1, 0),tf.reduce_mean(fcl2, 0)],0)
        sparse_cost = KL(p,pjj)
        regular = lamda * ( regular1+regular2+regular3) / 2
        # cross_entropy = tf.reduce_mean(tf.pow(output - x, 2)) / 2 + sparse_cost * beta + regular  # + regular+sparse_cost*beta
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))\
                        + regular  + sparse_cost * beta # + regular+sparse_cost*beta
        train_step = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)  # 调用优化器优化
        pred=tf.argmax(y_conv, 1)
        correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 记录日志
    with tf.name_scope('log') as scope:
        # 首先再源码中加入需要跟踪的变量：
        tf.summary.scalar("loss", cross_entropy)  # 损失函数值
        tf.summary.scalar("accuracy", accuracy)  # 损失函数值
        # )然后定义执行操作：
        merged_summary_op = tf.summary.merge_all()
    return x,keep_prob,y_,cross_entropy, fcl1, y_conv,y_,train_step,accuracy,sparse_cost,regular,pred,merged_summary_op


# 可视化自编码器：为了使隐藏单元得到最大激励（隐藏单元需要什么样的特征输入），将这些特征输入显示出来。
def show_image(w):
    sum = np.sqrt(np.sum(w ** 2, 0))
    changedw = w / sum
    a, b = changedw.shape
    c = np.sqrt(a * b)
    d = int(np.sqrt(a))
    e = int(c / d)
    buf = 1
    newimage = -np.ones((buf + (d + buf) * e, buf + (d + buf) * e))
    k = 0
    for i in range(e):
        for j in range(e):
            maxvalue = np.amax(changedw[:, k])
            if (maxvalue < 0):
                maxvalue = -maxvalue
            newimage[(buf + i * (d + buf)):(buf + i * (d + buf) + d),
            (buf + j * (d + buf)):(buf + j * (d + buf) + d)] = np.reshape(changedw[:, k], (d, d)) / maxvalue
            k += 1
    plt.figure("beauty")
    plt.imshow(newimage)
    plt.axis('off')
    plt.show()

def softmaxpredict(te):
    x, keep_prob, y_, cost, fcl1, y_conv, y_, train_step, accuracy, sparse_cost, regular, pred,_ = computecost()
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
    predict = sess.run(pred, feed_dict={x: te, y_: te_hot, keep_prob: 1.0})
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
        testacc=[]
        for i in range(50000):
            train_x, train_l = sampleImage()
            if i % 1000 == 0:
                #             print(fcl1_)
                #             print(output_)
                #训练数据
                cost_c,y_c,acctr,sparse,reg=sess.run([cost,y_conv,accuracy,sparse_cost,regular], feed_dict={x: train_x, y_: train_l,keep_prob:1.0})
                trainacc.append(acctr)
                print('time (%d)\n'%(int(i/1000)),'\tloss:%.4f'%(cost_c),' accuracy is %.4f sparse is %.4f regular is %.4f'%(acctr,sparse,reg))
                # 测试数据
                teshuffle, te_hotshuffle=gettest()
                if teshuffle!=0:
                    accte = sess.run(accuracy, feed_dict={x: teshuffle, y_: te_hotshuffle, keep_prob: 1.0})
                    testacc.append(accte)
                    print("\ttest data acc is %.4f"%(accte))
                #保存参数
                save_path = saver.save(sess, "save/model.model")
            sumarylog,_=sess.run([merged_summary_op,train_step], feed_dict={x: train_x, y_: train_l,keep_prob:0.5})
            summary_writer.add_summary(sumarylog, i)
        if te_lens != 0:
            #训练完之后,画出训练测试的acc图
            x_axis=np.arange(0,len(trainacc))
            plt.plot(x_axis,trainacc)
            plt.plot(x_axis,testacc)
            plt.legend(labels = ['trainacc','testacc'])
            plt.show()
    # np.save("weights1.npy", w_)
    # show_image(w_)
    #测试训练数据
    # 取训练集
    test_path = './dataset/test.csv'
    te, _, _, _, passengerIdlist = predata.predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                                   predataflag=predataflag)
    # te = (te - te.mean(0)) / (te.std(0) + eps)
    #测试数据
    predict=sess.run(pred, feed_dict={x: te,y_:te_hot,keep_prob:1.0})
    # print("predict is ",predict)
    predata.savepredictcsv(passengerIdlist, predict, 'softmax_test')

if __name__ == '__main__':
    main()