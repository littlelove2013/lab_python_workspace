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
# 取训练集
test_path = './dataset/test.csv'
te, _, _, _, passengerIdlist = predata.predata(test_path, savefile, savename='test', savepath=savepath, ratio=1,
                                           predataflag=predataflag)
#对tr每一列做0均值单位方差
eps=1e-4

testnum=100
te=tr[0:testnum]
te_l=tr_l[0:testnum]
tr=tr[testnum:]
tr_l=tr_l[testnum:]
tr=(tr-tr.mean(0))/(tr.std(0)+eps)
te=(te-te.mean(0))/(te.std(0)+eps)
te_lens=te_l.shape[0]
te_hot=np.zeros([te_lens,2])
te_hot[np.arange(0,te_lens),np.array(te_l,np.int32)]=1

lens=tr_l.shape[0]
one_hot=np.zeros([lens,2])
one_hot[np.arange(0,lens),np.array(tr_l,np.int32)]=1
lens = len(tr)
# 对数据归一化，然后扔到稀疏编码器内
input_nodes = 26  # 输入节点数
hidden_size = 100  # 隐藏节点数
output_nodes = 2  # 输出节点数


def sampleImage():
    batchsize = 100
    start = np.random.randint(0, lens - 100)
    return tr[start:100 + start],one_hot[start:100 + start]

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
def computecost(w, b, x, w1, b1,keep_prob):
    #逐步减小学习率和增大稀疏惩罚
    p = input_nodes/(hidden_size*4)
    beta = 8
    lamda = 8
    out=2
    learnrate=1e-5
    y_ = tf.placeholder(tf.float32, shape=[None, out], name='input_y')
    hidden_output = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w),b))
    fc2_drop = tf.nn.dropout(hidden_output, keep_prob)
    pjj = tf.reduce_mean(hidden_output, 0)
    sparse_cost = KL(p,pjj)
    fcl2 = tf.nn.bias_add(tf.matmul(fc2_drop, w1),b1)
    # softmax多分类器
    y_conv = tf.nn.softmax(fcl2)
    regular = lamda * (tf.reduce_sum(w * w) + tf.reduce_sum(w1 * w1)) / 2

    # cross_entropy = tf.reduce_mean(tf.pow(output - x, 2)) / 2 + sparse_cost * beta + regular  # + regular+sparse_cost*beta
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))\
                    + sparse_cost * beta + regular  # + regular+sparse_cost*beta
    train_step = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)  # 调用优化器优化
    pred=tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return cross_entropy, hidden_output, y_conv,y_,train_step,accuracy,sparse_cost,regular,pred


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


def main():
    w = tf.Variable(xvaier_init(input_nodes, hidden_size))
    b = tf.Variable(tf.truncated_normal([hidden_size], 0.1))
    w1 = tf.Variable(tf.truncated_normal([hidden_size, output_nodes], -0.1, 0.1))
    b1 = tf.Variable(tf.truncated_normal([output_nodes], 0.1))
    x = tf.placeholder(tf.float32, shape=[None, input_nodes])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    cost, hidden_output,y_conv,y_ ,train_step,accuracy,sparse_cost,regular,pred= computecost(w, b, x, w1, b1,keep_prob)

    train_x,train_l = sampleImage()
    sess = tf.Session()

    # 创建saver
    saver = tf.train.Saver()
    istrian=True
    # 然后再session执行的时候，保存：
    if os.path.exists('save/model.model.meta'):  # 判断模型是否存在
        print('restore weightes form model!')
        saver.restore(sess, tf.train.latest_checkpoint('save'))  # 存在就从模型中恢复变量
        # istrian = False
    else:
        # sess.run(tf.global_variables_initializer())
        print('init weightes!')
        #若不存在，则训练数据
        sess.run(tf.global_variables_initializer())
        istrian = True
    if istrian:
        for i in range(50000):
            train_x, train_l = sampleImage()
            if i % 1000 == 0:
                #             print(hidden_output_)
                #             print(output_)
                cost_c,y_c,acc,sparse,reg=sess.run([cost,y_conv,accuracy,sparse_cost,regular], feed_dict={x: train_x, y_: train_l,keep_prob:1.0})
                print('time (%d)'%(int(i/1000)),cost_c,' accuracy is %.4f sparse is %.4f regular is %.4f'%(acc,sparse,reg))
                #保存参数
                save_path = saver.save(sess, "save/model.model")
            sess.run(train_step, feed_dict={x: train_x, y_: train_l,keep_prob:0.5})
    # np.save("weights1.npy", w_)
    # show_image(w_)
    #测试数据
    acc=sess.run(accuracy, feed_dict={x: te,y_:te_hot,keep_prob:1.0})
    print("test acc is %.4f"%(acc))
    # predata.savepredictcsv(passengerIdlist, pred, 'softmax_test')

if __name__ == '__main__':
    main()