import tensorflow as tf
import numpy as np
import input_data
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 参数同上，ksize是池化块的大小


x = tf.placeholder("float", shape=[None, 784],name='input_x')
y_ = tf.placeholder("float", shape=[None, 10],name='input_y')

# 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定，第二三参数代表图像尺寸，最后一个参数代表图像通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1') as scope:
    # 第一层卷积加池化
    w_conv1 = weight_variable([5, 5, 1, 32])  # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
with tf.name_scope('conv2') as scope:
    # 第二层卷积加池化
    w_conv2 = weight_variable([5, 5, 32, 64])  # 多通道卷积，卷积出64个特征
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张

with tf.name_scope('fc1') as scope:
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开，第一个参数为样本数量，-1未知
    #relu是和sigmoid差不多的非线性激活函数，但是比sigmoid好
    f_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
with tf.name_scope('softmax1') as scope:
    # dropout操作，减少过拟合
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    h_fc1_drop = tf.nn.dropout(f_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    #softmax多分类器
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
with tf.name_scope('arg') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 调用优化器优化
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.add_to_collection('pred_network', y_conv)
    tf.add_to_collection('accuracy',accuracy)
sess = tf.InteractiveSession()
# 创建saver
saver = tf.train.Saver()
if os.path.exists('save/model.model.meta'):  # 判断模型是否存在
    print('restore weightes form model!')
    saver.restore(sess, tf.train.latest_checkpoint('save'))  # 存在就从模型中恢复变量
else:
    #sess.run(tf.initialize_all_variables())
    print('init weightes!')
    sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if (i+1) % 100 == 0:
        #keep_prob表示神经元按概率失活，=1则表示跳过该步骤
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # 保存checkpoint, 同时也默认导出一个meta_graph
        # graph名为'my-model-{global_step}.meta'.
        # saver.save(sess, 'my-model', global_step=i+1)
        save_path = saver.save(sess, "save/model.model")
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(
"test accuracy %g" % accuracy.eval(
    feed_dict={x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))
