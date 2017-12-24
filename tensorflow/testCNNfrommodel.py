import tensorflow as tf
import numpy as np
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('save/model.ckpt-100.meta')
    # saver.restore(sess, 'save/model.ckpt-100')
    saver.restore(sess, tf.train.latest_checkpoint('./save'))
    y = tf.get_collection('pred_network')[0]
    accuracy=tf.get_collection('accuracy')[0]
    graph = tf.get_default_graph()
    input_x = graph.get_operation_by_name('input_x').outputs[0]
    input_y=graph.get_operation_by_name('input_y').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    # 使用y进行预测
    feed={input_x: mnist.test.images[0:500], input_y: mnist.test.labels[0:500], keep_prob: 1.0}
    y_conv,acc=sess.run([y,accuracy], feed_dict=feed)
    print("acc:",acc)