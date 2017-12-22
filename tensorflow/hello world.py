import tensorflow as tf

hw = tf.constant("hello world")

with tf.Session() as sess:
    print(sess.run(hw))